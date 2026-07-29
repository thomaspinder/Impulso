"""Identification schemes for structural VAR analysis."""

import weakref
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field, PrivateAttr, model_validator

from impulso._base import ImpulsoBaseModel, ImpulsoModel
from impulso._linalg import lag_matrices
from impulso._ma import compute_ma_phi

if TYPE_CHECKING:
    from impulso.data import VARData

# Arbitrary (non-recursive) long-run zero patterns need the Arias-Rubio-
# Ramirez-Waggoner machinery; `LongRunRestriction` points users here.
_NONRECURSIVE_ISSUE = "https://github.com/thomaspinder/Impulso/issues/144"


def _companion_spectral_radius(A: list[np.ndarray]) -> np.ndarray:
    """Largest absolute eigenvalue of the VAR companion matrix, per draw.

    The companion form stacks the lag matrices in its top block-row and
    carries an identity subdiagonal. Its spectral radius is below one
    exactly when the VAR is stable, i.e. when the moving-average sum
    `sum_h Phi_h` converges.

    Args:
        A: Lag coefficient matrices `A_1, ..., A_p` in lag order, each of
            shape `(..., n, n)` with shared leading batch axes.

    Returns:
        Spectral radii with the shared leading batch shape, e.g.
        `(chains, draws)`.
    """
    n = A[0].shape[-1]
    n_lags = len(A)
    leading = A[0].shape[:-2]
    companion = np.zeros((*leading, n * n_lags, n * n_lags))
    companion[..., :n, :] = np.concatenate(A, axis=-1)
    if n_lags > 1:
        sub = np.arange(n * (n_lags - 1))
        companion[..., n + sub, sub] = 1.0
    return np.abs(np.linalg.eigvals(companion)).max(axis=-1)


_CACHE_MISS: Final = object()
"""Sentinel returned by `_PosteriorCache.get` when there is no valid entry.

A dedicated sentinel (rather than `None`) keeps `None` usable as a cached
value, and makes the hit test at call sites an unambiguous `is` check.
"""


class _PosteriorCache:
    """Single-slot memo keyed on object identity, validated by weak references.

    Several identification schemes memoise a quantity that depends only on
    the posterior (plus a few scalars) so that a per-period identification
    loop — `IdentifiedVAR._identify_per_t` calls `identify()` once per time
    index with the same posterior — pays the expensive part once instead of
    `T` times.

    Keying such a memo on `id(posterior)` is unsafe: once the posterior is
    garbage collected its address can be reused by a new object, and a
    subsequent lookup with a *different* posterior that happens to land on
    the recycled address returns a stale value silently (issue #203). This
    cache stores `weakref.ref(owner)` instead and treats a dead referent —
    or a live referent that is not the object being looked up — as a miss.
    The referent identity check (`ref() is owner`) subsumes an `id()`
    comparison exactly, so no `id()` is kept in the key.

    Usage is a get/compute/set triple:

        cached = self._cache.get(posterior, (n_lags,))
        if cached is _CACHE_MISS:
            cached = expensive(posterior, n_lags)
            self._cache.set(posterior, (n_lags,), cached)

    `owners` may be a single object or a tuple of objects when validity
    depends on more than one identity (`ProxySVAR` keys on the posterior
    *and* the `VARData`). Every owner must support weak references —
    `xr.Dataset` and Pydantic models such as `VARData` both do. If any
    owner does not (plain tuples, ints and strings do not), `set` declines
    to cache rather than falling back to an unsafe key, so the only cost is
    a lost speed-up.

    Entries in `key` are ordinary scalars compared with `==` (lag order, a
    variable name, a horizon). Do not put arrays there — elementwise
    comparison would not yield a bool.

    Intended adoption (issue #203). The other identity-keyed memos in this
    module collapse to the same three lines once their branches merge:

        MaxShare._spectral_cache:
            self._spectral_cache.get(posterior, (n_lags, target))
            self._spectral_cache.set(posterior, (n_lags, target), value)

        LongRunRestriction._lr_cache:
            self._lr_cache.get(posterior, (n_lags,))
            self._lr_cache.set(posterior, (n_lags,), value)

    Declare the attribute on the (frozen) scheme with
    `PrivateAttr(default_factory=_PosteriorCache)` so each instance gets
    its own slot, and mutate it in place — no `object.__setattr__` needed.
    """

    __slots__ = ("_key", "_refs", "_value")

    def __init__(self) -> None:
        self._refs: tuple[weakref.ref, ...] | None = None
        self._key: tuple[Any, ...] = ()
        self._value: Any = _CACHE_MISS

    @staticmethod
    def _as_tuple(owners: Any) -> tuple[Any, ...]:
        """Normalise a single owner or a tuple of owners to a tuple."""
        return owners if isinstance(owners, tuple) else (owners,)

    def get(self, owners: Any, key: tuple[Any, ...] = ()) -> Any:
        """Look up the memoised value.

        Args:
            owners: The object (or tuple of objects) whose identity the
                cached value is tied to.
            key: Scalar key tail — everything the value depends on that is
                not an owner identity.

        Returns:
            The cached value, or `_CACHE_MISS` if there is no entry, the
            key tail differs, the owners differ, or any owner has been
            garbage collected.
        """
        if self._refs is None or self._key != key:
            return _CACHE_MISS
        owner_tuple = self._as_tuple(owners)
        if len(owner_tuple) != len(self._refs):
            return _CACHE_MISS
        # A dead referent dereferences to None, which can never be an
        # owner (weakref.ref(None) is not constructible), so the same
        # check covers both "collected" and "different object".
        if any(ref() is not owner for ref, owner in zip(self._refs, owner_tuple, strict=True)):
            return _CACHE_MISS
        return self._value

    def set(self, owners: Any, key: tuple[Any, ...], value: Any) -> None:
        """Store `value`, tying its validity to the owners staying alive.

        Args:
            owners: The object (or tuple of objects) whose identity the
                value depends on.
            key: Scalar key tail.
            value: The value to memoise.
        """
        try:
            refs = tuple(weakref.ref(owner) for owner in self._as_tuple(owners))
        except TypeError:
            # An owner that cannot be weakly referenced has no validity
            # token, so caching it would reintroduce the id()-reuse bug.
            self.clear()
            return
        self._refs = refs
        self._key = key
        self._value = value

    def clear(self) -> None:
        """Drop any stored entry."""
        self._refs = None
        self._key = ()
        self._value = _CACHE_MISS


class Cholesky(ImpulsoModel):
    """Cholesky identification scheme.

    Uses the lower-triangular Cholesky decomposition of the residual
    covariance matrix to identify structural shocks. Variable ordering
    determines the causal ordering.

    Attributes:
        ordering: Ordered list of variable names (most exogenous first).
    """

    ordering: list[str]

    def identify(
        self,
        L: np.ndarray,
        var_names: list[str],
        posterior: "xr.Dataset | None" = None,
        data: "VARData | None" = None,
        n_lags: int | None = None,
    ) -> np.ndarray:
        """Apply Cholesky identification.

        When `self.ordering` matches `var_names` this is a no-op and `L` is
        returned unchanged. Otherwise the factor is re-derived so that it is
        lower-triangular in the requested causal ordering, then written back
        into the data's row order.

        Concretely, with `Pi` the permutation sending data order to
        `self.ordering`, the ordered factor is the LQ factor of `Pi @ L`:
        `qr((Pi @ L).T) = Q @ R` gives `G = R.T` lower-triangular with
        `G @ G.T = Pi @ Sigma @ Pi.T`. Columns are sign-fixed so `G` has a
        positive diagonal, matching the textbook `cholesky(Pi Sigma Pi.T)`.
        `Sigma` is never formed, so the conditioning of the decomposition is
        not squared.

        Args:
            L: Lower-triangular Cholesky factor, shape (..., n_vars, n_vars).
            var_names: Variable names in the data's natural order.
            posterior: Unused. Accepted for Protocol uniformity.
            data: Unused. Accepted for Protocol uniformity.
            n_lags: Unused. Accepted for Protocol uniformity.

        Returns:
            Structural shock matrix, same shape as `L`. Rows follow
            `var_names` (the data's order, matching the `response`
            coordinate downstream); columns follow `shock_coords`, i.e.
            `self.ordering`. Triangularity therefore holds in the *ordering*
            row coordinates: permuting the rows by `self.ordering` recovers
            an exactly lower-triangular factor with a positive diagonal.
        """
        del posterior, data, n_lags  # unused

        # Fast path: ordering matches data — identify is a no-op.
        if list(self.ordering) == list(var_names):
            return L

        perm = np.array([var_names.index(v) for v in self.ordering])
        inv = np.argsort(perm)

        # (Pi L)(Pi L).T = Pi Sigma Pi.T, so any lower-triangular factor of
        # Pi L is a Cholesky factor of the permuted covariance.
        L_ord = L[..., perm, :]
        _, R = np.linalg.qr(np.swapaxes(L_ord, -1, -2))
        signs = np.sign(np.diagonal(R, axis1=-2, axis2=-1))
        signs = np.where(signs == 0.0, 1.0, signs)
        P_ord = np.swapaxes(R, -1, -2) * signs[..., np.newaxis, :]

        return P_ord[..., inv, :]  # back to data row order

    def shock_coords(self, n_vars: int) -> list[str]:
        """Cholesky shock labels are simply the causal ordering."""
        del n_vars  # ordering already has the right length
        return list(self.ordering)


class SignRestriction(ImpulsoModel):
    """Sign restriction identification scheme.

    Uses random rotation matrices to find structural impact matrices
    satisfying sign restrictions on impulse responses.

    Attributes:
        restrictions: Dict mapping variable -> {shock_name: "+" or "-"}.
        n_rotations: Number of candidate rotations per draw.
        random_seed: Seed for reproducibility.
    """

    restrictions: dict[str, dict[str, str]]
    n_rotations: int = Field(default=1000, ge=1)
    restriction_horizon: int = Field(default=0, ge=0)
    random_seed: int | None = None

    # Rotation-sampling schemes cannot yet pin one rotation per posterior
    # draw across forecast steps; forecast-side scenario machinery checks
    # this capability flag and refuses time-varying volatility.
    _samples_rotations: ClassVar[bool] = True

    # Single-call scratchpad: identify() writes the rate; the pipeline
    # (IdentifiedVAR.shock_matrix) reads it immediately afterwards and
    # attaches it to the shock-matrix DataArray's attrs. Not reentrant —
    # overwritten on each identify() call. Do not rely on this between
    # calls; the surviving public surface is the shock-matrix attr.
    _last_acceptance_rate: float = PrivateAttr(default=0.0)

    def identify(
        self,
        L: np.ndarray,
        var_names: list[str],
        posterior: "xr.Dataset | None" = None,
        data: "VARData | None" = None,
        n_lags: int | None = None,
    ) -> np.ndarray:
        """Apply sign-restriction identification.

        Args:
            L: Lower-triangular Cholesky factor, shape (chains, draws, n_vars, n_vars).
            var_names: Variable names in the data's natural order.
            posterior: Required when `self.restriction_horizon > 0` because
                the multi-horizon check needs the VAR coefficients `B` from
                the posterior. Ignored for impact-only restrictions
                (`restriction_horizon == 0`).
            data: Unused. Accepted for Protocol uniformity.
            n_lags: Unused. Accepted for Protocol uniformity.

        Returns:
            Structural shock matrix, shape (chains, draws, n_vars, n_vars).
            Per-draw fallback to the supplied `L` for draws where no
            rotation satisfies the restrictions. The acceptance rate is
            available via the `sign_restriction_acceptance_rate` attr on
            `IdentifiedVAR.shock_matrix()` (attached by the pipeline).
        """
        del data, n_lags  # unused
        from scipy.stats import special_ortho_group

        n_chains, n_draws, n_vars, _ = L.shape
        rng = np.random.default_rng(self.random_seed)

        shock_names = list(next(iter(self.restrictions.values())).keys())

        # Multi-horizon path needs B — fail clearly if posterior wasn't provided.
        B_all: np.ndarray | None = None
        n_lags = 0
        if self.restriction_horizon > 0:
            if posterior is None or "B" not in posterior:
                raise ValueError(
                    "restriction_horizon > 0 requires the full posterior with 'B' "
                    "(VAR coefficients). Pass posterior=fitted.idata.posterior to identify()."
                )
            B_all = posterior["B"].values
            n_lags = B_all.shape[-1] // n_vars

        P = np.full((n_chains, n_draws, n_vars, n_vars), np.nan)
        accepted_count = 0
        total_count = n_chains * n_draws
        for c in range(n_chains):
            for d in range(n_draws):
                chol = L[c, d]
                found = False
                B_draw = B_all[c, d] if B_all is not None else None
                for _ in range(self.n_rotations):
                    Q = special_ortho_group.rvs(n_vars, random_state=rng)
                    candidate = chol @ Q
                    if self.restriction_horizon == 0:
                        ok = self._check_restrictions(candidate, var_names, shock_names)
                    else:
                        ok = self._check_restrictions_at_horizons(candidate, B_draw, var_names, shock_names, n_lags)
                    if ok:
                        P[c, d] = candidate
                        found = True
                        accepted_count += 1
                        break
                if not found:
                    P[c, d] = chol  # Fallback to the unrotated factor.

        fallback_count = total_count - accepted_count
        if fallback_count > 0:
            import warnings

            warnings.warn(
                f"Sign restrictions not satisfied for {fallback_count}/{total_count} draws "
                f"({fallback_count / total_count:.1%}). Those draws fell back to L (Cholesky).",
                stacklevel=2,
            )

        # Stash the acceptance rate as a side channel — the pipeline reads
        # it back to attach to the InferenceData.attrs.
        self._last_acceptance_rate = accepted_count / total_count
        return P

    @staticmethod
    def _build_shock_coords(shock_names: list[str], n_vars: int) -> list[str]:
        """Build shock coordinate labels for the structural shock matrix.

        Named shocks occupy their column positions; remaining columns
        are labeled 'unidentified_1', 'unidentified_2', etc.
        """
        if len(shock_names) == n_vars:
            return shock_names
        return shock_names + [f"unidentified_{i}" for i in range(1, n_vars - len(shock_names) + 1)]

    def shock_coords(self, n_vars: int) -> list[str]:
        """Sign-restriction shock labels: named shocks first, then padding."""
        shock_names = list(next(iter(self.restrictions.values())).keys())
        return self._build_shock_coords(shock_names, n_vars)

    def _check_restrictions_at_horizons(
        self,
        candidate: np.ndarray,
        B_draw: np.ndarray,
        var_names: list[str],
        shock_names: list[str],
        n_lags: int,
    ) -> bool:
        """Check sign restrictions at all horizons 0..restriction_horizon.

        Args:
            candidate: Candidate structural impact matrix (n_vars, n_vars).
            B_draw: VAR coefficient matrix (n_vars, n_vars * n_lags) for this draw.
            var_names: Variable names.
            shock_names: Shock names from restrictions.
            n_lags: Number of lags in the VAR.

        Returns:
            True if all restrictions satisfied at all horizons.
        """
        # Always check impact (h=0)
        if not self._check_restrictions(candidate, var_names, shock_names):
            return False

        A = lag_matrices(B_draw, n_lags)
        Phi = compute_ma_phi(A, self.restriction_horizon)  # (H+1, n, n)

        # Phi[0] (= I) handles the impact check above; iterate h=1..H here.
        for h in range(1, self.restriction_horizon + 1):
            irf_h = Phi[h] @ candidate
            if not self._check_restrictions(irf_h, var_names, shock_names):
                return False

        return True

    def _check_restrictions(self, candidate: np.ndarray, var_names: list[str], shock_names: list[str]) -> bool:
        """Check if a candidate matrix satisfies all sign restrictions."""
        for var_name, shocks in self.restrictions.items():
            var_idx = var_names.index(var_name)
            for shock_name, sign in shocks.items():
                shock_idx = shock_names.index(shock_name)
                val = candidate[var_idx, shock_idx]
                if sign == "+" and val < 0:
                    return False
                if sign == "-" and val > 0:
                    return False
        return True


class LongRunRestriction(ImpulsoModel):
    """Long-run (cumulative) zero restrictions, Blanchard-Quah style.

    Cholesky and sign restrictions constrain the *impact* matrix
    `Theta(0) = P`. This scheme constrains the *cumulative* matrix
    `Theta(1)`: the total effect of each structural shock on the level of
    each variable, summed over all horizons. For a stable reduced-form VAR
    with moving-average coefficients `Phi_h`, the long-run multiplier is

        C(1) = sum_h Phi_h = (I - sum_j A_j)^-1 = M^-1,   M = I - sum_j A_j,

    and the cumulative structural impact is `Theta(1) = C(1) P`. The
    restriction imposed here is that `Theta(1)` is lower-triangular in the
    coordinates given by `ordering`: shock `j` has no long-run effect on
    the variables ordered before it. In the Blanchard-Quah (1989) two-
    variable case — `ordering=["output_growth", "unemployment"]`,
    `shock_names=["supply", "demand"]` — the single zero says the demand
    shock has no permanent effect on the level of output.

    Construction is closed-form. `Theta(1) Theta(1)' = C(1) Sigma C(1)'`,
    so the lower-triangular positive-diagonal `Theta(1)` is the Cholesky
    factor of that matrix and `P = M Theta(1)`. Internally the equivalent
    QR route is used: with `L L' = Sigma`, factor `(M^-1 L)' = Q R` and set
    `P = L Q` (sign-fixed so `diag(R) > 0`). Because `Q` is orthogonal,
    `P P' = Sigma` holds to machine precision, and the conditioning of
    `C(1) Sigma C(1)'` is never squared. It also shows what the scheme is:
    a rotation of the Cholesky factor, chosen in closed form rather than
    searched for.

    Sign convention: the diagonal of `Theta(1)` is positive, so shock `j`
    raises variable `j`'s long-run level. The diagonal of `P` itself may
    be negative — the normalisation is on the cumulative matrix, not on
    impact.

    Restriction count: triangularity is `n(n-1)/2` restrictions, exactly
    the number needed for point identification. For `n = 2` that is the
    single restriction users usually want. For larger `n` it asserts every
    zero above the diagonal, which is a strong joint claim. Arbitrary
    (non-recursive) long-run zero patterns are out of scope, as are
    partial long-run identification and mixed short-run/long-run schemes
    (Gali 1999).

    Two failure modes are reported separately:

    - `M` near-singular (`C(1)` numerically undefined). Controlled by
      `on_undefined` and `max_condition`.
    - Explosive draws (companion spectral radius above one). `M` may be
      perfectly well conditioned while `sum_h Phi_h` diverges, so the
      arithmetic succeeds but the *interpretation* does not. Those draws
      are always returned finite and always warned about.

    Attributes:
        ordering: Variable names ordered most long-run-restricted first,
            mirroring `Cholesky.ordering`. The returned matrix keeps its
            rows in the data's own variable order; `ordering` only defines
            the coordinates in which `Theta(1)` is triangular.
        shock_names: Labels for the shock columns, in the same order as
            `ordering`. `None` (default) reuses `ordering` as the labels.
            Explicit naming is strongly preferred — the whole point of the
            scheme is that the columns mean something.
        on_undefined: `"nan"` (default) blanks draws whose `M` is
            numerically singular and warns; `"raise"` errors instead.
            NaN draws propagate into IRF/FEVD and are rejected outright by
            the scenario methods, so `"raise"` catches the problem early.
        max_condition: Condition-number threshold above which `M` counts
            as numerically singular. Default `1e8`.

    Note:
        `C(1)` is the long-run multiplier on the levels of the *modelled*
        variables. "No permanent effect on output" therefore requires
        output to enter the VAR as a growth rate; the library cannot check
        this for you.
    """

    ordering: list[str]
    shock_names: list[str] | None = None
    on_undefined: Literal["nan", "raise"] = "nan"
    max_condition: float = Field(default=1e8, gt=0.0)

    # Single-call scratchpad, mirroring ProxySVAR._last_diagnostics:
    # _screen() writes the long-run diagnostics; the pipeline reads them
    # back and attaches them to the shock-matrix attrs.
    _last_diagnostics: dict[str, float] = PrivateAttr(default_factory=dict)

    # Memoised screen. `M`, its conditioning and the spectral radii depend
    # only on (posterior, n_lags) — not on L — so under time-varying
    # volatility, where the pipeline calls identify() once per period with
    # the same posterior, the eigendecomposition runs once instead of T
    # times (and the warnings fire once, not T times). Keyed by object
    # identity: valid while the caller holds the same posterior object,
    # which is exactly the per-t loop's lifetime.
    _lr_cache: tuple | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_names(self) -> "LongRunRestriction":
        """Reject empty/duplicated orderings and unusable shock labels."""
        if not self.ordering:
            raise ValueError("ordering must name at least one variable")
        if len(set(self.ordering)) != len(self.ordering):
            raise ValueError(f"ordering contains duplicate variable names: {self.ordering}")
        if self.shock_names is None:
            return self
        if len(self.shock_names) != len(self.ordering):
            raise ValueError(
                f"shock_names and ordering must have the same length; got "
                f"{len(self.shock_names)} shock names for {len(self.ordering)} variables."
            )
        if len(set(self.shock_names)) != len(self.shock_names):
            raise ValueError(f"shock_names contains duplicate names: {self.shock_names}")
        reserved = [s for s in self.shock_names if s.startswith("unidentified_")]
        if reserved:
            raise ValueError(
                f"Shock names may not start with the reserved prefix 'unidentified_' (got {reserved}). "
                "Downstream guards read that prefix to mask rotation-arbitrary columns, so such a "
                "name would silently hide an identified shock from FEVD and historical decomposition."
            )
        return self

    @classmethod
    def from_zero_restrictions(
        cls,
        restrictions: dict[str, list[str]],
        var_names: list[str],
        shock_names: list[str],
        **kwargs,
    ) -> "LongRunRestriction":
        """Build the scheme from named long-run zeros instead of an ordering.

        Each entry maps a variable to the shocks that must have *no*
        long-run effect on it. The pattern must be triangular under some
        ordering of the variables — that is what a recursive long-run
        scheme means — and this constructor recovers that ordering.

        Args:
            restrictions: Variable name -> list of shock names with zero
                long-run effect on it. Variables with no zeros may be
                omitted.
            var_names: All endogenous variable names.
            shock_names: All shock names, ordered most long-run-restricted
                first (the shock that is zero-restricted nowhere comes
                first, and so on).
            **kwargs: Forwarded to the constructor (`on_undefined`,
                `max_condition`).

        Returns:
            A `LongRunRestriction` whose `ordering` reproduces the named
            pattern.

        Raises:
            ValueError: If a name is unknown, or the pattern is not
                triangular under any ordering.
        """
        n = len(var_names)
        if len(shock_names) != n:
            raise ValueError(f"shock_names must name {n} shocks, one per variable; got {len(shock_names)}.")
        unknown_vars = [v for v in restrictions if v not in var_names]
        if unknown_vars:
            raise ValueError(f"restrictions name variables absent from var_names: {unknown_vars}")
        named = {s for shocks in restrictions.values() for s in shocks}
        unknown_shocks = sorted(named - set(shock_names))
        if unknown_shocks:
            raise ValueError(f"restrictions name shocks absent from shock_names: {unknown_shocks}")

        counts = {v: len(set(restrictions.get(v, []))) for v in var_names}
        if sorted(counts.values()) != list(range(n)):
            raise ValueError(
                f"A recursive long-run structure needs one variable restricted by each of "
                f"{list(range(n))} shocks; got restriction counts {counts}. Arbitrary "
                f"(non-recursive) long-run zero patterns are not supported — see {_NONRECURSIVE_ISSUE}."
            )
        ordering = sorted(var_names, key=lambda v: -counts[v])
        for i, variable in enumerate(ordering):
            expected = set(shock_names[i + 1 :])
            got = set(restrictions.get(variable, []))
            if got != expected:
                raise ValueError(
                    f"Variable {variable!r} is restricted by {sorted(got)}, but its restriction "
                    f"count places it at position {i} of the ordering, where a recursive structure "
                    f"requires exactly {sorted(expected)}. Only recursive (triangular) long-run "
                    f"patterns are supported — see {_NONRECURSIVE_ISSUE}."
                )
        return cls(ordering=ordering, shock_names=list(shock_names), **kwargs)

    def identify(
        self,
        L: np.ndarray,
        var_names: list[str],
        posterior: "xr.Dataset | None" = None,
        data: "VARData | None" = None,
        n_lags: int | None = None,
    ) -> np.ndarray:
        """Apply long-run-restriction identification.

        Args:
            L: Lower-triangular Cholesky factor, shape (chains, draws, n_vars, n_vars).
            var_names: Variable names in the data's natural order.
            posterior: Full posterior; required, because the long-run
                multiplier is built from the lag coefficients `B`.
            data: Unused. Accepted for Protocol uniformity.
            n_lags: Lag order. Inferred from `B`'s trailing axis if omitted.

        Returns:
            Structural shock matrix, shape (chains, draws, n_vars, n_vars).
            Rows follow `var_names` (the data's order); columns follow
            `shock_coords`. Draws whose long-run multiplier is numerically
            undefined are NaN when `on_undefined="nan"`.

        Raises:
            ValueError: If `posterior` is missing or carries no `B`, if
                `ordering` does not match `var_names`, or if
                `on_undefined="raise"` and some draw is undefined.
        """
        del data  # unused
        if posterior is None or "B" not in posterior:
            raise ValueError(
                "LongRunRestriction.identify requires the full posterior with 'B' (the VAR lag "
                "coefficients): the long-run multiplier C(1) = (I - sum_j A_j)^-1 is built from "
                "them. Pass posterior=fitted.idata.posterior to identify() — "
                "FittedVAR.set_identification_strategy(...) supplies it automatically."
            )
        n_vars = L.shape[-1]
        if n_lags is None:
            n_lags = posterior["B"].shape[-1] // n_vars

        perm = self._permutation(var_names)
        inv = np.argsort(perm)
        M, bad = self._screen(posterior, n_lags, n_vars)

        # Work in `ordering` coordinates: permute M on both axes, and L on
        # its rows only (L_ord L_ord' is then the permuted Sigma).
        M_ord = M[..., perm, :][..., :, perm]
        L_ord = L[..., perm, :]
        # Batched solve raises for the whole batch if any slice is
        # singular, so sanitise first and blank the offenders afterwards.
        M_safe = np.where(bad[..., np.newaxis, np.newaxis], np.eye(n_vars), M_ord)

        K = np.linalg.solve(M_safe, L_ord)  # C(1) L, in ordering coords
        Q, R = np.linalg.qr(np.swapaxes(K, -1, -2))
        # numpy does not guarantee diag(R) > 0; fixing the signs pins the
        # sign convention (positive diagonal of Theta(1)) and makes
        # identify() deterministic.
        signs = np.sign(np.diagonal(R, axis1=-2, axis2=-1))
        signs = np.where(signs == 0.0, 1.0, signs)
        P = (L_ord @ (Q * signs[..., np.newaxis, :]))[..., inv, :]

        if bad.any():
            P = np.where(bad[..., np.newaxis, np.newaxis], np.nan, P)
        return P

    def _permutation(self, var_names: list[str]) -> np.ndarray:
        """Positions of `ordering` within `var_names`."""
        missing = [v for v in self.ordering if v not in var_names]
        if missing:
            raise ValueError(f"ordering names variables absent from the data: {missing}. Data has {list(var_names)}.")
        if len(self.ordering) != len(var_names):
            raise ValueError(
                f"ordering must cover every variable: got {len(self.ordering)} names "
                f"for {len(var_names)} variables ({list(var_names)})."
            )
        return np.array([var_names.index(v) for v in self.ordering])

    def _screen(self, posterior: "xr.Dataset", n_lags: int, n_vars: int) -> tuple[np.ndarray, np.ndarray]:
        """Build `M = I - sum_j A_j` and flag the draws where `C(1)` is undefined.

        Memoised on `(id(posterior), n_lags)` — see `_lr_cache`.

        Returns:
            Tuple `(M, bad)`: the long-run matrix, shape
            `(chains, draws, n_vars, n_vars)`, and a boolean mask of draws
            whose condition number exceeds `max_condition`.
        """
        cache_key = (id(posterior), n_lags)
        if self._lr_cache is not None and self._lr_cache[0] == cache_key:
            return self._lr_cache[1]

        A = lag_matrices(posterior["B"].values, n_lags)
        M = np.eye(n_vars) - np.sum(A, axis=0)
        # cond() returns inf for a singular matrix rather than raising,
        # which is what makes the sanitise-then-blank strategy possible.
        condition = np.linalg.cond(M)
        bad = ~np.isfinite(condition) | (condition > self.max_condition)
        rho = _companion_spectral_radius(A)
        explosive = rho > 1.0

        self._last_diagnostics = {
            "long_run_singular_draws": float(bad.sum()),
            "long_run_singular_fraction": float(bad.mean()),
            "long_run_condition_median": float(np.median(condition)),
            "long_run_condition_q95": float(np.quantile(condition, 0.95)),
            "long_run_condition_max": float(condition.max()),
            "long_run_explosive_draws": float(explosive.sum()),
            "long_run_explosive_fraction": float(explosive.mean()),
            "long_run_spectral_radius_median": float(np.median(rho)),
            "long_run_spectral_radius_max": float(rho.max()),
        }
        self._report(bad, explosive)
        self._lr_cache = (cache_key, (M, bad))
        return M, bad

    def _report(self, bad: np.ndarray, explosive: np.ndarray) -> None:
        """Warn (or raise) about undefined and explosive draws.

        The two conditions are distinct. A singular `M` is an arithmetic
        failure — `C(1)` does not exist. An explosive draw is an
        interpretation failure — the arithmetic is fine but the cumulative
        sum diverges, so "long-run effect" means nothing. Bayesian
        posteriors near a unit root routinely contain explosive draws, so
        those are reported, never blanked.
        """
        import warnings

        total = int(bad.size)
        n_bad = int(bad.sum())
        if n_bad:
            if self.on_undefined == "raise":
                raise ValueError(
                    f"The long-run multiplier C(1) = (I - sum_j A_j)^-1 is numerically undefined "
                    f"for {n_bad}/{total} posterior draws (condition number above "
                    f"max_condition={self.max_condition:g}). Pass on_undefined='nan' to blank "
                    "those draws instead, or raise max_condition if the loss of precision is "
                    "acceptable."
                )
            warnings.warn(
                f"The long-run multiplier C(1) = (I - sum_j A_j)^-1 is numerically undefined for "
                f"{n_bad}/{total} posterior draws (condition number above "
                f"max_condition={self.max_condition:g}); those draws are returned as NaN. NaN "
                "draws propagate into IRF and FEVD, and the scenario methods reject them.",
                UserWarning,
                stacklevel=4,
            )
        n_explosive = int(explosive.sum())
        if n_explosive:
            warnings.warn(
                f"{n_explosive}/{total} posterior draws are explosive (companion spectral radius "
                "above 1), so the cumulative moving-average sum does not converge and the long-run "
                "restriction has no interpretation for them. They are still identified "
                "arithmetically and returned finite — check long_run_diagnostics() before reading "
                "the results.",
                UserWarning,
                stacklevel=4,
            )

    def long_run_diagnostics(self, posterior: "xr.Dataset", n_lags: int | None = None) -> dict[str, np.ndarray]:
        """Per-draw conditioning and stability of the long-run multiplier.

        Args:
            posterior: Posterior Dataset carrying `B` (`fitted.idata.posterior`).
            n_lags: Lag order. Inferred from `B`'s trailing axis if omitted.

        Returns:
            Dict with `"condition"` — the condition number of
            `M = I - sum_j A_j`, shape `(chains, draws)` — and
            `"spectral_radius"`, the companion spectral radius, same shape.
            A large condition number means `C(1)` is barely defined; a
            spectral radius above one means the long-run sum diverges.
        """
        n_vars = posterior["B"].shape[-2]
        if n_lags is None:
            n_lags = posterior["B"].shape[-1] // n_vars
        A = lag_matrices(posterior["B"].values, n_lags)
        M = np.eye(n_vars) - np.sum(A, axis=0)
        return {
            "condition": np.linalg.cond(M),
            "spectral_radius": _companion_spectral_radius(A),
        }

    def shock_coords(self, n_vars: int) -> list[str]:
        """Explicit shock names if given, otherwise the ordering."""
        del n_vars  # ordering already has the right length
        return list(self.shock_names or self.ordering)


class ProxySVAR(ImpulsoBaseModel):
    """External-instrument (proxy) identification for one structural shock.

    Identifies a single structural shock from an external instrument `z_t`
    that is correlated with the target shock (relevance) and uncorrelated
    with all others (exogeneity). Under those conditions the covariance
    between the instrument and the reduced-form residuals is proportional
    to the target shock's impact column:
    `E[z_t u_t] = phi * p_1`.

    Per posterior draw, the impact column is estimated as the sample
    covariance between the (date-aligned) instrument and that draw's
    reconstructed residuals, normalised on `policy_variable`. The
    remaining columns are completed orthogonally, consistent with the
    draw's shock covariance, so downstream code that needs a full
    invertible matrix (historical decomposition) keeps working — but
    those columns are rotation-arbitrary and are labelled
    `unidentified_1..` accordingly. Downstream guard rails respond to
    that labelling: `IdentifiedVAR.fevd` masks the unidentified columns'
    shares to NaN, and `IdentifiedVAR.historical_decomposition` collapses
    them into a single `unidentified_remainder` column (their sum is
    well-defined even though the split is not).

    Attributes:
        instrument: Instrument series with a DatetimeIndex. Aligned to the
            estimation sample by date at identify() time (inner join —
            months missing from the instrument are dropped, matching the
            reindex-and-drop convention in the proxy-SVAR literature).
            Periods where no event occurred should be zero, not NaN.
        policy_variable: Endogenous variable used to normalise the shock.
        shock_name: Label of the identified shock column.
        scale: If None (default), the identified column is a one-standard-
            deviation shock, consistent with the draw's shock covariance
            (`P @ P.T = Sigma` holds exactly). If a float, the column is
            rescaled per draw so the shock moves `policy_variable` by
            `scale` units on impact (unit-effect normalisation, e.g.
            `scale=10.0` for a +10% impact on a log*100 variable); the
            matrix then no longer reproduces Sigma, which is inherent to
            unit-effect normalisation.
    """

    instrument: pd.Series
    policy_variable: str
    shock_name: str = "instrumented"
    scale: float | None = None

    # Single-call scratchpad, mirroring SignRestriction._last_acceptance_rate:
    # identify() writes first-stage diagnostics; the pipeline reads them
    # immediately afterwards and attaches to the shock matrix attrs.
    _last_diagnostics: dict[str, float] = PrivateAttr(default_factory=dict)

    # Memoised impact direction. The instrument-residual covariance (and
    # the first-stage diagnostics) depend only on (posterior, data, n_lags)
    # — not on L — so under time-varying volatility, where the pipeline
    # calls identify() once per period with the same posterior/data, the
    # expensive residual reconstruction runs once instead of T times.
    # Keyed by object identity, with weak references as the validity token:
    # valid while the caller holds the same posterior/data objects, which
    # is exactly the per-t loop's lifetime. See `_PosteriorCache`.
    _impact_cache: _PosteriorCache = PrivateAttr(default_factory=_PosteriorCache)

    def identify(
        self,
        L: np.ndarray,
        var_names: list[str],
        posterior: "xr.Dataset | None" = None,
        data: "VARData | None" = None,
        n_lags: int | None = None,
    ) -> np.ndarray:
        """Apply external-instrument identification.

        Args:
            L: Lower-triangular Cholesky factor, shape (chains, draws, n_vars, n_vars).
            var_names: Variable names in the data's natural order.
            posterior: Full posterior; required (residual reconstruction
                needs `B` and `intercept` draws).
            data: The VARData used at fit time; required for residual
                reconstruction and date alignment.
            n_lags: Lag order of the fitted VAR; required.

        Returns:
            Structural shock matrix, shape (chains, draws, n_vars, n_vars).
            Column 0 is the identified shock; columns 1.. are an arbitrary
            orthogonal completion.

        Raises:
            ValueError: If posterior/data/n_lags are missing, the policy
                variable is unknown, or the instrument does not overlap
                the estimation sample.
        """
        if posterior is None or data is None or n_lags is None:
            raise ValueError(
                "ProxySVAR.identify requires posterior, data, and n_lags — "
                "they are supplied automatically by "
                "FittedVAR.set_identification_strategy(...); pass them "
                "explicitly if calling identify() directly."
            )
        if self.policy_variable not in var_names:
            raise ValueError(f"policy_variable {self.policy_variable!r} not in var_names {var_names}")
        policy_idx = var_names.index(self.policy_variable)

        d = self._impact_cache.get((posterior, data), (n_lags,))
        if d is _CACHE_MISS:
            z, u = self._aligned_residuals(posterior, data, n_lags)

            # Impact direction: per-draw covariance between instrument and
            # residuals (both demeaned), normalised on the policy variable.
            z_c = z - z.mean()
            u_c = u - u.mean(axis=2, keepdims=True)
            s = np.einsum("t,cdti->cdi", z_c, u_c) / len(z)  # (C, D, n)
            d = s / s[:, :, policy_idx][:, :, np.newaxis]  # d[policy] = 1

            # First-stage strength: F of u_policy ~ const + z, per draw.
            f_draws = self._first_stage_f(z_c, u_c[:, :, :, policy_idx])
            f_median = float(np.median(f_draws))
            self._last_diagnostics = {
                "proxy_first_stage_f_median": f_median,
                "proxy_first_stage_f_q05": float(np.quantile(f_draws, 0.05)),
                "proxy_first_stage_f_q95": float(np.quantile(f_draws, 0.95)),
            }
            self._impact_cache.set((posterior, data), (n_lags,), d)
            if f_median < 10.0:
                import warnings

                warnings.warn(
                    f"Weak instrument: posterior-median first-stage F = {f_median:.2f} < 10. "
                    "The identified impact column is unreliable.",
                    UserWarning,
                    stacklevel=2,
                )

        # Complete the matrix: q1 = L^{-1} d normalised, extended to an
        # orthonormal basis via a Householder reflection; P = L @ Q gives
        # P @ P.T = Sigma with column 0 proportional to d (positive factor,
        # so the shock raises the policy variable by construction).
        n = len(var_names)
        v = np.linalg.solve(L, d[..., np.newaxis])[..., 0]  # (C, D, n)
        q1 = v / np.linalg.norm(v, axis=-1, keepdims=True)
        e1 = np.zeros(n)
        e1[0] = 1.0
        w = q1 - e1
        w_norm2 = np.einsum("cdi,cdi->cd", w, w)[..., np.newaxis, np.newaxis]
        outer = w[..., :, np.newaxis] * w[..., np.newaxis, :]
        eye = np.broadcast_to(np.eye(n), outer.shape)
        Q = np.where(w_norm2 > 1e-14, eye - 2.0 * outer / np.where(w_norm2 > 1e-14, w_norm2, 1.0), eye)
        P = L @ Q

        if self.scale is not None:
            # Unit-effect normalisation: the identified column moves the
            # policy variable by `scale` on impact, per draw.
            P = P.copy()
            P[..., 0] = d * self.scale
        return P

    def _aligned_residuals(
        self, posterior: "xr.Dataset", data: "VARData", n_lags: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct residuals and align the instrument to them by date.

        Returns:
            Tuple `(z, u)`: the instrument values on the overlap, shape
            `(T_z,)`, and the matching residual draws, `(C, D, T_z, n)`.

        Raises:
            ValueError: If the instrument index does not overlap the
                estimation sample, or the overlap is too short.
        """
        from impulso._residuals import reduced_form_residuals

        resid = reduced_form_residuals(posterior, data, n_lags)  # (C, D, T_eff, n)

        # Inner join on dates: months missing from the instrument are dropped.
        eff_index = data.index[n_lags:]
        common = eff_index.intersection(self.instrument.index)
        n_vars = resid.shape[-1]
        if len(common) == 0:
            raise ValueError(
                "Instrument index does not overlap the estimation sample "
                f"({eff_index[0]}..{eff_index[-1]}). Check the DatetimeIndex "
                "frequency and range."
            )
        if len(common) < 3 * n_vars:
            raise ValueError(
                f"Only {len(common)} instrument observations overlap the "
                "estimation sample — too few to identify the impact column."
            )
        positions = eff_index.get_indexer(common)
        z = self.instrument.loc[common].to_numpy(dtype=float)
        return z, resid[:, :, positions, :]

    def first_stage(self, posterior: "xr.Dataset", data: "VARData", n_lags: int) -> np.ndarray:
        """Posterior draws of the first-stage F statistic.

        Regresses the policy variable's reconstructed reduced-form
        residuals on the date-aligned instrument (with a constant), per
        posterior draw. Because the residuals differ draw by draw, the
        instrument-relevance F is itself a posterior quantity.

        Args:
            posterior: Posterior Dataset with `B` and `intercept` draws
                (`fitted.idata.posterior`).
            data: The VARData used at fit time.
            n_lags: Lag order of the fitted VAR.

        Returns:
            F statistics, shape `(chains, draws)`.
        """
        policy_idx = data.endog_names.index(self.policy_variable)
        z, u = self._aligned_residuals(posterior, data, n_lags)
        z_c = z - z.mean()
        u_c = u - u.mean(axis=2, keepdims=True)
        return self._first_stage_f(z_c, u_c[:, :, :, policy_idx])

    @staticmethod
    def _first_stage_f(z_c: np.ndarray, u_policy_c: np.ndarray) -> np.ndarray:
        """Per-draw F-stat of the first stage u_policy ~ const + z.

        Args:
            z_c: Demeaned instrument, shape (T_z,).
            u_policy_c: Demeaned policy-variable residuals, (C, D, T_z).

        Returns:
            F statistics, shape (C, D).
        """
        T = len(z_c)
        szz = z_c @ z_c
        szu = np.einsum("t,cdt->cd", z_c, u_policy_c)
        slope = szu / szz
        ess = slope**2 * szz  # explained sum of squares
        tss = np.einsum("cdt,cdt->cd", u_policy_c, u_policy_c)
        rss = tss - ess
        return ess / (rss / (T - 2))

    def shock_coords(self, n_vars: int) -> list[str]:
        """Identified shock first, then rotation-arbitrary padding."""
        return SignRestriction._build_shock_coords([self.shock_name], n_vars)
