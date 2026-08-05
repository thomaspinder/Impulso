"""Long-run (cumulative) zero-restriction identification."""

from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from pydantic import Field, PrivateAttr, model_validator

from impulso._base import ImpulsoModel
from impulso._linalg import lag_matrices
from impulso.identification._cache import _CACHE_MISS, _PosteriorCache

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

    # Single-call scratchpad backing `last_diagnostics`: _screen() writes
    # the long-run diagnostics; the pipeline reads them back and attaches
    # them to the shock-matrix attrs.
    _last_diagnostics: dict[str, float] = PrivateAttr(default_factory=dict)

    @property
    def last_diagnostics(self) -> dict[str, float]:
        """Diagnostics from the most recent `identify()` call.

        Scheme-prefixed scalars (see CONTEXT.md "Identification
        diagnostics"), overwritten per call and surfaced onto
        `IdentifiedVAR.shock_matrix().attrs` by the pipeline. Returns a copy.
        """
        return dict(self._last_diagnostics)

    # Memoised screen. `M`, its conditioning and the spectral radii depend
    # only on (posterior, n_lags) — not on L — so under time-varying
    # volatility, where the pipeline calls identify() once per period with
    # the same posterior, the eigendecomposition runs once instead of T
    # times (and the warnings fire once, not T times). Keyed by object
    # identity, with a weak reference as the validity token: valid while
    # the caller holds the same posterior object, which is exactly the
    # per-t loop's lifetime. See `_PosteriorCache`.
    _lr_cache: _PosteriorCache = PrivateAttr(default_factory=_PosteriorCache)

    @model_validator(mode="after")
    def _validate_names(self) -> "LongRunRestriction":
        """Reject empty/duplicated orderings and unusable shock labels."""
        if not self.ordering:
            raise ValueError("ordering must name at least one variable")
        if len(set(self.ordering)) != len(self.ordering):
            raise ValueError(f"ordering contains duplicate variable names: {self.ordering}")
        if self.shock_names is not None:
            if len(self.shock_names) != len(self.ordering):
                raise ValueError(
                    f"shock_names and ordering must have the same length; got "
                    f"{len(self.shock_names)} shock names for {len(self.ordering)} variables."
                )
            if len(set(self.shock_names)) != len(self.shock_names):
                raise ValueError(f"shock_names contains duplicate names: {self.shock_names}")
        # Check the EFFECTIVE labels: with shock_names=None, shock_coords falls
        # back to the ordering, so a variable named unidentified_* would leak
        # the reserved prefix into the shock labels through that path too.
        reserved = [s for s in (self.shock_names or self.ordering) if s.startswith("unidentified_")]
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
        ordering of the variables *and* some ordering of the shocks — that
        is what a recursive long-run scheme means — and this constructor
        recovers both from the restriction counts. Neither `var_names` nor
        `shock_names` need be given in that order; the names are labels,
        the pattern decides the positions.

        Args:
            restrictions: Variable name -> list of shock names with zero
                long-run effect on it. Variables with no zeros may be
                omitted.
            var_names: All endogenous variable names, in any order.
            shock_names: All shock names, in any order. Their order in the
                returned scheme is inferred from the pattern.
            **kwargs: Forwarded to the constructor (`on_undefined`,
                `max_condition`).

        Returns:
            A `LongRunRestriction` whose `ordering` and `shock_names`
            reproduce the named pattern.

        Raises:
            ValueError: If a name is unknown, or the pattern is not
                triangular under any pair of orderings.
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

        zeros = {v: set(restrictions.get(v, [])) for v in var_names}
        counts = {v: len(zeros[v]) for v in var_names}
        if sorted(counts.values()) != list(range(n)):
            raise ValueError(
                f"A recursive long-run structure needs one variable restricted by each of "
                f"{list(range(n))} shocks; got restriction counts {counts}. Arbitrary "
                f"(non-recursive) long-run zero patterns are not supported — see {_NONRECURSIVE_ISSUE}."
            )
        ordering = sorted(var_names, key=lambda v: -counts[v])
        # The shock order is the same count, read down the other axis. In a
        # triangular pattern the variable at position i is restricted by the
        # shocks at positions i+1..n-1, so the shock at position k is
        # restricted out of exactly the k variables at positions 0..k-1: a
        # shock's position is the number of restriction lists it appears in.
        # Ties mean no ordering makes the pattern triangular; they sort
        # stably and are rejected by the exact per-position check below.
        appearances = {s: sum(s in zeros[v] for v in var_names) for s in shock_names}
        shock_order = sorted(shock_names, key=lambda s: appearances[s])
        for i, variable in enumerate(ordering):
            expected = set(shock_order[i + 1 :])
            got = zeros[variable]
            if got != expected:
                raise ValueError(
                    f"Variable {variable!r} is restricted by {sorted(got)}, but its restriction "
                    f"count places it at position {i} of the ordering, where a recursive structure "
                    f"requires exactly {sorted(expected)}. Only recursive (triangular) long-run "
                    f"patterns are supported — see {_NONRECURSIVE_ISSUE}."
                )
        return cls(ordering=ordering, shock_names=shock_order, **kwargs)

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
                "them. Pass the fit's posterior group as an xarray.Dataset to identify() — "
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

        Memoised on the posterior's identity plus `n_lags` — see
        `_lr_cache` and `_PosteriorCache`.

        Returns:
            Tuple `(M, bad)`: the long-run matrix, shape
            `(chains, draws, n_vars, n_vars)`, and a boolean mask of draws
            whose condition number exceeds `max_condition`.
        """
        cached = self._lr_cache.get(posterior, (n_lags,))
        if cached is not _CACHE_MISS:
            self._last_diagnostics = {**self._last_diagnostics, "long_run_screen_cache_hit": 1.0}
            return cached

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
            "long_run_screen_cache_hit": 0.0,
        }
        self._report(bad, explosive)
        self._lr_cache.set(posterior, (n_lags,), (M, bad))
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
            posterior: Posterior Dataset carrying `B` (the fit's posterior group).
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
