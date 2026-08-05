"""Zero-and-sign restrictions (Arias, Rubio-Ramirez & Waggoner, 2018)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
import xarray as xr
from pydantic import Field, PrivateAttr, model_validator

from impulso._base import ImpulsoModel
from impulso._linalg import lag_matrices
from impulso._ma import compute_ma_phi
from impulso.identification._shared import pad_shock_coords

if TYPE_CHECKING:
    from impulso.data import VARData

#: Prefix reserved by the pipeline for rotation-arbitrary shock columns.
_RESERVED_SHOCK_PREFIX = "unidentified_"


def _signs_ok(values: np.ndarray, rows: np.ndarray, cols: np.ndarray, signs: np.ndarray) -> bool:
    """Check sign restrictions against a response matrix.

    Restrictions are supplied as flat index arrays so the check is one
    fancy-index plus one comparison, rather than a Python loop over the
    nested restriction dict.

    Args:
        values: Response matrix, shape `(n_vars, n_vars)`; rows are
            variables, columns are shocks (in data / user order).
        rows: Variable indices of the restricted cells.
        cols: Shock indices of the restricted cells.
        signs: `+1.0` for a `"+"` restriction, `-1.0` for `"-"`.

    Returns:
        True if every restricted cell has the required sign. Exact zeros
        pass, matching `SignRestriction._check_restrictions`.
    """
    if rows.size == 0:
        return True
    return bool(np.all(values[rows, cols] * signs >= 0.0))


def _signs_ok_at_horizons(
    P: np.ndarray,
    Phi: np.ndarray,
    horizon: int,
    rows: np.ndarray,
    cols: np.ndarray,
    signs: np.ndarray,
) -> bool:
    """Check sign restrictions at horizons `1..horizon`.

    Impact (`h = 0`) is *not* checked here: the recursive construction
    screens impact signs column by column while it builds `P`, so by the
    time this runs the impact restrictions already hold.

    Args:
        P: Candidate structural impact matrix, shape `(n_vars, n_vars)`.
        Phi: MA coefficient matrices for this draw, shape
            `(horizon + 1, n_vars, n_vars)`.
        horizon: Highest horizon to check.
        rows: Variable indices of the restricted cells.
        cols: Shock indices of the restricted cells.
        signs: `+1.0` for a `"+"` restriction, `-1.0` for `"-"`.

    Returns:
        True if all restrictions hold at every horizon `1..horizon`.
    """
    return all(_signs_ok(Phi[h] @ P, rows, cols, signs) for h in range(1, horizon + 1))


@dataclass(frozen=True)
class _ZeroSignLayout:
    """Draw-independent bookkeeping compiled once per `identify()` call.

    Two column orderings coexist and mixing them is the easiest way to get
    this wrong, so they are named explicitly here: *user order* is the order
    of `shock_coords` (what callers see), *construction order* is the
    zero-count-descending order the recursion requires.

    Attributes:
        labels: Effective shock labels, user order.
        order: `order[k]` is the user column built at construction step `k`.
        perm_back: Inverse of `order` — permutes construction-order columns
            back to user order.
        zero_rows: Zero-restricted variable rows per construction step.
        sign_rows: Variable indices of sign-restricted cells.
        sign_cols: Shock indices (user order) of sign-restricted cells.
        sign_vals: `+1.0` / `-1.0` targets matching `sign_rows`/`sign_cols`.
        col_rows: Impact-sign-restricted variable rows per construction step.
        col_signs: Matching sign targets per construction step.
        zero_idx_rows: Variable indices of zero-restricted cells (user order).
        zero_idx_cols: Shock indices of zero-restricted cells (user order).
    """

    labels: list[str]
    order: list[int]
    perm_back: np.ndarray
    zero_rows: list[np.ndarray]
    sign_rows: np.ndarray
    sign_cols: np.ndarray
    sign_vals: np.ndarray
    col_rows: list[np.ndarray]
    col_signs: list[np.ndarray]
    zero_idx_rows: np.ndarray
    zero_idx_cols: np.ndarray


class ZeroSignRestriction(ImpulsoModel):
    """Combined zero-and-sign restriction identification.

    Implements the recursive orthogonalisation of Arias,
    Rubio-Ramirez & Waggoner (2018). Writing the structural impact matrix
    as `P = L Q` with `Q` orthogonal, a zero restriction "variable `i`
    does not respond to shock `j` on impact" is the linear condition
    `e_i' L q_j = 0` on the `j`-th column of `Q`. Columns are built one at
    a time, each drawn uniformly from the unit sphere of the null space of

        R_k = [ Z_k L ; q_1' ; ... ; q_{k-1}' ]

    where `Z_k` selects the rows carrying zero restrictions on shock `k`.
    The null-space draw imposes the zeros *exactly* (to SVD precision) and
    orthogonality to the earlier columns by construction, so no rejection
    step is needed for the zeros — only the sign restrictions are checked
    by accept/reject.

    Shocks are ordered internally by their number of zero restrictions,
    descending (ties keep the order given in `shock_names`; unnamed padding
    columns go last), because the construction requires it. Rows of the
    returned matrix are always in data order and columns are permuted back
    to `shock_names` order, so the internal ordering is not observable.

    Attributes:
        shock_names: Structural shock labels, in the order the columns of
            the returned matrix should appear. May be shorter than the
            number of variables — remaining columns are labelled
            `unidentified_1`, ... and carry no restrictions.
        zero_restrictions: Dict mapping variable -> list of shocks that
            have zero impact on that variable. Keyed by variable for
            consistency with `sign_restrictions`.
        sign_restrictions: Dict mapping variable -> {shock: "+" or "-"},
            the same format `SignRestriction` uses.
        restriction_horizon: Sign restrictions are imposed at horizons
            `0..restriction_horizon`. Zero restrictions are always impact
            only (`h = 0`); long-run zeros are not supported.
        n_rotations: Maximum candidate draws per posterior draw.
        random_seed: Seed for reproducibility.
        on_failure: What to do for a posterior draw where no candidate
            satisfies the sign restrictions within `n_rotations` attempts.
            `"nan"` (default) fills that draw with NaN and warns once at
            the end; `"raise"` raises immediately.

    Note:
        Candidates are drawn *unweighted*: each accepted draw keeps the
        `Q` that the recursion produced, with no importance weight
        correcting for the volume element of the zero-restricted manifold.
        Arias, Rubio-Ramirez & Waggoner (2018) derive such a weight for
        their uniform-conditional prior over the identified set. The
        unweighted draws therefore do not represent that prior exactly
        when the restrictions leave a set (rather than a point) identified.
        Two regimes are unaffected: with no zero restrictions the draws are
        exactly Haar, and when the zeros exactly identify the system
        (`z_j = n - j` for every shock) the answer is a point up to column
        signs. See the explanation page for the full caveat.
    """

    shock_names: list[str]
    zero_restrictions: dict[str, list[str]] = Field(default_factory=dict)
    sign_restrictions: dict[str, dict[str, str]] = Field(default_factory=dict)
    restriction_horizon: int = Field(default=0, ge=0)
    n_rotations: int = Field(default=1000, ge=1)
    random_seed: int | None = None
    on_failure: Literal["nan", "raise"] = "nan"

    # Draws a fresh Q per identify() call, exactly as SignRestriction does —
    # forecast-side scenario machinery reads this flag and refuses
    # time-varying volatility for such schemes.
    _samples_rotations: ClassVar[bool] = True

    # Single-call scratchpad backing `last_diagnostics`: identify() writes,
    # IdentifiedVAR.shock_matrix reads it back immediately and attaches the
    # entries to the shock-matrix attrs. Not reentrant. Keys carry the
    # zero_sign_ prefix so they can never mislabel another scheme's
    # diagnostics (see CONTEXT.md "Scheme-prefixed diagnostic keys").
    _last_diagnostics: dict[str, float] = PrivateAttr(default_factory=dict)

    @property
    def last_diagnostics(self) -> dict[str, float]:
        """Diagnostics from the most recent `identify()` call.

        Scheme-prefixed scalars (see CONTEXT.md "Identification
        diagnostics"), overwritten per call and surfaced onto
        `IdentifiedVAR.shock_matrix().attrs` by the pipeline. Returns a copy.
        """
        return dict(self._last_diagnostics)

    @model_validator(mode="after")
    def _validate_restrictions(self) -> "ZeroSignRestriction":
        """Validate everything that does not depend on the number of variables.

        Variable names and the rank condition need `n_vars`, which is only
        known at `identify()` time; those are checked there.

        Returns:
            The validated instance.

        Raises:
            ValueError: On empty/duplicate shock names, reserved shock
                labels, unknown shocks, bad sign tokens, a cell restricted
                to zero *and* to a sign, or two empty restriction dicts.
        """
        import warnings

        self._validate_shock_names()
        self._validate_restriction_dicts()

        signed_shocks = {s for signed in self.sign_restrictions.values() for s in signed}
        unsigned = [s for s in self.shock_names if s not in signed_shocks]
        if unsigned:
            warnings.warn(
                f"Named shock(s) {unsigned} carry no sign restriction at any horizon, so their "
                "column sign is not identified: q and -q are both admissible and posterior "
                "summaries will mix the two directions. Add a sign restriction to pin the "
                "direction.",
                UserWarning,
                stacklevel=2,
            )
        return self

    def _validate_shock_names(self) -> None:
        """Check `shock_names` is a non-empty list of unreserved, unique labels.

        Raises:
            ValueError: On an empty list, duplicates, or a reserved prefix.
        """
        if not self.shock_names:
            raise ValueError("shock_names must name at least one structural shock.")

        duplicates = sorted({s for s in self.shock_names if self.shock_names.count(s) > 1})
        if duplicates:
            raise ValueError(f"Duplicate shock names in shock_names: {duplicates}.")

        reserved = [s for s in self.shock_names if s.startswith(_RESERVED_SHOCK_PREFIX)]
        if reserved:
            raise ValueError(
                f"Shock names {reserved} use the reserved prefix {_RESERVED_SHOCK_PREFIX!r}. "
                "The pipeline assigns that prefix to rotation-arbitrary columns under partial "
                "identification; pick a different label."
            )

    def _validate_restriction_dicts(self) -> None:
        """Check the two restriction dicts are non-empty, well-formed, and consistent.

        Raises:
            ValueError: If both dicts are empty, a shock is unknown, a sign
                token is not `"+"`/`"-"`, or a cell carries a zero and a
                sign at once.
        """
        if not self.zero_restrictions and not self.sign_restrictions:
            raise ValueError(
                "ZeroSignRestriction needs at least one of zero_restrictions or sign_restrictions. "
                "With neither, every orthogonal Q is admissible and nothing is identified."
            )

        known = set(self.shock_names)
        for variable, shocks in self.zero_restrictions.items():
            unknown = sorted(set(shocks) - known)
            if unknown:
                raise ValueError(
                    f"zero_restrictions[{variable!r}] references unknown shock(s) {unknown}. "
                    f"Known shocks: {self.shock_names}."
                )
            clash = sorted(set(shocks) & set(self.sign_restrictions.get(variable, {})))
            if clash:
                raise ValueError(
                    f"Restriction conflict on variable {variable!r}: shock(s) {clash} are "
                    "restricted to zero on impact and simultaneously given a sign. A signed "
                    "response contradicts a zero response at h = 0."
                )

        for variable, signed in self.sign_restrictions.items():
            unknown = sorted(set(signed) - known)
            if unknown:
                raise ValueError(
                    f"sign_restrictions[{variable!r}] references unknown shock(s) {unknown}. "
                    f"Known shocks: {self.shock_names}."
                )
            bad = sorted({s for s, direction in signed.items() if direction not in ("+", "-")})
            if bad:
                raise ValueError(
                    f"sign_restrictions[{variable!r}] has non-sign token(s) for shock(s) {bad}. Use '+' or '-'."
                )

    def identify(
        self,
        L: np.ndarray,
        var_names: list[str],
        posterior: "xr.Dataset | None" = None,
        data: "VARData | None" = None,
        n_lags: int | None = None,
    ) -> np.ndarray:
        """Apply zero-and-sign-restriction identification.

        Args:
            L: Lower-triangular Cholesky factor, shape (chains, draws, n_vars, n_vars).
            var_names: Variable names in the data's natural order.
            posterior: Required when `self.restriction_horizon > 0`, which
                needs the VAR coefficients `B` for the MA recursion.
                Ignored for impact-only restrictions.
            data: Unused. Accepted for Protocol uniformity.
            n_lags: Unused — the lag order is read off `B`. Accepted for
                Protocol uniformity.

        Returns:
            Structural shock matrix, shape (chains, draws, n_vars, n_vars),
            with columns in `shock_names` order (padding last). Draws where
            no candidate satisfied the sign restrictions are NaN. Acceptance
            diagnostics land on `IdentifiedVAR.shock_matrix()` attrs under
            the `zero_sign_` prefix.

        Raises:
            ValueError: If a restriction names an unknown variable, more
                shocks are named than there are variables, the zero pattern
                violates the rank condition, `restriction_horizon > 0`
                without a posterior, or `on_failure="raise"` and a draw
                found no admissible candidate.
        """
        del data, n_lags  # unused; lag order comes from B
        import warnings

        n_chains, n_draws, n_vars, _ = L.shape
        layout = self._compile_layout(var_names, n_vars)
        B_all = self._require_coefficients(posterior)
        horizon = self.restriction_horizon

        rng = np.random.default_rng(self.random_seed)
        eye = np.eye(n_vars)
        P = np.full((n_chains, n_draws, n_vars, n_vars), np.nan)
        n_total = n_chains * n_draws
        n_accepted = 0
        total_attempts = 0
        max_zero_violation = 0.0

        n_lags_b = B_all.shape[-1] // n_vars if B_all is not None else 0
        for c in range(n_chains):
            for d in range(n_draws):
                # Hoisted out of the candidate loop: the MA coefficients depend
                # on the draw only, not on the rotation being tried.
                Phi = compute_ma_phi(lag_matrices(B_all[c, d], n_lags_b), horizon) if B_all is not None else None
                accepted, attempts = self._identify_draw(L[c, d], Phi, layout, eye, rng, n_vars)
                total_attempts += attempts

                if accepted is None:
                    if self.on_failure == "raise":
                        raise ValueError(
                            f"No admissible rotation for draw (chain={c}, draw={d}) within "
                            f"n_rotations={self.n_rotations}. Increase n_rotations, relax the sign "
                            "restrictions, or set on_failure='nan' to keep the draw as NaN."
                        )
                    continue

                P[c, d] = accepted
                n_accepted += 1
                if layout.zero_idx_rows.size:
                    violation = float(np.max(np.abs(accepted[layout.zero_idx_rows, layout.zero_idx_cols])))
                    max_zero_violation = max(max_zero_violation, violation)

        failed = n_total - n_accepted
        self._last_diagnostics = {
            "zero_sign_acceptance_rate": n_accepted / n_total,
            "zero_sign_failed_draws": float(failed),
            "zero_sign_failed_fraction": failed / n_total,
            "zero_sign_mean_attempts": total_attempts / n_total,
            "zero_sign_max_zero_violation": max_zero_violation,
        }
        if failed:
            warnings.warn(
                f"Sign restrictions not satisfied for {failed}/{n_total} draws "
                f"({failed / n_total:.1%}); those draws are NaN. Increase n_rotations or relax the "
                "restrictions. NaN draws propagate into IRF/FEVD summaries and are rejected by the "
                "scenario methods.",
                UserWarning,
                stacklevel=2,
            )
        return P

    def _require_coefficients(self, posterior: "xr.Dataset | None") -> np.ndarray | None:
        """Fetch the `B` draws when horizon restrictions need them.

        Args:
            posterior: Posterior Dataset, or None.

        Returns:
            The `B` draws, or None for impact-only restrictions.

        Raises:
            ValueError: If `restriction_horizon > 0` and `B` is unavailable.
        """
        if self.restriction_horizon == 0:
            return None
        if posterior is None or "B" not in posterior:
            raise ValueError(
                "restriction_horizon > 0 requires the full posterior with 'B' "
                "(VAR coefficients). Pass the fit's posterior group as an xarray.Dataset "
                "to identify() — FittedVAR.set_identification_strategy(...) does this for you."
            )
        return posterior["B"].values

    def _compile_layout(self, var_names: list[str], n_vars: int) -> _ZeroSignLayout:
        """Resolve names to indices and fix the construction order, once per call.

        Everything here depends on the restriction pattern and the variable
        names only — not on the posterior draw — so it is hoisted out of the
        draw loop. The rank condition is checked here too, before any
        sampling happens.

        Args:
            var_names: Variable names in data order.
            n_vars: Number of endogenous variables.

        Returns:
            The compiled layout.

        Raises:
            ValueError: If more shocks are named than there are variables, a
                restriction names an unknown variable, or the zero pattern
                violates the rank condition.
        """
        if len(self.shock_names) > n_vars:
            raise ValueError(
                f"ZeroSignRestriction names {len(self.shock_names)} shocks "
                f"({self.shock_names}) but the VAR has only {n_vars} variables. "
                "A VAR admits at most n_vars structural shocks."
            )

        referenced = set(self.zero_restrictions) | set(self.sign_restrictions)
        unknown_vars = sorted(referenced - set(var_names))
        if unknown_vars:
            raise ValueError(f"Restrictions reference unknown variable(s) {unknown_vars}. Variables: {var_names}.")

        labels = self.shock_coords(n_vars)
        label_index = {label: j for j, label in enumerate(labels)}
        var_index = {name: i for i, name in enumerate(var_names)}

        # Zero-restricted variable rows per shock column, in user order.
        zero_rows_user: list[list[int]] = [[] for _ in range(n_vars)]
        for variable, shocks in self.zero_restrictions.items():
            i = var_index[variable]
            for shock in shocks:
                col = label_index[shock]
                if i not in zero_rows_user[col]:
                    zero_rows_user[col].append(i)
        zero_rows_user = [sorted(rows) for rows in zero_rows_user]

        # Construction order: most-restricted shock first. Python's sort is
        # stable, so ties keep user order and the unrestricted padding
        # columns (z = 0) stay last.
        order = sorted(range(n_vars), key=lambda u: -len(zero_rows_user[u]))
        self._check_rank_condition(order, zero_rows_user, labels, n_vars)

        # Flat sign-restriction index arrays, columns in user order.
        rows_list, cols_list, signs_list = [], [], []
        for variable, signed in self.sign_restrictions.items():
            i = var_index[variable]
            for shock, direction in signed.items():
                rows_list.append(i)
                cols_list.append(label_index[shock])
                signs_list.append(1.0 if direction == "+" else -1.0)
        sign_rows = np.asarray(rows_list, dtype=int)
        sign_cols = np.asarray(cols_list, dtype=int)
        sign_vals = np.asarray(signs_list, dtype=float)

        # Flat zero-cell indices (user order) for the violation diagnostic.
        zero_cells = [(i, u) for u in range(n_vars) for i in zero_rows_user[u]]

        return _ZeroSignLayout(
            labels=labels,
            order=order,
            perm_back=np.argsort(order),
            zero_rows=[np.asarray(zero_rows_user[u], dtype=int) for u in order],
            sign_rows=sign_rows,
            sign_cols=sign_cols,
            sign_vals=sign_vals,
            col_rows=[sign_rows[sign_cols == u] for u in order],
            col_signs=[sign_vals[sign_cols == u] for u in order],
            zero_idx_rows=np.asarray([i for i, _ in zero_cells], dtype=int),
            zero_idx_cols=np.asarray([u for _, u in zero_cells], dtype=int),
        )

    def _identify_draw(
        self,
        chol: np.ndarray,
        Phi: np.ndarray | None,
        layout: _ZeroSignLayout,
        eye: np.ndarray,
        rng: np.random.Generator,
        n_vars: int,
    ) -> tuple[np.ndarray | None, int]:
        """Rejection-sample one posterior draw's structural impact matrix.

        Args:
            chol: This draw's Cholesky factor, shape `(n_vars, n_vars)`.
            Phi: MA coefficients for this draw, or None when
                `restriction_horizon == 0`.
            layout: Compiled restriction bookkeeping.
            eye: Cached `n_vars` identity.
            rng: Random generator.
            n_vars: Number of endogenous variables.

        Returns:
            Tuple `(P, attempts)`: the accepted impact matrix with columns in
            user order, or None if the budget ran out, and the number of
            candidates drawn.
        """
        for attempt in range(1, self.n_rotations + 1):
            candidate = self._draw_candidate(chol, layout, eye, rng, n_vars)
            if candidate is None:
                continue
            P_cand = chol @ candidate[:, layout.perm_back]
            if Phi is not None and not _signs_ok_at_horizons(
                P_cand, Phi, self.restriction_horizon, layout.sign_rows, layout.sign_cols, layout.sign_vals
            ):
                continue
            return P_cand, attempt
        return None, self.n_rotations

    def _check_rank_condition(
        self,
        order: list[int],
        zero_rows_user: list[list[int]],
        labels: list[str],
        n_vars: int,
    ) -> None:
        """Verify the Rubio-Ramirez, Waggoner & Zha (2010) rank condition.

        With shocks sorted by zero count descending, the null space at
        position `j` (1-based) has dimension `n - z_j - (j - 1)`, so a
        non-degenerate column requires `z_j <= n - j`. The check is
        deterministic — it depends on the restriction pattern alone — so it
        runs once, before any sampling.

        Args:
            order: Construction order (user column indices, most-restricted first).
            zero_rows_user: Zero-restricted variable rows per user column.
            labels: Effective shock labels, user order.
            n_vars: Number of endogenous variables.

        Raises:
            ValueError: If any sorted position violates the bound.
        """
        for k, u in enumerate(order):
            z_k = len(zero_rows_user[u])
            bound = n_vars - k - 1
            if z_k > bound:
                raise ValueError(
                    f"Zero restrictions violate the rank condition of Rubio-Ramirez, Waggoner & "
                    f"Zha (2010): shock {labels[u]!r} carries {z_k} zero restriction(s), but at "
                    f"position j = {k + 1} of the zero-count ordering a shock may carry at most "
                    f"n - j = {bound}. With more, the null space for that column is empty and no "
                    "orthogonal matrix satisfies the restrictions."
                )

    def _draw_candidate(
        self,
        chol: np.ndarray,
        layout: _ZeroSignLayout,
        eye: np.ndarray,
        rng: np.random.Generator,
        n_vars: int,
    ) -> np.ndarray | None:
        """Draw one candidate orthogonal matrix in construction order.

        Args:
            chol: This draw's Cholesky factor, shape `(n_vars, n_vars)`.
            layout: Compiled restriction bookkeeping.
            eye: Cached `n_vars` identity.
            rng: Random generator.
            n_vars: Number of endogenous variables.

        Returns:
            An orthogonal matrix whose columns are in construction order
            and satisfy every zero restriction plus every *impact* sign
            restriction, or None if the impact screen failed.
        """
        Q = np.empty((n_vars, n_vars))
        for k in range(n_vars):
            rows_k = layout.zero_rows[k]
            # Rows of R: the z_k zero conditions Z_k L q = 0, plus the k - 1
            # orthogonality conditions against the columns already drawn.
            m = rows_k.size + k
            if m == 0:
                N = eye
            else:
                R = np.vstack((chol[rows_k, :], Q[:, :k].T))
                # Right singular vectors beyond the m-th span the null space
                # of R. Valid even when R is rank-deficient: those vectors
                # still have zero singular value, so they lie in the (then
                # larger) null space.
                _, _, Vh = np.linalg.svd(R, full_matrices=True)
                N = Vh[m:].T

            x = rng.standard_normal(n_vars - m)
            norm = float(np.linalg.norm(x))
            while norm < 1e-12:
                x = rng.standard_normal(n_vars - m)
                norm = float(np.linalg.norm(x))
            q = N @ (x / norm)

            if layout.col_rows[k].size:
                impact = chol @ q
                if not bool(np.all(impact[layout.col_rows[k]] * layout.col_signs[k] >= 0.0)):
                    # Abandon the WHOLE candidate and restart from column 1.
                    # Redrawing only this column would be a distribution bug:
                    # q_k's law is conditional on q_1..q_{k-1}, so retrying
                    # column k alone conditions the retained prefix on
                    # "produced a failure here", which is not the marginal of
                    # the accepted joint draw. Abandoning everything keeps the
                    # procedure plain rejection sampling over the whole Q.
                    return None
            Q[:, k] = q
        return Q

    def shock_coords(self, n_vars: int) -> list[str]:
        """Named shocks in user order, then rotation-arbitrary padding."""
        return pad_shock_coords(list(self.shock_names), n_vars)
