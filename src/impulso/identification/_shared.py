"""Shock-coordinate helpers shared across identification schemes."""


def pad_shock_coords(shock_names: list[str], n_vars: int) -> list[str]:
    """Build shock coordinate labels for the structural shock matrix.

    Named shocks occupy their column positions; remaining columns
    are labeled 'unidentified_1', 'unidentified_2', etc.
    """
    if len(shock_names) == n_vars:
        return shock_names
    return shock_names + [f"unidentified_{i}" for i in range(1, n_vars - len(shock_names) + 1)]
