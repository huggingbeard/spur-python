from __future__ import annotations
import numpy as np
import pandas as pd


def parse_single_var_formula(formula: str, data: pd.DataFrame, fn_name: str) -> dict:
    """Parse the selector string used by the single-variable SPUR APIs.

    This exists to normalize the selector wrapper input (`"y"` or `"y ~ 1"`)
    before dispatching to the dedicated single-variable SPUR functions.
    """
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError(f"`formula` for {fn_name} must be a non-empty string.")

    raw = formula.strip()

    if "+" in raw:
        raise ValueError(
            f"`formula` for {fn_name} must reference exactly one variable."
        )

    if "~" not in raw:
        var = raw
    else:
        lhs, rhs = [part.strip() for part in raw.split("~", 1)]
        if not lhs or rhs != "1":
            raise ValueError(f"For {fn_name} use `y` or `y ~ 1`.")
        var = lhs

    assert var in data.columns, f"Variable '{var}' not found in data."
    return {"var": var, "use_rows": np.ones(len(data), dtype=bool)}


def parse_residual_formula(formula: str, data: pd.DataFrame, fn_name: str) -> dict:
    """Parse a residual-test formula into `Y`, `X_in`, and the estimation sample.

    This keeps the public residual APIs formula-based while preserving the
    existing matrix-based SPUR residual implementation.
    """
    if not isinstance(formula, str) or "~" not in formula:
        raise ValueError(
            f"`formula` for {fn_name} must be two-sided, e.g. `y ~ x1 + x2`."
        )

    lhs, rhs = [part.strip() for part in formula.split("~", 1)]
    assert lhs, (
        f"`formula` for {fn_name} must have a dependent variable on the left-hand side."
    )

    rhs_terms = [] if rhs in ("", "1") else [term.strip() for term in rhs.split("+")]
    if any(not term for term in rhs_terms):
        raise ValueError(f"`formula` for {fn_name} is invalid.")
    print(f"[formula-parser] y = {lhs}, covars (selection) = {rhs_terms[:5]}")

    needed = [lhs] + rhs_terms
    missing = [name for name in needed if name not in data.columns]
    assert not missing, f"Variables not found in data: {', '.join(missing)}"

    frame = data[needed]
    use_rows = frame.notna().all(axis=1).to_numpy()
    assert use_rows.any(), f"No observations with complete data for {fn_name}."
    y = data.loc[use_rows, lhs].to_numpy(dtype=float).reshape(-1, 1)

    x_cols = []
    for term in rhs_terms:
        col = data.loc[use_rows, term].to_numpy(dtype=float).reshape(-1, 1)
        x_cols.append(col)

    if x_cols:
        X = np.column_stack(x_cols)
        X_in = np.column_stack([np.ones(len(y)), X])
    else:
        X_in = np.ones((len(y), 1))

    return {"Y": y, "X_in": X_in, "use_rows": use_rows}


def parse_transform_formula(formula: str, data: pd.DataFrame) -> list[str]:
    """Collect all variables referenced by `spurtransform()`.

    This lets the public transform API accept formulas while the existing
    transformation code continues to work with a flat variable list.
    """
    if not isinstance(formula, str) or "~" not in formula:
        raise ValueError("`formula` must be two-sided, e.g. `y ~ x1 + x2`.")

    lhs, rhs = [part.strip() for part in formula.split("~", 1)]
    vars = [lhs] if lhs and lhs != "1" else []

    if rhs not in ("", "1"):
        rhs_terms = [term.strip() for term in rhs.split("+")]
        if any(not term for term in rhs_terms):
            raise ValueError("`formula` is invalid.")
        vars.extend(rhs_terms)
    else:
        rhs_terms = []

    print(f"[formula-parser] y = {lhs}, covars (selection) = {rhs_terms[:5]}")

    vars = list(dict.fromkeys(vars))
    assert vars, "No variables found in `formula`."
    missing = [name for name in vars if name not in data.columns]
    assert not missing, f"Variables not found in data: {', '.join(missing)}"

    return vars


def rewrite_formula_with_prefix(formula: str, prefix: str) -> str:
    """Rewrite `y ~ x1 + x2` as `h_y ~ h_x1 + h_x2` for transformed SPUR models."""
    lhs, rhs = [part.strip() for part in formula.split("~", 1)]

    lhs_out = f"{prefix}{lhs}" if lhs and lhs != "1" else ""
    rhs_terms = [] if rhs in ("", "1") else [term.strip() for term in rhs.split("+")]
    rhs_out = " + ".join(f"{prefix}{term}" for term in rhs_terms) if rhs_terms else "1"

    return f"{lhs_out} ~ {rhs_out}"
