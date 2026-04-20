from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from scpc import scpc
import statsmodels.formula.api as smf
from .spurtest import SpurTestResult, spurtest_i0, spurtest_i1
from .spurtransform import spurtransform
from .utils import rewrite_formula_with_prefix


@dataclass
class SpurResult:
    """Container for the high-level SPUR pipeline output."""

    branch: str
    test_i0: SpurTestResult
    test_i1: SpurTestResult
    model: object
    scpc: object
    data_used: pd.DataFrame
    formula_used: str


def spur(
    formula: str,
    data: pd.DataFrame,
    *,
    lon: str | None = None,
    lat: str | None = None,
    coords_euclidean: list[str] | tuple[str, ...] | None = None,
    q: int = 15,
    nrep: int = 100000,
    seed: int = 42,
    avc: float = 0.03,
    uncond: bool = False,
    cvs: bool = False,
) -> SpurResult:
    """Run the full SPUR pipeline and pass the final model to `scpc()`.

    This function runs the two single-variable diagnostics on the dependent
    variable, chooses either the levels branch or the transformed branch, fits the
    regression, and then passes the fitted model to `scpc()`.

    Args:
        formula: Two-sided regression formula for the model of interest.
        data: DataFrame containing the model variables and coordinate columns.
        lon: Name of the longitude column. Use together with `lat`.
        lat: Name of the latitude column. Use together with `lon`.
        coords_euclidean: Names of Euclidean coordinate columns. Use instead of
            `lon` and `lat`.
        q: Number of low-frequency weights used in the SPUR tests.
        nrep: Number of Monte Carlo draws used in the SPUR tests.
        seed: Random seed used to generate the SPUR simulation draws.
        avc: SCPC variance-covariance tuning parameter.
        uncond: Passed through to `scpc()`.
        cvs: Passed through to `scpc()`.

    Returns:
        A `SpurResult` containing the branch decision, the two diagnostic test
        results, the fitted model, the `scpc()` result, the data used for
        estimation, and the formula ultimately fit.

    Raises:
        ValueError: If `formula` is not two-sided or does not contain a dependent
            variable on the left-hand side.
        ValueError: Propagated from the SPUR test and transformation steps if the
            inputs are invalid.

    Example:
        >>> from spur import load_chetty_data, standardize, spur
        >>> df = load_chetty_data()
        >>> df = df[["am", "fracblack", "lat", "lon"]].dropna()
        >>> df = standardize(df, ["am", "fracblack"])
        >>> result = spur("am ~ fracblack", df, lon="lon", lat="lat", q=10, nrep=500, seed=42)
        >>> result.branch
        >>> result.formula_used
    """
    if not isinstance(formula, str) or "~" not in formula:
        raise ValueError("`formula` must be two-sided, e.g. `y ~ x1 + x2`.")

    depvar = formula.split("~", 1)[0].strip()
    if not depvar:
        raise ValueError(
            "`formula` must have a dependent variable on the left-hand side."
        )

    test_i0 = spurtest_i0(
        depvar,
        data,
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
        q=q,
        nrep=nrep,
        seed=seed,
    )
    test_i1 = spurtest_i1(
        depvar,
        data,
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
        q=q,
        nrep=nrep,
        seed=seed,
    )

    if test_i1.pvalue < 0.1 and test_i0.pvalue >= 0.1:
        branch = "levels"
        data_used = data
        formula_used = formula
    else:
        branch = "transformed"
        data_used = spurtransform(
            formula,
            data,
            lon=lon,
            lat=lat,
            coords_euclidean=coords_euclidean,
            prefix="h_",
            transformation="lbmgls",
        )
        formula_used = rewrite_formula_with_prefix(formula, "h_")

    model = smf.ols(formula_used, data=data_used).fit()
    scpc_result = scpc(
        model,
        data_used,
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
        avc=avc,
        uncond=uncond,
        cvs=cvs,
    )

    return SpurResult(
        branch=branch,
        test_i0=test_i0,
        test_i1=test_i1,
        model=model,
        scpc=scpc_result,
        data_used=data_used,
        formula_used=formula_used,
    )
