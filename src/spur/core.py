from __future__ import annotations
from collections.abc import Sequence
from typing import Literal, Optional
import numpy as np
import pandas as pd
from scpc import scpc
import statsmodels.formula.api as smf

from .types import (
    Fits,
    HalfLifeResult,
    PipelineResult,
    RegressionResult,
    TestResult,
    Tests,
)
from .utils.dist import (
    get_cbar,
    get_distance_matrix,
    normalized_distmat,
    resolve_spur_coords,
)
from .utils.formula import (
    parse_residual_formula,
    parse_single_var_formula,
    parse_transform_formula,
    rewrite_formula_with_prefix,
)
from .utils.inference import (
    get_ha_param_i0,
    get_ha_param_i1,
    get_ha_param_i1_residual,
    spur_persistence,
)
from .utils.matrix import (
    cholesky_upper,
    cluster_matrix,
    demean_matrix,
    get_r,
    get_sigma_dm,
    get_sigma_lbm,
    get_sigma_residual,
    iso_matrix,
    lbmgls_matrix,
    nn_matrix,
)


def spurtest_i1(
    var: str,
    data: pd.DataFrame,
    *,
    lon: str | None = None,
    lat: str | None = None,
    coords_euclidean: Sequence[str] | None = None,
    q: int = 15,
    nrep: int = 100000,
    seed: int = 42,
) -> TestResult:
    """Conduct the spatial I(1) unit-root test for a single variable.

    Use this function when you want to test one variable directly, without using
    the formula-based wrapper. Provide either longitude/latitude columns or
    Euclidean coordinate columns.

    Args:
        var: Name of the variable to test.
        data: DataFrame containing the variable and coordinate columns.
        lon: Name of the longitude column. Use together with `lat`.
        lat: Name of the latitude column. Use together with `lon`.
        coords_euclidean: Names of Euclidean coordinate columns. Use instead of
            `lon` and `lat`.
        q: Number of low-frequency weights used in the test statistic.
        nrep: Number of Monte Carlo draws used to simulate the null distribution.
        seed: Random seed used to generate the simulation draws.

    Returns:
        A `TestResult` containing the LR statistic, p-value, critical values,
        and the calibrated local-alternative parameter.

    Raises:
        ValueError: If `data` is not a DataFrame, `var` is missing or contains
            non-finite values, the coordinate specification is invalid, `q` is out
            of range, or `nrep` is less than 1.

    Example:
        >>> from spur import load_chetty_data, standardize, spurtest_i1
        >>> df = load_chetty_data()
        >>> df = df[["am", "lat", "lon"]].dropna()
        >>> df = standardize(df, ["am"])
        >>> result = spurtest_i1("am", df, lon="lon", lat="lat", q=10, nrep=500, seed=42)
        >>> print(result.summary())
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a pandas DataFrame.")
    if not isinstance(var, str) or not var:
        raise ValueError("`var` must be a non-empty column name.")
    if var not in data.columns:
        raise ValueError(f"Variable '{var}' not found in data.")

    Y = data[var].to_numpy(dtype=float)
    if not np.all(np.isfinite(Y)):
        raise ValueError(
            f"Variable '{var}' contains NaN or inf values. All values must be finite."
        )

    coord_info = resolve_spur_coords(
        data=data,
        use_rows=np.ones(len(data), dtype=bool),
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
    )

    distmat = normalized_distmat(coord_info["coords"], latlon=coord_info["latlong"])
    n = distmat.shape[0]
    if q >= n:
        raise ValueError(f"q={q} must be less than n={n}. Use q <= {n - 1}.")
    if q < 1:
        raise ValueError(f"q={q} must be >= 1.")
    if nrep < 1:
        raise ValueError(f"nrep={nrep} must be >= 1.")

    emat = np.random.default_rng(seed).standard_normal((q, nrep))
    q = emat.shape[0]

    sigdm_bm = demean_matrix(get_sigma_lbm(distmat))
    R = get_r(sigdm_bm, q)
    om_ho = R.T @ sigdm_bm @ R

    ha_parm = get_ha_param_i1(om_ho, distmat, R, emat)
    sigdm_ha = get_sigma_dm(distmat, ha_parm)
    om_ha = R.T @ sigdm_ha @ R

    ch_om_ho = cholesky_upper(om_ho)
    omi_ho = np.linalg.inv(om_ho)
    omi_ha = np.linalg.inv(om_ha)
    ch_omi_ho = cholesky_upper(omi_ho)
    ch_omi_ha = cholesky_upper(omi_ha)

    y_ho = ch_om_ho.T @ emat
    y_ho_ho = ch_omi_ho @ y_ho
    y_ho_ha = ch_omi_ha @ y_ho
    q_ho_ho = np.sum(y_ho_ho**2, axis=0)
    q_ho_ha = np.sum(y_ho_ha**2, axis=0)
    lr_ho = q_ho_ho / q_ho_ha

    sz_vec = np.array([0.01, 0.05, 0.10])
    cv_vec = np.quantile(lr_ho, 1 - sz_vec)

    X = Y - np.mean(Y)
    P = R.T @ X
    LR = float((P.T @ omi_ho @ P) / (P.T @ omi_ha @ P))
    pvalue = float(np.mean(lr_ho > LR))

    return TestResult(test_type="i1", LR=LR, pvalue=pvalue, cv=cv_vec, ha_param=ha_parm)


def spurtest_i0(
    var: str,
    data: pd.DataFrame,
    *,
    lon: str | None = None,
    lat: str | None = None,
    coords_euclidean: Sequence[str] | None = None,
    q: int = 15,
    nrep: int = 100000,
    seed: int = 42,
) -> TestResult:
    """Conduct the spatial I(0) test for a single variable.

    Use this function when you want to test one variable directly, without using
    the formula-based wrapper. Provide either longitude/latitude columns or
    Euclidean coordinate columns.

    Args:
        var: Name of the variable to test.
        data: DataFrame containing the variable and coordinate columns.
        lon: Name of the longitude column. Use together with `lat`.
        lat: Name of the latitude column. Use together with `lon`.
        coords_euclidean: Names of Euclidean coordinate columns. Use instead of
            `lon` and `lat`.
        q: Number of low-frequency weights used in the test statistic.
        nrep: Number of Monte Carlo draws used to simulate the null distribution.
        seed: Random seed used to generate the simulation draws.

    Returns:
        A `TestResult` containing the LR statistic, p-value, critical values,
        and the calibrated local-alternative parameter.

    Raises:
        ValueError: If `data` is not a DataFrame, `var` is missing or contains
            non-finite values, the coordinate specification is invalid, `q` is out
            of range, or `nrep` is less than 1.

    Example:
        >>> from spur import load_chetty_data, standardize, spurtest_i0
        >>> df = load_chetty_data()
        >>> df = df[["am", "lat", "lon"]].dropna()
        >>> df = standardize(df, ["am"])
        >>> result = spurtest_i0("am", df, lon="lon", lat="lat", q=10, nrep=500, seed=42)
        >>> print(result.summary())
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a pandas DataFrame.")
    if not isinstance(var, str) or not var:
        raise ValueError("`var` must be a non-empty column name.")
    if var not in data.columns:
        raise ValueError(f"Variable '{var}' not found in data.")

    Y = data[var].to_numpy(dtype=float)
    if not np.all(np.isfinite(Y)):
        raise ValueError(
            f"Variable '{var}' contains NaN or inf values. All values must be finite."
        )

    coord_info = resolve_spur_coords(
        data=data,
        use_rows=np.ones(len(data), dtype=bool),
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
    )

    distmat = normalized_distmat(coord_info["coords"], latlon=coord_info["latlong"])
    n = distmat.shape[0]
    if q >= n:
        raise ValueError(f"q={q} must be less than n={n}. Use q <= {n - 1}.")
    if q < 1:
        raise ValueError(f"q={q} must be >= 1.")
    if nrep < 1:
        raise ValueError(f"nrep={nrep} must be >= 1.")

    emat = np.random.default_rng(seed).standard_normal((q, nrep))
    q = emat.shape[0]

    sigdm_bm = demean_matrix(get_sigma_lbm(distmat))
    R = get_r(sigdm_bm, q)

    rho = 0.001
    c = get_cbar(rho, distmat)
    sigdm_rho = get_sigma_dm(distmat, c)

    om_rho = R.T @ sigdm_rho @ R
    om_bm = R.T @ sigdm_bm @ R

    om_i0 = om_rho
    om_ho = om_rho
    ha_parm = get_ha_param_i0(om_ho, om_i0, om_bm, emat)
    om_ha = om_i0 + ha_parm * om_bm

    ch_omi_ho = cholesky_upper(np.linalg.inv(om_ho))
    ch_omi_ha = cholesky_upper(np.linalg.inv(om_ha))

    X = Y - np.mean(Y)
    P = R.T @ X
    y_P_ho = ch_omi_ho @ P
    y_P_ha = ch_omi_ha @ P
    q_P_ho = np.sum(y_P_ho**2)
    q_P_ha = np.sum(y_P_ha**2)
    LR = float(q_P_ho / q_P_ha)

    rho_min = 0.0001
    rho_max = 0.03
    n_rho = 30
    rho_grid = np.linspace(rho_min, rho_max, n_rho)

    ch_om_ho_list = []
    for i in range(n_rho):
        rho_i = rho_grid[i]
        if rho_i > 0:
            c_i = get_cbar(rho_i, distmat)
            sigdm_ho_i = get_sigma_dm(distmat, c_i)
            om_ho_i = R.T @ sigdm_ho_i @ R
        else:
            om_ho_i = np.eye(q)
        ch_om_ho_list.append(cholesky_upper(om_ho_i))

    pvalue_vec = np.zeros(n_rho)
    cvalue_mat = np.zeros((n_rho, 3))
    sz_vec = np.array([0.01, 0.05, 0.10])

    for ir in range(n_rho):
        ch_om_ho_ir = ch_om_ho_list[ir]
        y_ho = ch_om_ho_ir.T @ emat
        y_ho_ho = ch_omi_ho @ y_ho
        y_ho_ha = ch_omi_ha @ y_ho
        q_ho_ho = np.sum(y_ho_ho**2, axis=0)
        q_ho_ha = np.sum(y_ho_ha**2, axis=0)
        lr_ho = q_ho_ho / q_ho_ha
        cvalue_mat[ir, :] = np.quantile(lr_ho, 1 - sz_vec)
        pvalue_vec[ir] = np.mean(lr_ho > LR)

    cvalue = cvalue_mat.max(axis=0)
    pvalue = float(pvalue_vec.max())

    return TestResult(test_type="i0", LR=LR, pvalue=pvalue, cv=cvalue, ha_param=ha_parm)


def spurtest_i1resid(
    formula: str,
    data: pd.DataFrame,
    *,
    lon: str | None = None,
    lat: str | None = None,
    coords_euclidean: Sequence[str] | None = None,
    q: int = 15,
    nrep: int = 100000,
    seed: int = 42,
) -> TestResult:
    """Conduct the spatial I(1) residual test for a regression formula.

    Use this function when you want to test the residual dependence implied by a
    formula such as `y ~ x1 + x2`. Provide either longitude/latitude columns or
    Euclidean coordinate columns.

    Args:
        formula: Two-sided regression formula identifying the dependent variable
            and covariates.
        data: DataFrame containing the variables and coordinate columns.
        lon: Name of the longitude column. Use together with `lat`.
        lat: Name of the latitude column. Use together with `lon`.
        coords_euclidean: Names of Euclidean coordinate columns. Use instead of
            `lon` and `lat`.
        q: Number of low-frequency weights used in the test statistic.
        nrep: Number of Monte Carlo draws used to simulate the null distribution.
        seed: Random seed used to generate the simulation draws.

    Returns:
        A `TestResult` containing the LR statistic, p-value, critical values,
        and the calibrated local-alternative parameter.

    Raises:
        ValueError: If `data` is not a DataFrame, `formula` is invalid, the
            coordinate specification is invalid, the regressor matrix is
            rank-deficient, `q` is out of range, or `nrep` is less than 1.

    Example:
        >>> from spur import load_chetty_data, standardize, spurtest_i1resid
        >>> df = load_chetty_data()
        >>> df = df[["am", "fracblack", "lat", "lon"]].dropna()
        >>> df = standardize(df, ["am", "fracblack"])
        >>> result = spurtest_i1resid(
        ...     "am ~ fracblack",
        ...     df,
        ...     lon="lon",
        ...     lat="lat",
        ...     q=10,
        ...     nrep=500,
        ...     seed=42,
        ... )
        >>> print(result.summary())
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a pandas DataFrame.")

    parsed = parse_residual_formula(formula, data, "spurtest_i1resid()")
    coord_info = resolve_spur_coords(
        data=data,
        use_rows=parsed["use_rows"],
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
    )

    distmat = normalized_distmat(coord_info["coords"], latlon=coord_info["latlong"])
    n = distmat.shape[0]
    if q >= n:
        raise ValueError(f"q={q} must be less than n={n}. Use q <= {n - 1}.")
    if q < 1:
        raise ValueError(f"q={q} must be >= 1.")
    if nrep < 1:
        raise ValueError(f"nrep={nrep} must be >= 1.")

    Y = parsed["Y"].reshape(-1)
    X_in = parsed["X_in"]
    emat = np.random.default_rng(seed).standard_normal((q, nrep))
    q = emat.shape[0]
    n = distmat.shape[0]

    if np.linalg.matrix_rank(X_in) < X_in.shape[1]:
        raise ValueError(
            "Regressor matrix X is rank-deficient (collinear regressors or dummy trap). "
            "Check for perfectly collinear columns or a full set of category dummies."
        )
    XtX_inv = np.linalg.inv(X_in.T @ X_in)
    M = np.eye(n) - X_in @ XtX_inv @ X_in.T

    rho_bm = 0.999
    c_bm = get_cbar(rho_bm, distmat)
    sigdm_bm = get_sigma_residual(distmat, c_bm, M)

    R = get_r(sigdm_bm, q)
    om_ho = R.T @ sigdm_bm @ R

    ha_parm = get_ha_param_i1_residual(om_ho, distmat, R, emat, M)
    sigdm_ha = get_sigma_residual(distmat, ha_parm, M)
    om_ha = R.T @ sigdm_ha @ R

    ch_om_ho = cholesky_upper(om_ho)
    omi_ho = np.linalg.inv(om_ho)
    omi_ha = np.linalg.inv(om_ha)
    ch_omi_ho = cholesky_upper(omi_ho)
    ch_omi_ha = cholesky_upper(omi_ha)

    y_ho = ch_om_ho.T @ emat
    y_ho_ho = ch_omi_ho @ y_ho
    y_ho_ha = ch_omi_ha @ y_ho
    q_ho_ho = np.sum(y_ho_ho**2, axis=0)
    q_ho_ha = np.sum(y_ho_ha**2, axis=0)
    lr_ho = q_ho_ho / q_ho_ha

    sz_vec = np.array([0.01, 0.05, 0.10])
    cv_vec = np.quantile(lr_ho, 1 - sz_vec)

    X = Y - np.mean(Y)
    P = R.T @ X
    LR = float((P.T @ omi_ho @ P) / (P.T @ omi_ha @ P))
    pvalue = float(np.mean(lr_ho > LR))

    return TestResult(
        test_type="i1resid", LR=LR, pvalue=pvalue, cv=cv_vec, ha_param=ha_parm
    )


def spurtest_i0resid(
    formula: str,
    data: pd.DataFrame,
    *,
    lon: str | None = None,
    lat: str | None = None,
    coords_euclidean: Sequence[str] | None = None,
    q: int = 15,
    nrep: int = 100000,
    seed: int = 42,
) -> TestResult:
    """Conduct the spatial I(0) residual test for a regression formula.

    Use this function when you want to test the residual dependence implied by a
    formula such as `y ~ x1 + x2`. Provide either longitude/latitude columns or
    Euclidean coordinate columns.

    Args:
        formula: Two-sided regression formula identifying the dependent variable
            and covariates.
        data: DataFrame containing the variables and coordinate columns.
        lon: Name of the longitude column. Use together with `lat`.
        lat: Name of the latitude column. Use together with `lon`.
        coords_euclidean: Names of Euclidean coordinate columns. Use instead of
            `lon` and `lat`.
        q: Number of low-frequency weights used in the test statistic.
        nrep: Number of Monte Carlo draws used to simulate the null distribution.
        seed: Random seed used to generate the simulation draws.

    Returns:
        A `TestResult` containing the LR statistic, p-value, critical values,
        and the calibrated local-alternative parameter.

    Raises:
        ValueError: If `data` is not a DataFrame, `formula` is invalid, the
            coordinate specification is invalid, the regressor matrix is
            rank-deficient, `q` is out of range, or `nrep` is less than 1.

    Example:
        >>> from spur import load_chetty_data, standardize, spurtest_i0resid
        >>> df = load_chetty_data()
        >>> df = df[["am", "fracblack", "lat", "lon"]].dropna()
        >>> df = standardize(df, ["am", "fracblack"])
        >>> result = spurtest_i0resid(
        ...     "am ~ fracblack",
        ...     df,
        ...     lon="lon",
        ...     lat="lat",
        ...     q=10,
        ...     nrep=500,
        ...     seed=42,
        ... )
        >>> print(result.summary())
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a pandas DataFrame.")

    parsed = parse_residual_formula(formula, data, "spurtest_i0resid()")
    coord_info = resolve_spur_coords(
        data=data,
        use_rows=parsed["use_rows"],
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
    )

    distmat = normalized_distmat(coord_info["coords"], latlon=coord_info["latlong"])
    n = distmat.shape[0]
    if q >= n:
        raise ValueError(f"q={q} must be less than n={n}. Use q <= {n - 1}.")
    if q < 1:
        raise ValueError(f"q={q} must be >= 1.")
    if nrep < 1:
        raise ValueError(f"nrep={nrep} must be >= 1.")

    Y = parsed["Y"].reshape(-1)
    X_in = parsed["X_in"]
    emat = np.random.default_rng(seed).standard_normal((q, nrep))
    q = emat.shape[0]
    n = distmat.shape[0]

    if np.linalg.matrix_rank(X_in) < X_in.shape[1]:
        raise ValueError(
            "Regressor matrix X is rank-deficient (collinear regressors or dummy trap). "
            "Check for perfectly collinear columns or a full set of category dummies."
        )
    XtX_inv = np.linalg.inv(X_in.T @ X_in)
    M = np.eye(n) - X_in @ XtX_inv @ X_in.T

    rho_bm = 0.999
    c_bm = get_cbar(rho_bm, distmat)
    sigdm_bm = get_sigma_residual(distmat, c_bm, M)

    R = get_r(sigdm_bm, q)

    rho = 0.001
    c = get_cbar(rho, distmat)
    sigdm_rho = get_sigma_residual(distmat, c, M)

    om_rho = R.T @ sigdm_rho @ R
    om_bm = R.T @ sigdm_bm @ R

    om_i0 = om_rho
    om_ho = om_rho
    ha_parm = get_ha_param_i0(om_ho, om_i0, om_bm, emat)
    om_ha = om_i0 + ha_parm * om_bm

    ch_omi_ho = cholesky_upper(np.linalg.inv(om_ho))
    ch_omi_ha = cholesky_upper(np.linalg.inv(om_ha))

    X = Y - np.mean(Y)
    P = R.T @ X
    y_P_ho = ch_omi_ho @ P
    y_P_ha = ch_omi_ha @ P
    q_P_ho = np.sum(y_P_ho**2)
    q_P_ha = np.sum(y_P_ha**2)
    LR = float(q_P_ho / q_P_ha)

    rho_min = 0.0001
    rho_max = 0.03
    n_rho = 30
    rho_grid = np.linspace(rho_min, rho_max, n_rho)

    ch_om_ho_list = []
    for i in range(n_rho):
        rho_i = rho_grid[i]
        if rho_i > 0:
            c_i = get_cbar(rho_i, distmat)
            sigdm_ho_i = get_sigma_residual(distmat, c_i, M)
            om_ho_i = R.T @ sigdm_ho_i @ R
        else:
            om_ho_i = np.eye(q)
        ch_om_ho_list.append(cholesky_upper(om_ho_i))

    pvalue_vec = np.zeros(n_rho)
    cvalue_mat = np.zeros((n_rho, 3))
    sz_vec = np.array([0.01, 0.05, 0.10])

    for ir in range(n_rho):
        ch_om_ho_ir = ch_om_ho_list[ir]
        y_ho = ch_om_ho_ir.T @ emat
        y_ho_ho = ch_omi_ho @ y_ho
        y_ho_ha = ch_omi_ha @ y_ho
        q_ho_ho = np.sum(y_ho_ho**2, axis=0)
        q_ho_ha = np.sum(y_ho_ha**2, axis=0)
        lr_ho = q_ho_ho / q_ho_ha
        cvalue_mat[ir, :] = np.quantile(lr_ho, 1 - sz_vec)
        pvalue_vec[ir] = np.mean(lr_ho > LR)

    cvalue = cvalue_mat.max(axis=0)
    pvalue = float(pvalue_vec.max())

    return TestResult(
        test_type="i0resid", LR=LR, pvalue=pvalue, cv=cvalue, ha_param=ha_parm
    )


def spurtest(
    formula: str,
    data: pd.DataFrame,
    *,
    test: Literal["i1", "i0", "i1resid", "i0resid"],
    lon: str | None = None,
    lat: str | None = None,
    coords_euclidean: Sequence[str] | None = None,
    q: int = 15,
    nrep: int = 100000,
    seed: int = 42,
) -> TestResult:
    """Run one of the four public SPUR diagnostic tests.

    Use this wrapper when you want one entrypoint for all test types. For
    single-variable tests, pass either a bare variable name like `"am"` or
    `"am ~ 1"`. For residual tests, pass a two-sided formula such as
    `"am ~ fracblack"`.

    Args:
        formula: Variable string or formula, depending on the selected test.
        data: DataFrame containing the variables and coordinate columns.
        test: Which SPUR test to run. Must be one of `"i0"`, `"i1"`,
            `"i0resid"`, or `"i1resid"`.
        lon: Name of the longitude column. Use together with `lat`.
        lat: Name of the latitude column. Use together with `lon`.
        coords_euclidean: Names of Euclidean coordinate columns. Use instead of
            `lon` and `lat`.
        q: Number of low-frequency weights used in the test statistic.
        nrep: Number of Monte Carlo draws used to simulate the null distribution.
        seed: Random seed used to generate the simulation draws.

    Returns:
        A `TestResult` for the selected test.

    Raises:
        ValueError: If `test` is invalid or if the selected test receives invalid
            variables, formula inputs, coordinates, or simulation settings.

    Example:
        >>> from spur import load_chetty_data, standardize, spurtest
        >>> df = load_chetty_data()
        >>> df = df[["am", "lat", "lon"]].dropna()
        >>> df = standardize(df, ["am"])
        >>> result = spurtest("am", df, test="i1", lon="lon", lat="lat", q=10, nrep=500, seed=42)
        >>> print(result.summary())
    """
    if test == "i0":
        parsed = parse_single_var_formula(formula, data, "spurtest()")
        return spurtest_i0(
            parsed["var"],
            data,
            lon=lon,
            lat=lat,
            coords_euclidean=coords_euclidean,
            q=q,
            nrep=nrep,
            seed=seed,
        )
    if test == "i1":
        parsed = parse_single_var_formula(formula, data, "spurtest()")
        return spurtest_i1(
            parsed["var"],
            data,
            lon=lon,
            lat=lat,
            coords_euclidean=coords_euclidean,
            q=q,
            nrep=nrep,
            seed=seed,
        )
    if test == "i0resid":
        return spurtest_i0resid(
            formula,
            data,
            lon=lon,
            lat=lat,
            coords_euclidean=coords_euclidean,
            q=q,
            nrep=nrep,
            seed=seed,
        )
    if test == "i1resid":
        return spurtest_i1resid(
            formula,
            data,
            lon=lon,
            lat=lat,
            coords_euclidean=coords_euclidean,
            q=q,
            nrep=nrep,
            seed=seed,
        )
    raise ValueError("`test` must be one of 'i0', 'i1', 'i0resid', 'i1resid'.")


def spurtransform(
    formula: str,
    data: pd.DataFrame,
    *,
    lon: str | None = None,
    lat: str | None = None,
    coords_euclidean: Sequence[str] | None = None,
    prefix: str = "h_",
    transformation: str = "lbmgls",
    radius: Optional[float] = None,
    clustvar: Optional[str] = None,
) -> pd.DataFrame:
    """Apply a SPUR transformation to the variables referenced in a formula.

    Use this function when you want transformed versions of the variables in a
    model formula. The function builds one transformation matrix and applies it to
    each referenced variable, adding new columns with the chosen prefix.

    Args:
        formula: Formula identifying the variables to transform.
        data: DataFrame containing the variables and any required coordinate or
            cluster columns.
        lon: Name of the longitude column. Use together with `lat`.
        lat: Name of the latitude column. Use together with `lon`.
        coords_euclidean: Names of Euclidean coordinate columns. Use instead of
            `lon` and `lat`.
        prefix: Prefix added to each transformed variable name.
        transformation: Transformation method. Must be one of `"nn"`, `"iso"`,
            `"lbmgls"`, or `"cluster"`.
        radius: Radius used by the isotropic transformation. Required when
            `transformation="iso"`.
        clustvar: Cluster label column used by the cluster transformation.
            Required when `transformation="cluster"`.

    Returns:
        A copy of `data` with transformed columns added.

    Raises:
        ValueError: If the formula is invalid, variables are missing, coordinates
            are invalid, or the transformation name is unknown.
        AssertionError: If a required `radius` or `clustvar` is missing, or if the
            cluster column contains missing values.

    Example:
        >>> from spur import load_chetty_data, standardize, spurtransform
        >>> df = load_chetty_data()
        >>> df = df[["am", "fracblack", "lat", "lon"]].dropna()
        >>> df = standardize(df, ["am", "fracblack"])
        >>> df_out = spurtransform(
        ...     "am ~ fracblack",
        ...     df,
        ...     lon="lon",
        ...     lat="lat",
        ...     transformation="lbmgls",
        ... )
        >>> df_out[["h_am", "h_fracblack"]].head()
    """
    df = data.copy()
    varlist = parse_transform_formula(formula, df)

    if transformation == "cluster":
        assert clustvar is not None, (
            "clustvar must be specified for transformation='cluster'"
        )
        assert clustvar in df.columns, (
            f"Cluster variable '{clustvar}' not found in DataFrame"
        )
        assert not df[clustvar].isna().any(), (
            f"Cluster column '{clustvar}' contains missing values."
        )
        cluster = df[clustvar].to_numpy()
        M = cluster_matrix(cluster)
    else:
        coord_info = resolve_spur_coords(
            data=df,
            use_rows=np.ones(len(df), dtype=bool),
            lon=lon,
            lat=lat,
            coords_euclidean=coords_euclidean,
        )

        coords = coord_info["coords"]
        latlon = coord_info["latlong"]

        if coords.shape[1] != 2:
            raise ValueError(
                f"Distance-based transformations require exactly 2 coordinate columns, got {coords.shape[1]}"
            )
        if transformation == "nn":
            M = nn_matrix(coords, latlon=latlon)
        elif transformation == "iso":
            assert radius is not None, (
                "radius must be specified for transformation='iso'"
            )
            M = iso_matrix(coords, radius, latlon=latlon)
        elif transformation == "lbmgls":
            M = lbmgls_matrix(coords, latlon=latlon)
        else:
            raise ValueError(
                f"Unknown transformation: {transformation}. "
                "Use 'nn', 'iso', 'lbmgls', or 'cluster'."
            )

    for var in varlist:
        if var not in df.columns:
            raise ValueError(f"Variable '{var}' not found in DataFrame")

        data = df[var].values

        if np.any(np.isnan(data)):
            print(
                f"Warning: Variable '{var}' contains {np.isnan(data).sum()} missing values. "
                f"Transformed values involving missing data will be NaN."
            )

        transformed = M @ data
        new_name = f"{prefix}{var}"
        df[new_name] = transformed

    return df


def spurhalflife(
    var: str,
    data: pd.DataFrame,
    *,
    lon: str | None = None,
    lat: str | None = None,
    coords_euclidean: Sequence[str] | None = None,
    q: int = 15,
    nrep: int = 100000,
    level: float = 95,
    normdist: bool = False,
    seed: int = 42,
) -> HalfLifeResult:
    """
    Compute confidence interval for spatial half-life.

    The half-life is the distance at which spatial correlation = 1/2.

    Parameters
    ----------
    data : DataFrame
        Input data
    var : str
        Variable to analyze
    lon, lat : str, optional
        Geographic coordinate column names
    coords_euclidean : sequence of str, optional
        Euclidean coordinate column names
    q : int, default 15
        Number of low-frequency weights
    nrep : int, default 100000
        Monte Carlo draws
    level : float, default 95
        Confidence level in percent
    normdist : bool, default False
        If True, return CI in fractions of max pairwise distance.
        If False, return in meters (if latlon) or coordinate units.
    seed : int, optional
        Random seed

    Returns
    -------
    HalfLifeResult
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a pandas DataFrame.")
    if not isinstance(var, str) or not var:
        raise ValueError("`var` must be a non-empty column name.")
    if var not in data.columns:
        raise ValueError(f"Variable '{var}' not found in data.")
    if not (0 < level < 100):
        raise ValueError(f"level={level} must be strictly between 0 and 100.")
    if q < 1:
        raise ValueError(f"q={q} must be >= 1.")
    if nrep < 1:
        raise ValueError(f"nrep={nrep} must be >= 1.")

    Y = data[var].to_numpy(dtype=float)
    if not np.all(np.isfinite(Y)):
        raise ValueError(
            f"Variable '{var}' contains NaN or inf values. All values must be finite."
        )

    coord_info = resolve_spur_coords(
        data=data,
        use_rows=np.ones(len(data), dtype=bool),
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
    )
    coords = coord_info["coords"]
    latlon = coord_info["latlong"]

    Y = Y - Y.mean()

    distmat_raw = get_distance_matrix(coords, latlon=latlon)

    max_dist_norm = distmat_raw.max()
    if max_dist_norm <= 1e-10:
        raise ValueError(
            "All coordinates are identical (or nearly so) — half-life normalization "
            "requires distinct locations."
        )
    distmat = distmat_raw / max_dist_norm

    if latlon:
        max_dist = distmat_raw.max()
    else:
        max_dist = distmat_raw.max()

    emat = np.random.default_rng(seed).standard_normal((q, nrep))

    ci_l, ci_u = spur_persistence(Y, distmat, emat, level / 100)

    if ci_u >= 100:
        ci_u = np.inf

    if not normdist:
        ci_l = ci_l * max_dist if not np.isnan(ci_l) else ci_l
        ci_u = ci_u * max_dist if not np.isinf(ci_u) else ci_u

    return HalfLifeResult(
        ci_lower=ci_l,
        ci_upper=ci_u,
        max_dist=max_dist,
        level=level,
        normdist=normdist,
    )


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
) -> PipelineResult:
    """Run the full SPUR pipeline and return all major outputs.

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
        A `PipelineResult` containing the test results and both fitted branches.

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
        >>> print(result.summary())
    """
    if not isinstance(formula, str) or "~" not in formula:
        raise ValueError("`formula` must be two-sided, e.g. `y ~ x1 + x2`.")

    depvar = formula.split("~", 1)[0].strip()
    if not depvar:
        raise ValueError(
            "`formula` must have a dependent variable on the left-hand side."
        )

    i0 = spurtest_i0(
        depvar,
        data,
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
        q=q,
        nrep=nrep,
        seed=seed,
    )
    i1 = spurtest_i1(
        depvar,
        data,
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
        q=q,
        nrep=nrep,
        seed=seed,
    )

    i0resid = spurtest_i0resid(
        formula,
        data,
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
        q=q,
        nrep=nrep,
        seed=seed,
    )
    i1resid = spurtest_i1resid(
        formula,
        data,
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
        q=q,
        nrep=nrep,
        seed=seed,
    )

    levels_model = smf.ols(formula, data=data).fit()
    levels_scpc = scpc(
        levels_model,
        data,
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
        avc=avc,
        uncond=uncond,
        cvs=cvs,
    )

    transformed_data = spurtransform(
        formula,
        data,
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
        prefix="h_",
        transformation="lbmgls",
    )
    transformed_formula = rewrite_formula_with_prefix(formula, "h_")
    transformed_model = smf.ols(transformed_formula, data=transformed_data).fit()
    transformed_scpc = scpc(
        transformed_model,
        transformed_data,
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
        avc=avc,
        uncond=uncond,
        cvs=cvs,
    )

    return PipelineResult(
        tests=Tests(
            i0=i0,
            i1=i1,
            i0resid=i0resid,
            i1resid=i1resid,
        ),
        fits=Fits(
            levels=RegressionResult(
                model=levels_model,
                scpc=levels_scpc,
            ),
            transformed=RegressionResult(
                model=transformed_model,
                scpc=transformed_scpc,
            ),
        ),
    )
