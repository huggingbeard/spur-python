"""
SPUR Test: Diagnostic tests for spatial unit roots.

Implements four tests from Muller-Watson (2024):
- i1: Test I(1) null (unit root) on a single variable
- i0: Test I(0) null (stationarity) on a single variable
- i1resid: Test I(1) null on regression residuals
- i0resid: Test I(0) null on regression residuals

Reference: Becker, Boll, Voth (2025) SPUR Stata Package
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from dataclasses import dataclass
from spur import get_distance_matrix, get_sigma_lbm, demean_matrix


@dataclass
class SpurTestResult:
    """Container for spurtest results."""

    test_type: str  # 'i1', 'i0', 'i1resid', 'i0resid'
    LR: float  # Likelihood ratio test statistic
    pvalue: float  # P-value
    cv: np.ndarray  # Critical values at 1%, 5%, 10%
    ha_param: float  # Alternative hypothesis parameter

    def summary(self) -> str:
        """Format test results for display."""
        stat_name = "LFUR" if self.test_type.startswith("i1") else "LFST"
        lines = [
            f"Spatial {self.test_type.upper()} Test Results",
            "-" * 45,
            f"Test Statistic ({stat_name}):  {self.LR:9.4f}",
            f"P-value:                {self.pvalue:9.4f}",
            f"CV 1%:                  {self.cv[0]:9.4f}",
            f"CV 5%:                  {self.cv[1]:9.4f}",
            f"CV 10%:                 {self.cv[2]:9.4f}",
            "-" * 45,
        ]
        return "\n".join(lines)


def get_distmat_normalized(coords: np.ndarray, latlon: bool = True) -> np.ndarray:
    """Get distance matrix normalized so max distance = 1."""
    distmat = get_distance_matrix(coords, latlon=latlon)
    return distmat / distmat.max()


def get_R(sigma: np.ndarray, qmax: int) -> np.ndarray:
    """
    Get the top qmax eigenvectors of covariance matrix (sorted by eigenvalue desc).

    Parameters
    ----------
    sigma : ndarray (n, n)
        Symmetric covariance matrix
    qmax : int
        Number of top eigenvectors to return

    Returns
    -------
    ndarray (n, qmax)
        Matrix of top eigenvectors
    """
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    return eigenvectors[:, :qmax]


def get_sigma_dm(distmat: np.ndarray, c: float) -> np.ndarray:
    """
    Compute demeaned exponential covariance matrix.

    sigma(i,j) = exp(-c * distmat(i,j)), then double-demean.
    """
    sigma = np.exp(-c * distmat)
    return demean_matrix(sigma)


def lvech(S: np.ndarray) -> np.ndarray:
    """Extract lower triangular part (below diagonal) as vector."""
    n = S.shape[0]
    # Get indices below diagonal
    i, j = np.tril_indices(n, k=-1)
    return S[i, j]


def getcbar(rhobar: float, distmat: np.ndarray) -> float:
    """
    Bisection method to find c such that mean(exp(-c*d)) = rhobar.

    Parameters
    ----------
    rhobar : float
        Target average correlation
    distmat : ndarray
        Distance matrix

    Returns
    -------
    float
        c value
    """
    vd = lvech(distmat)

    # Initial bounds
    c0 = 10.0
    c1 = 10.0

    # Find c0 such that v(c0) > rhobar
    i1 = False
    jj = 0
    while not i1:
        v = np.mean(np.exp(-c0 * vd))
        i1 = v > rhobar
        if not i1:
            c1 = c0
            c0 = c0 / 2
            jj += 1
        if jj > 500:
            raise ValueError("rhobar too large")

    # Find c1 such that v(c1) < rhobar
    i1 = False
    jj = 0
    while not i1:
        v = np.mean(np.exp(-c1 * vd))
        i1 = v < rhobar
        if not i1:
            c0 = c1
            c1 = 2 * c1
            jj += 1
        if c1 > 10000:
            i1 = True
        if jj > 500:
            raise ValueError("rhobar too small")

    # Bisection (geometric mean)
    while (c1 - c0) > 0.001:
        cm = np.sqrt(c0 * c1)
        v = np.mean(np.exp(-cm * vd))
        if v < rhobar:
            c1 = cm
        else:
            c0 = cm

    return np.sqrt(c0 * c1)


def _cholesky_upper(M: np.ndarray) -> np.ndarray:
    """
    Get upper triangular Cholesky factor.

    Stata's cholesky() returns lower triangular, and the Stata code uses
    cholesky(X)' which is the transpose (upper triangular).
    """
    # numpy returns lower by default, transpose for upper
    L = np.linalg.cholesky(M)
    return L.T


def getpow_qf(om0: np.ndarray, om1: np.ndarray, e: np.ndarray) -> float:
    """
    Compute power of test via quadratic forms.

    Parameters
    ----------
    om0 : ndarray (q, q)
        Null covariance matrix
    om1 : ndarray (q, q)
        Alternative covariance matrix
    e : ndarray (q, nrep)
        Random Monte Carlo draws

    Returns
    -------
    float
        Power (probability of rejecting H0 at 5% level)
    """
    om0i = np.linalg.inv(om0)
    om1i = np.linalg.inv(om1)

    # Upper triangular Cholesky factors
    ch_om0 = _cholesky_upper(om0)
    ch_om1 = _cholesky_upper(om1)
    ch_om0i = _cholesky_upper(om0i)
    ch_om1i = _cholesky_upper(om1i)

    # Transform matrices (matching Stata: ch_om1i * ch_om0')
    # Note: ch_om0' in Stata means transpose of upper-tri = lower-tri = L
    # But cholesky(om0)' is upper, so cholesky(om0)'' = lower (the L)
    # Actually looking more carefully at Stata code:
    # ch_om0 = cholesky(om0)'  -> upper triangular
    # ho = ch_om1i * ch_om0'   -> ch_om1i @ (upper)' = ch_om1i @ lower
    # So ch_om0' in Stata expression = transpose of upper = lower triangular = numpy's L
    ho = ch_om1i @ _cholesky_upper(om0).T
    ha = ch_om0i @ _cholesky_upper(om1).T

    # Quadratic forms
    qe = np.sum(e**2, axis=0)  # sum of squares of each column
    ya_o = ho @ e
    yo_a = ha @ e
    qa_o = np.sum(ya_o**2, axis=0)
    qo_a = np.sum(yo_a**2, axis=0)

    # Likelihood ratios
    lr_o = qe / qa_o
    lr_a = qo_a / qe

    # 95th percentile of lr_o
    cv = np.quantile(lr_o, 0.95)
    pow_ = np.mean(lr_a > cv)

    return pow_


def get_ha_parm_I1(
    om_ho: np.ndarray, distmat: np.ndarray, R: np.ndarray, e: np.ndarray
) -> float:
    """
    Find alternative hypothesis parameter c that yields ~50% power for I(1) test.
    """
    pow50 = 0.5
    pow_ = 1.0
    ctry = getcbar(0.95, distmat)

    # Step 1: Decrease c until pow < 0.5
    while pow_ > pow50:
        c = ctry
        sigdm_c = get_sigma_dm(distmat, c)
        om_c = R.T @ sigdm_c @ R
        pow_ = getpow_qf(om_ho, om_c, e)
        ctry = ctry / 2

    c1 = c

    # Step 2: Increase c until pow > 0.5
    pow_ = 0.0
    ctry = getcbar(0.01, distmat)
    while pow_ < pow50:
        c = ctry
        sigdm_c = get_sigma_dm(distmat, c)
        om_c = R.T @ sigdm_c @ R
        pow_ = getpow_qf(om_ho, om_c, e)
        ctry = 2 * ctry

    c2 = c

    # Step 3: Bisection
    ii = 0
    while abs(pow_ - pow50) > 0.01:
        c = (c1 + c2) / 2
        sigdm_c = get_sigma_dm(distmat, c)
        om_c = R.T @ sigdm_c @ R
        pow_ = getpow_qf(om_ho, om_c, e)

        if pow_ > pow50:
            c2 = c
        else:
            c1 = c

        ii += 1
        if ii > 20:
            break

    return c


def get_ha_parm_I0(
    om_ho: np.ndarray, om_i0: np.ndarray, om_bm: np.ndarray, e: np.ndarray
) -> float:
    """
    Find alternative hypothesis parameter g that yields ~50% power for I(0) test.
    """
    pow_ = 1.0
    gtry = 1.0

    # Step 1: Find lower bound g1
    while pow_ > 0.5:
        g = gtry
        pow_ = getpow_qf(om_ho, om_i0 + g * om_bm, e)
        gtry = g / 2
    g1 = g

    # Step 2: Find upper bound g2
    pow_ = 0.0
    gtry = 30.0
    while pow_ < 0.5:
        g = gtry
        pow_ = getpow_qf(om_ho, om_i0 + g * om_bm, e)
        gtry = g * 2
    g2 = g

    # Step 3: Bisection
    ii = 1  # DG: changed from 0 for consistency with stata
    # https://github.com/pdavidboll/SPUR/blob/main/mata/get_ha_parm_I0.mata#L33
    while abs(pow_ - 0.5) > 0.01:
        g = (g1 + g2) / 2
        pow_ = getpow_qf(om_ho, om_i0 + g * om_bm, e)
        if pow_ > 0.5:
            g2 = g
        else:
            g1 = g
        ii += 1
        if ii > 20:
            break

    return g


def spatial_i1_test(
    Y: np.ndarray, distmat: np.ndarray, emat: np.ndarray
) -> SpurTestResult:
    """
    Test H0: I(1) (unit root) for variable Y.

    Parameters
    ----------
    Y : ndarray (n,)
        Variable to test
    distmat : ndarray (n, n)
        Normalized distance matrix
    emat : ndarray (q, nrep)
        Monte Carlo draws (standard normal)

    Returns
    -------
    SpurTestResult
    """
    q = emat.shape[0]
    n = distmat.shape[0]

    # BM covariance matrix (demeaned)
    sigdm_bm = demean_matrix(get_sigma_lbm(distmat))

    # Eigenvectors for low-frequency weights
    R = get_R(sigdm_bm, q)

    # Null hypothesis covariance
    om_ho = R.T @ sigdm_bm @ R

    # Alternative hypothesis parameter (~50% power)
    ha_parm = get_ha_parm_I1(om_ho, distmat, R, emat)
    sigdm_ha = get_sigma_dm(distmat, ha_parm)
    om_ha = R.T @ sigdm_ha @ R

    # Simulate LR distribution under H0
    ch_om_ho = _cholesky_upper(om_ho)
    omi_ho = np.linalg.inv(om_ho)
    omi_ha = np.linalg.inv(om_ha)
    ch_omi_ho = _cholesky_upper(omi_ho)
    ch_omi_ha = _cholesky_upper(omi_ha)

    # Draws under H0: y_ho has distribution om_ho
    y_ho = ch_om_ho.T @ emat
    y_ho_ho = ch_omi_ho @ y_ho
    y_ho_ha = ch_omi_ha @ y_ho
    q_ho_ho = np.sum(y_ho_ho**2, axis=0)
    q_ho_ha = np.sum(y_ho_ha**2, axis=0)
    lr_ho = q_ho_ho / q_ho_ha

    # Critical values
    sz_vec = np.array([0.01, 0.05, 0.10])
    cv_vec = np.quantile(lr_ho, 1 - sz_vec)

    # Test statistic for data
    X = Y - np.mean(Y)
    P = R.T @ X
    LR = float((P.T @ omi_ho @ P) / (P.T @ omi_ha @ P))
    pvalue = float(np.mean(lr_ho > LR))

    return SpurTestResult(
        test_type="i1", LR=LR, pvalue=pvalue, cv=cv_vec, ha_param=ha_parm
    )


def spatial_i0_test(
    Y: np.ndarray, distmat: np.ndarray, emat: np.ndarray
) -> SpurTestResult:
    """
    Test H0: I(0) (stationarity) for variable Y.

    Parameters
    ----------
    Y : ndarray (n,)
        Variable to test
    distmat : ndarray (n, n)
        Normalized distance matrix
    emat : ndarray (q, nrep)
        Monte Carlo draws

    Returns
    -------
    SpurTestResult
    """
    q = emat.shape[0]
    n = distmat.shape[0]

    # BM covariance matrix (demeaned)
    sigdm_bm = demean_matrix(get_sigma_lbm(distmat))

    # Eigenvectors
    R = get_R(sigdm_bm, q)

    # om_ho for rhobar = 0.001 (near-stationary)
    rho = 0.001
    c = getcbar(rho, distmat)
    sigdm_rho = get_sigma_dm(distmat, c)

    om_rho = R.T @ sigdm_rho @ R
    om_bm = R.T @ sigdm_bm @ R

    # Find alternative parameter
    om_i0 = om_rho
    om_ho = om_rho
    ha_parm = get_ha_parm_I0(om_ho, om_i0, om_bm, emat)
    om_ha = om_i0 + ha_parm * om_bm

    # Cholesky factors
    ch_omi_ho = _cholesky_upper(np.linalg.inv(om_ho))
    ch_omi_ha = _cholesky_upper(np.linalg.inv(om_ha))

    # LR statistic for data
    X = Y - np.mean(Y)
    P = R.T @ X
    y_P_ho = ch_omi_ho @ P
    y_P_ha = ch_omi_ha @ P
    q_P_ho = np.sum(y_P_ho**2)
    q_P_ha = np.sum(y_P_ha**2)
    LR = float(q_P_ho / q_P_ha)

    # Grid over rho for least-favorable p-value
    rho_min = 0.0001
    rho_max = 0.03
    n_rho = 30
    rho_grid = np.linspace(rho_min, rho_max, n_rho)

    ch_om_ho_list = []
    for i in range(n_rho):
        rho_i = rho_grid[i]
        if rho_i > 0:
            c_i = getcbar(rho_i, distmat)
            sigdm_ho_i = get_sigma_dm(distmat, c_i)
            om_ho_i = R.T @ sigdm_ho_i @ R
        else:
            om_ho_i = np.eye(q)
        ch_om_ho_list.append(_cholesky_upper(om_ho_i))

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

    # Least-favorable p-value and critical values (max over grid)
    cvalue = cvalue_mat.max(axis=0)
    pvalue = float(pvalue_vec.max())

    return SpurTestResult(
        test_type="i0", LR=LR, pvalue=pvalue, cv=cvalue, ha_param=ha_parm
    )


def get_sigma_residual(distmat: np.ndarray, c: float, M: np.ndarray) -> np.ndarray:
    """
    Compute residualized covariance matrix: M @ exp(-c * distmat) @ M.T

    Used for I(1) and I(0) residual tests. The projection matrix M already
    accounts for the regressors, so we project the spatial covariance.
    """
    sigma = np.exp(-c * distmat)
    return M @ sigma @ M.T


def get_ha_parm_I1_residual(
    om_ho: np.ndarray, distmat: np.ndarray, R: np.ndarray, e: np.ndarray, M: np.ndarray
) -> float:
    """
    Find alternative parameter c yielding ~50% power for I(1) residual test.
    Same structure as get_ha_parm_I1 but uses get_sigma_residual.
    """
    pow50 = 0.5
    pow_ = 1.0
    ctry = getcbar(0.95, distmat)

    while pow_ > pow50:
        c = ctry
        sigdm_c = get_sigma_residual(distmat, c, M)
        om_c = R.T @ sigdm_c @ R
        pow_ = getpow_qf(om_ho, om_c, e)
        ctry = ctry / 2
    c1 = c

    pow_ = 0.0
    ctry = getcbar(0.01, distmat)
    while pow_ < pow50:
        c = ctry
        sigdm_c = get_sigma_residual(distmat, c, M)
        om_c = R.T @ sigdm_c @ R
        pow_ = getpow_qf(om_ho, om_c, e)
        ctry = 2 * ctry
    c2 = c

    ii = 0
    while abs(pow_ - pow50) > 0.01:
        c = (c1 + c2) / 2
        sigdm_c = get_sigma_residual(distmat, c, M)
        om_c = R.T @ sigdm_c @ R
        pow_ = getpow_qf(om_ho, om_c, e)
        if pow_ > pow50:
            c2 = c
        else:
            c1 = c
        ii += 1
        if ii > 20:
            break

    return c


def spatial_i1_test_residual(
    Y: np.ndarray, X_in: np.ndarray, distmat: np.ndarray, emat: np.ndarray
) -> SpurTestResult:
    """
    Test H0: I(1) for residuals of regression Y ~ X.

    Note: The test uses the projection matrix M = I - X(X'X)^(-1)X' built into
    the covariance, NOT the OLS residuals directly. The test statistic uses
    Y - mean(Y), and the covariance accounts for the regression structure.
    """
    q = emat.shape[0]
    n = distmat.shape[0]

    # Projection matrix (annihilator)
    XtX_inv = np.linalg.inv(X_in.T @ X_in)
    M = np.eye(n) - X_in @ XtX_inv @ X_in.T

    # BM covariance matrix (approximation for demeaned value)
    rho_bm = 0.999
    c_bm = getcbar(rho_bm, distmat)
    sigdm_bm = get_sigma_residual(distmat, c_bm, M)

    # Eigenvectors
    R = get_R(sigdm_bm, q)
    om_ho = R.T @ sigdm_bm @ R

    # Alternative parameter
    ha_parm = get_ha_parm_I1_residual(om_ho, distmat, R, emat, M)
    sigdm_ha = get_sigma_residual(distmat, ha_parm, M)
    om_ha = R.T @ sigdm_ha @ R

    # Simulate LR distribution
    ch_om_ho = _cholesky_upper(om_ho)
    omi_ho = np.linalg.inv(om_ho)
    omi_ha = np.linalg.inv(om_ha)
    ch_omi_ho = _cholesky_upper(omi_ho)
    ch_omi_ha = _cholesky_upper(omi_ha)

    y_ho = ch_om_ho.T @ emat
    y_ho_ho = ch_omi_ho @ y_ho
    y_ho_ha = ch_omi_ha @ y_ho
    q_ho_ho = np.sum(y_ho_ho**2, axis=0)
    q_ho_ha = np.sum(y_ho_ha**2, axis=0)
    lr_ho = q_ho_ho / q_ho_ha

    # Critical values
    sz_vec = np.array([0.01, 0.05, 0.10])
    cv_vec = np.quantile(lr_ho, 1 - sz_vec)

    # Test statistic: use Y - mean(Y), not OLS residuals
    X = Y - np.mean(Y)
    P = R.T @ X
    LR = float((P.T @ omi_ho @ P) / (P.T @ omi_ha @ P))
    pvalue = float(np.mean(lr_ho > LR))

    return SpurTestResult(
        test_type="i1resid", LR=LR, pvalue=pvalue, cv=cv_vec, ha_param=ha_parm
    )


def spatial_i0_test_residual(
    Y: np.ndarray, X_in: np.ndarray, distmat: np.ndarray, emat: np.ndarray
) -> SpurTestResult:
    """Test H0: I(0) for residuals of regression Y ~ X."""
    q = emat.shape[0]
    n = distmat.shape[0]

    # Projection matrix
    XtX_inv = np.linalg.inv(X_in.T @ X_in)
    M = np.eye(n) - X_in @ XtX_inv @ X_in.T

    # BM covariance using c_bm from rho_bm=0.999
    rho_bm = 0.999
    c_bm = getcbar(rho_bm, distmat)
    sigdm_bm = get_sigma_residual(distmat, c_bm, M)

    R = get_R(sigdm_bm, q)

    # om_ho for rho=0.001
    rho = 0.001
    c = getcbar(rho, distmat)
    sigdm_rho = get_sigma_residual(distmat, c, M)

    om_rho = R.T @ sigdm_rho @ R
    om_bm = R.T @ sigdm_bm @ R

    om_i0 = om_rho
    om_ho = om_rho
    ha_parm = get_ha_parm_I0(om_ho, om_i0, om_bm, emat)
    om_ha = om_i0 + ha_parm * om_bm

    ch_omi_ho = _cholesky_upper(np.linalg.inv(om_ho))
    ch_omi_ha = _cholesky_upper(np.linalg.inv(om_ha))

    # LR for data: uses Y - mean(Y)
    X = Y - np.mean(Y)
    P = R.T @ X
    y_P_ho = ch_omi_ho @ P
    y_P_ha = ch_omi_ha @ P
    q_P_ho = np.sum(y_P_ho**2)
    q_P_ha = np.sum(y_P_ha**2)
    LR = float(q_P_ho / q_P_ha)

    # Grid over rho
    rho_min = 0.0001
    rho_max = 0.03
    n_rho = 30
    rho_grid = np.linspace(rho_min, rho_max, n_rho)

    ch_om_ho_list = []
    for i in range(n_rho):
        rho_i = rho_grid[i]
        if rho_i > 0:
            c_i = getcbar(rho_i, distmat)
            sigdm_ho_i = get_sigma_residual(distmat, c_i, M)
            om_ho_i = R.T @ sigdm_ho_i @ R
        else:
            om_ho_i = np.eye(q)
        ch_om_ho_list.append(_cholesky_upper(om_ho_i))

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

    return SpurTestResult(
        test_type="i0resid", LR=LR, pvalue=pvalue, cv=cvalue, ha_param=ha_parm
    )


def spurtest(
    df: pd.DataFrame,
    test_type: str,
    varname: str,
    coord_cols: List[str],
    indepvars: Optional[List[str]] = None,
    q: int = 15,
    nrep: int = 100000,
    latlon: bool = True,
    seed: Optional[int] = None,
) -> SpurTestResult:
    """
    Main user-facing function for SPUR diagnostic tests.

    Parameters
    ----------
    df : DataFrame
        Input data
    test_type : str
        One of 'i1', 'i0', 'i1resid', 'i0resid'
    varname : str
        Variable to test (dependent variable for residual tests)
    coord_cols : list of str
        [lat_col, lon_col] or [x_col, y_col]
    indepvars : list of str, optional
        Independent variables (for residual tests)
    q : int, default 15
        Number of low-frequency weights (eigenvectors)
    nrep : int, default 100000
        Monte Carlo draws
    latlon : bool, default True
        If True, use Haversine distance
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    SpurTestResult
    """
    # Validate inputs
    if test_type not in ("i1", "i0", "i1resid", "i0resid"):
        raise ValueError(
            f"Unknown test_type: {test_type}. Use 'i1', 'i0', 'i1resid', or 'i0resid'."
        )

    # Extract data
    coords = df[coord_cols].values
    Y = df[varname].values

    # Get normalized distance matrix
    distmat = get_distmat_normalized(coords, latlon=latlon)

    # Generate Monte Carlo draws
    rng = np.random.default_rng(seed)
    emat = rng.standard_normal((q, nrep))

    # Run test
    if test_type == "i1":
        result = spatial_i1_test(Y, distmat, emat)
    elif test_type == "i0":
        result = spatial_i0_test(Y, distmat, emat)
    elif test_type in ("i1resid", "i0resid"):
        # Build X_in with constant (always included like Stata)
        n = len(Y)
        if indepvars is None or len(indepvars) == 0:
            X_in = np.ones((n, 1))
        else:
            X = df[indepvars].values
            X_in = np.column_stack([np.ones(n), X])

        if test_type == "i1resid":
            result = spatial_i1_test_residual(Y, X_in, distmat, emat)
        else:
            result = spatial_i0_test_residual(Y, X_in, distmat, emat)

    return result


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    n = 30
    lat = np.random.uniform(40, 50, n)
    lon = np.random.uniform(5, 15, n)
    y = np.random.randn(n) + 0.2 * lat  # mild spatial correlation

    df = pd.DataFrame({"lat": lat, "lon": lon, "y": y})

    print("Testing spurtest i1 (small N=30, nrep=1000)...")
    result = spurtest(df, "i1", "y", ["lat", "lon"], q=10, nrep=1000, seed=42)
    print(result.summary())

    print("\nTesting spurtest i0 (small N=30, nrep=1000)...")
    result = spurtest(df, "i0", "y", ["lat", "lon"], q=10, nrep=1000, seed=42)
    print(result.summary())
