"""
SPUR Half-Life: Confidence sets for spatial half-life.

Implements spurhalflife from Muller-Watson (2024), which constructs confidence
intervals for the half-life parameter in the Spatial Local-to-Unity model.

The half-life is the distance at which spatial correlation equals 1/2.

Reference: Becker, Boll, Voth (2025) SPUR Stata Package
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from dataclasses import dataclass
from scipy.special import gamma as gamma_func

from spur import get_distance_matrix
from .spurtest import (
    get_R,
    get_sigma_dm,
    getcbar,
    _cholesky_upper,
)


@dataclass
class HalfLifeResult:
    """Container for spurhalflife results."""

    ci_lower: float  # Lower bound of CI
    ci_upper: float  # Upper bound (inf if unbounded)
    max_dist: float  # Max pairwise distance in sample (for unit conversion)
    level: float  # Confidence level (e.g., 0.95)
    normdist: bool  # If True, CI is in fractions of max_dist; else meters

    def summary(self) -> str:
        """Format results for display."""
        units = "fractions of max distance" if self.normdist else "meters"
        level_pct = int(self.level * 100)
        upper_str = "inf" if np.isinf(self.ci_upper) else f"{self.ci_upper:.4f}"
        lines = [
            f"Spatial half-life {level_pct}% confidence interval ({units})",
            "-" * 45,
            f"Lower bound: {self.ci_lower:.4f}",
            f"Upper bound: {upper_str}",
            f"Max distance in sample: {self.max_dist:.4f}",
            "-" * 45,
        ]
        return "\n".join(lines)


def _c_ci(
    Y: np.ndarray,
    distmat: np.ndarray,
    emat: np.ndarray,
    c_grid_ho: np.ndarray,
    c_grid_ha: np.ndarray,
) -> np.ndarray:
    """
    Compute p-values over c_grid_ho using LR test with averaged Ha likelihood.

    Returns
    -------
    ndarray (n_c_ho,)
        P-value at each c in c_grid_ho
    """
    q = emat.shape[0]
    # n = distmat.shape[0] # DG unused
    n_c_ho = len(c_grid_ho)
    n_c_ha = len(c_grid_ha)

    # BM covariance (approximation for demeaned values)
    rho_bm = 0.999
    c_bm = getcbar(rho_bm, distmat)
    sigdm_bm = get_sigma_dm(distmat, c_bm)

    # Eigenvectors
    R = get_R(sigdm_bm, q)

    # Precompute Omega matrices and constants for c_grid_ho
    ch_om_list = []
    ch_omi_list = []
    const_den_list = np.zeros(n_c_ho)
    const_factor = 0.5 * gamma_func(q / 2) / (np.pi ** (q / 2))

    for i in range(n_c_ho):
        c = c_grid_ho[i]
        sigdm = get_sigma_dm(distmat, c)
        om = R.T @ sigdm @ R
        ch_om_list.append(_cholesky_upper(om))
        omi = np.linalg.inv(om)
        ch_omi_list.append(_cholesky_upper(omi))
        const_den_list[i] = np.sqrt(np.linalg.det(omi)) * const_factor

    # Precompute for c_grid_ha
    ch_omi_ha_list = []
    const_den_ha_list = np.zeros(n_c_ha)

    for i in range(n_c_ha):
        c = c_grid_ha[i]
        sigdm = get_sigma_dm(distmat, c)
        om = R.T @ sigdm @ R
        omi = np.linalg.inv(om)
        ch_omi_ha_list.append(_cholesky_upper(omi))
        const_den_ha_list[i] = np.sqrt(np.linalg.det(omi)) * const_factor

    # Test statistic loop
    X = R.T @ Y
    pv_mat = np.zeros(n_c_ho)

    for i in range(n_c_ho):
        # Draws under null i
        ch_null = ch_om_list[i]  # upper triangular
        e = ch_null.T @ emat  # shape (q, nrep)

        # Null density for draws
        ch_omi = ch_omi_list[i]
        const_den = const_den_list[i]
        xc = ch_omi @ e
        den_ho = const_den * (np.sum(xc**2, axis=0) ** (-q / 2))

        # Null density for data
        Xc = ch_omi @ X
        den_ho_X = const_den * (np.sum(Xc**2) ** (-q / 2))

        # Alternative densities (average over ha grid)
        den_ha_mat = np.zeros((emat.shape[1], n_c_ha))
        den_ha_mat_X = np.zeros(n_c_ha)

        for j in range(n_c_ha):
            ch_omi_j = ch_omi_ha_list[j]
            const_den_j = const_den_ha_list[j]
            xc_j = ch_omi_j @ e
            den_ha_mat[:, j] = const_den_j * (np.sum(xc_j**2, axis=0) ** (-q / 2))
            Xc_j = ch_omi_j @ X
            den_ha_mat_X[j] = const_den_j * (np.sum(Xc_j**2) ** (-q / 2))

        den_ha_avg = den_ha_mat.mean(axis=1)
        lr = den_ha_avg / den_ho

        # Data LR
        den_ha_avg_X = den_ha_mat_X.mean()
        lr_X = den_ha_avg_X / den_ho_X

        # P-value
        pv_mat[i] = np.mean(lr > lr_X)

    return pv_mat


def spatial_persistence(
    Y: np.ndarray, distmat: np.ndarray, emat: np.ndarray, level: float
) -> tuple:
    """
    Compute confidence interval for half-life parameter.

    Parameters
    ----------
    Y : ndarray (n,)
        Variable (centered)
    distmat : ndarray (n, n)
        Normalized distance matrix
    emat : ndarray (q, nrep)
        Monte Carlo draws
    level : float
        Confidence level in [0, 1]

    Returns
    -------
    tuple (ci_lower, ci_upper)
        Bounds on half-life (normalized; i.e., fraction of max distance)
    """
    # Grid of half-lives under H0 (dense near 0, sparse at tail)
    n_hl = 100
    hl_grid_ho = np.concatenate(
        [np.linspace(0.001, 1, n_hl), np.linspace(1.01, 3, 30), [100.0]]
    )

    # Grid of half-lives under Ha
    n_hl_ha = 50
    hl_grid_ha = np.linspace(0.001, 1.0, n_hl_ha)

    # Convert half-life to c: exp(-c * hl) = 0.5 => c = -log(0.5) / hl
    c_grid_ho = -np.log(0.5) / hl_grid_ho
    c_grid_ha = -np.log(0.5) / hl_grid_ha

    # Compute p-values
    pv_mat = _c_ci(Y, distmat, emat, c_grid_ho, c_grid_ha)

    # CI: half-lives where p-value > 1 - level
    ii = pv_mat > (1 - level)
    if not ii.any():
        # No half-life in CI - return point estimate area
        return np.nan, np.nan

    hl_ci = hl_grid_ho[ii]
    return float(hl_ci.min()), float(hl_ci.max())


def spurhalflife(
    df: pd.DataFrame,
    varname: str,
    coord_cols: List[str],
    q: int = 15,
    nrep: int = 100000,
    level: float = 0.95,
    latlon: bool = True,
    normdist: bool = False,
    seed: Optional[int] = None,
) -> HalfLifeResult:
    """
    Compute confidence interval for spatial half-life.

    The half-life is the distance at which spatial correlation = 1/2.

    Parameters
    ----------
    df : DataFrame
        Input data
    varname : str
        Variable to analyze
    coord_cols : list of str
        [lat_col, lon_col] or [x_col, y_col]
    q : int, default 15
        Number of low-frequency weights
    nrep : int, default 100000
        Monte Carlo draws
    level : float, default 0.95
        Confidence level
    latlon : bool, default True
        If True, use Haversine distance
    normdist : bool, default False
        If True, return CI in fractions of max pairwise distance.
        If False, return in meters (if latlon) or coordinate units.
    seed : int, optional
        Random seed

    Returns
    -------
    HalfLifeResult
    """
    # Validate parameters
    if not (0 < level < 1):
        raise ValueError(f"level={level} must be strictly between 0 and 1.")
    if q < 1:
        raise ValueError(f"q={q} must be >= 1.")
    if nrep < 1:
        raise ValueError(f"nrep={nrep} must be >= 1.")

    # Extract data
    coords = df[coord_cols].values
    Y = df[varname].values

    # Center Y
    Y = Y - Y.mean()

    # Distance matrices
    distmat_raw = get_distance_matrix(coords, latlon=latlon)

    # Stata's getcbar uses a normalized distmat (max = 1) where latlon distances
    # come out in units of "fraction of max distance". The spatial_persistence
    # function expects a normalized distmat.
    max_dist_norm = distmat_raw.max()
    if max_dist_norm <= 1e-10:
        raise ValueError(
            "All coordinates are identical (or nearly so) — half-life normalization "
            "requires distinct locations."
        )
    distmat = distmat_raw / max_dist_norm

    # For converting CI to meters, Stata uses:
    # max_dist = max(getdistmat(s)) * pi * 6371000.009 * 2
    # Note: Stata's getdistmat already normalizes by pi for latlon, so max
    # comes out as fraction of half circumference. Multiplying by pi * R * 2
    # gives great-circle distance in meters.
    # Our get_distance_matrix returns raw distance in meters for latlon, so:
    if latlon:
        max_dist = distmat_raw.max()  # already in meters
    else:
        max_dist = distmat_raw.max()  # in coordinate units

    # Generate MC draws
    rng = np.random.default_rng(seed)
    emat = rng.standard_normal((q, nrep))

    # Compute CI (in normalized units)
    ci_l, ci_u = spatial_persistence(Y, distmat, emat, level)

    # Upper bound check: if at the tail (100), treat as infinity
    if ci_u >= 100:
        ci_u = np.inf

    # Convert to requested units
    if not normdist:
        ci_l = ci_l * max_dist if not np.isnan(ci_l) else ci_l
        ci_u = ci_u * max_dist if not np.isinf(ci_u) else ci_u

    return HalfLifeResult(
        ci_lower=ci_l, ci_upper=ci_u, max_dist=max_dist, level=level, normdist=normdist
    )


if __name__ == "__main__":
    # Quick self-test
    np.random.seed(42)
    n = 30
    lat = np.random.uniform(40, 50, n)
    lon = np.random.uniform(5, 15, n)
    y = np.cumsum(np.random.randn(n)) + 0.5 * lat

    df = pd.DataFrame({"lat": lat, "lon": lon, "y": y})

    print("Spatial half-life test (n=30, nrep=5000)...")
    result = spurhalflife(df, "y", ["lat", "lon"], q=10, nrep=5000, seed=42)
    print(result.summary())
