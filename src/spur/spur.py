"""
SPUR: Spatial Unit Root Transformations for Python

A Python implementation of Muller-Watson spatial differencing methods for
diagnosing and correcting spatial unit roots in regressions.

Reference: Becker, Boll, Voth (2025) - SPUR Stata Package

This module implements `spurtransform` - the core spatial differencing
functionality that removes spatial unit roots from variables.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional


def haversine_distance(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    Compute great-circle distance between points using Haversine formula.

    Parameters
    ----------
    lat1, lon1 : array-like
        Latitude and longitude of first point(s) in degrees
    lat2, lon2 : array-like
        Latitude and longitude of second point(s) in degrees

    Returns
    -------
    ndarray
        Distance in meters
    """
    R = 6371000  # Earth radius in meters

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def get_distance_matrix(coords: np.ndarray, latlon: bool = True) -> np.ndarray:
    """
    Compute pairwise distance matrix between all observations.

    Parameters
    ----------
    coords : ndarray of shape (n, 2)
        Coordinates array. If latlon=True, columns are [latitude, longitude].
        If latlon=False, columns are [x, y].
    latlon : bool, default True
        If True, use Haversine formula for great-circle distance (in meters).
        If False, use Euclidean distance.

    Returns
    -------
    ndarray of shape (n, n)
        Symmetric distance matrix where entry (i,j) is distance between
        observations i and j.
    """
    # n = coords.shape[0] # DG: unused

    if latlon:
        # Haversine: vectorized computation for all pairs
        lat = coords[:, 0]
        lon = coords[:, 1]

        # Create meshgrid for all pairs
        lat1, lat2 = np.meshgrid(lat, lat, indexing="ij")
        lon1, lon2 = np.meshgrid(lon, lon, indexing="ij")

        distmat = haversine_distance(lat1, lon1, lat2, lon2)
    else:
        # Euclidean distance
        from scipy.spatial.distance import cdist

        distmat = cdist(coords, coords, metric="euclidean")

    return distmat


def nn_matrix(coords: np.ndarray, latlon: bool = True) -> np.ndarray:
    """
    Compute nearest-neighbor transformation matrix.

    For each observation, identifies its nearest neighbor and creates a
    row-normalized weight matrix. The transformation matrix is I - W,
    which effectively differences each observation from its nearest neighbor.

    Parameters
    ----------
    coords : ndarray of shape (n, 2)
        Coordinates array
    latlon : bool, default True
        If True, use Haversine distance; if False, use Euclidean

    Returns
    -------
    ndarray of shape (n, n)
        Transformation matrix M = I - W where W is the normalized
        nearest-neighbor weight matrix
    """
    n = coords.shape[0]
    distmat = get_distance_matrix(coords, latlon=latlon)

    # Set diagonal to infinity to exclude self as nearest neighbor
    np.fill_diagonal(distmat, np.inf)

    # Find nearest neighbor for each observation
    # Handle ties by identifying all points at minimum distance
    min_dist = np.min(distmat, axis=1, keepdims=True)

    # Binary indicator: 1 if this neighbor is at minimum distance (handles ties)
    NN = (distmat == min_dist).astype(float)

    # Row-normalize (divide by number of tied nearest neighbors)
    row_sums = NN.sum(axis=1, keepdims=True)
    NN = NN / row_sums

    # Transformation matrix: I - NN
    M = np.eye(n) - NN

    return M


def iso_matrix(coords: np.ndarray, radius: float, latlon: bool = True) -> np.ndarray:
    """
    Compute isotropic transformation matrix.

    For each observation, identifies all neighbors within the specified radius
    and creates a row-normalized weight matrix. The transformation matrix is
    I - W, which differences each observation from the average of its neighbors.

    Parameters
    ----------
    coords : ndarray of shape (n, 2)
        Coordinates array
    radius : float
        Radius threshold for neighbor inclusion. Units depend on latlon:
        - If latlon=True: radius in meters
        - If latlon=False: radius in same units as coordinates
    latlon : bool, default True
        If True, use Haversine distance; if False, use Euclidean

    Returns
    -------
    ndarray of shape (n, n)
        Transformation matrix M = I - W where W is the normalized
        isotropic weight matrix

    Notes
    -----
    Observations with no neighbors within the radius have their transformed
    value set to 0 (matching Stata's behavior). The entire row of M is set
    to zeros for these observations.
    """
    n = coords.shape[0]
    distmat = get_distance_matrix(coords, latlon=latlon)

    # Identify neighbors within radius (excluding self)
    neighbors = (distmat < radius) & (distmat > 0)

    # Row sums for normalization
    row_sums = neighbors.sum(axis=1, keepdims=True).astype(float)

    # Identify isolated observations (no neighbors)
    no_neighbors = (row_sums == 0).flatten()

    # Avoid division by zero - set to 1 temporarily
    row_sums[row_sums == 0] = 1

    # Normalized weight matrix
    W = neighbors.astype(float) / row_sums

    # Transformation matrix: I - W
    M = np.eye(n) - W

    # For isolated observations: set entire row to 0 (Stata behavior)
    # This means transformed value = 0 for these observations
    M[no_neighbors, :] = 0

    # Report if any observations have no neighbors
    if no_neighbors.any():
        n_isolated = no_neighbors.sum()
        print(
            f"Warning: {n_isolated} observation(s) have no neighbors within radius {radius}. "
            f"These observations will be set to 0."
        )

    return M


def demean_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Double-demean a matrix (row then column demeaning).

    Parameters
    ----------
    mat : ndarray of shape (n, n)
        Input matrix

    Returns
    -------
    ndarray of shape (n, n)
        Demeaned matrix
    """
    # Row demean
    mat = mat - mat.mean(axis=1, keepdims=True)
    # Column demean
    mat = mat - mat.mean(axis=0, keepdims=True)
    return mat


def get_sigma_lbm(distmat: np.ndarray) -> np.ndarray:
    """
    Compute the Lévy Brownian Motion (LBM) covariance matrix.

    Uses the first observation as the origin. The covariance between
    locations i and j is: 0.5 * (d(i,0) + d(j,0) - d(i,j))

    Parameters
    ----------
    distmat : ndarray of shape (n, n)
        Normalized distance matrix (max distance = 1)

    Returns
    -------
    ndarray of shape (n, n)
        LBM covariance matrix
    """
    # n = distmat.shape[0] # DG: n not needed
    # d[:,0] broadcast as column, d[0,:] broadcast as row
    sigma_lbm = 0.5 * (distmat[:, 0:1] + distmat[0:1, :] - distmat)
    return sigma_lbm


def lbmgls_matrix(coords: np.ndarray, latlon: bool = True) -> np.ndarray:
    """
    Compute the LBM-GLS transformation matrix.

    This is the default transformation recommended by Muller & Watson (2024).
    It uses GLS based on the covariance matrix of a Lévy-Brownian motion.

    Parameters
    ----------
    coords : ndarray of shape (n, 2)
        Coordinates array
    latlon : bool, default True
        If True, use Haversine distance; if False, use Euclidean

    Returns
    -------
    ndarray of shape (n, n)
        LBM-GLS transformation matrix

    Notes
    -----
    Algorithm:
    1. Compute normalized distance matrix (max = 1)
    2. Compute LBM covariance matrix
    3. Double-demean the covariance matrix
    4. Eigendecomposition
    5. GLS transform: V @ diag(1/sqrt(eigenvalues)) @ V'
    """
    small = 1e-10

    # Get distance matrix and normalize
    distmat = get_distance_matrix(coords, latlon=latlon)
    distmat = distmat / distmat.max()

    # Get demeaned LBM covariance
    sigma_lbm = get_sigma_lbm(distmat)
    sigma_lbm_dm = demean_matrix(sigma_lbm)

    # Eigendecomposition (symmetric matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(sigma_lbm_dm)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Keep only eigenvalues > small (filter near-zero)
    mask = eigenvalues > small
    eigenvalues = eigenvalues[mask]
    eigenvectors = eigenvectors[:, mask]

    # GLS transformation: V @ diag(1/sqrt(eigenvalues)) @ V'
    dsi = 1.0 / np.sqrt(eigenvalues)
    LBMGLS_mat = eigenvectors @ np.diag(dsi) @ eigenvectors.T

    return LBMGLS_mat


def cluster_matrix(cluster: np.ndarray) -> np.ndarray:
    """
    Compute cluster demeaning transformation matrix.

    This is equivalent to within-cluster demeaning (like fixed effects).
    Each observation is differenced from its cluster mean.

    Parameters
    ----------
    cluster : ndarray of shape (n,)
        Cluster identifiers for each observation

    Returns
    -------
    ndarray of shape (n, n)
        Cluster transformation matrix
    """
    n = len(cluster)

    # Create indicator matrix: clust_mat[i,j] = 1 if same cluster
    clust_mat = (cluster.reshape(-1, 1) == cluster.reshape(1, -1)).astype(float)

    # Row normalize (divide by cluster size)
    clust_mat = clust_mat / clust_mat.sum(axis=1, keepdims=True)

    # Transformation: I - cluster_average
    M = np.eye(n) - clust_mat

    # Report cluster info
    n_clusters = len(np.unique(cluster))
    print(f"Number of observations: {n}, number of clusters: {n_clusters}")

    return M


def transform(
    data: np.ndarray,
    coords: np.ndarray,
    method: str = "nn",
    radius: Optional[float] = None,
    latlon: bool = True,
    cluster: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply spatial differencing transformation to data.

    Parameters
    ----------
    data : ndarray of shape (n,) or (n, k)
        Data to transform. Can be a single variable (1D) or multiple
        variables (2D with observations in rows).
    coords : ndarray of shape (n, 2)
        Coordinates for each observation. Not used for method='cluster'.
    method : str, default 'nn'
        Transformation method:
        - 'nn': nearest-neighbor differencing
        - 'iso': isotropic (radius-based) differencing
        - 'lbmgls': LBM-GLS transformation (recommended by Muller-Watson)
        - 'cluster': within-cluster demeaning
    radius : float, optional
        Radius for isotropic method (required if method='iso')
    latlon : bool, default True
        If True, use Haversine distance; if False, use Euclidean
    cluster : ndarray, optional
        Cluster identifiers (required if method='cluster')

    Returns
    -------
    ndarray
        Transformed data with same shape as input
    """
    if method == "nn":
        M = nn_matrix(coords, latlon=latlon)
    elif method == "iso":
        if radius is None:
            raise ValueError("radius must be specified for method='iso'")
        M = iso_matrix(coords, radius, latlon=latlon)
    elif method == "lbmgls":
        M = lbmgls_matrix(coords, latlon=latlon)
    elif method == "cluster":
        if cluster is None:
            raise ValueError("cluster must be specified for method='cluster'")
        M = cluster_matrix(cluster)
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'nn', 'iso', 'lbmgls', or 'cluster'."
        )

    # Apply transformation: M @ data
    # Handle both 1D and 2D data
    if data.ndim == 1:
        return M @ data
    else:
        return M @ data


def spurtransform(
    df: pd.DataFrame,
    varlist: Union[str, List[str]],
    coord_cols: List[str],
    method: str = "nn",
    radius: Optional[float] = None,
    latlon: bool = True,
    prefix: str = "d_",
    cluster_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply SPUR transformation to variables in a DataFrame.

    This is the main user-facing function for spatial differencing in pandas.
    For each variable in varlist, it applies the spatial transformation and
    adds a new column with the specified prefix.

    Parameters
    ----------
    df : DataFrame
        Input data containing variables and coordinates
    varlist : str or list of str
        Variable name(s) to transform
    coord_cols : list of str, length 2
        Column names for coordinates. Order depends on latlon:
        - If latlon=True: [latitude_col, longitude_col]
        - If latlon=False: [x_col, y_col]
        Not used for method='cluster'.
    method : str, default 'nn'
        Transformation method:
        - 'nn': nearest-neighbor differencing
        - 'iso': isotropic (radius-based) differencing
        - 'lbmgls': LBM-GLS transformation (recommended by Muller-Watson)
        - 'cluster': within-cluster demeaning
    radius : float, optional
        Radius for isotropic method (required if method='iso')
    latlon : bool, default True
        If True, interpret coordinates as lat/lon and use Haversine distance
    prefix : str, default 'd_'
        Prefix for new transformed variable names
    cluster_col : str, optional
        Column name for cluster identifiers (required if method='cluster')

    Returns
    -------
    DataFrame
        Copy of input DataFrame with new transformed columns added

    Examples
    --------
    >>> # Nearest-neighbor transformation
    >>> df_out = spurtransform(df, ['gdp', 'population'],
    ...                        ['latitude', 'longitude'], method='nn')

    >>> # Isotropic transformation with 100km radius
    >>> df_out = spurtransform(df, 'income', ['lat', 'lon'],
    ...                        method='iso', radius=100000)

    >>> # LBM-GLS transformation (default recommended by Muller-Watson)
    >>> df_out = spurtransform(df, 'income', ['lat', 'lon'],
    ...                        method='lbmgls')

    >>> # Cluster demeaning
    >>> df_out = spurtransform(df, 'income', ['lat', 'lon'],
    ...                        method='cluster', cluster_col='state')
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure varlist is a list
    if isinstance(varlist, str):
        varlist = [varlist]

    # Extract coordinates
    coords = df[coord_cols].values

    # Validate coordinate columns
    if coords.shape[1] != 2:
        raise ValueError(
            f"coord_cols must specify exactly 2 columns, got {len(coord_cols)}"
        )

    # Check for missing coordinates
    if np.any(np.isnan(coords)):
        raise ValueError(
            "Coordinate columns contain missing values. "
            "Remove or impute missing coordinates before transformation."
        )

    # Build transformation matrix once (reuse for all variables)
    if method == "nn":
        M = nn_matrix(coords, latlon=latlon)
    elif method == "iso":
        if radius is None:
            raise ValueError("radius must be specified for method='iso'")
        M = iso_matrix(coords, radius, latlon=latlon)
    elif method == "lbmgls":
        M = lbmgls_matrix(coords, latlon=latlon)
    elif method == "cluster":
        if cluster_col is None:
            raise ValueError("cluster_col must be specified for method='cluster'")
        if cluster_col not in df.columns:
            raise ValueError(f"Cluster column '{cluster_col}' not found in DataFrame")
        cluster = df[cluster_col].values
        M = cluster_matrix(cluster)
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'nn', 'iso', 'lbmgls', or 'cluster'."
        )

    # Transform each variable
    for var in varlist:
        if var not in df.columns:
            raise ValueError(f"Variable '{var}' not found in DataFrame")

        data = df[var].values

        # Handle missing values in data
        if np.any(np.isnan(data)):
            print(
                f"Warning: Variable '{var}' contains {np.isnan(data).sum()} missing values. "
                f"Transformed values involving missing data will be NaN."
            )

        # Apply transformation
        transformed = M @ data

        # Add new column
        new_name = f"{prefix}{var}"
        df[new_name] = transformed

    return df


def get_transformation_stats(
    coords: np.ndarray,
    method: str = "nn",
    radius: Optional[float] = None,
    latlon: bool = True,
) -> dict:
    """
    Compute summary statistics for the transformation matrix.

    Useful for diagnosing the spatial structure and checking for potential issues.

    Parameters
    ----------
    coords : ndarray of shape (n, 2)
        Coordinates array
    method : str, default 'nn'
        Transformation method: 'nn' or 'iso'
    radius : float, optional
        Radius for isotropic method
    latlon : bool, default True
        If True, use Haversine distance

    Returns
    -------
    dict
        Dictionary containing:
        - n_obs: number of observations
        - method: transformation method used
        - dist_min: minimum pairwise distance
        - dist_max: maximum pairwise distance
        - dist_mean: mean pairwise distance
        - dist_median: median pairwise distance
        - nn_dist_mean: mean nearest-neighbor distance
        - nn_dist_max: max nearest-neighbor distance
        - n_isolated: (for iso method) number of isolated observations
    """
    distmat = get_distance_matrix(coords, latlon=latlon)
    n = coords.shape[0]

    # Distance statistics (exclude diagonal)
    upper_tri = distmat[np.triu_indices(n, k=1)]

    # Nearest neighbor distances
    distmat_no_diag = distmat.copy()
    np.fill_diagonal(distmat_no_diag, np.inf)
    nn_distances = np.min(distmat_no_diag, axis=1)

    stats = {
        "n_obs": n,
        "method": method,
        "dist_min": upper_tri.min(),
        "dist_max": upper_tri.max(),
        "dist_mean": upper_tri.mean(),
        "dist_median": np.median(upper_tri),
        "nn_dist_mean": nn_distances.mean(),
        "nn_dist_max": nn_distances.max(),
    }

    if method == "iso" and radius is not None:
        stats["radius"] = radius
        neighbors = (distmat < radius) & (distmat > 0)
        neighbor_counts = neighbors.sum(axis=1)
        stats["n_isolated"] = (neighbor_counts == 0).sum()
        stats["neighbors_mean"] = neighbor_counts.mean()
        stats["neighbors_min"] = neighbor_counts.min()
        stats["neighbors_max"] = neighbor_counts.max()

    return stats


if __name__ == "__main__":
    # Quick self-test with synthetic data
    np.random.seed(42)
    n = 10

    # Generate random lat/lon (roughly in Europe)
    lat = np.random.uniform(45, 55, n)
    lon = np.random.uniform(5, 15, n)
    coords = np.column_stack([lat, lon])

    # Generate some data
    y = np.random.randn(n) + 0.1 * lat  # data correlated with latitude

    # Test distance matrix
    distmat = get_distance_matrix(coords, latlon=True)
    print("Distance matrix shape:", distmat.shape)
    print("Distance range (km):", distmat.min() / 1000, "-", distmat.max() / 1000)

    # Test NN transformation
    M_nn = nn_matrix(coords, latlon=True)
    y_nn = M_nn @ y
    print("\nNearest-neighbor transformation:")
    print("  Original mean:", y.mean())
    print("  Transformed mean:", y_nn.mean())

    # Test ISO transformation
    M_iso = iso_matrix(coords, radius=200000, latlon=True)  # 200km radius
    y_iso = M_iso @ y
    print("\nIsotropic transformation (200km):")
    print("  Transformed mean:", y_iso.mean())

    # Test DataFrame interface
    df = pd.DataFrame({"lat": lat, "lon": lon, "y": y, "x": np.random.randn(n)})

    df_out = spurtransform(df, ["y", "x"], ["lat", "lon"], method="nn")
    print("\nDataFrame columns after transform:", list(df_out.columns))

    print("\nAll tests passed!")
