from typing import Optional
import numpy as np
from .dist import get_distance_matrix


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
    mat = mat - mat.mean(axis=1, keepdims=True)
    mat = mat - mat.mean(axis=0, keepdims=True)
    return mat


def get_r(sigma: np.ndarray, qmax: int) -> np.ndarray:
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
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    return eigenvectors[:, :qmax]


def cholesky_upper(M: np.ndarray) -> np.ndarray:
    """
    Get upper triangular Cholesky factor.

    NumPy returns a lower-triangular factor, while R `chol()` already returns
    the upper-triangular factor used in the SPUR translations. This helper
    bridges that library convention difference in Python.
    """
    L = np.linalg.cholesky(M)
    return L.T


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
    sigma_lbm = 0.5 * (distmat[:, 0:1] + distmat[0:1, :] - distmat)
    return sigma_lbm


def get_sigma_dm(distmat: np.ndarray, c: float) -> np.ndarray:
    """
    Compute demeaned exponential covariance matrix.

    sigma(i,j) = exp(-c * distmat(i,j)), then double-demean.
    """
    sigma = np.exp(-c * distmat)
    return demean_matrix(sigma)


def get_sigma_residual(distmat: np.ndarray, c: float, M: np.ndarray) -> np.ndarray:
    """
    Compute residualized covariance matrix: M @ exp(-c * distmat) @ M.T

    Used for I(1) and I(0) residual tests. The projection matrix M already
    accounts for the regressors, so we project the spatial covariance.
    """
    sigma = np.exp(-c * distmat)
    return M @ sigma @ M.T


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

    np.fill_diagonal(distmat, np.inf)
    min_dist = np.min(distmat, axis=1, keepdims=True)
    NN = (distmat == min_dist).astype(float)
    row_sums = NN.sum(axis=1, keepdims=True)
    NN = NN / row_sums
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

    neighbors = (distmat < radius) & (distmat > 0)
    row_sums = neighbors.sum(axis=1, keepdims=True).astype(float)
    no_neighbors = (row_sums == 0).flatten()
    row_sums[row_sums == 0] = 1
    W = neighbors.astype(float) / row_sums
    M = np.eye(n) - W
    M[no_neighbors, :] = 0

    if no_neighbors.any():
        n_isolated = no_neighbors.sum()
        print(
            f"Warning: {n_isolated} observation(s) have no neighbors within radius {radius}. "
            f"These observations will be set to 0."
        )

    return M


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

    distmat = get_distance_matrix(coords, latlon=latlon)
    max_dist = distmat.max()
    if max_dist <= 1e-10:
        raise ValueError(
            "All coordinates are identical (or nearly so) — LBM-GLS normalization "
            "requires distinct locations."
        )
    distmat = distmat / max_dist

    sigma_lbm = get_sigma_lbm(distmat)
    sigma_lbm_dm = demean_matrix(sigma_lbm)

    eigenvalues, eigenvectors = np.linalg.eigh(sigma_lbm_dm)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    mask = eigenvalues > small
    eigenvalues = eigenvalues[mask]
    eigenvectors = eigenvectors[:, mask]
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
    clust_mat = (cluster.reshape(-1, 1) == cluster.reshape(1, -1)).astype(float)
    clust_mat = clust_mat / clust_mat.sum(axis=1, keepdims=True)
    M = np.eye(n) - clust_mat

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

    if data.ndim == 1:
        return M @ data
    else:
        return M @ data


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
    upper_tri = distmat[np.triu_indices(n, k=1)]

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
