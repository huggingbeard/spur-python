from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def resolve_spur_coords(
    data: pd.DataFrame,
    use_rows: np.ndarray,
    lon: str | None = None,
    lat: str | None = None,
    coords_euclidean: Sequence[str] | None = None,
) -> dict:
    """Resolve the public coordinate inputs to SPUR's internal coordinate payload.

    This is the bridge between the harmonized public API (`lon`/`lat` or
    `coords_euclidean`) and the current SPUR numerical code.
    """
    use_geodesic = lon is not None or lat is not None
    use_euclidean = coords_euclidean is not None

    if use_geodesic and use_euclidean:
        raise ValueError("Specify either `lon`/`lat` or `coords_euclidean`, not both.")
    if not use_geodesic and not use_euclidean:
        raise ValueError("Specify coordinates via `lon`/`lat` or `coords_euclidean`.")

    if use_geodesic:
        assert lon is not None and lat is not None, (
            "Both `lon` and `lat` must be specified for geodesic coordinates."
        )
        missing = [name for name in (lon, lat) if name not in data.columns]
        assert not missing, (
            f"Coordinate variables not found in data: {', '.join(missing)}"
        )

        coords_lon_lat = data.loc[use_rows, [lon, lat]]
        assert all(
            np.issubdtype(dtype, np.number) for dtype in coords_lon_lat.dtypes
        ), "`lon` and `lat` must reference numeric columns."
        arr = coords_lon_lat.to_numpy(dtype=float)
        assert np.isfinite(arr).all(), "Geodesic coordinates must be finite."
        assert ((coords_lon_lat[lon] >= -180) & (coords_lon_lat[lon] <= 180)).all(), (
            "Longitude values must be in [-180, 180]."
        )
        assert ((coords_lon_lat[lat] >= -90) & (coords_lon_lat[lat] <= 90)).all(), (
            "Latitude values must be in [-90, 90]."
        )
        return {
            "coords": data.loc[use_rows, [lat, lon]].to_numpy(dtype=float),
            "latlong": True,
        }

    if (
        isinstance(coords_euclidean, str)
        or coords_euclidean is None
        or len(coords_euclidean) < 1
    ):
        raise ValueError(
            "`coords_euclidean` must be a sequence with at least one column name."
        )

    missing = [name for name in coords_euclidean if name not in data.columns]
    assert not missing, f"Coordinate variables not found in data: {', '.join(missing)}"

    coords = data.loc[use_rows, list(coords_euclidean)]
    assert all(np.issubdtype(dtype, np.number) for dtype in coords.dtypes), (
        "`coords_euclidean` columns must be numeric."
    )
    arr = coords.to_numpy(dtype=float)
    assert np.isfinite(arr).all(), "Euclidean coordinates must be finite."

    return {"coords": arr, "latlong": False}


def haversine_distance(
    lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike
) -> np.ndarray:
    """
    Compute great-circle distance between points using Haversine formula.

    Parameters
    ----------
    lat1, lon1 : array-like
        Latitude and longitude of first point(s) in degrees; accepts scalars or arrays.
    lat2, lon2 : array-like
        Latitude and longitude of second point(s) in degrees; accepts scalars or arrays.

    Returns
    -------
    ndarray
        Distance in meters
    """
    lat1_arr = np.asarray(lat1, dtype=float)
    lon1_arr = np.asarray(lon1, dtype=float)
    lat2_arr = np.asarray(lat2, dtype=float)
    lon2_arr = np.asarray(lon2, dtype=float)

    R = 6371000  # Earth radius in meters

    phi1 = np.radians(lat1_arr)
    phi2 = np.radians(lat2_arr)
    dphi = np.radians(lat2_arr - lat1_arr)
    dlam = np.radians(lon2_arr - lon1_arr)

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
    n = coords.shape[0]

    if latlon:
        if n > 10_000:
            raise ValueError(
                f"n={n} exceeds the safe limit for full n×n haversine distance computation "
                f"(limit: 10,000). At this size the matrix requires >{n**2 * 8 // 1_000_000} MB "
                "and will likely OOM. Use a blocked/sparse implementation for large datasets."
            )
        lat = coords[:, 0]
        lon = coords[:, 1]
        lat1, lat2 = np.meshgrid(lat, lat, indexing="ij")
        lon1, lon2 = np.meshgrid(lon, lon, indexing="ij")
        distmat = haversine_distance(lat1, lon1, lat2, lon2)
    else:
        from scipy.spatial.distance import cdist

        distmat = cdist(coords, coords, metric="euclidean")

    return distmat


def normalized_distmat(coords: np.ndarray, latlon: bool = True) -> np.ndarray:
    """Get distance matrix normalized so max distance = 1."""
    distmat = get_distance_matrix(coords, latlon=latlon)
    return distmat / distmat.max()


def lvech(S: np.ndarray) -> np.ndarray:
    """Extract lower triangular part (below diagonal) as vector."""
    n = S.shape[0]
    i, j = np.tril_indices(n, k=-1)
    return S[i, j]


def get_cbar(rhobar: float, distmat: np.ndarray) -> float:
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

    c0 = 10.0
    c1 = 10.0

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

    while (c1 - c0) > 0.001:
        cm = np.sqrt(c0 * c1)
        v = np.mean(np.exp(-cm * vd))
        if v < rhobar:
            c1 = cm
        else:
            c0 = cm

    return np.sqrt(c0 * c1)
