import numpy as np
import numpy.testing as npt
import pytest
import scipy.spatial.distance as distance

from spur.utils.dist import get_distance_matrix
from tests.config import ATOL, RTOL


@pytest.fixture
def grid_coords() -> np.ndarray:
    rng = np.random.default_rng(42)
    lat = rng.uniform(45, 55, 20)
    lon = rng.uniform(5, 15, 20)
    return np.column_stack([lat, lon])


def test_get_distance_matrix_is_symmetric_with_zero_diagonal(
    grid_coords: np.ndarray,
) -> None:
    distmat = get_distance_matrix(grid_coords, latlon=True)

    npt.assert_allclose(distmat, distmat.T, atol=ATOL, rtol=RTOL)
    npt.assert_allclose(np.diag(distmat), 0.0, atol=ATOL, rtol=RTOL)
    assert np.all(distmat >= 0)


def test_get_distance_matrix_euclidean_differs_from_haversine(
    grid_coords: np.ndarray,
) -> None:
    haversine = get_distance_matrix(grid_coords, latlon=True)
    euclidean = get_distance_matrix(grid_coords, latlon=False)

    assert not np.allclose(haversine, euclidean)


def test_get_distance_matrix_rejects_large_haversine_problem() -> None:
    coords = np.zeros((10_001, 2))

    with pytest.raises(ValueError, match="10,000"):
        get_distance_matrix(coords, latlon=True)


def test_get_distance_matrix_large_euclidean_problem_does_not_hit_haversine_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coords = np.zeros((10_001, 2))
    sentinel = np.zeros((2, 2))
    called: dict[str, tuple[tuple[int, int], tuple[int, int], str]] = {}

    def fake_cdist(left: np.ndarray, right: np.ndarray, metric: str) -> np.ndarray:
        called["args"] = (left.shape, right.shape, metric)
        return sentinel

    monkeypatch.setattr(distance, "cdist", fake_cdist)

    distmat = get_distance_matrix(coords, latlon=False)

    assert called["args"] == ((10_001, 2), (10_001, 2), "euclidean")
    assert distmat is sentinel
