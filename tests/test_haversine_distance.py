import pytest

from spur.utils.dist import haversine_distance


def test_haversine_distance_zero_for_same_point() -> None:
    assert haversine_distance(48.0, 11.0, 48.0, 11.0) == 0.0


def test_haversine_distance_is_symmetric() -> None:
    d1 = haversine_distance(40.7, -74.0, 51.5, -0.1)
    d2 = haversine_distance(51.5, -0.1, 40.7, -74.0)

    assert d1 == pytest.approx(d2, rel=1e-12)


def test_haversine_distance_matches_known_scale() -> None:
    dist = haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)

    assert 5_500_000 < dist < 5_700_000


def test_haversine_distance_satisfies_triangle_inequality() -> None:
    d_ab = haversine_distance(48, 10, 50, 12)
    d_bc = haversine_distance(50, 12, 52, 8)
    d_ac = haversine_distance(48, 10, 52, 8)

    assert d_ac <= d_ab + d_bc + 1e-6
