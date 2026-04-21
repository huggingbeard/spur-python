import numpy as np

from spur.utils.matrix import get_transformation_stats


def test_get_transformation_stats_reports_core_distance_summary() -> None:
    rng = np.random.default_rng(42)
    coords = np.column_stack([rng.uniform(45, 55, 20), rng.uniform(5, 15, 20)])

    stats = get_transformation_stats(coords, method="nn", latlon=True)

    assert stats["n_obs"] == 20
    assert stats["method"] == "nn"
    assert stats["dist_min"] > 0
    assert stats["dist_max"] > stats["dist_min"]
    assert stats["nn_dist_mean"] > 0


def test_get_transformation_stats_reports_iso_specific_fields() -> None:
    rng = np.random.default_rng(42)
    coords = np.column_stack([rng.uniform(48, 52, 20), rng.uniform(8, 12, 20)])

    stats = get_transformation_stats(coords, method="iso", radius=200_000, latlon=True)

    assert stats["method"] == "iso"
    assert stats["radius"] == 200_000
    assert "n_isolated" in stats
    assert "neighbors_mean" in stats
