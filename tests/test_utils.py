"""
SPUR Python - Verification Tests

Validates mathematical properties of transformation matrices and
checks edge cases.
"""

import numpy as np
import pandas as pd
from spur import (
    get_distance_matrix,
    nn_matrix,
    iso_matrix,
    transform,
    spurtransform,
    haversine_distance,
    get_transformation_stats,
)

np.random.seed(42)


def test_haversine_distance():
    """Test Haversine formula against known values."""
    print("Testing Haversine distance...")

    # New York to London: ~5570 km
    ny_lat, ny_lon = 40.7128, -74.0060
    london_lat, london_lon = 51.5074, -0.1278

    dist = haversine_distance(ny_lat, ny_lon, london_lat, london_lon)
    dist_km = dist / 1000

    assert 5500 < dist_km < 5700, f"NY-London should be ~5570km, got {dist_km:.0f}km"
    print(f"  NY to London: {dist_km:.0f} km (expected ~5570 km) OK")

    # Same point should be 0
    dist_same = haversine_distance(ny_lat, ny_lon, ny_lat, ny_lon)
    assert dist_same == 0, f"Same point should have 0 distance, got {dist_same}"
    print(f"  Same point: {dist_same:.0f} m (expected 0 m) OK")

    # Antipodal points: ~20,000 km (half Earth circumference)
    dist_antipodal = haversine_distance(0, 0, 0, 180)
    dist_km = dist_antipodal / 1000
    assert 19000 < dist_km < 21000, f"Antipodal should be ~20000km, got {dist_km:.0f}km"
    print(f"  Antipodal: {dist_km:.0f} km (expected ~20,000 km) OK")


def test_distance_matrix_properties():
    """Test distance matrix is symmetric, non-negative, with zero diagonal."""
    print("\nTesting distance matrix properties...")

    n = 20
    lat = np.random.uniform(45, 55, n)
    lon = np.random.uniform(5, 15, n)
    coords = np.column_stack([lat, lon])

    # Test Haversine
    D = get_distance_matrix(coords, latlon=True)

    # Symmetric
    assert np.allclose(D, D.T), "Distance matrix should be symmetric"
    print("  Symmetric: OK")

    # Non-negative
    assert np.all(D >= 0), "Distances should be non-negative"
    print("  Non-negative: OK")

    # Zero diagonal
    assert np.allclose(np.diag(D), 0), "Diagonal should be zero"
    print("  Zero diagonal: OK")

    # Test Euclidean
    D_euc = get_distance_matrix(coords, latlon=False)
    assert np.allclose(D_euc, D_euc.T), "Euclidean matrix should be symmetric"
    assert np.all(D_euc >= 0), "Euclidean distances should be non-negative"
    print("  Euclidean properties: OK")


def test_nn_matrix_properties():
    """Test nearest-neighbor matrix properties."""
    print("\nTesting NN matrix properties...")

    n = 20
    lat = np.random.uniform(45, 55, n)
    lon = np.random.uniform(5, 15, n)
    coords = np.column_stack([lat, lon])

    M = nn_matrix(coords, latlon=True)

    # M = I - W where W is row-stochastic (rows sum to 1)
    # So rows of M should sum to 0
    row_sums = M.sum(axis=1)
    assert np.allclose(row_sums, 0, atol=1e-10), f"Rows should sum to 0, got {row_sums}"
    print("  Row sums = 0: OK")

    # Diagonal should be 1 (from identity matrix)
    diag = np.diag(M)
    assert np.allclose(diag, 1), "Diagonal should be 1"
    print("  Diagonal = 1: OK")

    # Off-diagonal should be <= 0 (since M = I - W with W >= 0)
    off_diag = M - np.diag(np.diag(M))
    assert np.all(off_diag <= 1e-10), "Off-diagonal should be <= 0"
    print("  Off-diagonal <= 0: OK")

    # At least one -1 per row (or ties split)
    nn_weights = -off_diag.sum(axis=1)
    assert np.allclose(nn_weights, 1), "Each row should have NN weights summing to 1"
    print("  NN weights sum to 1: OK")


def test_iso_matrix_properties():
    """Test isotropic matrix properties."""
    print("\nTesting ISO matrix properties...")

    n = 20
    lat = np.random.uniform(48, 52, n)  # Smaller region to ensure neighbors
    lon = np.random.uniform(8, 12, n)
    coords = np.column_stack([lat, lon])

    # Use large radius to ensure everyone has neighbors
    radius = 500000  # 500 km
    M = iso_matrix(coords, radius=radius, latlon=True)

    # Rows should sum to 0 (all have neighbors with large radius)
    row_sums = M.sum(axis=1)
    assert np.allclose(row_sums, 0, atol=1e-10), f"Rows should sum to 0, got {row_sums}"
    print("  Row sums = 0 (all have neighbors): OK")

    # Diagonal should be 1
    diag = np.diag(M)
    assert np.allclose(diag, 1), "Diagonal should be 1"
    print("  Diagonal = 1: OK")

    # Test with smaller radius - some may be isolated
    # Isolated observations have row set to 0 (Stata behavior)
    radius_small = 50000  # 50 km
    M_small = iso_matrix(coords, radius=radius_small, latlon=True)
    diag_small = np.diag(M_small)

    # Diagonal tells us which are isolated: 0 = isolated, 1 = has neighbors
    isolated = np.isclose(diag_small, 0)
    has_neighbors = np.isclose(diag_small, 1)

    # Verify diagonal is either 0 or 1
    assert np.all(isolated | has_neighbors), "Diagonal must be 0 or 1"

    # For isolated observations, entire row should be 0
    for i in np.where(isolated)[0]:
        assert np.allclose(M_small[i, :], 0), f"Isolated row {i} should be all zeros"

    print("  Small radius - isolated rows zeroed: OK")


def test_nn_ties():
    """Test that ties in nearest neighbors are handled correctly."""
    print("\nTesting NN ties handling...")

    # In an equilateral triangle, point 2 sees both 0 and 1 at equal distance.
    # Points 0 and 1 might find ties or not (depends on floating point).
    # The key test: for point 2, it should split weight between 0 and 1.
    coords = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])

    M = nn_matrix(coords, latlon=False)

    # Point 2 (row 2) should have two neighbors with -0.5 weight
    row2 = M[2, :]
    neg_vals = row2[row2 < 0]
    assert len(neg_vals) == 2, "Row 2 should have 2 negative entries (ties)"
    assert np.allclose(neg_vals, -0.5), "Row 2 NN weights should be -0.5 each"
    print("  Ties split equally for point 2: OK")

    # All rows should sum to 0 (key invariant)
    assert np.allclose(M.sum(axis=1), 0), "All rows should sum to 0"
    print("  Row sums = 0: OK")


def test_transform_constant():
    """Test that transform of constant is zero."""
    print("\nTesting constant vector transformation...")

    n = 20
    lat = np.random.uniform(45, 55, n)
    lon = np.random.uniform(5, 15, n)
    coords = np.column_stack([lat, lon])

    # Constant vector
    c = np.ones(n) * 5.0

    # NN transform
    c_nn = transform(c, coords, method="nn")
    assert np.allclose(c_nn, 0, atol=1e-10), "NN transform of constant should be 0"
    print("  NN of constant = 0: OK")

    # ISO transform
    c_iso = transform(c, coords, method="iso", radius=500000)
    assert np.allclose(c_iso, 0, atol=1e-10), "ISO transform of constant should be 0"
    print("  ISO of constant = 0: OK")


def test_spurtransform_dataframe():
    """Test DataFrame interface."""
    print("\nTesting DataFrame interface...")

    df = pd.DataFrame(
        {
            "lat": [45, 46, 47, 48, 49],
            "lon": [10, 11, 12, 13, 14],
            "y": [1, 2, 3, 4, 5],
            "x": [10, 20, 30, 40, 50],
        }
    )

    # NN transform
    df_nn = spurtransform(df, ["y", "x"], ["lat", "lon"], method="nn", prefix="nn_")

    assert "nn_y" in df_nn.columns, "Should have nn_y column"
    assert "nn_x" in df_nn.columns, "Should have nn_x column"
    assert len(df_nn) == len(df), "Should preserve row count"
    print("  NN columns created: OK")

    # ISO transform
    df_iso = spurtransform(
        df, "y", ["lat", "lon"], method="iso", radius=200000, prefix="iso_"
    )
    assert "iso_y" in df_iso.columns, "Should have iso_y column"
    print("  ISO columns created: OK")

    # Original data unchanged
    assert df_nn["y"].tolist() == [1, 2, 3, 4, 5], "Original y should be unchanged"
    print("  Original data preserved: OK")


def test_spurtransform_cluster_with_string_dtype():
    """Test cluster transformation with pandas string dtype labels."""
    print("\nTesting cluster transformation with string dtype labels...")

    df = pd.DataFrame(
        {
            "lat": [45, 46, 47, 48, 49],
            "lon": [10, 11, 12, 13, 14],
            "state": pd.Series(["TN", "TN", "NC", "NC", "VA"], dtype="string"),
            "y": [1, 2, 3, 4, 5],
        }
    )

    df_cluster = spurtransform(
        df,
        "y",
        ["lat", "lon"],
        method="cluster",
        cluster_col="state",
        prefix="cluster_",
    )

    assert "cluster_y" in df_cluster.columns, "Should have cluster_y column"
    assert len(df_cluster) == len(df), "Should preserve row count"
    print("  Cluster transform works with string dtype labels: OK")


def test_missing_coordinates_error():
    """Test that missing coordinates raise error."""
    print("\nTesting missing coordinate handling...")

    df = pd.DataFrame(
        {
            "lat": [45, np.nan, 47, 48, 49],
            "lon": [10, 11, 12, 13, 14],
            "y": [1, 2, 3, 4, 5],
        }
    )

    try:
        spurtransform(df, "y", ["lat", "lon"], method="nn")
        assert False, "Should have raised error for missing coordinates"
    except ValueError as e:
        assert "missing" in str(e).lower(), f"Error should mention missing values: {e}"
        print("  Missing coords raise ValueError: OK")


def test_get_transformation_stats():
    """Test statistics function."""
    print("\nTesting transformation stats...")

    n = 50
    lat = np.random.uniform(45, 55, n)
    lon = np.random.uniform(5, 15, n)
    coords = np.column_stack([lat, lon])

    stats_nn = get_transformation_stats(coords, method="nn", latlon=True)

    assert stats_nn["n_obs"] == n, "Should have correct observation count"
    assert stats_nn["method"] == "nn", "Should have correct method"
    assert stats_nn["dist_min"] > 0, "Min distance should be positive"
    assert stats_nn["dist_max"] > stats_nn["dist_min"], "Max > min"
    assert stats_nn["nn_dist_mean"] > 0, "NN distance mean should be positive"
    print("  NN stats computed: OK")

    stats_iso = get_transformation_stats(
        coords, method="iso", radius=200000, latlon=True
    )
    assert "n_isolated" in stats_iso, "ISO should have isolated count"
    assert "neighbors_mean" in stats_iso, "ISO should have mean neighbors"
    print("  ISO stats computed: OK")


def test_euclidean_vs_haversine():
    """Test that Euclidean and Haversine give different results."""
    print("\nTesting Euclidean vs Haversine...")

    n = 20
    lat = np.random.uniform(45, 55, n)
    lon = np.random.uniform(5, 15, n)
    coords = np.column_stack([lat, lon])

    D_hav = get_distance_matrix(coords, latlon=True)
    D_euc = get_distance_matrix(coords, latlon=False)

    # They should be different (scaled differently)
    assert not np.allclose(D_hav, D_euc), "Haversine and Euclidean should differ"
    print("  Different distance matrices: OK")

    # But order of nearest neighbors might be similar
    # (for small regions, Euclidean approximates Haversine)


def run_all_tests():
    """Run all verification tests."""
    print("=" * 60)
    print("SPUR Python - Verification Tests")
    print("=" * 60)

    test_haversine_distance()
    test_distance_matrix_properties()
    test_nn_matrix_properties()
    test_iso_matrix_properties()
    test_nn_ties()
    test_transform_constant()
    test_spurtransform_dataframe()
    test_spurtransform_cluster_with_string_dtype()
    test_missing_coordinates_error()
    test_get_transformation_stats()
    test_euclidean_vs_haversine()

    print("\n" + "=" * 60)
    print("All tests passed! OK")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
