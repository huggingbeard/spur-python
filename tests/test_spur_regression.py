from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spur import (
    get_distance_matrix,
    haversine_distance,
    iso_matrix,
    lbmgls_matrix,
    nn_matrix,
    spurhalflife,
    spurtest,
    spurtransform,
    transform,
)
from tests.config import (
    CSV_ATOL,
    CSV_RTOL,
    DIST_ATOL,
    EXACT_ATOL,
    NUMERICAL_ATOL,
    STRICT_ATOL,
    STRICT_REL,
    TRIANGLE_SLACK,
)

VALIDATION_CSV = Path(__file__).parent.parent / "validation_python_output.csv"
SEED = 42


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grid_coords() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    lat = rng.uniform(45, 55, 20)
    lon = rng.uniform(5, 15, 20)
    return np.column_stack([lat, lon])


@pytest.fixture
def small_coords() -> np.ndarray:
    return np.array([[float(i), 0.0] for i in range(5)])


@pytest.fixture
def sample_transform_df() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    return pd.DataFrame(
        {
            "lat": rng.uniform(45, 55, 10),
            "lon": rng.uniform(5, 15, 10),
            "y": rng.standard_normal(10),
            "x": rng.standard_normal(10),
        }
    )


@pytest.fixture
def val_df() -> pd.DataFrame:
    if not VALIDATION_CSV.exists():
        pytest.skip("validation CSV not present")
    return pd.read_csv(VALIDATION_CSV)


# ---------------------------------------------------------------------------
# haversine_distance
# ---------------------------------------------------------------------------


def test_haversine_zero_for_same_point() -> None:
    assert haversine_distance(48.0, 11.0, 48.0, 11.0) == 0.0


def test_haversine_symmetric() -> None:
    d1 = haversine_distance(40.7, -74.0, 51.5, -0.1)
    d2 = haversine_distance(51.5, -0.1, 40.7, -74.0)
    assert d1 == pytest.approx(d2, rel=STRICT_REL)


def test_haversine_positive_for_distinct_points() -> None:
    assert haversine_distance(0, 0, 1, 1) > 0


def test_haversine_triangle_inequality() -> None:
    d_ab = haversine_distance(48, 10, 50, 12)
    d_bc = haversine_distance(50, 12, 52, 8)
    d_ac = haversine_distance(48, 10, 52, 8)
    assert d_ac <= d_ab + d_bc + TRIANGLE_SLACK


# ---------------------------------------------------------------------------
# get_distance_matrix
# ---------------------------------------------------------------------------


def test_distance_matrix_symmetric(grid_coords: np.ndarray) -> None:
    dist = get_distance_matrix(grid_coords, latlon=True)
    np.testing.assert_allclose(dist, dist.T, atol=DIST_ATOL)


def test_distance_matrix_zero_diagonal(grid_coords: np.ndarray) -> None:
    dist = get_distance_matrix(grid_coords, latlon=True)
    np.testing.assert_allclose(np.diag(dist), 0.0, atol=DIST_ATOL)


def test_distance_matrix_non_negative(grid_coords: np.ndarray) -> None:
    dist = get_distance_matrix(grid_coords, latlon=True)
    assert np.all(dist >= 0)


def test_distance_matrix_euclidean_differs_from_haversine(
    grid_coords: np.ndarray,
) -> None:
    dist_hav = get_distance_matrix(grid_coords, latlon=True)
    dist_euc = get_distance_matrix(grid_coords, latlon=False)
    assert not np.allclose(dist_hav, dist_euc)


# ---------------------------------------------------------------------------
# nn_matrix
# ---------------------------------------------------------------------------


def test_nn_matrix_rows_sum_to_zero(grid_coords: np.ndarray) -> None:
    matrix = nn_matrix(grid_coords, latlon=True)
    np.testing.assert_allclose(matrix.sum(axis=1), 0.0, atol=STRICT_ATOL)


def test_nn_matrix_diagonal_is_one(grid_coords: np.ndarray) -> None:
    matrix = nn_matrix(grid_coords, latlon=True)
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=STRICT_ATOL)


def test_nn_matrix_off_diagonal_nonpositive(grid_coords: np.ndarray) -> None:
    matrix = nn_matrix(grid_coords, latlon=True)
    off_diagonal = matrix - np.diag(np.diag(matrix))
    assert np.all(off_diagonal <= STRICT_ATOL)


def test_nn_matrix_each_row_has_at_least_one_negative_entry(
    grid_coords: np.ndarray,
) -> None:
    matrix = nn_matrix(grid_coords, latlon=True)
    for row in matrix:
        assert np.any(row < 0)


def test_nn_matrix_weights_sum_to_one_per_row(grid_coords: np.ndarray) -> None:
    matrix = nn_matrix(grid_coords, latlon=True)
    for row in matrix:
        assert row[row < 0].sum() == pytest.approx(-1.0, abs=STRICT_ATOL)


def test_nn_matrix_euclidean_also_satisfies_row_sum(
    small_coords: np.ndarray,
) -> None:
    matrix = nn_matrix(small_coords, latlon=False)
    np.testing.assert_allclose(matrix.sum(axis=1), 0.0, atol=EXACT_ATOL)


def test_nn_matrix_tie_splitting() -> None:
    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    matrix = nn_matrix(coords, latlon=False)
    for i in range(4):
        neg = matrix[i][matrix[i] < 0]
        assert len(neg) == 2, f"row {i} should have 2 tied NNs, got {neg}"
        np.testing.assert_allclose(neg, -0.5, atol=EXACT_ATOL)


# ---------------------------------------------------------------------------
# iso_matrix
# ---------------------------------------------------------------------------


def test_iso_matrix_rows_sum_to_zero_with_large_radius(
    grid_coords: np.ndarray,
) -> None:
    matrix = iso_matrix(grid_coords, radius=2_000_000, latlon=True)
    np.testing.assert_allclose(matrix.sum(axis=1), 0.0, atol=STRICT_ATOL)


def test_iso_matrix_diagonal_is_zero_for_isolated() -> None:
    coords = np.array([[0.0, 0.0], [50.0, 0.0]])
    matrix = iso_matrix(coords, radius=0.1, latlon=False)
    np.testing.assert_allclose(matrix, 0.0, atol=EXACT_ATOL)


def test_iso_matrix_diagonal_one_for_obs_with_neighbours(
    grid_coords: np.ndarray,
) -> None:
    matrix = iso_matrix(grid_coords, radius=2_000_000, latlon=True)
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=STRICT_ATOL)


def test_iso_matrix_weights_row_normalized(grid_coords: np.ndarray) -> None:
    matrix = iso_matrix(grid_coords, radius=500_000, latlon=True)
    for i, row in enumerate(matrix):
        if np.diag(matrix)[i] != 0:
            assert row[row < 0].sum() == pytest.approx(-1.0, abs=STRICT_ATOL)


# ---------------------------------------------------------------------------
# transform — invariants
# ---------------------------------------------------------------------------


def test_transform_constant_maps_to_zero_nn(grid_coords: np.ndarray) -> None:
    constant = np.ones(len(grid_coords)) * 3.14
    result = transform(constant, grid_coords, method="nn", latlon=True)
    np.testing.assert_allclose(result, 0.0, atol=STRICT_ATOL)


def test_transform_constant_maps_to_zero_iso(grid_coords: np.ndarray) -> None:
    constant = np.ones(len(grid_coords)) * 3.14
    result = transform(
        constant, grid_coords, method="iso", radius=1_000_000, latlon=True
    )
    np.testing.assert_allclose(result, 0.0, atol=STRICT_ATOL)


def test_transform_constant_maps_to_zero_lbmgls(grid_coords: np.ndarray) -> None:
    constant = np.ones(len(grid_coords)) * 2.71
    result = transform(constant, grid_coords, method="lbmgls", latlon=True)
    np.testing.assert_allclose(result, 0.0, atol=NUMERICAL_ATOL)


def test_transform_constant_maps_to_zero_cluster() -> None:
    coords = np.zeros((6, 2))
    cluster = np.array([1, 1, 2, 2, 3, 3])
    constant = np.ones(6) * 7.0
    result = transform(constant, coords, method="cluster", cluster=cluster)
    np.testing.assert_allclose(result, 0.0, atol=STRICT_ATOL)


def test_transform_linearity_nn(grid_coords: np.ndarray) -> None:
    rng = np.random.default_rng(SEED)
    y = rng.standard_normal(len(grid_coords))
    z = rng.standard_normal(len(grid_coords))
    a, b = 2.5, -1.3
    lhs = transform(a * y + b * z, grid_coords, method="nn", latlon=True)
    rhs = a * transform(y, grid_coords, method="nn", latlon=True) + b * transform(
        z, grid_coords, method="nn", latlon=True
    )
    np.testing.assert_allclose(lhs, rhs, atol=NUMERICAL_ATOL)


def test_transform_unknown_method_raises(grid_coords: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        transform(np.ones(len(grid_coords)), grid_coords, method="bogus")


def test_transform_iso_requires_radius(grid_coords: np.ndarray) -> None:
    with pytest.raises(ValueError, match="radius"):
        transform(np.ones(len(grid_coords)), grid_coords, method="iso")


def test_transform_cluster_requires_cluster_arg(grid_coords: np.ndarray) -> None:
    with pytest.raises(ValueError, match="cluster"):
        transform(np.ones(len(grid_coords)), grid_coords, method="cluster")


# ---------------------------------------------------------------------------
# spurtransform (DataFrame interface)
# ---------------------------------------------------------------------------


def test_spurtransform_output_columns_created(
    sample_transform_df: pd.DataFrame,
) -> None:
    out = spurtransform(
        sample_transform_df,
        ["y", "x"],
        ["lat", "lon"],
        method="nn",
        prefix="nn_",
    )
    assert "nn_y" in out.columns
    assert "nn_x" in out.columns


def test_spurtransform_original_columns_unchanged(
    sample_transform_df: pd.DataFrame,
) -> None:
    y_before = sample_transform_df["y"].values.copy()
    spurtransform(sample_transform_df, "y", ["lat", "lon"], method="nn")
    np.testing.assert_array_equal(sample_transform_df["y"].values, y_before)


def test_spurtransform_row_count_preserved(
    sample_transform_df: pd.DataFrame,
) -> None:
    out = spurtransform(sample_transform_df, "y", ["lat", "lon"], method="nn")
    assert len(out) == len(sample_transform_df)


def test_spurtransform_matches_direct_transform(
    sample_transform_df: pd.DataFrame,
) -> None:
    out = spurtransform(
        sample_transform_df,
        "y",
        ["lat", "lon"],
        method="nn",
        prefix="nn_",
    )
    coords = sample_transform_df[["lat", "lon"]].values
    direct = transform(
        sample_transform_df["y"].values, coords, method="nn", latlon=True
    )
    np.testing.assert_allclose(out["nn_y"].values, direct, atol=EXACT_ATOL)


def test_spurtransform_missing_coordinates_raises() -> None:
    df = pd.DataFrame(
        {
            "lat": [45.0, np.nan, 47.0],
            "lon": [10.0, 11.0, 12.0],
            "y": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="missing"):
        spurtransform(df, "y", ["lat", "lon"], method="nn")


def test_spurtransform_missing_variable_raises(
    sample_transform_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="not found"):
        spurtransform(sample_transform_df, "nonexistent", ["lat", "lon"], method="nn")


def test_spurtransform_cluster_demeaning_within_group() -> None:
    df = pd.DataFrame(
        {
            "lat": [0.0] * 6,
            "lon": [0.0] * 6,
            "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            "grp": [1, 1, 1, 2, 2, 2],
        }
    )
    out = spurtransform(
        df, "y", ["lat", "lon"], method="cluster", cluster_col="grp", prefix="cl_"
    )
    for group in [1, 2]:
        group_result = out.loc[df["grp"] == group, "cl_y"]
        assert group_result.sum() == pytest.approx(0.0, abs=STRICT_ATOL)


def test_spurtransform_cluster_nullable_string_with_na_raises() -> None:
    df = pd.DataFrame(
        {
            "lat": [0.0] * 6,
            "lon": [0.0] * 6,
            "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            "province": pd.array(
                ["Zurich", "Zurich", pd.NA, "Bern", "Bern", "Bern"], dtype="string"
            ),
        }
    )
    with pytest.raises(ValueError, match="missing"):
        spurtransform(df, "y", ["lat", "lon"], method="cluster", cluster_col="province")


def test_spurtransform_cluster_string_province_ids() -> None:
    df = pd.DataFrame(
        {
            "lat": [0.0] * 6,
            "lon": [0.0] * 6,
            "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            "province": pd.array(
                ["Zurich", "Zurich", "Zurich", "Bern", "Bern", "Bern"],
                dtype="string",
            ),
        }
    )
    out = spurtransform(
        df,
        "y",
        ["lat", "lon"],
        method="cluster",
        cluster_col="province",
        prefix="cl_",
    )
    for group in ["Zurich", "Bern"]:
        group_result = out.loc[df["province"] == group, "cl_y"]
        assert group_result.sum() == pytest.approx(0.0, abs=STRICT_ATOL)


# ---------------------------------------------------------------------------
# Batch-1 issue fixes
# ---------------------------------------------------------------------------


def test_batch1_cluster_works_without_coord_cols_in_df() -> None:
    df = pd.DataFrame(
        {
            "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            "grp": ["A", "A", "A", "B", "B", "B"],
        }
    )
    out = spurtransform(df, "y", ["lat", "lon"], method="cluster", cluster_col="grp")
    for group in ["A", "B"]:
        assert out.loc[df["grp"] == group, "d_y"].sum() == pytest.approx(
            0.0, abs=STRICT_ATOL
        )


def test_batch1_lbmgls_matrix_raises_on_identical_coords() -> None:
    coords = np.zeros((5, 2))
    with pytest.raises(ValueError, match="identical"):
        lbmgls_matrix(coords, latlon=False)


def test_batch1_spurtest_rejects_q_ge_n() -> None:
    rng = np.random.default_rng(SEED)
    n = 10
    df = pd.DataFrame(
        {
            "lat": rng.uniform(45, 55, n),
            "lon": rng.uniform(5, 15, n),
            "y": rng.standard_normal(n),
        }
    )
    with pytest.raises(ValueError, match="q="):
        spurtest(df, "i1", "y", ["lat", "lon"], q=n, nrep=100, seed=0)


def test_batch1_spurtest_i1resid_raises_on_collinear_x() -> None:
    rng = np.random.default_rng(SEED)
    n = 20
    x = rng.standard_normal(n)
    df = pd.DataFrame(
        {
            "lat": rng.uniform(45, 55, n),
            "lon": rng.uniform(5, 15, n),
            "y": rng.standard_normal(n),
            "x1": x,
            "x2": x,
        }
    )
    with pytest.raises(ValueError, match="rank-deficient"):
        spurtest(
            df,
            "i1resid",
            "y",
            ["lat", "lon"],
            indepvars=["x1", "x2"],
            q=5,
            nrep=100,
            seed=0,
        )


def test_batch1_spurtest_i0resid_raises_on_collinear_x() -> None:
    rng = np.random.default_rng(SEED)
    n = 20
    x = rng.standard_normal(n)
    df = pd.DataFrame(
        {
            "lat": rng.uniform(45, 55, n),
            "lon": rng.uniform(5, 15, n),
            "y": rng.standard_normal(n),
            "x1": x,
            "x2": x,
        }
    )
    with pytest.raises(ValueError, match="rank-deficient"):
        spurtest(
            df,
            "i0resid",
            "y",
            ["lat", "lon"],
            indepvars=["x1", "x2"],
            q=5,
            nrep=100,
            seed=0,
        )


# ---------------------------------------------------------------------------
# Batch-2 issue fixes
# ---------------------------------------------------------------------------


def test_batch2_spurtest_rejects_nan_y() -> None:
    rng = np.random.default_rng(SEED)
    n = 20
    df = pd.DataFrame(
        {
            "lat": rng.uniform(45, 55, n),
            "lon": rng.uniform(5, 15, n),
            "y": np.where(np.arange(n) == 5, np.nan, rng.standard_normal(n)),
        }
    )
    with pytest.raises(ValueError, match="NaN or inf"):
        spurtest(df, "i1", "y", ["lat", "lon"], q=5, nrep=100, seed=0)


def test_batch2_spurtest_rejects_inf_y() -> None:
    rng = np.random.default_rng(SEED)
    n = 20
    df = pd.DataFrame(
        {
            "lat": rng.uniform(45, 55, n),
            "lon": rng.uniform(5, 15, n),
            "y": np.where(np.arange(n) == 3, np.inf, rng.standard_normal(n)),
        }
    )
    with pytest.raises(ValueError, match="NaN or inf"):
        spurtest(df, "i1", "y", ["lat", "lon"], q=5, nrep=100, seed=0)


def test_batch2_spurhalflife_rejects_level_gt_1() -> None:
    rng = np.random.default_rng(SEED)
    n = 20
    df = pd.DataFrame(
        {
            "lat": rng.uniform(45, 55, n),
            "lon": rng.uniform(5, 15, n),
            "y": rng.standard_normal(n),
        }
    )
    with pytest.raises(ValueError, match="level="):
        spurhalflife(df, "y", ["lat", "lon"], level=1.5, q=5, nrep=100, seed=0)


def test_batch2_spurhalflife_rejects_level_zero() -> None:
    rng = np.random.default_rng(SEED)
    n = 20
    df = pd.DataFrame(
        {
            "lat": rng.uniform(45, 55, n),
            "lon": rng.uniform(5, 15, n),
            "y": rng.standard_normal(n),
        }
    )
    with pytest.raises(ValueError, match="level="):
        spurhalflife(df, "y", ["lat", "lon"], level=0.0, q=5, nrep=100, seed=0)


def test_batch2_spurhalflife_rejects_q_zero() -> None:
    rng = np.random.default_rng(SEED)
    n = 20
    df = pd.DataFrame(
        {
            "lat": rng.uniform(45, 55, n),
            "lon": rng.uniform(5, 15, n),
            "y": rng.standard_normal(n),
        }
    )
    with pytest.raises(ValueError, match="q="):
        spurhalflife(df, "y", ["lat", "lon"], level=0.95, q=0, nrep=100, seed=0)


def test_batch2_spurhalflife_raises_on_identical_coords() -> None:
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame(
        {
            "lat": [48.0] * 10,
            "lon": [11.0] * 10,
            "y": rng.standard_normal(10),
        }
    )
    with pytest.raises(ValueError, match="identical"):
        spurhalflife(df, "y", ["lat", "lon"], level=0.95, q=5, nrep=100, seed=0)


def test_batch2_get_distance_matrix_raises_for_large_n() -> None:
    coords = np.zeros((10_001, 2))
    with pytest.raises(ValueError, match="10,000"):
        get_distance_matrix(coords, latlon=True)


def test_batch2_get_distance_matrix_euclidean_no_limit() -> None:
    coords = np.zeros((10_001, 2))
    dist = get_distance_matrix(coords, latlon=False)
    assert dist.shape == (10_001, 10_001)


# ---------------------------------------------------------------------------
# Validation CSV: Python vs saved expected output
# ---------------------------------------------------------------------------


def test_validation_csv_nn_y_matches_saved(val_df: pd.DataFrame) -> None:
    coords = val_df[["lat", "lon"]].values
    result = transform(val_df["y"].values, coords, method="nn", latlon=True)
    np.testing.assert_allclose(
        result,
        val_df["py_nn_y"].values,
        rtol=CSV_RTOL,
        atol=CSV_ATOL,
    )


def test_validation_csv_nn_x_matches_saved(val_df: pd.DataFrame) -> None:
    coords = val_df[["lat", "lon"]].values
    result = transform(val_df["x"].values, coords, method="nn", latlon=True)
    np.testing.assert_allclose(
        result,
        val_df["py_nn_x"].values,
        rtol=CSV_RTOL,
        atol=CSV_ATOL,
    )


def test_validation_csv_iso_y_matches_saved(val_df: pd.DataFrame) -> None:
    coords = val_df[["lat", "lon"]].values
    result = transform(
        val_df["y"].values, coords, method="iso", radius=200_000, latlon=True
    )
    np.testing.assert_allclose(
        result,
        val_df["py_iso_y"].values,
        rtol=CSV_RTOL,
        atol=CSV_ATOL,
    )


def test_validation_csv_cluster_y_matches_saved(val_df: pd.DataFrame) -> None:
    out = spurtransform(
        val_df,
        "y",
        ["lat", "lon"],
        method="cluster",
        cluster_col="cluster",
        prefix="cl_",
    )
    np.testing.assert_allclose(
        out["cl_y"].values,
        val_df["py_cl_y"].values,
        rtol=CSV_RTOL,
        atol=CSV_ATOL,
    )
