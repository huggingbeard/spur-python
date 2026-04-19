"""
pytest tests for spur.py

Tests mathematical properties and behavioral invariants.
No hardcoded numeric expectations — all assertions derive from the
structure of the problem (symmetry, zero-sum rows, etc.).
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from spur import (
    haversine_distance,
    get_distance_matrix,
    nn_matrix,
    iso_matrix,
    lbmgls_matrix,
    transform,
    spurtransform,
    get_transformation_stats,
    spurtest,
    spurhalflife,
)

VALIDATION_CSV = Path(__file__).parent.parent / "validation_python_output.csv"

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grid_coords():
    """20 random lat/lon points in central Europe."""
    lat = RNG.uniform(45, 55, 20)
    lon = RNG.uniform(5, 15, 20)
    return np.column_stack([lat, lon])


@pytest.fixture
def small_coords():
    """5 points in a line, Euclidean."""
    return np.array([[float(i), 0.0] for i in range(5)])


@pytest.fixture
def val_df():
    if not VALIDATION_CSV.exists():
        pytest.skip("validation CSV not present")
    return pd.read_csv(VALIDATION_CSV)


# ---------------------------------------------------------------------------
# haversine_distance
# ---------------------------------------------------------------------------


class TestHaversine:
    def test_zero_for_same_point(self):
        assert haversine_distance(48.0, 11.0, 48.0, 11.0) == 0.0

    def test_symmetric(self):
        d1 = haversine_distance(40.7, -74.0, 51.5, -0.1)
        d2 = haversine_distance(51.5, -0.1, 40.7, -74.0)
        assert d1 == pytest.approx(d2, rel=1e-12)

    def test_positive_for_distinct_points(self):
        assert haversine_distance(0, 0, 1, 1) > 0

    def test_triangle_inequality(self):
        # A→C ≤ A→B + B→C
        dAB = haversine_distance(48, 10, 50, 12)
        dBC = haversine_distance(50, 12, 52, 8)
        dAC = haversine_distance(48, 10, 52, 8)
        assert dAC <= dAB + dBC + 1e-6  # tiny tol for floating point


# ---------------------------------------------------------------------------
# get_distance_matrix
# ---------------------------------------------------------------------------


class TestDistanceMatrix:
    def test_symmetric(self, grid_coords):
        D = get_distance_matrix(grid_coords, latlon=True)
        np.testing.assert_allclose(D, D.T, atol=1e-8)

    def test_zero_diagonal(self, grid_coords):
        D = get_distance_matrix(grid_coords, latlon=True)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-8)

    def test_non_negative(self, grid_coords):
        D = get_distance_matrix(grid_coords, latlon=True)
        assert np.all(D >= 0)

    def test_euclidean_differs_from_haversine(self, grid_coords):
        D_hav = get_distance_matrix(grid_coords, latlon=True)
        D_euc = get_distance_matrix(grid_coords, latlon=False)
        assert not np.allclose(D_hav, D_euc)


# ---------------------------------------------------------------------------
# nn_matrix
# ---------------------------------------------------------------------------


class TestNNMatrix:
    def test_rows_sum_to_zero(self, grid_coords):
        M = nn_matrix(grid_coords, latlon=True)
        np.testing.assert_allclose(M.sum(axis=1), 0.0, atol=1e-12)

    def test_diagonal_is_one(self, grid_coords):
        M = nn_matrix(grid_coords, latlon=True)
        np.testing.assert_allclose(np.diag(M), 1.0, atol=1e-12)

    def test_off_diagonal_nonpositive(self, grid_coords):
        M = nn_matrix(grid_coords, latlon=True)
        off = M - np.diag(np.diag(M))
        assert np.all(off <= 1e-12)

    def test_each_row_has_at_least_one_negative_entry(self, grid_coords):
        M = nn_matrix(grid_coords, latlon=True)
        for row in M:
            assert np.any(row < 0)

    def test_nn_weights_sum_to_one_per_row(self, grid_coords):
        """Negative off-diagonal entries (the NN weights) sum to -1 per row."""
        M = nn_matrix(grid_coords, latlon=True)
        for row in M:
            assert row[row < 0].sum() == pytest.approx(-1.0, abs=1e-12)

    def test_euclidean_also_satisfies_row_sum(self, small_coords):
        M = nn_matrix(small_coords, latlon=False)
        np.testing.assert_allclose(M.sum(axis=1), 0.0, atol=1e-14)

    def test_tie_splitting(self):
        """Square corners: each point ties between two equidistant neighbours.
        Uses integer coords so tie distances are exact in floating point."""
        # (0,0) NN = (0,1) and (1,0) at distance 1; (1,1) at sqrt(2)
        coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        M = nn_matrix(coords, latlon=False)
        for i in range(4):
            neg = M[i][M[i] < 0]
            assert len(neg) == 2, f"row {i} should have 2 tied NNs, got {neg}"
            np.testing.assert_allclose(neg, -0.5, atol=1e-14)


# ---------------------------------------------------------------------------
# iso_matrix
# ---------------------------------------------------------------------------


class TestIsoMatrix:
    def test_rows_sum_to_zero_with_large_radius(self, grid_coords):
        M = iso_matrix(grid_coords, radius=2_000_000, latlon=True)
        np.testing.assert_allclose(M.sum(axis=1), 0.0, atol=1e-12)

    def test_diagonal_is_zero_for_isolated(self):
        """Observations with no neighbours get a zero row."""
        # Two far-apart singletons
        coords = np.array([[0.0, 0.0], [50.0, 0.0]])
        M = iso_matrix(coords, radius=0.1, latlon=False)
        np.testing.assert_allclose(M, 0.0, atol=1e-14)

    def test_diagonal_one_for_obs_with_neighbours(self, grid_coords):
        M = iso_matrix(grid_coords, radius=2_000_000, latlon=True)
        np.testing.assert_allclose(np.diag(M), 1.0, atol=1e-12)

    def test_weights_row_normalized(self, grid_coords):
        """NN weights (negative entries) sum to -1 for non-isolated rows."""
        M = iso_matrix(grid_coords, radius=500_000, latlon=True)
        for i, row in enumerate(M):
            if np.diag(M)[i] != 0:  # not isolated
                assert row[row < 0].sum() == pytest.approx(-1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# transform — invariants
# ---------------------------------------------------------------------------


class TestTransform:
    def test_constant_maps_to_zero_nn(self, grid_coords):
        c = np.ones(len(grid_coords)) * 3.14
        result = transform(c, grid_coords, method="nn", latlon=True)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_constant_maps_to_zero_iso(self, grid_coords):
        c = np.ones(len(grid_coords)) * 3.14
        result = transform(c, grid_coords, method="iso", radius=1_000_000, latlon=True)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_constant_maps_to_zero_lbmgls(self, grid_coords):
        c = np.ones(len(grid_coords)) * 2.71
        result = transform(c, grid_coords, method="lbmgls", latlon=True)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_constant_maps_to_zero_cluster(self):
        coords = np.zeros((6, 2))  # coords irrelevant for cluster
        cluster = np.array([1, 1, 2, 2, 3, 3])
        c = np.ones(6) * 7.0
        result = transform(c, coords, method="cluster", cluster=cluster)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_linearity_nn(self, grid_coords):
        """M(a*y + b*z) == a*M(y) + b*M(z)."""
        y = RNG.standard_normal(len(grid_coords))
        z = RNG.standard_normal(len(grid_coords))
        a, b = 2.5, -1.3
        lhs = transform(a * y + b * z, grid_coords, method="nn", latlon=True)
        rhs = a * transform(y, grid_coords, method="nn", latlon=True) + b * transform(
            z, grid_coords, method="nn", latlon=True
        )
        np.testing.assert_allclose(lhs, rhs, atol=1e-10)

    def test_unknown_method_raises(self, grid_coords):
        with pytest.raises(ValueError, match="Unknown method"):
            transform(np.ones(len(grid_coords)), grid_coords, method="bogus")

    def test_iso_requires_radius(self, grid_coords):
        with pytest.raises(ValueError, match="radius"):
            transform(np.ones(len(grid_coords)), grid_coords, method="iso")

    def test_cluster_requires_cluster_arg(self, grid_coords):
        with pytest.raises(ValueError, match="cluster"):
            transform(np.ones(len(grid_coords)), grid_coords, method="cluster")


# ---------------------------------------------------------------------------
# spurtransform (DataFrame interface)
# ---------------------------------------------------------------------------


class TestSpurTransform:
    @pytest.fixture
    def df(self):
        return pd.DataFrame(
            {
                "lat": RNG.uniform(45, 55, 10),
                "lon": RNG.uniform(5, 15, 10),
                "y": RNG.standard_normal(10),
                "x": RNG.standard_normal(10),
            }
        )

    def test_output_columns_created(self, df):
        out = spurtransform(df, ["y", "x"], ["lat", "lon"], method="nn", prefix="nn_")
        assert "nn_y" in out.columns
        assert "nn_x" in out.columns

    def test_original_columns_unchanged(self, df):
        y_before = df["y"].values.copy()
        spurtransform(df, "y", ["lat", "lon"], method="nn")
        np.testing.assert_array_equal(df["y"].values, y_before)

    def test_row_count_preserved(self, df):
        out = spurtransform(df, "y", ["lat", "lon"], method="nn")
        assert len(out) == len(df)

    def test_matches_direct_transform(self, df):
        """spurtransform must produce same result as calling transform directly."""
        out = spurtransform(df, "y", ["lat", "lon"], method="nn", prefix="nn_")
        coords = df[["lat", "lon"]].values
        direct = transform(df["y"].values, coords, method="nn", latlon=True)
        np.testing.assert_allclose(out["nn_y"].values, direct, atol=1e-14)

    def test_missing_coordinates_raises(self):
        df = pd.DataFrame(
            {
                "lat": [45.0, np.nan, 47.0],
                "lon": [10.0, 11.0, 12.0],
                "y": [1.0, 2.0, 3.0],
            }
        )
        with pytest.raises(ValueError, match="missing"):
            spurtransform(df, "y", ["lat", "lon"], method="nn")

    def test_missing_variable_raises(self, df):
        with pytest.raises(ValueError, match="not found"):
            spurtransform(df, "nonexistent", ["lat", "lon"], method="nn")

    def test_cluster_demeaning_within_group(self):
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
        for g in [1, 2]:
            group_result = out.loc[df["grp"] == g, "cl_y"]
            assert group_result.sum() == pytest.approx(0.0, abs=1e-12)

    def test_cluster_nullable_string_with_na_raises(self):
        """Nullable StringDtype cluster column containing NA must raise, not silently misbehave."""
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
            spurtransform(
                df, "y", ["lat", "lon"], method="cluster", cluster_col="province"
            )

    def test_cluster_string_province_ids(self):
        """Province names (strings) must work — pandas StringDtype fix."""
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
        for g in ["Zurich", "Bern"]:
            group_result = out.loc[df["province"] == g, "cl_y"]
            assert group_result.sum() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Batch-1 issue fixes
# ---------------------------------------------------------------------------


class TestBatch1Fixes:
    """Regression tests for the four issues fixed in batch 1."""

    # Issue 1: cluster method must not require coord columns
    def test_cluster_works_without_coord_cols_in_df(self):
        """spurtransform(method='cluster') must succeed even if coord_cols are absent."""
        df = pd.DataFrame(
            {
                "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
                "grp": ["A", "A", "A", "B", "B", "B"],
            }
        )
        # coord_cols listed but not present — cluster should never touch them
        out = spurtransform(
            df, "y", ["lat", "lon"], method="cluster", cluster_col="grp"
        )
        for g in ["A", "B"]:
            assert out.loc[df["grp"] == g, "d_y"].sum() == pytest.approx(0.0, abs=1e-12)

    # Issue 2: LBM-GLS div-by-zero on degenerate coords
    def test_lbmgls_matrix_raises_on_identical_coords(self):
        """lbmgls_matrix must raise ValueError when all points are identical."""
        coords = np.zeros((5, 2))
        with pytest.raises(ValueError, match="identical"):
            lbmgls_matrix(coords, latlon=False)

    # Issue 3: q >= n must be rejected upfront
    def test_spurtest_rejects_q_ge_n(self):
        """spurtest must raise ValueError when q >= n."""
        n = 10
        df = pd.DataFrame(
            {
                "lat": RNG.uniform(45, 55, n),
                "lon": RNG.uniform(5, 15, n),
                "y": RNG.standard_normal(n),
            }
        )
        with pytest.raises(ValueError, match="q="):
            spurtest(df, "i1", "y", ["lat", "lon"], q=n, nrep=100, seed=0)

    # Issue 4: collinear X raises on residual tests
    def test_spurtest_i1resid_raises_on_collinear_X(self):
        """i1resid must raise ValueError when regressors are collinear."""
        n = 20
        x = RNG.standard_normal(n)
        df = pd.DataFrame(
            {
                "lat": RNG.uniform(45, 55, n),
                "lon": RNG.uniform(5, 15, n),
                "y": RNG.standard_normal(n),
                "x1": x,
                "x2": x,  # perfect duplicate → rank-deficient X
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

    def test_spurtest_i0resid_raises_on_collinear_X(self):
        """i0resid must raise ValueError when regressors are collinear."""
        n = 20
        x = RNG.standard_normal(n)
        df = pd.DataFrame(
            {
                "lat": RNG.uniform(45, 55, n),
                "lon": RNG.uniform(5, 15, n),
                "y": RNG.standard_normal(n),
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


class TestBatch2Fixes:
    """Regression tests for the five issues fixed in batch 2."""

    # Issue 5: NaN/inf in Y must fail fast with a clear error
    def test_spurtest_rejects_nan_y(self):
        n = 20
        df = pd.DataFrame(
            {
                "lat": RNG.uniform(45, 55, n),
                "lon": RNG.uniform(5, 15, n),
                "y": np.where(np.arange(n) == 5, np.nan, RNG.standard_normal(n)),
            }
        )
        with pytest.raises(ValueError, match="NaN or inf"):
            spurtest(df, "i1", "y", ["lat", "lon"], q=5, nrep=100, seed=0)

    def test_spurtest_rejects_inf_y(self):
        n = 20
        df = pd.DataFrame(
            {
                "lat": RNG.uniform(45, 55, n),
                "lon": RNG.uniform(5, 15, n),
                "y": np.where(np.arange(n) == 3, np.inf, RNG.standard_normal(n)),
            }
        )
        with pytest.raises(ValueError, match="NaN or inf"):
            spurtest(df, "i1", "y", ["lat", "lon"], q=5, nrep=100, seed=0)

    # Issue 6: spurhalflife invalid level/q/nrep
    def test_spurhalflife_rejects_level_gt_1(self):
        n = 20
        df = pd.DataFrame(
            {
                "lat": RNG.uniform(45, 55, n),
                "lon": RNG.uniform(5, 15, n),
                "y": RNG.standard_normal(n),
            }
        )
        with pytest.raises(ValueError, match="level="):
            spurhalflife(df, "y", ["lat", "lon"], level=1.5, q=5, nrep=100, seed=0)

    def test_spurhalflife_rejects_level_zero(self):
        n = 20
        df = pd.DataFrame(
            {
                "lat": RNG.uniform(45, 55, n),
                "lon": RNG.uniform(5, 15, n),
                "y": RNG.standard_normal(n),
            }
        )
        with pytest.raises(ValueError, match="level="):
            spurhalflife(df, "y", ["lat", "lon"], level=0.0, q=5, nrep=100, seed=0)

    def test_spurhalflife_rejects_q_zero(self):
        n = 20
        df = pd.DataFrame(
            {
                "lat": RNG.uniform(45, 55, n),
                "lon": RNG.uniform(5, 15, n),
                "y": RNG.standard_normal(n),
            }
        )
        with pytest.raises(ValueError, match="q="):
            spurhalflife(df, "y", ["lat", "lon"], level=0.95, q=0, nrep=100, seed=0)

    # Issue 7: spurhalflife raises on degenerate (identical) coordinates
    def test_spurhalflife_raises_on_identical_coords(self):
        df = pd.DataFrame(
            {
                "lat": [48.0] * 10,
                "lon": [11.0] * 10,
                "y": RNG.standard_normal(10),
            }
        )
        with pytest.raises(ValueError, match="identical"):
            spurhalflife(df, "y", ["lat", "lon"], level=0.95, q=5, nrep=100, seed=0)

    # Issue 8: large-N haversine distance matrix raises with informative error
    def test_get_distance_matrix_raises_for_large_n(self):
        coords = np.zeros((10_001, 2))
        with pytest.raises(ValueError, match="10,000"):
            get_distance_matrix(coords, latlon=True)

    # Issue 8: Euclidean path (no limit) still works for large-ish N
    def test_get_distance_matrix_euclidean_no_limit(self):
        coords = np.zeros((10_001, 2))
        D = get_distance_matrix(coords, latlon=False)
        assert D.shape == (10_001, 10_001)


# ---------------------------------------------------------------------------
# Validation CSV: Python vs saved expected output
# ---------------------------------------------------------------------------


class TestValidationCSV:
    """Checks Python output agrees with the saved validation run."""

    def test_nn_y_matches_saved(self, val_df):
        coords = val_df[["lat", "lon"]].values
        result = transform(val_df["y"].values, coords, method="nn", latlon=True)
        np.testing.assert_allclose(
            result, val_df["py_nn_y"].values, rtol=1e-10, atol=1e-10
        )

    def test_nn_x_matches_saved(self, val_df):
        coords = val_df[["lat", "lon"]].values
        result = transform(val_df["x"].values, coords, method="nn", latlon=True)
        np.testing.assert_allclose(
            result, val_df["py_nn_x"].values, rtol=1e-10, atol=1e-10
        )

    def test_iso_y_matches_saved(self, val_df):
        # Validation CSV was generated with 200 km radius (see validate_against_stata.py)
        coords = val_df[["lat", "lon"]].values
        result = transform(
            val_df["y"].values, coords, method="iso", radius=200_000, latlon=True
        )
        np.testing.assert_allclose(
            result, val_df["py_iso_y"].values, rtol=1e-10, atol=1e-10
        )

    def test_cluster_y_matches_saved(self, val_df):
        out = spurtransform(
            val_df,
            "y",
            ["lat", "lon"],
            method="cluster",
            cluster_col="cluster",
            prefix="cl_",
        )
        np.testing.assert_allclose(
            out["cl_y"].values, val_df["py_cl_y"].values, rtol=1e-10, atol=1e-10
        )
