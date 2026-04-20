import textwrap
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from spur.utils.matrix import nn_matrix
from tests.config import ATOL, PARITY_ATOL, PARITY_RTOL, RTOL
from tests.utils import (
    STATA,
    ensure_spur_stata_installed,
    execute_stata_command,
    stata_path,
)


@pytest.fixture
def grid_coords() -> np.ndarray:
    rng = np.random.default_rng(42)
    lat = rng.uniform(45, 55, 8)
    lon = rng.uniform(5, 15, 8)
    return np.column_stack([lat, lon])


@pytest.fixture
def small_coords() -> np.ndarray:
    return np.array([[float(i), 0.0] for i in range(5)])


def run_stata_nn_matrix(tmp_path: Path, coords: np.ndarray) -> np.ndarray:
    stata_root = ensure_spur_stata_installed()
    plus = stata_root / "plus"
    personal = stata_root / "personal"
    input_csv = tmp_path / "nn_input.csv"
    output_csv = tmp_path / "nn_output.csv"

    df = pd.DataFrame({"lat": coords[:, 0], "lon": coords[:, 1]})
    for i in range(len(coords)):
        basis = np.zeros(len(coords))
        basis[i] = 1.0
        df[f"e{i}"] = basis
    df.to_csv(input_csv, index=False)

    varlist = " ".join(f"e{i}" for i in range(len(coords)))
    script = textwrap.dedent(
        f"""
        clear all
        set more off

        sysdir set PLUS "{stata_path(plus)}"
        sysdir set PERSONAL "{stata_path(personal)}"

        import delimited using "{stata_path(input_csv)}", clear asdouble
        rename lat s_1
        rename lon s_2

        spurtransform {varlist}, prefix(m_) transformation(nn) latlong

        export delimited using "{stata_path(output_csv)}", replace
        """
    )
    execute_stata_command(script, tmp_path)

    stata_df = pd.read_csv(output_csv)
    return np.column_stack([stata_df[f"m_e{i}"].to_numpy() for i in range(len(coords))])


def test_nn_matrix_rows_sum_to_zero_and_diagonal_is_one(
    grid_coords: np.ndarray,
) -> None:
    matrix = nn_matrix(grid_coords, latlon=True)

    npt.assert_allclose(matrix.sum(axis=1), 0.0, atol=ATOL, rtol=RTOL)
    npt.assert_allclose(np.diag(matrix), 1.0, atol=ATOL, rtol=RTOL)


def test_nn_matrix_off_diagonal_entries_are_nonpositive(
    grid_coords: np.ndarray,
) -> None:
    matrix = nn_matrix(grid_coords, latlon=True)
    off_diagonal = matrix - np.diag(np.diag(matrix))

    assert np.all(off_diagonal <= 1e-12)


def test_nn_matrix_each_row_has_a_negative_weight(grid_coords: np.ndarray) -> None:
    matrix = nn_matrix(grid_coords, latlon=True)

    for row in matrix:
        assert np.any(row < 0)


def test_nn_matrix_splits_ties_evenly() -> None:
    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

    matrix = nn_matrix(coords, latlon=False)

    for i in range(4):
        neg = matrix[i][matrix[i] < 0]
        assert len(neg) == 2
        npt.assert_allclose(neg, -0.5, atol=1e-14, rtol=0.0)


def test_nn_matrix_euclidean_rows_also_sum_to_zero(
    small_coords: np.ndarray,
) -> None:
    matrix = nn_matrix(small_coords, latlon=False)

    npt.assert_allclose(matrix.sum(axis=1), 0.0, atol=1e-14, rtol=0.0)


@pytest.mark.skipif(STATA is None, reason="stata-mp not installed")
def test_nn_matrix_matches_stata(tmp_path: Path, grid_coords: np.ndarray) -> None:
    py_value = nn_matrix(grid_coords, latlon=True)
    st_value = run_stata_nn_matrix(tmp_path, grid_coords)

    npt.assert_allclose(py_value, st_value, atol=PARITY_ATOL, rtol=PARITY_RTOL)
