import textwrap
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from spur.utils.matrix import iso_matrix
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
    lat = rng.uniform(48, 52, 8)
    lon = rng.uniform(8, 12, 8)
    return np.column_stack([lat, lon])


def run_stata_iso_matrix(
    tmp_path: Path, coords: np.ndarray, radius: float
) -> np.ndarray:
    stata_root = ensure_spur_stata_installed()
    plus = stata_root / "plus"
    personal = stata_root / "personal"
    input_csv = tmp_path / "iso_input.csv"
    output_csv = tmp_path / "iso_output.csv"

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

        spurtransform {varlist}, prefix(m_) transformation(iso) radius({radius}) latlong

        export delimited using "{stata_path(output_csv)}", replace
        """
    )
    execute_stata_command(script, tmp_path)

    stata_df = pd.read_csv(output_csv)
    return np.column_stack([stata_df[f"m_e{i}"].to_numpy() for i in range(len(coords))])


def test_iso_matrix_rows_sum_to_zero_when_everyone_has_neighbors(
    grid_coords: np.ndarray,
) -> None:
    matrix = iso_matrix(grid_coords, radius=500_000, latlon=True)

    npt.assert_allclose(matrix.sum(axis=1), 0.0, atol=ATOL, rtol=RTOL)
    npt.assert_allclose(np.diag(matrix), 1.0, atol=ATOL, rtol=RTOL)


def test_iso_matrix_zeroes_rows_for_isolated_observations() -> None:
    coords = np.array([[0.0, 0.0], [50.0, 0.0]])

    matrix = iso_matrix(coords, radius=0.1, latlon=False)

    npt.assert_allclose(matrix, 0.0, atol=ATOL, rtol=RTOL)


def test_iso_matrix_diagonal_is_one_when_neighbors_exist(
    grid_coords: np.ndarray,
) -> None:
    matrix = iso_matrix(grid_coords, radius=500_000, latlon=True)

    npt.assert_allclose(np.diag(matrix), 1.0, atol=ATOL, rtol=RTOL)


def test_iso_matrix_negative_weights_sum_to_minus_one_for_nonisolated_rows(
    grid_coords: np.ndarray,
) -> None:
    matrix = iso_matrix(grid_coords, radius=300_000, latlon=True)

    for i, row in enumerate(matrix):
        if matrix[i, i] != 0:
            assert row[row < 0].sum() == pytest.approx(-1.0, abs=ATOL)


@pytest.mark.skipif(STATA is None, reason="stata-mp not installed")
def test_iso_matrix_matches_stata(tmp_path: Path, grid_coords: np.ndarray) -> None:
    py_value = iso_matrix(grid_coords, radius=300_000, latlon=True)
    st_value = run_stata_iso_matrix(tmp_path, grid_coords, radius=300_000)

    npt.assert_allclose(py_value, st_value, atol=PARITY_ATOL, rtol=PARITY_RTOL)
