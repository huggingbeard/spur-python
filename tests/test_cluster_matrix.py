import textwrap
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from spur.utils.matrix import cluster_matrix
from tests.config import ATOL, PARITY_ATOL, PARITY_RTOL, RTOL
from tests.utils import (
    STATA,
    ensure_spur_stata_installed,
    execute_stata_command,
    stata_path,
)


def run_stata_cluster_matrix(tmp_path: Path, cluster: np.ndarray) -> np.ndarray:
    stata_root = ensure_spur_stata_installed()
    plus = stata_root / "plus"
    personal = stata_root / "personal"
    input_csv = tmp_path / "cluster_input.csv"
    output_csv = tmp_path / "cluster_output.csv"

    df = pd.DataFrame(
        {
            "lat": np.zeros(len(cluster)),
            "lon": np.zeros(len(cluster)),
            "cluster": cluster,
        }
    )
    for i in range(len(cluster)):
        basis = np.zeros(len(cluster))
        basis[i] = 1.0
        df[f"e{i}"] = basis
    df.to_csv(input_csv, index=False)

    varlist = " ".join(f"e{i}" for i in range(len(cluster)))
    script = textwrap.dedent(
        f"""
        clear all
        set more off

        sysdir set PLUS "{stata_path(plus)}"
        sysdir set PERSONAL "{stata_path(personal)}"

        import delimited using "{stata_path(input_csv)}", clear asdouble
        rename lat s_1
        rename lon s_2

        spurtransform {varlist}, prefix(m_) transformation(cluster) clustvar(cluster)

        export delimited using "{stata_path(output_csv)}", replace
        """
    )
    execute_stata_command(script, tmp_path)

    stata_df = pd.read_csv(output_csv)
    return np.column_stack(
        [stata_df[f"m_e{i}"].to_numpy() for i in range(len(cluster))]
    )


def test_cluster_matrix_demeans_within_cluster() -> None:
    cluster = np.array(["A", "A", "A", "B", "B", "B"])
    values = np.array([10.0, 20.0, 30.0, 100.0, 200.0, 300.0])

    matrix = cluster_matrix(cluster)
    transformed = matrix @ values

    npt.assert_allclose(matrix.sum(axis=1), 0.0, atol=ATOL, rtol=RTOL)
    assert transformed[:3].sum() == pytest.approx(0.0, abs=ATOL)
    assert transformed[3:].sum() == pytest.approx(0.0, abs=ATOL)


@pytest.mark.skipif(STATA is None, reason="stata-mp not installed")
def test_cluster_matrix_matches_stata(tmp_path: Path) -> None:
    cluster = np.array(["A", "A", "A", "B", "B", "B"])

    py_value = cluster_matrix(cluster)
    st_value = run_stata_cluster_matrix(tmp_path, cluster)

    npt.assert_allclose(py_value, st_value, atol=PARITY_ATOL, rtol=PARITY_RTOL)
