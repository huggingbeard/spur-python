import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spur import load_chetty_data, standardize
from spur.utils.dist import get_distance_matrix
from spur.utils.inference import spur_persistence
from tests.utils import (
    STATA,
    ensure_spur_stata_installed,
    execute_stata_command,
    stata_path,
)

HALFLIFE_ATOL = 1e-2
NREP = 100000


@pytest.fixture
def chetty_df() -> pd.DataFrame:
    df = load_chetty_data()
    df = df[~df["state"].isin(["AK", "HI"])][["am", "lat", "lon", "state"]]
    df = df.dropna(subset=["am", "lat", "lon"]).reset_index(drop=True)
    return standardize(df, ["am"])


def run_stata_spur_persistence(tmp_path: Path, df: pd.DataFrame) -> tuple[float, float]:
    stata_root = ensure_spur_stata_installed()
    plus = stata_root / "plus"
    personal = stata_root / "personal"
    input_csv = tmp_path / "spatial_persistence_input.csv"
    output_csv = tmp_path / "spatial_persistence_output.csv"

    df.to_csv(input_csv, index=False)

    script = textwrap.dedent(
        f"""
        clear all
        set more off

        sysdir set PLUS "{stata_path(plus)}"
        sysdir set PERSONAL "{stata_path(personal)}"

        import delimited using "{stata_path(input_csv)}", clear asdouble
        rename lat s_1
        rename lon s_2
        set seed 42

        spurhalflife am, q(15) nrep({NREP}) level(95) latlong normdist

        scalar ci_l = r(ci_l)
        scalar ci_u = r(ci_u)

        clear
        set obs 1
        gen double ci_lower = ci_l
        gen double ci_upper = ci_u

        export delimited using "{stata_path(output_csv)}", replace
        """
    )
    execute_stata_command(script, tmp_path)
    row = pd.read_csv(output_csv).iloc[0]
    return float(row["ci_lower"]), float(row["ci_upper"])


def test_spur_persistence_returns_ordered_interval(chetty_df: pd.DataFrame) -> None:
    coords = chetty_df[["lat", "lon"]].to_numpy()
    y = chetty_df["am"].to_numpy()
    distmat = get_distance_matrix(coords, latlon=True)
    distmat = distmat / distmat.max()
    emat = np.random.default_rng(42).standard_normal((15, 200))

    ci_lower, ci_upper = spur_persistence(y - y.mean(), distmat, emat, 0.95)

    assert np.isfinite(ci_lower)
    assert np.isfinite(ci_upper)
    assert 0.0 < ci_lower <= ci_upper


@pytest.mark.skipif(STATA is None, reason="stata-mp not installed")
def test_spur_persistence_matches_stata(
    tmp_path: Path,
    chetty_df: pd.DataFrame,
) -> None:
    coords = chetty_df[["lat", "lon"]].to_numpy()
    y = chetty_df["am"].to_numpy()
    distmat = get_distance_matrix(coords, latlon=True)
    distmat = distmat / distmat.max()
    emat = np.random.default_rng(42).standard_normal((15, NREP))

    py_lower, py_upper = spur_persistence(y - y.mean(), distmat, emat, 0.95)
    st_lower, st_upper = run_stata_spur_persistence(tmp_path, chetty_df)

    assert py_lower == pytest.approx(st_lower, abs=HALFLIFE_ATOL)
    if np.isnan(st_upper):
        assert py_upper == 100.0
    else:
        assert py_upper == pytest.approx(st_upper, abs=HALFLIFE_ATOL)
