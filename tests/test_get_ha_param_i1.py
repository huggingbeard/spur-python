import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spur import load_chetty_data, standardize
from spur.utils.dist import normalized_distmat
from spur.utils.inference import get_ha_param_i1
from spur.utils.matrix import demean_matrix, get_r, get_sigma_lbm
from tests.config import PARITY_ATOL
from tests.utils import (
    STATA,
    ensure_spur_stata_installed,
    execute_stata_command,
    stata_path,
)

NREP = 100000


@pytest.fixture
def chetty_df() -> pd.DataFrame:
    df = load_chetty_data()
    df = df[~df["state"].isin(["AK", "HI"])][["am", "fracblack", "lat", "lon", "state"]]
    df = df.dropna(subset=["am", "fracblack", "lat", "lon"]).reset_index(drop=True)
    return standardize(df, ["am", "fracblack"])


def run_stata_ha_param_i1(tmp_path: Path, df: pd.DataFrame) -> float:
    stata_root = ensure_spur_stata_installed()
    plus = stata_root / "plus"
    personal = stata_root / "personal"
    input_csv = tmp_path / "i1_input.csv"
    output_csv = tmp_path / "i1_output.csv"

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

        _spurtest_i1 am, q(10) nrep({NREP}) latlong

        scalar h = r(ha_param)

        clear
        set obs 1
        gen double ha_param = h

        export delimited using "{stata_path(output_csv)}", replace
        """
    )
    execute_stata_command(script, tmp_path)
    return float(pd.read_csv(output_csv).iloc[0]["ha_param"])


def test_get_ha_param_i1_returns_positive_value() -> None:
    rng = np.random.default_rng(42)
    coords = np.column_stack([rng.uniform(45, 55, 30), rng.uniform(5, 15, 30)])
    distmat = normalized_distmat(coords, latlon=True)
    sigdm_bm = demean_matrix(get_sigma_lbm(distmat))
    R = get_r(sigdm_bm, 10)
    om_ho = R.T @ sigdm_bm @ R
    emat = rng.standard_normal((10, 10_000))

    ha_param = get_ha_param_i1(om_ho, distmat, R, emat)

    assert np.isfinite(ha_param)
    assert ha_param > 0


@pytest.mark.skipif(STATA is None, reason="stata-mp not installed")
def test_get_ha_param_i1_matches_stata(
    tmp_path: Path,
    chetty_df: pd.DataFrame,
) -> None:
    coords = chetty_df[["lat", "lon"]].to_numpy()
    distmat = normalized_distmat(coords, latlon=True)
    sigdm_bm = demean_matrix(get_sigma_lbm(distmat))
    R = get_r(sigdm_bm, 10)
    om_ho = R.T @ sigdm_bm @ R
    emat = np.random.default_rng(42).standard_normal((10, NREP))

    py_value = get_ha_param_i1(om_ho, distmat, R, emat)
    st_value = run_stata_ha_param_i1(tmp_path, chetty_df)

    assert py_value == pytest.approx(st_value, abs=PARITY_ATOL)
