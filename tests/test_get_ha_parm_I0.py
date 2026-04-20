import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spur import (
    demean_matrix,
    get_R,
    get_distmat_normalized,
    get_ha_parm_I0,
    get_sigma_dm,
    get_sigma_lbm,
    getcbar,
    load_chetty_data,
    standardize,
)
from tests.config import PARITY_ATOL
from tests.utils import STATA, ensure_spur_stata_installed, stata_path

NREP = 100000


@pytest.fixture
def chetty_df() -> pd.DataFrame:
    df = load_chetty_data()
    df = df[~df["state"].isin(["AK", "HI"])][["am", "fracblack", "lat", "lon", "state"]]
    df = df.dropna(subset=["am", "fracblack", "lat", "lon"]).reset_index(drop=True)
    return standardize(df, ["am", "fracblack"])


def run_stata_ha_param_i0(tmp_path: Path, df: pd.DataFrame) -> float:
    stata_root = ensure_spur_stata_installed()
    plus = stata_root / "plus"
    personal = stata_root / "personal"
    input_csv = tmp_path / "i0_input.csv"
    output_csv = tmp_path / "i0_output.csv"

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

        _spurtest_i0 am, q(10) nrep({NREP}) latlong

        scalar h = r(ha_param)

        clear
        set obs 1
        gen double ha_param = h

        export delimited using "{stata_path(output_csv)}", replace
        """
    )
    result = subprocess.run(
        [STATA, "-q"],
        cwd=tmp_path,
        input=script + "\nexit, clear\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.strip() or result.stdout.strip()
    return float(pd.read_csv(output_csv).iloc[0]["ha_param"])


def test_get_ha_parm_i0_returns_positive_value() -> None:
    rng = np.random.default_rng(42)
    coords = np.column_stack([rng.uniform(45, 55, 30), rng.uniform(5, 15, 30)])
    distmat = get_distmat_normalized(coords, latlon=True)
    sigdm_bm = demean_matrix(get_sigma_lbm(distmat))
    R = get_R(sigdm_bm, 10)
    c = getcbar(0.001, distmat)
    om_i0 = R.T @ get_sigma_dm(distmat, c) @ R
    om_bm = R.T @ sigdm_bm @ R
    emat = rng.standard_normal((10, 10_000))

    ha_param = get_ha_parm_I0(om_i0, om_i0, om_bm, emat)

    assert np.isfinite(ha_param)
    assert ha_param > 0


@pytest.mark.skipif(STATA is None, reason="stata-mp not installed")
def test_get_ha_parm_i0_matches_stata(
    tmp_path: Path,
    chetty_df: pd.DataFrame,
) -> None:
    coords = chetty_df[["lat", "lon"]].to_numpy()
    distmat = get_distmat_normalized(coords, latlon=True)
    sigdm_bm = demean_matrix(get_sigma_lbm(distmat))
    R = get_R(sigdm_bm, 10)
    c = getcbar(0.001, distmat)
    om_i0 = R.T @ get_sigma_dm(distmat, c) @ R
    om_bm = R.T @ sigdm_bm @ R
    emat = np.random.default_rng(42).standard_normal((10, NREP))

    py_value = get_ha_parm_I0(om_i0, om_i0, om_bm, emat)
    st_value = run_stata_ha_param_i0(tmp_path, chetty_df)

    assert py_value == pytest.approx(st_value, abs=PARITY_ATOL)
