import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spur import load_chetty_data, spurtest_i1, standardize
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


def run_stata_spurtest_i1(tmp_path: Path, df: pd.DataFrame) -> tuple[float, float]:
    stata_root = ensure_spur_stata_installed()
    plus = stata_root / "plus"
    personal = stata_root / "personal"
    input_csv = tmp_path / "spurtest_i1_input.csv"
    output_csv = tmp_path / "spurtest_i1_output.csv"

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

        scalar teststat = r(teststat)
        scalar ha = r(ha_param)

        clear
        set obs 1
        gen double teststat = teststat
        gen double ha = ha

        export delimited using "{stata_path(output_csv)}", replace
        """
    )
    execute_stata_command(script, tmp_path)
    row = pd.read_csv(output_csv).iloc[0]
    return float(row["teststat"]), float(row["ha"])


def test_spurtest_i1_returns_valid_result() -> None:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "lat": rng.uniform(45, 55, 30),
            "lon": rng.uniform(5, 15, 30),
            "y": rng.standard_normal(30),
        }
    )

    result = spurtest_i1("y", df, lon="lon", lat="lat", q=10, nrep=10_000, seed=42)

    assert np.isfinite(result.LR)
    assert 0.0 <= result.pvalue <= 1.0
    assert result.cv.shape == (3,)
    assert np.isfinite(result.ha_param)


@pytest.mark.skipif(STATA is None, reason="stata-mp not installed")
def test_spurtest_i1_matches_stata(
    tmp_path: Path,
    chetty_df: pd.DataFrame,
) -> None:
    py_value = spurtest_i1(
        "am",
        chetty_df,
        lon="lon",
        lat="lat",
        q=10,
        nrep=NREP,
        seed=42,
    )
    st_LR, st_ha = run_stata_spurtest_i1(tmp_path, chetty_df)

    assert py_value.LR == pytest.approx(st_LR, abs=PARITY_ATOL)
    assert py_value.ha_param == pytest.approx(st_ha, abs=PARITY_ATOL)
