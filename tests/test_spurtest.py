import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spur import SpurTestResult, load_chetty_data, spurtest, standardize
from tests.utils import (
    STATA,
    ensure_spur_stata_installed,
    execute_stata_command,
    stata_path,
)

PARITY_ATOL = 1e-5
NREP = 100000


@pytest.fixture
def chetty_df() -> pd.DataFrame:
    df = load_chetty_data()
    df = df[~df["state"].isin(["AK", "HI"])][["am", "fracblack", "lat", "lon", "state"]]
    df = df.dropna(subset=["am", "fracblack", "lat", "lon"]).reset_index(drop=True)
    return standardize(df, ["am", "fracblack"])


def run_stata_spurtest(tmp_path: Path, df: pd.DataFrame) -> float:
    stata_root = ensure_spur_stata_installed()
    plus = stata_root / "plus"
    personal = stata_root / "personal"
    input_csv = tmp_path / "spurtest_input.csv"
    output_csv = tmp_path / "spurtest_output.csv"

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

        spurtest i1 am, q(10) nrep({NREP}) latlong

        scalar spurtest_teststat = r(teststat)

        clear
        set obs 1
        gen double teststat = spurtest_teststat

        export delimited using "{stata_path(output_csv)}", replace
        """
    )
    execute_stata_command(script, tmp_path)

    return float(pd.read_csv(output_csv).iloc[0]["teststat"])


def test_spurtest_validates_q_and_formats_summary(chetty_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="q="):
        spurtest(
            chetty_df,
            "i1",
            "am",
            ["lat", "lon"],
            q=len(chetty_df),
            nrep=100,
            seed=0,
        )

    result = spurtest(chetty_df, "i1", "am", ["lat", "lon"], q=10, nrep=200, seed=42)

    assert isinstance(result, SpurTestResult)
    assert np.isfinite(result.LR)
    assert 0.0 <= result.pvalue <= 1.0
    assert result.cv.shape == (3,)
    assert "Spatial I1 Test Results" in result.summary()


def test_spurtest_rejects_nan_and_inf_in_dependent_variable() -> None:
    rng = np.random.default_rng(42)
    n = 20

    df_nan = pd.DataFrame(
        {
            "lat": rng.uniform(45, 55, n),
            "lon": rng.uniform(5, 15, n),
            "y": np.where(np.arange(n) == 5, np.nan, rng.standard_normal(n)),
        }
    )
    with pytest.raises(ValueError, match="NaN or inf"):
        spurtest(df_nan, "i1", "y", ["lat", "lon"], q=10, nrep=100, seed=0)

    df_inf = pd.DataFrame(
        {
            "lat": rng.uniform(45, 55, n),
            "lon": rng.uniform(5, 15, n),
            "y": np.where(np.arange(n) == 3, np.inf, rng.standard_normal(n)),
        }
    )
    with pytest.raises(ValueError, match="NaN or inf"):
        spurtest(df_inf, "i1", "y", ["lat", "lon"], q=10, nrep=100, seed=0)


def test_spurtest_residual_variants_reject_rank_deficient_regressors() -> None:
    rng = np.random.default_rng(42)
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
            q=10,
            nrep=100,
            seed=0,
        )

    with pytest.raises(ValueError, match="rank-deficient"):
        spurtest(
            df,
            "i0resid",
            "y",
            ["lat", "lon"],
            indepvars=["x1", "x2"],
            q=10,
            nrep=100,
            seed=0,
        )


@pytest.mark.skipif(STATA is None, reason="stata-mp not installed")
def test_spurtest_matches_stata(
    tmp_path: Path,
    chetty_df: pd.DataFrame,
) -> None:
    py_value = spurtest(
        chetty_df,
        "i1",
        "am",
        ["lat", "lon"],
        q=10,
        nrep=NREP,
        seed=42,
    )
    st_teststat = run_stata_spurtest(tmp_path, chetty_df)

    assert py_value.LR == pytest.approx(st_teststat, abs=PARITY_ATOL)
