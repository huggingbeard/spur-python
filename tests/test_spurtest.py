import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spur import (
    TestResult,
    load_chetty_data,
    spurtest,
    spurtest_i0,
    spurtest_i0resid,
    spurtest_i1,
    spurtest_i1resid,
    standardize,
)
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
        spurtest_i1(
            "am",
            chetty_df,
            lon="lon",
            lat="lat",
            q=len(chetty_df),
            nrep=100,
            seed=0,
        )

    result = spurtest_i1("am", chetty_df, lon="lon", lat="lat", q=10, nrep=200, seed=42)

    assert isinstance(result, TestResult)
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
        spurtest_i1("y", df_nan, lon="lon", lat="lat", q=10, nrep=100, seed=0)

    df_inf = pd.DataFrame(
        {
            "lat": rng.uniform(45, 55, n),
            "lon": rng.uniform(5, 15, n),
            "y": np.where(np.arange(n) == 3, np.inf, rng.standard_normal(n)),
        }
    )
    with pytest.raises(ValueError, match="NaN or inf"):
        spurtest_i1("y", df_inf, lon="lon", lat="lat", q=10, nrep=100, seed=0)


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
        spurtest_i1resid(
            "y ~ x1 + x2",
            df,
            lon="lon",
            lat="lat",
            q=10,
            nrep=100,
            seed=0,
        )

    with pytest.raises(ValueError, match="rank-deficient"):
        spurtest_i0resid(
            "y ~ x1 + x2",
            df,
            lon="lon",
            lat="lat",
            q=10,
            nrep=100,
            seed=0,
        )


def test_spurtest_selector_accepts_bare_variable_and_y_tilde_1(
    chetty_df: pd.DataFrame,
) -> None:
    bare = spurtest(
        "am", chetty_df, test="i1", lon="lon", lat="lat", q=10, nrep=200, seed=42
    )
    rhs1 = spurtest(
        "am ~ 1",
        chetty_df,
        test="i1",
        lon="lon",
        lat="lat",
        q=10,
        nrep=200,
        seed=42,
    )

    assert bare.LR == pytest.approx(rhs1.LR)
    assert bare.pvalue == pytest.approx(rhs1.pvalue)
    assert np.allclose(bare.cv, rhs1.cv)
    assert bare.ha_param == pytest.approx(rhs1.ha_param)


def test_spurtest_i_wrappers_take_plain_variable_strings(
    chetty_df: pd.DataFrame,
) -> None:
    i0 = spurtest_i0("am", chetty_df, lon="lon", lat="lat", q=10, nrep=200, seed=42)
    i1 = spurtest_i1("am", chetty_df, lon="lon", lat="lat", q=10, nrep=200, seed=42)

    assert isinstance(i0, TestResult)
    assert isinstance(i1, TestResult)


def test_spurtest_selector_rejects_rhs_for_single_variable_tests(
    chetty_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):
        spurtest(
            "am ~ fracblack",
            chetty_df,
            test="i1",
            lon="lon",
            lat="lat",
            q=10,
            nrep=100,
            seed=0,
        )


def test_spurtest_selector_matches_wrappers(chetty_df: pd.DataFrame) -> None:
    sel_i0 = spurtest(
        "am", chetty_df, test="i0", lon="lon", lat="lat", q=10, nrep=100, seed=42
    )
    wrap_i0 = spurtest_i0(
        "am", chetty_df, lon="lon", lat="lat", q=10, nrep=100, seed=42
    )
    sel_i1 = spurtest(
        "am", chetty_df, test="i1", lon="lon", lat="lat", q=10, nrep=100, seed=42
    )
    wrap_i1 = spurtest_i1(
        "am", chetty_df, lon="lon", lat="lat", q=10, nrep=100, seed=42
    )
    sel_i0r = spurtest(
        "am ~ fracblack",
        chetty_df,
        test="i0resid",
        lon="lon",
        lat="lat",
        q=10,
        nrep=100,
        seed=42,
    )
    wrap_i0r = spurtest_i0resid(
        "am ~ fracblack",
        chetty_df,
        lon="lon",
        lat="lat",
        q=10,
        nrep=100,
        seed=42,
    )
    sel_i1r = spurtest(
        "am ~ fracblack",
        chetty_df,
        test="i1resid",
        lon="lon",
        lat="lat",
        q=10,
        nrep=100,
        seed=42,
    )
    wrap_i1r = spurtest_i1resid(
        "am ~ fracblack",
        chetty_df,
        lon="lon",
        lat="lat",
        q=10,
        nrep=100,
        seed=42,
    )

    for left, right in [
        (sel_i0, wrap_i0),
        (sel_i1, wrap_i1),
        (sel_i0r, wrap_i0r),
        (sel_i1r, wrap_i1r),
    ]:
        assert left.LR == pytest.approx(right.LR)
        assert left.pvalue == pytest.approx(right.pvalue)
        assert np.allclose(left.cv, right.cv)
        assert left.ha_param == pytest.approx(right.ha_param)


@pytest.mark.skipif(STATA is None, reason="stata-mp not installed")
def test_spurtest_matches_stata(
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
    st_teststat = run_stata_spurtest(tmp_path, chetty_df)

    assert py_value.LR == pytest.approx(st_teststat, abs=PARITY_ATOL)
