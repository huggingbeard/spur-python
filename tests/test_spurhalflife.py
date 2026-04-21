import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spur import HalfLifeResult, load_chetty_data, spurhalflife, standardize
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


def run_stata_spurhalflife(tmp_path: Path, df: pd.DataFrame) -> pd.Series:
    stata_root = ensure_spur_stata_installed()
    plus = stata_root / "plus"
    personal = stata_root / "personal"
    input_csv = tmp_path / "spurhalflife_input.csv"
    output_csv = tmp_path / "spurhalflife_output.csv"

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

        spurhalflife am, q(15) nrep({NREP}) level(95) latlong

        scalar hl_ci_l = r(ci_l)
        scalar hl_ci_u = r(ci_u)
        scalar hl_max_dist = r(max_dist)

        clear
        set obs 1
        gen double ci_lower = hl_ci_l
        gen double ci_upper = hl_ci_u
        gen double max_dist = hl_max_dist

        export delimited using "{stata_path(output_csv)}", replace
        """
    )
    execute_stata_command(script, tmp_path)

    return pd.read_csv(output_csv).iloc[0]


def test_spurhalflife_validates_level_and_summary(chetty_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="level="):
        spurhalflife(
            "am",
            chetty_df,
            lon="lon",
            lat="lat",
            level=150.0,
            q=5,
            nrep=100,
            seed=0,
        )

    result = spurhalflife("am", chetty_df, lon="lon", lat="lat", q=5, nrep=200, seed=42)

    assert isinstance(result, HalfLifeResult)
    assert result.ci_lower >= 0 or np.isnan(result.ci_lower)
    assert "Spatial half-life" in result.summary()


def test_spurhalflife_rejects_more_invalid_parameters() -> None:
    rng = np.random.default_rng(42)
    n = 20
    df = pd.DataFrame(
        {
            "lat": rng.uniform(45, 55, n),
            "lon": rng.uniform(5, 15, n),
            "y": rng.standard_normal(n),
        }
    )

    with pytest.raises(ValueError, match="level="):
        spurhalflife("y", df, lon="lon", lat="lat", level=0.0, q=5, nrep=100, seed=0)

    with pytest.raises(ValueError, match="q="):
        spurhalflife("y", df, lon="lon", lat="lat", level=95, q=0, nrep=100, seed=0)


def test_spurhalflife_rejects_identical_coordinates() -> None:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "lat": [48.0] * 10,
            "lon": [11.0] * 10,
            "y": rng.standard_normal(10),
        }
    )

    with pytest.raises(ValueError, match="identical"):
        spurhalflife("y", df, lon="lon", lat="lat", level=95, q=5, nrep=100, seed=0)


@pytest.mark.skipif(STATA is None, reason="stata-mp not installed")
def test_spurhalflife_matches_stata(
    tmp_path: Path,
    chetty_df: pd.DataFrame,
) -> None:
    py_value = spurhalflife(
        "am",
        chetty_df,
        lon="lon",
        lat="lat",
        q=15,
        nrep=NREP,
        level=95,
        normdist=False,
        seed=42,
    )
    st_value = run_stata_spurhalflife(tmp_path, chetty_df)

    assert py_value.ci_lower == pytest.approx(st_value["ci_lower"], abs=HALFLIFE_ATOL)
    if np.isinf(py_value.ci_upper):
        assert pd.isna(st_value["ci_upper"])
    else:
        assert py_value.ci_upper == pytest.approx(
            st_value["ci_upper"], abs=HALFLIFE_ATOL
        )
    assert py_value.max_dist == pytest.approx(st_value["max_dist"], abs=HALFLIFE_ATOL)
