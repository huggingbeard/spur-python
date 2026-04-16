from __future__ import annotations
import shutil
import subprocess
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from spur import load_chetty_data, spurhalflife, spurtest, spurtransform, standardize


# constants
STATA = shutil.which("stata-mp")
SPUR_COMMIT = "e694f9b09f04657554321ce90e03190280464792"
SPUR_STATA = f"https://raw.githubusercontent.com/pdavidboll/SPUR/{SPUR_COMMIT}/"
SPUR_FILES = ["spurtransform.ado", "spurtest.ado", "spurhalflife.ado"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATA_CACHE_ROOT = PROJECT_ROOT / ".pytest_cache" / "stata_spur" / SPUR_COMMIT
ABS_TOLERANCE = 1e-5
NREP = 100000

pytestmark = pytest.mark.skipif(STATA is None, reason="stata-mp not installed")


def stata_path(path: Path) -> str:
    return path.resolve().as_posix()


def execute_stata_command(script: str, root: Path) -> None:
    assert STATA is not None
    result = subprocess.run(
        [STATA, "-q"],
        cwd=root,
        input=script + "\nexit, clear\n",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def stata_packages_exist(plus: Path) -> bool:
    moremata_installed = (plus / "l" / "lmoremata.mlib").exists()
    spur_installed = all((plus / "s" / name).exists() for name in SPUR_FILES)
    return moremata_installed and spur_installed


@pytest.fixture(scope="module")
def stata_root() -> Path:
    root = STATA_CACHE_ROOT
    plus = root / "plus"
    personal = root / "personal"

    plus.mkdir(parents=True, exist_ok=True)
    personal.mkdir(parents=True, exist_ok=True)

    if not stata_packages_exist(plus):
        script = textwrap.dedent(
            f"""
            clear all
            set more off

            sysdir set PLUS "{stata_path(plus)}"
            sysdir set PERSONAL "{stata_path(personal)}"

            ssc install moremata, replace
            net install spur, replace from("{SPUR_STATA}")
            """
        )
        execute_stata_command(script, root)

    return root


@pytest.fixture(scope="module")
def chetty_df() -> pd.DataFrame:
    df = load_chetty_data()
    df = df[~df["state"].isin(["AK", "HI"])][["am", "fracblack", "lat", "lon", "state"]]
    df = df.dropna(subset=["am", "fracblack", "lat", "lon"]).reset_index(drop=True)
    df = standardize(df, ["am", "fracblack"])
    return df


@pytest.fixture(scope="module")
def input_csv(stata_root: Path, chetty_df: pd.DataFrame) -> Path:
    path = stata_root / "input.csv"
    chetty_df.to_csv(path, index=False)
    return path


def test_spurtransform_matches_stata(
    stata_root: Path,
    chetty_df: pd.DataFrame,
    input_csv: Path,
) -> None:
    plus = stata_root / "plus"
    personal = stata_root / "personal"
    output_csv = stata_root / "spurtransform.csv"

    py = spurtransform(
        chetty_df.copy(),
        ["am", "fracblack"],
        ["lat", "lon"],
        method="lbmgls",
        prefix="d_",
    )

    script = textwrap.dedent(
        f"""
        clear all
        set more off

        sysdir set PLUS "{stata_path(plus)}"
        sysdir set PERSONAL "{stata_path(personal)}"

        import delimited using "{stata_path(input_csv)}", clear asdouble
        rename lat s_1
        rename lon s_2

        spurtransform am fracblack, prefix(d_) transformation(lbmgls) latlong

        export delimited using "{stata_path(output_csv)}", replace
        """
    )
    execute_stata_command(script, stata_root)

    st = pd.read_csv(output_csv)

    assert np.allclose(py["d_am"], st["d_am"], atol=ABS_TOLERANCE, rtol=0)
    assert np.allclose(py["d_fracblack"], st["d_fracblack"], atol=ABS_TOLERANCE, rtol=0)


def test_spurtest_matches_stata(
    stata_root: Path,
    chetty_df: pd.DataFrame,
    input_csv: Path,
) -> None:
    plus = stata_root / "plus"
    personal = stata_root / "personal"
    output_csv = stata_root / "spurtest.csv"
    output_csv.unlink(missing_ok=True)

    py = spurtest(
        chetty_df,
        "i1",
        "am",
        ["lat", "lon"],
        q=10,
        nrep=NREP,
        seed=42,
    )

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
    execute_stata_command(script, stata_root)

    st = pd.read_csv(output_csv).iloc[0]

    assert np.isclose(py.LR, st["teststat"], atol=ABS_TOLERANCE, rtol=0)


def test_spurhalflife_matches_stata(
    stata_root: Path,
    chetty_df: pd.DataFrame,
    input_csv: Path,
) -> None:
    plus = stata_root / "plus"
    personal = stata_root / "personal"
    output_csv = stata_root / "spurhalflife.csv"
    output_csv.unlink(missing_ok=True)

    py = spurhalflife(
        chetty_df,
        "am",
        ["lat", "lon"],
        q=10,
        nrep=NREP,
        level=0.95,
        latlon=True,
        normdist=False,
        seed=42,
    )

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

        spurhalflife am, q(10) nrep({NREP}) level(95) latlong

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
    execute_stata_command(script, stata_root)

    st = pd.read_csv(output_csv).iloc[0]

    assert np.isclose(py.ci_lower, st["ci_lower"], atol=ABS_TOLERANCE, rtol=0)

    # stata returns missing for inf upper bound --> this
    # checks that if python returned inf for upper bound,
    # stata returned missing (NaN)
    if np.isinf(py.ci_upper):
        assert pd.isna(st["ci_upper"])
    else:
        assert np.isclose(py.ci_upper, st["ci_upper"], atol=ABS_TOLERANCE, rtol=0)

    assert np.isclose(py.max_dist, st["max_dist"], atol=ABS_TOLERANCE, rtol=0)
