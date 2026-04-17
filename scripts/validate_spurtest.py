"""
Validate Python spurtest against Stata spurtest.

Uses small N (30 obs) and small nrep (1000) for speed.

Notes:
- LR test statistic is DETERMINISTIC given data -> should match exactly
- P-values and CVs depend on Monte Carlo draws -> expect ~Monte Carlo error
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spur import spurtest

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPUR_CODE = Path("D:/UZHechist Dropbox/Joachim Voth/SPUR-Stata/SPUR_code")
TEST_DATA = PROJECT_DIR / "spurtest_data.csv"
STATA_LOG = PROJECT_DIR / "spurtest_stata.log"
DO_FILE = PROJECT_DIR / "spurtest_run.do"


def create_test_data(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """Create reproducible test data with mild spatial correlation."""
    np.random.seed(seed)
    lat = np.random.uniform(40, 50, n)
    lon = np.random.uniform(5, 15, n)

    # Variable with spatial trend (I(1)-ish)
    y_i1 = np.cumsum(np.random.randn(n)) + 0.5 * lat

    # Variable that is more stationary (I(0)-ish)
    y_i0 = np.random.randn(n)

    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "lat": lat,
            "lon": lon,
            "y_i1": y_i1,
            "y_i0": y_i0,
            "x": np.random.randn(n),
        }
    )

    return df


def generate_stata_do_file(q: int = 10, nrep: int = 1000):
    """Generate Stata do-file for validation."""
    do_content = f'''
* SPUR Test Validation

clear all
set more off
adopath + "{SPUR_CODE}"

* Load data
import delimited using "{TEST_DATA}", clear

* Rename coords
rename lat s_1
rename lon s_2

* Set seed for MC (not exactly matching Python, but stable across Stata runs)
set seed 42

* I(1) test on y_i1
spurtest i1 y_i1, q({q}) nrep({nrep}) latlong
scalar lfur_i1_yi1 = r(teststat)
scalar pvalue_i1_yi1 = r(p)
scalar haparm_i1_yi1 = r(ha_param)

* I(0) test on y_i1
spurtest i0 y_i1, q({q}) nrep({nrep}) latlong
scalar lfst_i0_yi1 = r(teststat)
scalar pvalue_i0_yi1 = r(p)

* I(1) test on y_i0
spurtest i1 y_i0, q({q}) nrep({nrep}) latlong
scalar lfur_i1_yi0 = r(teststat)
scalar pvalue_i1_yi0 = r(p)

* I(0) test on y_i0
spurtest i0 y_i0, q({q}) nrep({nrep}) latlong
scalar lfst_i0_yi0 = r(teststat)
scalar pvalue_i0_yi0 = r(p)

* Display all results
di ""
di "===== STATA RESULTS ====="
di "i1 test on y_i1: LR=" %9.4f lfur_i1_yi1 ", p=" %9.4f pvalue_i1_yi1 ", ha=" %9.4f haparm_i1_yi1
di "i0 test on y_i1: LR=" %9.4f lfst_i0_yi1 ", p=" %9.4f pvalue_i0_yi1
di "i1 test on y_i0: LR=" %9.4f lfur_i1_yi0 ", p=" %9.4f pvalue_i1_yi0
di "i0 test on y_i0: LR=" %9.4f lfst_i0_yi0 ", p=" %9.4f pvalue_i0_yi0
di "========================="
'''
    DO_FILE.write_text(do_content)
    print(f"Generated do-file: {DO_FILE}")


def run_python_tests(df: pd.DataFrame, q: int = 10, nrep: int = 1000):
    """Run Python tests."""
    results = {}

    print("Running Python tests...")

    # i1 on y_i1
    r = spurtest(df, "i1", "y_i1", ["lat", "lon"], q=q, nrep=nrep, seed=42)
    results["i1_yi1"] = {"LR": r.LR, "p": r.pvalue, "ha_param": r.ha_param}

    # i0 on y_i1
    r = spurtest(df, "i0", "y_i1", ["lat", "lon"], q=q, nrep=nrep, seed=42)
    results["i0_yi1"] = {"LR": r.LR, "p": r.pvalue, "ha_param": r.ha_param}

    # i1 on y_i0
    r = spurtest(df, "i1", "y_i0", ["lat", "lon"], q=q, nrep=nrep, seed=42)
    results["i1_yi0"] = {"LR": r.LR, "p": r.pvalue, "ha_param": r.ha_param}

    # i0 on y_i0
    r = spurtest(df, "i0", "y_i0", ["lat", "lon"], q=q, nrep=nrep, seed=42)
    results["i0_yi0"] = {"LR": r.LR, "p": r.pvalue, "ha_param": r.ha_param}

    return results


def main():
    print("=" * 60)
    print("SPUR Test Validation: Python vs Stata")
    print("=" * 60)

    # Create test data
    df = create_test_data(n=30, seed=42)
    df.to_csv(TEST_DATA, index=False)
    print(f"\nTest data: n={len(df)}, saved to {TEST_DATA}")

    # Generate do-file with larger nrep for MC convergence
    # i0 test has higher MC noise in ha_param - use 200k for good convergence
    q, nrep = 10, 200000
    generate_stata_do_file(q=q, nrep=nrep)

    # Run Python tests
    py_results = run_python_tests(df, q=q, nrep=nrep)

    print("\n" + "=" * 60)
    print("PYTHON RESULTS")
    print("=" * 60)
    for key, vals in py_results.items():
        print(
            f"{key}: LR={vals['LR']:.4f}, p={vals['p']:.4f}, ha={vals['ha_param']:.4f}"
        )

    print("\nNext: run do-file in Stata:")
    print(f'  do "{DO_FILE}"')


if __name__ == "__main__":
    main()
