"""
Cross-validate Python SPUR against Stata SPUR package.

This script:
1. Creates test data
2. Saves to CSV
3. Runs Stata spurtransform
4. Compares Python vs Stata output
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spur import spurtransform, get_transformation_stats

# Paths
PROJECT_DIR = Path("D:/UZHechist Dropbox/Joachim Voth/claudecode/spur-python")
SPUR_CODE = Path("D:/UZHechist Dropbox/Joachim Voth/SPUR-Stata/SPUR_code")
TEST_DATA = PROJECT_DIR / "validation_data.csv"
STATA_OUTPUT = PROJECT_DIR / "validation_stata_output.csv"
DO_FILE = PROJECT_DIR / "validation_run.do"


def create_test_data(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Create reproducible test data."""
    np.random.seed(seed)

    # Coordinates in contiguous US region
    lat = np.random.uniform(30, 45, n)
    lon = np.random.uniform(-100, -80, n)

    # Variables with spatial structure
    y = lat * 0.1 + np.random.randn(n) * 0.5
    x = lon * 0.05 + np.random.randn(n) * 0.3

    df = pd.DataFrame(
        {"id": np.arange(1, n + 1), "lat": lat, "lon": lon, "y": y, "x": x}
    )

    return df


def generate_stata_do_file(radius: float = 200000):
    """Generate Stata do-file for validation."""

    do_content = f'''
* SPUR Validation - Run from Stata
* This file runs spurtransform on test data

clear all
set more off

* Add SPUR to adopath
adopath + "{SPUR_CODE}"

* Load test data
import delimited using "{TEST_DATA}", clear

* NN transform
spurtransform y x, prefix(nn_) transformation(nn) latlong

* ISO transform (200km radius)
spurtransform y x, prefix(iso_) transformation(iso) radius({radius}) latlong

* Export results
export delimited using "{STATA_OUTPUT}", replace

di "Stata validation complete!"
'''

    DO_FILE.write_text(do_content)
    print(f"Generated do-file: {DO_FILE}")


def run_python_transforms(df: pd.DataFrame, radius: float = 200000) -> pd.DataFrame:
    """Run Python transforms."""
    # NN
    df = spurtransform(
        df, ["y", "x"], ["lat", "lon"], method="nn", latlon=True, prefix="py_nn_"
    )

    # ISO
    df = spurtransform(
        df,
        ["y", "x"],
        ["lat", "lon"],
        method="iso",
        radius=radius,
        latlon=True,
        prefix="py_iso_",
    )

    # LBM-GLS
    df = spurtransform(
        df, ["y", "x"], ["lat", "lon"], method="lbmgls", latlon=True, prefix="py_lbm_"
    )

    # Cluster (create same cluster variable as Stata: latitude tercile)
    df["cluster"] = pd.qcut(df["lat"], q=3, labels=[1, 2, 3]).astype(int)
    df = spurtransform(
        df,
        ["y", "x"],
        ["lat", "lon"],
        method="cluster",
        cluster_col="cluster",
        prefix="py_cl_",
    )

    return df


def compare_results():
    """Compare Stata vs Python outputs."""
    if not STATA_OUTPUT.exists():
        print(f"ERROR: Stata output not found at {STATA_OUTPUT}")
        print("Please run the do-file in Stata first:")
        print(f'  do "{DO_FILE}"')
        return

    # Load Stata output
    stata_df = pd.read_csv(STATA_OUTPUT)

    # Load test data and run Python
    test_df = pd.read_csv(TEST_DATA)
    python_df = run_python_transforms(test_df)

    print("\n" + "=" * 60)
    print("SPUR Validation: Python vs Stata")
    print("=" * 60)

    # Compare NN transforms
    print("\nNearest-Neighbor (NN) Transform:")
    for var in ["y", "x"]:
        py_col = f"py_nn_{var}"
        stata_col = f"nn_{var}"

        if stata_col not in stata_df.columns:
            print(f"  WARNING: {stata_col} not in Stata output")
            continue

        py_vals = python_df[py_col].values
        stata_vals = stata_df[stata_col].values

        diff = np.abs(py_vals - stata_vals)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        corr = np.corrcoef(py_vals, stata_vals)[0, 1]

        print(f"  {var}:")
        print(f"    Max absolute diff: {max_diff:.2e}")
        print(f"    Mean absolute diff: {mean_diff:.2e}")
        print(f"    Correlation: {corr:.10f}")

        if max_diff < 1e-6:
            print("    Status: MATCH (< 1e-6)")
        elif max_diff < 1e-3:
            print("    Status: CLOSE (< 1e-3)")
        else:
            print("    Status: DIFFERS")

    # Compare ISO transforms
    print("\nIsotropic (ISO) Transform:")
    for var in ["y", "x"]:
        py_col = f"py_iso_{var}"
        stata_col = f"iso_{var}"

        if stata_col not in stata_df.columns:
            print(f"  WARNING: {stata_col} not in Stata output")
            continue

        py_vals = python_df[py_col].values
        stata_vals = stata_df[stata_col].values

        diff = np.abs(py_vals - stata_vals)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        corr = np.corrcoef(py_vals, stata_vals)[0, 1]

        print(f"  {var}:")
        print(f"    Max absolute diff: {max_diff:.2e}")
        print(f"    Mean absolute diff: {mean_diff:.2e}")
        print(f"    Correlation: {corr:.10f}")

        if max_diff < 1e-6:
            print("    Status: MATCH (< 1e-6)")
        elif max_diff < 1e-3:
            print("    Status: CLOSE (< 1e-3)")
        else:
            print("    Status: DIFFERS")

    # Compare LBM-GLS transforms
    print("\nLBM-GLS Transform:")
    for var in ["y", "x"]:
        py_col = f"py_lbm_{var}"
        stata_col = f"lbm_{var}"

        if stata_col not in stata_df.columns:
            print(f"  WARNING: {stata_col} not in Stata output")
            continue

        py_vals = python_df[py_col].values
        stata_vals = stata_df[stata_col].values

        diff = np.abs(py_vals - stata_vals)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        corr = np.corrcoef(py_vals, stata_vals)[0, 1]

        print(f"  {var}:")
        print(f"    Max absolute diff: {max_diff:.2e}")
        print(f"    Mean absolute diff: {mean_diff:.2e}")
        print(f"    Correlation: {corr:.10f}")

        if max_diff < 1e-6:
            print("    Status: MATCH (< 1e-6)")
        elif max_diff < 1e-3:
            print("    Status: CLOSE (< 1e-3)")
        else:
            print("    Status: DIFFERS")

    # Compare cluster transforms
    print("\nCluster Transform:")
    for var in ["y", "x"]:
        py_col = f"py_cl_{var}"
        stata_col = f"cl_{var}"

        if stata_col not in stata_df.columns:
            print(f"  WARNING: {stata_col} not in Stata output")
            continue

        py_vals = python_df[py_col].values
        stata_vals = stata_df[stata_col].values

        diff = np.abs(py_vals - stata_vals)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        corr = np.corrcoef(py_vals, stata_vals)[0, 1]

        print(f"  {var}:")
        print(f"    Max absolute diff: {max_diff:.2e}")
        print(f"    Mean absolute diff: {mean_diff:.2e}")
        print(f"    Correlation: {corr:.10f}")

        if max_diff < 1e-6:
            print("    Status: MATCH (< 1e-6)")
        elif max_diff < 1e-3:
            print("    Status: CLOSE (< 1e-3)")
        else:
            print("    Status: DIFFERS")

    print("\n" + "=" * 60)


def main():
    print("SPUR Python-Stata Validation")
    print("=" * 60)

    # Step 1: Create test data
    print("\n1. Creating test data...")
    df = create_test_data(n=50, seed=42)
    df.to_csv(TEST_DATA, index=False)
    print(f"   Saved: {TEST_DATA}")

    # Step 2: Generate do-file
    print("\n2. Generating Stata do-file...")
    generate_stata_do_file(radius=200000)

    # Step 3: Run Python transforms
    print("\n3. Running Python transforms...")
    python_df = run_python_transforms(df)

    # Show Python stats
    coords = df[["lat", "lon"]].values
    stats = get_transformation_stats(coords, method="nn", latlon=True)
    print(f"   Mean NN distance: {stats['nn_dist_mean'] / 1000:.1f} km")

    stats_iso = get_transformation_stats(
        coords, method="iso", radius=200000, latlon=True
    )
    print(f"   ISO neighbors (mean): {stats_iso['neighbors_mean']:.1f}")

    # Save Python output for manual comparison
    python_df.to_csv(PROJECT_DIR / "validation_python_output.csv", index=False)
    print("   Python output saved")

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("\n1. Run in Stata:")
    print(f'   do "{DO_FILE}"')
    print("\n2. Then run this script again to compare:")
    print("   python validate_against_stata.py --compare")

    # Check if Stata output exists
    if STATA_OUTPUT.exists():
        print("\nStata output found - comparing now...")
        compare_results()


if __name__ == "__main__":
    import sys

    if "--compare" in sys.argv:
        compare_results()
    else:
        main()
