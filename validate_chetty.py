"""
Validate SPUR Python against Chetty data in Muller-Watson (2024).

The MW replication package includes Chetty Commuting Zone data (n=693 after
dropping AK/HI and missing values) with reference LBM-GLS transformed values.

This script:
1. Loads original Chetty data
2. Applies Python's LBM-GLS transformation
3. Compares against MW's reference AM_d, TLFPR_d
4. Runs the key regression (AM_d on TLFPR_d) and compares to MW's results
"""

import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from spur import lbmgls_matrix

DATA_DIR = Path(
    "D:/UZHechist Dropbox/Joachim Voth/claudecode/spur-python/test_data/ReplicationPackage_MS21654_MuellerWatson_r2"
)


def load_chetty():
    """Load Chetty Commuting Zone data (contiguous US).

    Note: CSV has precision loss. Use XLSX for exact match with MW.
    """
    df = pd.read_excel(DATA_DIR / "Data" / "Chetty_Data_1.xlsx")

    # Drop Alaska and Hawaii
    df = df[~df["State"].isin(["AK", "HI"])].reset_index(drop=True)

    return df


def load_mw_reference():
    """Load MW's reference LBM-GLS transformed values."""
    return pd.read_excel(DATA_DIR / "STATA_Matlab_Example" / "Chetty_Example.xlsx")


def run_validation():
    print("=" * 70)
    print("Chetty Data Validation: Python SPUR vs Muller-Watson (2024)")
    print("=" * 70)

    # Load data
    chetty = load_chetty()
    mw = load_mw_reference()

    print(f"\nLoaded Chetty data: {len(chetty)} observations")
    print(f"Loaded MW reference: {len(mw)} observations")

    # Merge on CZ to ensure alignment
    df = chetty[["CZ", "State", "Lat", "Lon", "AM", "TLFPR"]].merge(
        mw[["CZ", "AM_d", "TLFPR_d"]], on="CZ", how="inner"
    )
    print(f"Merged: {len(df)} observations")

    # Keep only rows with all data present (MW's final sample)
    df_complete = df.dropna(subset=["AM", "TLFPR"]).reset_index(drop=True)
    print(f"Complete cases (AM & TLFPR): {len(df_complete)}")

    # MW's procedure:
    # 1. Standardize: y -> (y - mean) / std
    # 2. Apply H = lbmgls_matrix(coords)
    # 3. Result: y_d = H @ ys

    # Build H matrix
    coords = df_complete[["Lat", "Lon"]].values
    H = lbmgls_matrix(coords, latlon=True)

    # Standardize and transform
    AM_std = (df_complete["AM"] - df_complete["AM"].mean()) / df_complete["AM"].std()
    TLFPR_std = (df_complete["TLFPR"] - df_complete["TLFPR"].mean()) / df_complete[
        "TLFPR"
    ].std()

    AM_d_py = H @ AM_std.values
    TLFPR_d_py = H @ TLFPR_std.values

    # Compare against MW
    # mw_complete = df_complete[["AM_d", "TLFPR_d"]].dropna() # DG: was unused

    print("\n" + "=" * 70)
    print("LBM-GLS TRANSFORMATION: Python vs MW Reference")
    print("=" * 70)

    # AM_d comparison
    am_diff = np.abs(AM_d_py - df_complete["AM_d"].values)
    am_corr = np.corrcoef(AM_d_py, df_complete["AM_d"].values)[0, 1]
    print("\nAM_d:")
    print(f"  Max abs diff:  {am_diff.max():.4e}")
    print(f"  Mean abs diff: {am_diff.mean():.4e}")
    print(f"  Correlation:   {am_corr:.10f}")
    print(f"  Status: {'MATCH' if am_diff.max() < 1e-3 else 'DIFFERS'}")

    # TLFPR_d comparison
    tlfpr_diff = np.abs(TLFPR_d_py - df_complete["TLFPR_d"].values)
    tlfpr_corr = np.corrcoef(TLFPR_d_py, df_complete["TLFPR_d"].values)[0, 1]
    print("\nTLFPR_d:")
    print(f"  Max abs diff:  {tlfpr_diff.max():.4e}")
    print(f"  Mean abs diff: {tlfpr_diff.mean():.4e}")
    print(f"  Correlation:   {tlfpr_corr:.10f}")
    print(f"  Status: {'MATCH' if tlfpr_diff.max() < 1e-3 else 'DIFFERS'}")

    # Run key regressions
    print("\n" + "=" * 70)
    print("REGRESSION RESULTS (MW Table / Figure 4)")
    print("=" * 70)

    # Original level regression (no differencing)
    # MW log: regress am_level tlfpr_level (standardized)
    X = sm.add_constant(TLFPR_std.values)
    reg_level = sm.OLS(AM_std.values, X).fit()
    print("\nLevel regression: std(AM) ~ std(TLFPR)")
    print("  MW:     beta=0.6601, t=23.10, R²=0.4358, N=693")
    print(
        f"  Python: beta={reg_level.params[1]:.4f}, t={reg_level.tvalues[1]:.2f}, "
        f"R²={reg_level.rsquared:.4f}, N={reg_level.nobs:.0f}"
    )

    # After LBM-GLS differencing (Python's values)
    X_d = sm.add_constant(TLFPR_d_py)
    reg_d = sm.OLS(AM_d_py, X_d).fit()

    # Cluster-robust by state
    state_ids = pd.factorize(df_complete["State"])[0]
    reg_d_cluster = sm.OLS(AM_d_py, X_d).fit(
        cov_type="cluster", cov_kwds={"groups": state_ids}
    )
    print("\nAfter LBM-GLS (Python): AM_d ~ TLFPR_d, cluster(state)")
    print("  MW:     beta=0.2599, SE_cluster=0.0971, t=2.68, R²=0.0447")
    print(
        f"  Python: beta={reg_d_cluster.params[1]:.4f}, "
        f"SE={reg_d_cluster.bse[1]:.4f}, t={reg_d_cluster.tvalues[1]:.2f}, "
        f"R²={reg_d.rsquared:.4f}"
    )

    # Save Python transformed values for the report
    df_out = df_complete.copy()
    df_out["AM_d_py"] = AM_d_py
    df_out["TLFPR_d_py"] = TLFPR_d_py
    df_out.to_csv("chetty_python_validation.csv", index=False)

    return {
        "n_obs": len(df_complete),
        "am_d_max_diff": am_diff.max(),
        "am_d_corr": am_corr,
        "tlfpr_d_max_diff": tlfpr_diff.max(),
        "tlfpr_d_corr": tlfpr_corr,
        "reg_level_beta": reg_level.params[1],
        "reg_level_r2": reg_level.rsquared,
        "reg_d_beta": reg_d_cluster.params[1],
        "reg_d_se": reg_d_cluster.bse[1],
        "reg_d_r2": reg_d.rsquared,
    }


if __name__ == "__main__":
    results = run_validation()
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k}: {v}")
