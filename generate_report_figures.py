"""
Generate all figures for the SPUR Python report.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm

from spur import lbmgls_matrix

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "figure.dpi": 120,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

FIG_DIR = Path("D:/UZHechist Dropbox/Joachim Voth/claudecode/spur-python/figures")
FIG_DIR.mkdir(exist_ok=True)

# ============================================================
# Figure 1: Chetty data maps (AM, TLFPR, AM_d, TLFPR_d)
# ============================================================


def fig_chetty_maps():
    """2x2 map of original and transformed variables."""
    chetty = pd.read_excel(
        "D:/UZHechist Dropbox/Joachim Voth/claudecode/spur-python/test_data/"
        "ReplicationPackage_MS21654_MuellerWatson_r2/Data/Chetty_Data_1.xlsx"
    )
    chetty = chetty[~chetty["State"].isin(["AK", "HI"])]
    chetty = chetty.dropna(subset=["AM", "TLFPR"]).reset_index(drop=True)

    # Compute LBM-GLS transformation
    coords = chetty[["Lat", "Lon"]].values
    H = lbmgls_matrix(coords, latlon=True)

    AM_std = (chetty["AM"] - chetty["AM"].mean()) / chetty["AM"].std()
    TLFPR_std = (chetty["TLFPR"] - chetty["TLFPR"].mean()) / chetty["TLFPR"].std()
    AM_d = H @ AM_std.values
    TLFPR_d = H @ TLFPR_std.values

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    lat = chetty["Lat"].values
    lon = chetty["Lon"].values

    for ax, data, title in zip(
        axes.flat,
        [AM_std.values, TLFPR_std.values, AM_d, TLFPR_d],
        [
            "AM (standardized)",
            "TLFPR (standardized)",
            "AM after LBM-GLS",
            "TLFPR after LBM-GLS",
        ],
    ):
        # Color by decile
        deciles = pd.qcut(data, 10, labels=False, duplicates="drop")
        sc = ax.scatter(lon, lat, c=deciles, cmap="RdBu_r", s=4, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, label="Decile", fraction=0.03)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "chetty_maps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'chetty_maps.png'}")


# ============================================================
# Figure 2: Chetty regression - before and after transformation
# ============================================================


def fig_chetty_regressions():
    """Scatter plots showing the regression before and after LBM-GLS."""
    chetty = pd.read_excel(
        "D:/UZHechist Dropbox/Joachim Voth/claudecode/spur-python/test_data/"
        "ReplicationPackage_MS21654_MuellerWatson_r2/Data/Chetty_Data_1.xlsx"
    )
    chetty = chetty[~chetty["State"].isin(["AK", "HI"])]
    chetty = chetty.dropna(subset=["AM", "TLFPR"]).reset_index(drop=True)

    coords = chetty[["Lat", "Lon"]].values
    H = lbmgls_matrix(coords, latlon=True)

    AM_std = (chetty["AM"] - chetty["AM"].mean()) / chetty["AM"].std()
    TLFPR_std = (chetty["TLFPR"] - chetty["TLFPR"].mean()) / chetty["TLFPR"].std()
    AM_d = H @ AM_std.values
    TLFPR_d = H @ TLFPR_std.values

    # Regressions
    X_level = sm.add_constant(TLFPR_std.values)
    reg_level = sm.OLS(AM_std.values, X_level).fit()
    X_d = sm.add_constant(TLFPR_d)
    reg_d = sm.OLS(AM_d, X_d).fit()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Level
    ax = axes[0]
    ax.scatter(TLFPR_std, AM_std, s=8, alpha=0.4, color="navy")
    xx = np.linspace(TLFPR_std.min(), TLFPR_std.max(), 100)
    yy = reg_level.params[0] + reg_level.params[1] * xx
    ax.plot(xx, yy, "r-", lw=2)
    ax.set_xlabel("TLFPR (standardized)")
    ax.set_ylabel("AM (standardized)")
    ax.set_title(
        f"Before LBM-GLS\n"
        rf"$\beta$={reg_level.params[1]:.3f}, "
        rf"$R^2$={reg_level.rsquared:.3f}, "
        f"t={reg_level.tvalues[1]:.1f}"
    )

    # Differenced
    ax = axes[1]
    ax.scatter(TLFPR_d, AM_d, s=8, alpha=0.4, color="darkgreen")
    xx = np.linspace(TLFPR_d.min(), TLFPR_d.max(), 100)
    yy = reg_d.params[0] + reg_d.params[1] * xx
    ax.plot(xx, yy, "r-", lw=2)
    ax.set_xlabel(r"TLFPR$_d$ (LBM-GLS differenced)")
    ax.set_ylabel(r"AM$_d$ (LBM-GLS differenced)")
    ax.set_title(
        f"After LBM-GLS\n"
        rf"$\beta$={reg_d.params[1]:.3f}, "
        rf"$R^2$={reg_d.rsquared:.3f}, "
        f"t={reg_d.tvalues[1]:.1f}"
    )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "chetty_regressions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'chetty_regressions.png'}")


# ============================================================
# Figure 3: MC variability comparison (already generated)
# ============================================================


def fig_mc_variability():
    """Copy/regenerate the MC variability plot."""
    import shutil

    src = Path(
        "D:/UZHechist Dropbox/Joachim Voth/claudecode/spur-python/mc_variability_comparison.png"
    )
    if src.exists():
        shutil.copy(src, FIG_DIR / "mc_variability.png")
        print(f"Copied: {FIG_DIR / 'mc_variability.png'}")


# ============================================================
# Figure 4: Validation summary plot
# ============================================================


def fig_validation_summary():
    """Bar chart of max differences for each validated function."""
    data = {
        "spurtransform nn": 4.3e-7,
        "spurtransform iso": 3.5e-7,
        "spurtransform lbmgls": 1.4e-5,
        "spurtransform cluster": 1e-10,
        "spurtest i1 LR": 1e-10,
        "spurtest i0 LR": 1e-10,
        "spurtest i1resid LR": 1e-5,
        "spurtest i0resid LR": 1e-10,
        "spurhalflife ci_l": 2e-4,
        "Chetty LBM-GLS": 6.1e-9,
    }

    _, ax = plt.subplots(figsize=(10, 6))
    names = list(data.keys())
    diffs = list(data.values())
    colors = ["green" if d < 1e-4 else "orange" for d in diffs]
    ax.barh(names, diffs, color=colors, alpha=0.7, edgecolor="black")
    ax.set_xscale("log")
    ax.set_xlabel("Max absolute difference (Python vs Stata)")
    ax.set_title("SPUR Python validation: max differences vs Stata reference")
    ax.axvline(
        1e-6, color="red", linestyle="--", alpha=0.5, label="1e-6 (floating point)"
    )
    ax.axvline(
        1e-3, color="orange", linestyle="--", alpha=0.5, label="1e-3 (MC noise OK)"
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "validation_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'validation_summary.png'}")


# ============================================================
# Figure 5: MC convergence as nrep grows
# ============================================================


def fig_mc_convergence():
    """Show how LR and ha_param converge as nrep grows."""
    from spur import spurtest

    df = pd.read_csv(
        "D:/UZHechist Dropbox/Joachim Voth/claudecode/spur-python/spurtest_data.csv"
    )

    nreps = [100, 500, 1000, 5000, 10000, 50000]
    LR_runs = {n: [] for n in nreps}
    ha_runs = {n: [] for n in nreps}

    for seed in range(10):
        for n in nreps:
            r = spurtest(df, "i1", "y_i1", ["lat", "lon"], q=10, nrep=n, seed=seed)
            LR_runs[n].append(r.LR)
            ha_runs[n].append(r.ha_param)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    means = [np.mean(LR_runs[n]) for n in nreps]
    stds = [np.std(LR_runs[n]) for n in nreps]
    ax.errorbar(nreps, means, yerr=stds, marker="o", capsize=5, lw=2, color="navy")
    ax.set_xscale("log")
    ax.set_xlabel("Monte Carlo draws (nrep)")
    ax.set_ylabel("LR test statistic")
    ax.set_title("LR convergence (10 seeds, y_i1 i1 test)")

    ax = axes[1]
    means = [np.mean(ha_runs[n]) for n in nreps]
    stds = [np.std(ha_runs[n]) for n in nreps]
    ax.errorbar(nreps, means, yerr=stds, marker="o", capsize=5, lw=2, color="darkred")
    ax.set_xscale("log")
    ax.set_xlabel("Monte Carlo draws (nrep)")
    ax.set_ylabel(r"$c_a$ (alternative parameter)")
    ax.set_title(r"$c_a$ convergence (10 seeds, y_i1 i1 test)")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "mc_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'mc_convergence.png'}")


if __name__ == "__main__":
    print("Generating figures for SPUR Python report...")
    fig_chetty_maps()
    fig_chetty_regressions()
    fig_mc_variability()
    fig_validation_summary()
    fig_mc_convergence()
    print("\nAll figures saved to:", FIG_DIR)
