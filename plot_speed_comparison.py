"""
Plot Python vs Stata speed comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "figure.dpi": 120,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

ROOT = Path(__file__).resolve().parent
FIG_DIR = Path(__file__).resolve().parent / "figures"

# Load results
with open(FIG_DIR / "speed_test_results.json") as f:
    py = pd.DataFrame(json.load(f))

st = pd.read_csv(ROOT / "stata_speed_results.csv")
st = st.rename(columns={"func": "function"})

# Merge
key = ["function", "n", "nrep"]
cmp = py.merge(st, on=key, suffixes=("_py", "_st"))
cmp["speedup"] = cmp["time_sec_st"] / cmp["time_sec_py"]
cmp.to_csv(FIG_DIR / "speed_comparison.csv", index=False)
print("Speed comparison saved")
print(
    cmp[["function", "n", "nrep", "time_sec_py", "time_sec_st", "speedup"]].to_string(
        index=False
    )
)

# =============================
# Figure 1: Transformation speeds
# =============================
tf = cmp[cmp["function"].str.startswith("transform")].copy()
tf["method"] = tf["function"].str.replace("transform_", "")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
width = 0.35
methods = ["nn", "iso", "lbmgls"]
ns = sorted(tf["n"].unique())
x = np.arange(len(ns))

for i, method in enumerate(methods):
    sub = tf[tf["method"] == method].sort_values("n")
    offset = (i - 1) * width / 1.5
    # Plot Python and Stata side by side
    ax.bar(
        x + offset - 0.12,
        sub["time_sec_py"].values,
        width=0.1,
        label=f"Py {method}",
        alpha=0.85,
    )
    ax.bar(
        x + offset - 0.02,
        sub["time_sec_st"].values,
        width=0.1,
        label=f"St {method}",
        alpha=0.6,
        hatch="//",
    )

ax.set_xticks(x)
ax.set_xticklabels([f"N={n}" for n in ns])
ax.set_ylabel("Time (s)")
ax.set_yscale("log")
ax.set_title("Transformation speed: Python vs Stata (log scale)")
ax.legend(ncol=3, fontsize=8, loc="upper left")

ax = axes[1]
# Speedup bar chart
for i, method in enumerate(methods):
    sub = tf[tf["method"] == method].sort_values("n")
    offset = (i - 1) * 0.25
    bars = ax.bar(
        x + offset, sub["speedup"].values, width=0.2, label=f"{method}", alpha=0.85
    )

ax.axhline(1, color="red", linestyle="--", alpha=0.5, label="equal speed")
ax.set_xticks(x)
ax.set_xticklabels([f"N={n}" for n in ns])
ax.set_ylabel("Speedup (Stata time / Python time)")
ax.set_title("Python speedup relative to Stata")
ax.legend()
ax.set_yscale("log")

plt.tight_layout()
plt.savefig(FIG_DIR / "speed_transform.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR / 'speed_transform.png'}")

# =============================
# Figure 2: Test speeds
# =============================
tests = cmp[cmp["function"].isin(["spurtest_i1", "spurtest_i0"])].copy()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
# Group by test type, N, and nrep
test_types = ["spurtest_i1", "spurtest_i0"]
ns = sorted(tests["n"].unique())
nreps = sorted(tests["nrep"].unique())

# Just show speedup for simplicity
for i, tt in enumerate(test_types):
    for j, nrep in enumerate(nreps):
        sub = tests[(tests["function"] == tt) & (tests["nrep"] == nrep)].sort_values(
            "n"
        )
        label = f"{tt.replace('spurtest_', '')}, nrep={nrep}"
        ax.plot(sub["n"], sub["speedup"], marker="o", lw=2, label=label)

ax.axhline(1, color="red", linestyle="--", alpha=0.5)
ax.set_xlabel("N observations")
ax.set_ylabel("Python speedup (Stata / Python)")
ax.set_title("Test speedup by N")
ax.legend()
ax.set_xscale("log")
ax.set_xticks(ns)
ax.set_xticklabels([str(n) for n in ns])
ax.minorticks_off()

ax = axes[1]
# Direct timing comparison
for i, tt in enumerate(test_types):
    sub = tests[(tests["function"] == tt) & (tests["nrep"] == 50000)].sort_values("n")
    tt_short = tt.replace("spurtest_", "")
    ax.plot(
        sub["n"],
        sub["time_sec_py"],
        marker="o",
        lw=2,
        label=f"Python {tt_short}",
        linestyle="-",
    )
    ax.plot(
        sub["n"],
        sub["time_sec_st"],
        marker="s",
        lw=2,
        label=f"Stata {tt_short}",
        linestyle="--",
    )

ax.set_xlabel("N observations")
ax.set_ylabel("Time (s)")
ax.set_title("spurtest time at nrep=50000")
ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(ns)
ax.set_xticklabels([str(n) for n in ns])
ax.minorticks_off()

plt.tight_layout()
plt.savefig(FIG_DIR / "speed_tests.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR / 'speed_tests.png'}")

# =============================
# Figure 3: Halflife speed
# =============================
hl = cmp[cmp["function"] == "spurhalflife"].copy()

fig, ax = plt.subplots(figsize=(8, 5))
ns = sorted(hl["n"].unique())
x = np.arange(len(ns))
width = 0.35

for i, nrep in enumerate([10000, 50000]):
    sub = hl[hl["nrep"] == nrep].sort_values("n")
    ax.bar(
        x + (i - 0.5) * width - 0.1,
        sub["time_sec_py"].values,
        width=0.18,
        label=f"Python nrep={nrep}",
        alpha=0.85,
    )
    ax.bar(
        x + (i - 0.5) * width + 0.1,
        sub["time_sec_st"].values,
        width=0.18,
        label=f"Stata nrep={nrep}",
        alpha=0.85,
        hatch="//",
    )

ax.set_xticks(x)
ax.set_xticklabels([f"N={n}" for n in ns])
ax.set_ylabel("Time (s)")
ax.set_title("spurhalflife: Python vs Stata")
ax.legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "speed_halflife.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR / 'speed_halflife.png'}")

# =============================
# Summary table
# =============================
summary = (
    cmp.groupby("function")
    .apply(
        lambda g: pd.Series(
            {
                "median_speedup": g["speedup"].median(),
                "min_speedup": g["speedup"].min(),
                "max_speedup": g["speedup"].max(),
            }
        ),
        include_groups=False,
    )
    .round(2)
)
print("\nSummary:")
print(summary)
summary.to_csv(FIG_DIR / "speed_summary.csv")
