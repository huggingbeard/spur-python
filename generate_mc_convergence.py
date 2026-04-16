"""
Generate mc_convergence figure - with safeguards and checkpointing.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

from spur import spurtest

plt.rcParams.update({"font.family": "serif", "font.size": 10})
FIG_DIR = Path("D:/UZHechist Dropbox/Joachim Voth/claudecode/spur-python/figures")
FIG_DIR.mkdir(exist_ok=True)
CHECKPOINT = FIG_DIR / "mc_convergence_data.json"

df = pd.read_csv(
    "D:/UZHechist Dropbox/Joachim Voth/claudecode/spur-python/spurtest_data.csv"
)

# Skip nrep=100 - too noisy and triggers overflow
nreps = [1000, 5000, 10000, 50000]
seeds = list(range(10))

# Load checkpoint if exists
if CHECKPOINT.exists():
    with open(CHECKPOINT) as f:
        results = json.load(f)
    print(f"Resuming from checkpoint: {len(results)} entries done")
else:
    results = []

done_keys = {(r["seed"], r["nrep"]) for r in results}

for seed in seeds:
    for n in nreps:
        key = (seed, n)
        if key in done_keys:
            continue
        print(f"  seed={seed}, nrep={n}...", flush=True)
        r = spurtest(df, "i1", "y_i1", ["lat", "lon"], q=10, nrep=n, seed=seed)
        results.append(
            {
                "seed": seed,
                "nrep": n,
                "LR": r.LR,
                "ha_param": r.ha_param,
                "pvalue": r.pvalue,
            }
        )
        # Checkpoint after each run
        with open(CHECKPOINT, "w") as f:
            json.dump(results, f, indent=2)

print(f"\nAll {len(results)} runs done, generating plot...")

# Build plot
df_r = pd.DataFrame(results)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
means = df_r.groupby("nrep")["LR"].mean()
stds = df_r.groupby("nrep")["LR"].std()
ax.errorbar(
    means.index,
    means.values,
    yerr=stds.values,
    marker="o",
    capsize=5,
    lw=2,
    color="navy",
)
ax.set_xscale("log")
ax.set_xlabel("Monte Carlo draws (nrep)")
ax.set_ylabel("LR test statistic")
ax.set_title("LR convergence (10 seeds, y_i1 i1 test)")
ax.grid(alpha=0.3)

ax = axes[1]
means = df_r.groupby("nrep")["ha_param"].mean()
stds = df_r.groupby("nrep")["ha_param"].std()
ax.errorbar(
    means.index,
    means.values,
    yerr=stds.values,
    marker="o",
    capsize=5,
    lw=2,
    color="darkred",
)
ax.set_xscale("log")
ax.set_xlabel("Monte Carlo draws (nrep)")
ax.set_ylabel(r"$c_a$ (alternative parameter)")
ax.set_title(r"$c_a$ convergence (10 seeds, y_i1 i1 test)")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "mc_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR / 'mc_convergence.png'}")
