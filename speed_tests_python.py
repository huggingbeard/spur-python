"""
Speed tests for SPUR Python across N and function.
Saves intermediate results after each run.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
from pathlib import Path

from spur import spurtransform
from spur import spurtest
from spur import spurhalflife

plt.rcParams.update({"font.family": "serif", "font.size": 10})
FIG_DIR = Path("D:/UZHechist Dropbox/Joachim Voth/claudecode/spur-python/figures")
FIG_DIR.mkdir(exist_ok=True)
CHECKPOINT = FIG_DIR / "speed_test_results.json"


def make_data(n, seed=42):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(30, 50, n)
    lon = rng.uniform(-100, -80, n)
    y = rng.standard_normal(n) + 0.5 * lat
    x = rng.standard_normal(n)
    return pd.DataFrame({"lat": lat, "lon": lon, "y": y, "x": x})


def time_call(func, n_repeats=3):
    """Median wall-clock time over n_repeats."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return np.median(times)


# Load existing checkpoint
if CHECKPOINT.exists():
    with open(CHECKPOINT) as f:
        results = json.load(f)
    print(f"Resuming with {len(results)} entries done")
else:
    results = []

done_keys = {(r["function"], r["n"], r.get("nrep", 0)) for r in results}


def run_and_save(function, n, nrep, func):
    key = (function, n, nrep)
    if key in done_keys:
        return
    print(f"  {function}, n={n}, nrep={nrep}...", flush=True)
    t = time_call(func)
    results.append({"function": function, "n": n, "nrep": nrep, "time_sec": t})
    done_keys.add(key)
    with open(CHECKPOINT, "w") as f:
        json.dump(results, f, indent=2)


# =============================
# Transformation speed tests
# =============================
print("=== TRANSFORMATION TIMING ===")
for n in [100, 300, 1000, 3000]:
    df = make_data(n)

    run_and_save(
        "transform_nn",
        n,
        0,
        lambda df=df: spurtransform(df, "y", ["lat", "lon"], method="nn"),
    )
    run_and_save(
        "transform_iso",
        n,
        0,
        lambda df=df: spurtransform(
            df, "y", ["lat", "lon"], method="iso", radius=500000
        ),
    )
    run_and_save(
        "transform_lbmgls",
        n,
        0,
        lambda df=df: spurtransform(df, "y", ["lat", "lon"], method="lbmgls"),
    )

# =============================
# Test speed tests
# =============================
print("\n=== TEST TIMING ===")
for n in [30, 100, 300]:
    df = make_data(n)
    for nrep in [10000, 50000]:
        run_and_save(
            "spurtest_i1",
            n,
            nrep,
            lambda df=df, nr=nrep: spurtest(
                df, "i1", "y", ["lat", "lon"], q=15, nrep=nr, seed=42
            ),
        )
        run_and_save(
            "spurtest_i0",
            n,
            nrep,
            lambda df=df, nr=nrep: spurtest(
                df, "i0", "y", ["lat", "lon"], q=15, nrep=nr, seed=42
            ),
        )

# =============================
# Halflife
# =============================
print("\n=== HALFLIFE TIMING ===")
for n in [30, 100, 300]:
    df = make_data(n)
    for nrep in [10000, 50000]:
        run_and_save(
            "spurhalflife",
            n,
            nrep,
            lambda df=df, nr=nrep: spurhalflife(
                df, "y", ["lat", "lon"], q=15, nrep=nr, seed=42
            ),
        )

print(f"\nAll done. {len(results)} measurements.")
print("Results saved to:", CHECKPOINT)
