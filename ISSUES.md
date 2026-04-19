# spur-python: Known Issues (Codex Adversarial Review, 2026-04-19)

Nine findings from a full adversarial review of `spur.py`, `spurtest.py`, and `spurhalflife.py`.
Status column: open / fixed.

---

## Batch 1 (issues 1–4)

### 1. `spurtransform(method='cluster')` requires coord columns it doesn't use
**File:** `spur.py:453-462`  **Severity:** high  **Status:** fixed

The function validates and extracts `coord_cols` unconditionally before branching on `method`.
Cluster mode never uses coordinates, so passing a DataFrame without coord columns (or with
non-numeric coord columns) raises an avoidable error.

**Fix:** branch on `method` before touching coordinates; for `method='cluster'`, skip coord
extraction and NaN checks entirely.

---

### 2. LBM-GLS normalization divides by zero on degenerate coordinates
**File:** `spur.py:269-277`  **Severity:** high  **Status:** fixed

`lbmgls_matrix` normalizes the distance matrix by `distmat.max()`. If all points are
identical (duplicate geocodes, collapsed dataset), `distmat.max() == 0`, producing NaN/inf
and crashing the eigendecomposition.

**Fix:** guard `distmat.max() > eps`; raise `ValueError` with a clear message before normalization.

---

### 3. `spurtest` allows `q` values that cause dimension-mismatch crashes
**File:** `spurtest.py:330-356`  **Severity:** high  **Status:** fixed

`get_R` silently truncates to available eigenvectors, but Monte Carlo draws continue to use
the user-supplied `q`. When `q > n-1` (or effective rank), downstream matrix multiplications
crash with an opaque shape error instead of a clear upfront rejection.

**Fix:** validate `q <= n-1` at entry to `spurtest`; raise `ValueError` with feasible range.

---

### 4. Residual tests crash on collinear regressors (unchecked `X'X` inversion)
**File:** `spurtest.py:551-610`  **Severity:** high  **Status:** fixed

Both residual test paths invert `X'X` directly via `np.linalg.inv`. Collinearity (dummy
traps, constant-like regressors, sparse categories) throws `LinAlgError` with no controlled
handling.

**Fix:** check matrix rank before inversion; raise a clear `ValueError` on rank deficiency,
or switch to `np.linalg.lstsq` / pseudo-inverse with explicit warning.

---

## Batch 2 (issues 5–9)

### 5. NaN/inf in `Y` yields silent invalid p-values
**File:** `spurtest.py:367-370`  **Severity:** high  **Status:** open

If `Y` contains NaN/inf, `LR` becomes NaN and `np.mean(lr_ho > LR)` returns all-False,
producing p-values of 0 with no warning.

**Fix:** validate finite `Y`, coords, and regressors at `spurtest` entry.

---

### 6. `spurhalflife` accepts invalid confidence levels
**File:** `spurhalflife.py:184-194`  **Severity:** high  **Status:** open

`level` is never validated. `level > 1` makes the inclusion threshold negative (CI always
full), `level <= 0` makes it > 1 (CI always empty). Both fail silently.

**Fix:** validate `0 < level < 1` (and `q >= 1`, `nrep >= 1`) at function entry.

---

### 7. Half-life normalization divides by zero for identical locations
**File:** `spurhalflife.py:241-243`  **Severity:** high  **Status:** open

`max_dist_norm` can be zero if all coordinates coincide, propagating NaN/inf throughout CI
computation.

**Fix:** guard `max_dist_norm > eps` before normalization; raise `ValueError`.

---

### 8. Distance matrix has quadratic memory blow-up for large N
**File:** `spur.py:65-76`  **Severity:** medium  **Status:** open

The haversine path allocates multiple full `n×n` arrays simultaneously. For large N this
becomes an OOM failure rather than just slow performance.

**Fix:** add an explicit size guard (raise informative error for N above a threshold);
longer-term: chunked/blockwise computation or sparse neighbour search.

---

### 9. Power-target bracketing loops can hang indefinitely
**File:** `spurtest.py:231-293`  **Severity:** medium  **Status:** open

`get_ha_parm_I1` / `get_ha_parm_I0` have no iteration cap. Non-monotone power behaviour or
numerical pathologies can cause them to loop forever.

**Fix:** add a `max_iter` cap (e.g. 200) with a clear convergence-failure error.
