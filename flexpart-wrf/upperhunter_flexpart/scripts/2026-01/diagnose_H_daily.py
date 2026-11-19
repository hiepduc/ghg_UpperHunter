#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path

# ================================================================
# Configuration
# ================================================================

BASE = Path.home() / "flexpart-wrf/upperhunter_flexpart"

HFILE = (
    BASE
    / "hmatrix/daily_20260105_20260131"
    / "H_daily_13mine_1124x351_ppm.npy"
)

OBSFILE = (
    BASE
    / "hmatrix/daily_20260105_20260131"
    / "observation_vector.csv"
)

METAFILE = (
    BASE
    / "hmatrix/daily_20260105_20260131"
    / "H_metadata.csv"
)

OUTDIR = BASE / "hmatrix/daily_20260105_20260131/H_diagnostics"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# Load H
# ================================================================

print("=" * 80)
print("LOADING H MATRIX")
print("=" * 80)

H = np.load(HFILE)

print(f"H shape              : {H.shape}")
print(f"H dtype              : {H.dtype}")
print(f"H min/max            : {H.min():.8e} / {H.max():.8e}")
print(f"H mean               : {H.mean():.8e}")
print(f"H median             : {np.median(H):.8e}")

nobs, nsrc = H.shape

# ================================================================
# Basic sparsity
# ================================================================

nonzero = np.count_nonzero(H)
total = H.size
sparsity = 100.0 * nonzero / total

print()
print("=" * 80)
print("SPARSITY")
print("=" * 80)

print(f"Nonzero elements     : {nonzero}")
print(f"Total elements       : {total}")
print(f"Nonzero fraction     : {nonzero / total:.8f}")
print(f"Nonzero percentage   : {sparsity:.4f}%")

# ================================================================
# Column names
# ================================================================

try:
    meta = pd.read_csv(METAFILE)
    print()
    print("Metadata columns:")
    print(meta.columns.tolist())

    print()
    print(meta.head())
except Exception as e:
    print(f"WARNING: Could not read metadata: {e}")
    meta = None

# Expected column order:
# OC1_20260105 ... OC13_20260105 ... OC1_20260131

mine_names = [f"OC{i}" for i in range(1, 14)]

dates = pd.date_range(
    start="2026-01-05",
    end="2026-01-31",
    freq="D"
)

column_names = [
    f"{mine}_{date:%Y%m%d}"
    for date in dates
    for mine in mine_names
]

if len(column_names) != nsrc:
    raise RuntimeError(
        f"Expected {len(column_names)} columns but H has {nsrc}"
    )

# ================================================================
# Nonzero counts per source-day
# ================================================================

column_nnz = np.count_nonzero(H, axis=0)

source_day = pd.DataFrame({
    "source_day": column_names,
    "nonzero_obs": column_nnz,
    "max_H": np.max(H, axis=0),
    "sum_H": np.sum(H, axis=0),
    "mean_nonzero_H": [
        np.mean(H[:, j][H[:, j] != 0])
        if np.count_nonzero(H[:, j]) > 0
        else 0.0
        for j in range(nsrc)
    ],
})

source_day.to_csv(
    OUTDIR / "source_day_sensitivity.csv",
    index=False
)

print()
print("=" * 80)
print("SOURCE-DAY SENSITIVITY")
print("=" * 80)

print(
    source_day.sort_values(
        "nonzero_obs",
        ascending=False
    ).head(20).to_string(index=False)
)

# ================================================================
# Nonzero counts per mine
# ================================================================

mine_rows = []

for mine in mine_names:

    cols = [
        j for j, name in enumerate(column_names)
        if name.startswith(mine + "_")
    ]

    block = H[:, cols]

    mine_rows.append({
        "mine": mine,
        "nonzero": np.count_nonzero(block),
        "nonzero_fraction": np.count_nonzero(block) / block.size,
        "max_H": block.max(),
        "sum_H": block.sum(),
        "mean_H": block.mean(),
        "active_source_days": np.count_nonzero(
            np.count_nonzero(block, axis=0)
        ),
    })

mine_df = pd.DataFrame(mine_rows)

mine_df.to_csv(
    OUTDIR / "mine_sensitivity.csv",
    index=False
)

print()
print("=" * 80)
print("SENSITIVITY BY MINE")
print("=" * 80)

print(
    mine_df.to_string(index=False)
)

# ================================================================
# Nonzero counts per day
# ================================================================

day_rows = []

for date in dates:

    cols = [
        j for j, name in enumerate(column_names)
        if name.endswith(date.strftime("%Y%m%d"))
    ]

    block = H[:, cols]

    day_rows.append({
        "date": date.strftime("%Y-%m-%d"),
        "nonzero": np.count_nonzero(block),
        "nonzero_fraction": np.count_nonzero(block) / block.size,
        "max_H": block.max(),
        "sum_H": block.sum(),
        "active_mines": np.count_nonzero(
            np.count_nonzero(block, axis=0)
        ),
    })

day_df = pd.DataFrame(day_rows)

day_df.to_csv(
    OUTDIR / "daily_sensitivity.csv",
    index=False
)

print()
print("=" * 80)
print("SENSITIVITY BY DAY")
print("=" * 80)

print(day_df.to_string(index=False))

# ================================================================
# Nonzero observations per row
# ================================================================

row_nnz = np.count_nonzero(H, axis=1)

row_df = pd.DataFrame({
    "observation_index": np.arange(nobs),
    "nonzero_sources": row_nnz,
    "row_sum": H.sum(axis=1),
    "row_max": H.max(axis=1),
})

row_df.to_csv(
    OUTDIR / "observation_sensitivity.csv",
    index=False
)

print()
print("=" * 80)
print("OBSERVATION ROW SENSITIVITY")
print("=" * 80)

print(f"Minimum sources/row  : {row_nnz.min()}")
print(f"Maximum sources/row  : {row_nnz.max()}")
print(f"Mean sources/row     : {row_nnz.mean():.3f}")
print(f"Median sources/row   : {np.median(row_nnz):.3f}")

# ================================================================
# Column norms
# ================================================================

column_norm = np.linalg.norm(H, axis=0)

norm_df = pd.DataFrame({
    "source_day": column_names,
    "column_norm": column_norm,
    "nonzero_obs": column_nnz,
})

norm_df.to_csv(
    OUTDIR / "column_norms.csv",
    index=False
)

print()
print("=" * 80)
print("COLUMN NORMS")
print("=" * 80)

print(
    norm_df.sort_values(
        "column_norm",
        ascending=False
    ).head(20).to_string(index=False)
)

print()
print(
    "Columns with zero sensitivity:",
    np.sum(column_norm == 0)
)

# ================================================================
# SVD
# ================================================================

print()
print("=" * 80)
print("SINGULAR VALUE DECOMPOSITION")
print("=" * 80)

# H is only 1124 x 351, so full SVD is manageable.
U, s, VT = np.linalg.svd(H, full_matrices=False)

np.save(OUTDIR / "singular_values.npy", s)

print(f"Number of singular values : {len(s)}")
print(f"Maximum singular value    : {s[0]:.8e}")
print(f"Minimum singular value    : {s[-1]:.8e}")

tol = np.max(H.shape) * np.finfo(H.dtype).eps * s[0]

rank = np.sum(s > tol)

print(f"Numerical tolerance       : {tol:.8e}")
print(f"Numerical rank             : {rank}")

if s[-1] > 0:
    print(f"Condition number           : {s[0] / s[-1]:.8e}")
else:
    print("Condition number           : infinite")

svd_df = pd.DataFrame({
    "index": np.arange(1, len(s) + 1),
    "singular_value": s,
    "relative_to_max": s / s[0],
})

svd_df.to_csv(
    OUTDIR / "singular_values.csv",
    index=False
)

print()
print("First 20 singular values:")
print(svd_df.head(20).to_string(index=False))

# ================================================================
# Effective rank
# ================================================================

p = s / np.sum(s)

effective_rank = np.exp(
    -np.sum(
        p[p > 0] * np.log(p[p > 0])
    )
)

print()
print(f"Effective rank             : {effective_rank:.3f}")

# ================================================================
# Correlation between source-day columns
# ================================================================

print()
print("=" * 80)
print("SOURCE-DAY COLUMN CORRELATION")
print("=" * 80)

# Normalize columns and calculate correlation.
# Zero columns are excluded.

valid = column_norm > 0

Hvalid = H[:, valid]
valid_names = np.array(column_names)[valid]

# np.corrcoef can produce NaNs for constant columns.
corr = np.corrcoef(Hvalid, rowvar=False)

np.fill_diagonal(corr, 0.0)

# Find strongest absolute correlations.
pairs = []

for i in range(len(valid_names)):
    for j in range(i + 1, len(valid_names)):

        value = corr[i, j]

        if np.isfinite(value):
            pairs.append((
                abs(value),
                value,
                valid_names[i],
                valid_names[j],
            ))

pairs.sort(reverse=True)

corr_df = pd.DataFrame(
    pairs[:100],
    columns=[
        "abs_correlation",
        "correlation",
        "source_day_1",
        "source_day_2",
    ]
)

corr_df.to_csv(
    OUTDIR / "strongest_column_correlations.csv",
    index=False
)

print()
print(corr_df.head(30).to_string(index=False))

# ================================================================
# Receptor identification
# ================================================================

# The observation_vector.csv should contain the receptor information.
try:

    obs = pd.read_csv(OBSFILE)

    print()
    print("=" * 80)
    print("OBSERVATION FILE")
    print("=" * 80)

    print(f"Rows: {len(obs)}")
    print("Columns:")
    print(obs.columns.tolist())

    print()
    print(obs.head())

    # Try to identify receptor column automatically.
    receptor_col = None

    for c in obs.columns:
        cl = c.lower()

        if (
            "receptor" in cl
            or "site" in cl
            or "station" in cl
            or "location" in cl
        ):
            receptor_col = c
            break

    if receptor_col is not None:

        print()
        print("Receptor column:", receptor_col)

        receptor_summary = []

        for receptor, idx in obs.groupby(receptor_col).groups.items():

            idx = np.asarray(list(idx))

            block = H[idx, :]

            receptor_summary.append({
                "receptor": receptor,
                "observations": len(idx),
                "nonzero_H": np.count_nonzero(block),
                "max_H": block.max(),
                "sum_H": block.sum(),
                "mean_H": block.mean(),
            })

        receptor_df = pd.DataFrame(receptor_summary)

        receptor_df.to_csv(
            OUTDIR / "receptor_sensitivity.csv",
            index=False
        )

        print()
        print(receptor_df.to_string(index=False))

    else:

        print(
            "WARNING: Could not automatically identify receptor column."
        )

except Exception as e:

    print(
        f"WARNING: Could not analyze observation file: {e}"
    )

# ================================================================
# Save summary
# ================================================================

summary = {
    "n_observations": nobs,
    "n_source_days": nsrc,
    "n_mines": 13,
    "n_days": 27,
    "nonzero_H": nonzero,
    "total_H": total,
    "nonzero_percent": sparsity,
    "zero_columns": int(np.sum(column_norm == 0)),
    "numerical_rank": int(rank),
    "effective_rank": float(effective_rank),
    "largest_singular_value": float(s[0]),
    "smallest_singular_value": float(s[-1]),
}

pd.DataFrame([summary]).to_csv(
    OUTDIR / "H_diagnostic_summary.csv",
    index=False
)

print()
print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

print(f"Output directory:")
print(OUTDIR)

print()
print("Files created:")

for f in sorted(OUTDIR.iterdir()):
    print(" ", f.name)

