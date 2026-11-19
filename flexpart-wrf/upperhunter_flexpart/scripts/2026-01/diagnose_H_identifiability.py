#!/usr/bin/env python3
"""
=====================================================================
H-MATRIX IDENTIFIABILITY DIAGNOSTIC
Upper Hunter FLEXPART-WRF CH4 inversion
=====================================================================

Input:
    H_daily_13mine_1124x351_ppm.npy

Dimensions:
    observations = 1124
    source-days  = 13 mines x 27 days = 351

Diagnostics:
    1. Zero-sensitivity source-days
    2. Weak source-days based on column-norm quantiles
    3. Mine-to-mine correlation
    4. Day-to-day correlation
    5. Source-day correlation pairs > selected thresholds
    6. Summary statistics

No modification of H.
No MCMC.
=====================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

BASE = Path.home() / "flexpart-wrf/upperhunter_flexpart"

HFILE = (
    BASE
    / "hmatrix/daily_20260105_20260131"
    / "H_daily_13mine_1124x351_ppm.npy"
)

OUTDIR = (
    BASE
    / "hmatrix/daily_20260105_20260131"
    / "H_identifiability"
)

OUTDIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# SETTINGS
# =====================================================================

N_MINE = 13
N_DAY = 27

MINES = [f"OC{i}" for i in range(1, 14)]

DATES = pd.date_range(
    "2026-01-05",
    "2026-01-31",
    freq="D"
)

COLUMN_NAMES = [
    f"{mine}_{date.strftime('%Y%m%d')}"
    for date in DATES
    for mine in MINES
]


# =====================================================================
# READ H
# =====================================================================

print("=" * 80)
print("READING H MATRIX")
print("=" * 80)

H = np.load(HFILE)

print(f"H file       : {HFILE}")
print(f"H shape      : {H.shape}")
print(f"Expected     : (1124, 351)")
print(f"dtype        : {H.dtype}")


if H.shape != (1124, 351):
    raise ValueError(
        f"Unexpected H shape: {H.shape}. "
        "Expected (1124, 351)."
    )


# =====================================================================
# CHECK COLUMN NAMES
# =====================================================================

if len(COLUMN_NAMES) != H.shape[1]:
    raise ValueError("Number of generated column names does not match H.")


# =====================================================================
# 1. COLUMN NORMS
# =====================================================================

print()
print("=" * 80)
print("1. COLUMN SENSITIVITY")
print("=" * 80)

column_norm = np.linalg.norm(H, axis=0)

column_sum = np.sum(H, axis=0)

nonzero_count = np.count_nonzero(H, axis=0)

column_mean = np.mean(H, axis=0)

column_df = pd.DataFrame({
    "column": COLUMN_NAMES,
    "mine": [x.split("_")[0] for x in COLUMN_NAMES],
    "date": [x.split("_")[1] for x in COLUMN_NAMES],
    "column_norm": column_norm,
    "column_sum": column_sum,
    "nonzero_observations": nonzero_count,
    "column_mean": column_mean,
})

column_df.to_csv(
    OUTDIR / "source_day_sensitivity.csv",
    index=False
)


# =====================================================================
# 2. ZERO-SENSITIVITY SOURCE-DAYS
# =====================================================================

print()
print("=" * 80)
print("2. ZERO-SENSITIVITY SOURCE-DAYS")
print("=" * 80)

zero_df = column_df[column_df["column_norm"] == 0].copy()

print(f"Number of zero-sensitivity columns: {len(zero_df)}")
print(
    f"Percentage of all source-days: "
    f"{100 * len(zero_df) / len(column_df):.2f}%"
)

zero_df.to_csv(
    OUTDIR / "zero_sensitivity_source_days.csv",
    index=False
)

print()
print("Zero-sensitivity source-days by mine:")

zero_by_mine = (
    zero_df
    .groupby("mine")
    .size()
    .reindex(MINES, fill_value=0)
    .reset_index(name="zero_source_days")
)

zero_by_mine["total_source_days"] = N_DAY
zero_by_mine["zero_fraction"] = (
    zero_by_mine["zero_source_days"] / N_DAY
)

print(zero_by_mine.to_string(index=False))

zero_by_mine.to_csv(
    OUTDIR / "zero_sensitivity_by_mine.csv",
    index=False
)

print()
print("Zero-sensitivity source-days by date:")

zero_by_day = (
    zero_df
    .groupby("date")
    .size()
    .reindex(
        [d.strftime("%Y%m%d") for d in DATES],
        fill_value=0
    )
    .reset_index(name="zero_source_days")
)

zero_by_day["total_mines"] = N_MINE
zero_by_day["zero_fraction"] = (
    zero_by_day["zero_source_days"] / N_MINE
)

print(zero_by_day.to_string(index=False))

zero_by_day.to_csv(
    OUTDIR / "zero_sensitivity_by_day.csv",
    index=False
)


# =====================================================================
# 3. WEAK SOURCE-DAYS
# =====================================================================

print()
print("=" * 80)
print("3. WEAK SOURCE-DAYS")
print("=" * 80)

nonzero_norms = column_norm[column_norm > 0]

quantiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

print()
print("Column-norm quantiles for NONZERO source-days:")

for q in quantiles:
    print(
        f"{q:>3}% : "
        f"{np.percentile(nonzero_norms, q):.8e}"
    )


# Define weak groups by percentile.
# This avoids choosing an arbitrary absolute threshold.

q10 = np.percentile(nonzero_norms, 10)
q25 = np.percentile(nonzero_norms, 25)

weak10 = column_df[
    (column_df["column_norm"] > 0)
    & (column_df["column_norm"] <= q10)
].copy()

weak25 = column_df[
    (column_df["column_norm"] > 0)
    & (column_df["column_norm"] <= q25)
].copy()

print()
print(f"10th percentile threshold : {q10:.8e}")
print(f"25th percentile threshold : {q25:.8e}")

print()
print(f"Weakest 10% nonzero columns : {len(weak10)}")
print(f"Weakest 25% nonzero columns : {len(weak25)}")

weak10.to_csv(
    OUTDIR / "weakest_10_percent_source_days.csv",
    index=False
)

weak25.to_csv(
    OUTDIR / "weakest_25_percent_source_days.csv",
    index=False
)


# =====================================================================
# 4. MINE-LEVEL AGGREGATION
# =====================================================================

print()
print("=" * 80)
print("4. MINE-LEVEL AGGREGATION")
print("=" * 80)

# For each mine, sum its 27 daily sensitivity columns.
# This gives the total response pattern of each mine
# across the 1124 observations.

H_mine = np.zeros((H.shape[0], N_MINE))

for m, mine in enumerate(MINES):

    cols = [
        i for i, name in enumerate(COLUMN_NAMES)
        if name.startswith(mine + "_")
    ]

    H_mine[:, m] = np.sum(H[:, cols], axis=1)


mine_norm = np.linalg.norm(H_mine, axis=0)

mine_sum = np.sum(H_mine, axis=0)

mine_nonzero = np.count_nonzero(H_mine, axis=0)

mine_df = pd.DataFrame({
    "mine": MINES,
    "norm": mine_norm,
    "sum": mine_sum,
    "nonzero_observations": mine_nonzero,
})

print(mine_df.to_string(index=False))

mine_df.to_csv(
    OUTDIR / "mine_aggregate_sensitivity.csv",
    index=False
)


# =====================================================================
# 5. MINE-TO-MINE CORRELATION
# =====================================================================

print()
print("=" * 80)
print("5. MINE-TO-MINE CORRELATION")
print("=" * 80)

mine_corr = np.corrcoef(H_mine, rowvar=False)

mine_corr_df = pd.DataFrame(
    mine_corr,
    index=MINES,
    columns=MINES
)

mine_corr_df.to_csv(
    OUTDIR / "mine_to_mine_correlation.csv"
)

print()
print(mine_corr_df.round(4).to_string())


# Strong mine pairs
mine_pairs = []

for i in range(N_MINE):
    for j in range(i + 1, N_MINE):

        mine_pairs.append({
            "mine1": MINES[i],
            "mine2": MINES[j],
            "correlation": mine_corr[i, j],
        })

mine_pairs_df = (
    pd.DataFrame(mine_pairs)
    .sort_values("correlation", ascending=False)
)

mine_pairs_df.to_csv(
    OUTDIR / "mine_correlation_pairs.csv",
    index=False
)

print()
print("Strongest mine-to-mine correlations:")

print(
    mine_pairs_df.head(20).to_string(index=False)
)


# =====================================================================
# 6. DAY-LEVEL AGGREGATION
# =====================================================================

print()
print("=" * 80)
print("6. DAY-LEVEL AGGREGATION")
print("=" * 80)

# For each day, sum the 13 mine sensitivities.
#
# H_day[:, d] represents the total response pattern of
# all 13 mines for that source day.

H_day = np.zeros((H.shape[0], N_DAY))

for d, date in enumerate(DATES):

    date_string = date.strftime("%Y%m%d")

    cols = [
        i for i, name in enumerate(COLUMN_NAMES)
        if name.endswith("_" + date_string)
    ]

    H_day[:, d] = np.sum(H[:, cols], axis=1)


day_norm = np.linalg.norm(H_day, axis=0)

day_sum = np.sum(H_day, axis=0)

day_nonzero = np.count_nonzero(H_day, axis=0)

day_df = pd.DataFrame({
    "date": [d.strftime("%Y%m%d") for d in DATES],
    "norm": day_norm,
    "sum": day_sum,
    "nonzero_observations": day_nonzero,
})

print(day_df.to_string(index=False))

day_df.to_csv(
    OUTDIR / "day_aggregate_sensitivity.csv",
    index=False
)


# =====================================================================
# 7. DAY-TO-DAY CORRELATION
# =====================================================================

print()
print("=" * 80)
print("7. DAY-TO-DAY CORRELATION")
print("=" * 80)

day_corr = np.corrcoef(H_day, rowvar=False)

day_labels = [
    d.strftime("%Y%m%d")
    for d in DATES
]

day_corr_df = pd.DataFrame(
    day_corr,
    index=day_labels,
    columns=day_labels
)

day_corr_df.to_csv(
    OUTDIR / "day_to_day_correlation.csv"
)

print()
print(day_corr_df.round(4).to_string())


# Strong day pairs
day_pairs = []

for i in range(N_DAY):
    for j in range(i + 1, N_DAY):

        day_pairs.append({
            "date1": day_labels[i],
            "date2": day_labels[j],
            "correlation": day_corr[i, j],
        })

day_pairs_df = (
    pd.DataFrame(day_pairs)
    .sort_values("correlation", ascending=False)
)

day_pairs_df.to_csv(
    OUTDIR / "day_correlation_pairs.csv",
    index=False
)

print()
print("Strongest day-to-day correlations:")

print(
    day_pairs_df.head(20).to_string(index=False)
)


# =====================================================================
# 8. SOURCE-DAY CORRELATIONS
# =====================================================================

print()
print("=" * 80)
print("8. SOURCE-DAY CORRELATIONS")
print("=" * 80)

# Correlation between all 351 individual source-day columns.
#
# This can be memory intensive, but 351 x 351 is small.

source_corr = np.corrcoef(H, rowvar=False)

source_pairs = []

for i in range(H.shape[1]):
    for j in range(i + 1, H.shape[1]):

        r = source_corr[i, j]

        if np.isfinite(r):

            source_pairs.append({
                "source1": COLUMN_NAMES[i],
                "source2": COLUMN_NAMES[j],
                "correlation": r,
            })

source_pairs_df = (
    pd.DataFrame(source_pairs)
    .sort_values("correlation", ascending=False)
)

source_pairs_df.to_csv(
    OUTDIR / "source_day_correlation_pairs.csv",
    index=False
)

print()
print("Strongest source-day correlations:")

print(
    source_pairs_df.head(30).to_string(index=False)
)


# =====================================================================
# 9. CORRELATION COUNTS
# =====================================================================

print()
print("=" * 80)
print("9. HIGH-CORRELATION COUNTS")
print("=" * 80)

for threshold in [0.90, 0.95, 0.99, 0.999]:

    n = np.sum(
        np.abs(source_corr[np.triu_indices(H.shape[1], k=1)])
        >= threshold
    )

    print(
        f"|r| >= {threshold:.3f}: "
        f"{n} source-day pairs"
    )


# =====================================================================
# 10. SUMMARY
# =====================================================================

print()
print("=" * 80)
print("10. SUMMARY")
print("=" * 80)

print(f"Observations                 : {H.shape[0]}")
print(f"Source-day parameters        : {H.shape[1]}")
print(f"Zero-sensitivity columns     : {np.sum(column_norm == 0)}")
print(
    f"Zero-sensitivity percentage  : "
    f"{100*np.mean(column_norm == 0):.2f}%"
)
print(
    f"Nonzero source-day columns   : "
    f"{np.sum(column_norm > 0)}"
)
print(
    f"Numerical rank from previous : 247"
)
print(
    f"Effective rank from previous : ~78.45"
)

print()
print(f"Output directory:")
print(OUTDIR)

print()
print("Files written:")
for f in sorted(OUTDIR.glob("*")):
    print(f"  {f.name}")

print()
print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

