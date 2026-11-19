#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd


BASE = "/home/duch/flexpart-wrf/upperhunter_flexpart/H_matrix_flexpart"

KF_FILE = os.path.join(
    BASE,
    "KF_bounded_full_footprint_13mine_Jan2026",
    "daily_bounded_full_footprint_KF_scaling_factors.csv"
)

MAP_FILE = os.path.join(
    BASE,
    "Bounded_batch_MAP_full_footprint_13mine_Jan2026",
    "daily_bounded_batch_MAP_scaling_factors.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "Bounded_batch_MAP_full_footprint_13mine_Jan2026"
)

MINES = [f"OC{i}" for i in range(1, 14)]


print("=" * 100)
print("BOUNDED KALMAN FILTER vs BOUNDED BATCH MAP")
print("Upper Hunter CH4 — 13 mines × 27 days")
print("=" * 100)


# ============================================================================
# READ
# ============================================================================

kf = pd.read_csv(KF_FILE)
mp = pd.read_csv(MAP_FILE)

kf["date"] = pd.to_datetime(kf["date"])
mp["date"] = pd.to_datetime(mp["date"])

kf = kf.sort_values("date").reset_index(drop=True)
mp = mp.sort_values("date").reset_index(drop=True)

print("\nKF shape:", kf.shape)
print("MAP shape:", mp.shape)


# ============================================================================
# CHECK DATES
# ============================================================================

if not np.array_equal(
    kf["date"].to_numpy(),
    mp["date"].to_numpy()
):
    raise RuntimeError("KF and MAP dates do not match.")


# ============================================================================
# CHECK MINES
# ============================================================================

for mine in MINES:

    if mine not in kf.columns:
        raise RuntimeError(f"{mine} missing from KF file.")

    if mine not in mp.columns:
        raise RuntimeError(f"{mine} missing from MAP file.")


# ============================================================================
# BUILD 351-VECTOR
# ============================================================================

kf_values = kf[MINES].to_numpy(dtype=float)
map_values = mp[MINES].to_numpy(dtype=float)

difference = (
    kf_values - map_values
)

absolute_difference = np.abs(difference)


# ============================================================================
# GLOBAL METRICS
# ============================================================================

print("\n" + "=" * 100)
print("GLOBAL COMPARISON")
print("=" * 100)

print(
    "Number of parameters:",
    difference.size
)

print(
    "Maximum absolute difference:",
    np.max(absolute_difference)
)

print(
    "RMSE:",
    np.sqrt(np.mean(difference ** 2))
)

print(
    "MAE:",
    np.mean(absolute_difference)
)

print(
    "Mean difference:",
    np.mean(difference)
)

print(
    "Correlation:",
    np.corrcoef(
        kf_values.ravel(),
        map_values.ravel()
    )[0, 1]
)


# ============================================================================
# TOLERANCE COUNTS
# ============================================================================

print("\n" + "=" * 100)
print("DIFFERENCE TOLERANCES")
print("=" * 100)

for tol in [
    1e-1,
    1e-2,
    1e-3,
    1e-4,
    1e-5,
    1e-6,
]:

    count = np.sum(
        absolute_difference > tol
    )

    print(
        f"|KF - MAP| > {tol:.0e}:",
        count,
        f"({100.0 * count / difference.size:.3f}%)"
    )


# ============================================================================
# WORST PARAMETERS
# ============================================================================

rows = []

for d in range(len(kf)):

    for m, mine in enumerate(MINES):

        rows.append({

            "date": kf.iloc[d]["date"],

            "mine": mine,

            "KF": kf_values[d, m],

            "bounded_batch_MAP": map_values[d, m],

            "difference": difference[d, m],

            "absolute_difference":
                absolute_difference[d, m],

        })


comparison = pd.DataFrame(rows)

comparison = comparison.sort_values(
    "absolute_difference",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 100)
print("TOP 20 DIFFERENCES")
print("=" * 100)

print(
    comparison.head(20).to_string(
        index=False,
        float_format=lambda x: f"{x:.10f}"
    )
)


# ============================================================================
# BY MINE
# ============================================================================

mine_rows = []

for mine in MINES:

    sub = comparison[
        comparison["mine"] == mine
    ]

    mine_rows.append({

        "mine": mine,

        "RMSE":
            np.sqrt(
                np.mean(
                    sub["difference"] ** 2
                )
            ),

        "MAE":
            np.mean(
                np.abs(
                    sub["difference"]
                )
            ),

        "maximum_absolute_difference":
            np.max(
                np.abs(
                    sub["difference"]
                )
            ),

        "mean_difference":
            np.mean(
                sub["difference"]
            ),

    })


mine_summary = pd.DataFrame(
    mine_rows
)


print("\n" + "=" * 100)
print("COMPARISON BY MINE")
print("=" * 100)

print(
    mine_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.10f}"
    )
)


# ============================================================================
# BY DAY
# ============================================================================

day_rows = []

for date in sorted(
    comparison["date"].unique()
):

    sub = comparison[
        comparison["date"] == date
    ]

    day_rows.append({

        "date": date,

        "RMSE":
            np.sqrt(
                np.mean(
                    sub["difference"] ** 2
                )
            ),

        "MAE":
            np.mean(
                np.abs(
                    sub["difference"]
                )
            ),

        "maximum_absolute_difference":
            np.max(
                np.abs(
                    sub["difference"]
                )
            ),

        "mean_difference":
            np.mean(
                sub["difference"]
            ),

    })


day_summary = pd.DataFrame(
    day_rows
)


print("\n" + "=" * 100)
print("COMPARISON BY DAY")
print("=" * 100)

print(
    day_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.10f}"
    )
)


# ============================================================================
# SAVE
# ============================================================================

comparison_file = os.path.join(
    OUTPUT_DIR,
    "bounded_KF_vs_bounded_batch_MAP_351.csv"
)

mine_file = os.path.join(
    OUTPUT_DIR,
    "bounded_KF_vs_bounded_batch_MAP_by_mine.csv"
)

day_file = os.path.join(
    OUTPUT_DIR,
    "bounded_KF_vs_bounded_batch_MAP_by_day.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)

mine_summary.to_csv(
    mine_file,
    index=False
)

day_summary.to_csv(
    day_file,
    index=False
)


print("\n" + "=" * 100)
print("FILES SAVED")
print("=" * 100)

print(comparison_file)
print(mine_file)
print(day_file)


print("\n" + "=" * 100)
print("DONE")
print("=" * 100)

