#!/usr/bin/env python3

"""
Compare mine-weighted FLEXPART sensitivities for 2-layer vs 3-layer
vertical configurations.

For each mine and each FLEXPART output hour, calculate:

    H_mine = sum(K(x,y) * mine_area_fraction(x,y))

where K = CONC / volume.

Compare:
    3L 0-10 m
    3L 10-100 m
    3L 0-100 m
    2L 0-100 m

The 100-20000 m contribution is also checked.

Units:
    ppmv / (kg CH4 s-1)

No PPMV conversion is applied here because the saved K arrays are already
the exact concentration/volume quantity used by the H calculation.
The script applies PPMV_FACTOR at the end.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE = Path.home() / "flexpart-wrf" / "upperhunter_flexpart"

SPATIAL_DIR = (
    BASE
    / "hmatrix"
    / "daily_test"
    / "spatial_2L_vs_3L"
)

WEIGHTS_FILE = (
    BASE
    / "hmatrix"
    / "test_20260105"
    / "mine_area_fraction_20260105.npy"
)

OUTDIR = (
    BASE
    / "hmatrix"
    / "daily_test"
    / "mine_weighted_2L_vs_3L"
)

OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONSTANTS
# ============================================================

PPMV_FACTOR = 1e6 * 28.97 / 16.043

MINES = [
    "OC1",
    "OC2",
    "OC3",
    "OC4",
    "OC5",
    "OC6",
    "OC7",
    "OC8",
    "OC9",
    "OC10",
    "OC11",
    "OC12",
    "OC13",
]

TIMES = [
    "2026-01-10 01:00 UTC",
    "2026-01-10 00:00 UTC",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("MINE-WEIGHTED 2-LAYER VS 3-LAYER FLEXPART COMPARISON")
print("=" * 80)

print("\nLoading mine weights:")
print(WEIGHTS_FILE)

weights = np.load(WEIGHTS_FILE)

print("weights shape:", weights.shape)

if weights.shape[0] != 13:
    raise ValueError(
        f"Expected 13 mines in weights, got {weights.shape[0]}"
    )

if weights.ndim != 3:
    raise ValueError(
        f"Expected weights dimensions (13,ny,nx), got {weights.shape}"
    )


# ------------------------------------------------------------
# Load saved K fields
# ------------------------------------------------------------

files = {
    "3L_0_10m": SPATIAL_DIR / "K_3L_0_10m.npy",
    "3L_10_100m": SPATIAL_DIR / "K_3L_10_100m.npy",
    "3L_0_100m": SPATIAL_DIR / "K_3L_0_100m.npy",
    "3L_100_20000m": SPATIAL_DIR / "K_3L_100_20000m.npy",
    "2L_0_100m": SPATIAL_DIR / "K_2L_0_100m.npy",
    "2L_100_20000m": SPATIAL_DIR / "K_2L_100_20000m.npy",
}

K = {}

for name, path in files.items():

    print(f"\nLoading {name}:")
    print(path)

    if not path.exists():
        raise FileNotFoundError(path)

    arr = np.load(path)

    print("  shape:", arr.shape)
    print("  min:", np.nanmin(arr))
    print("  max:", np.nanmax(arr))
    print("  sum:", np.nansum(arr))

    if arr.ndim != 3:
        raise ValueError(
            f"{name}: expected (time,ny,nx), got {arr.shape}"
        )

    if arr.shape[1:] != weights.shape[1:]:
        raise ValueError(
            f"{name}: spatial dimensions {arr.shape[1:]} "
            f"do not match weights {weights.shape[1:]}"
        )

    K[name] = arr


# ============================================================
# MINE-WEIGHTED SENSITIVITY
# ============================================================

rows = []

for t in range(2):

    print("\n" + "=" * 80)
    print(f"TIME {t}: {TIMES[t]}")
    print("=" * 80)

    for i, mine in enumerate(MINES):

        w = weights[i]

        values = {}

        for name, arr in K.items():

            # Mine-weighted sensitivity
            #
            # arr[t,:,:] = CONC / volume
            # w         = mine area fraction
            #
            # H = sum(K * w) * PPMV_FACTOR

            values[name] = np.sum(
                arr[t] * w
            ) * PPMV_FACTOR

        h3_010 = values["3L_0_10m"]
        h3_10100 = values["3L_10_100m"]
        h3_0100 = values["3L_0_100m"]

        h2_0100 = values["2L_0_100m"]

        h3_upper = values["3L_100_20000m"]
        h2_upper = values["2L_100_20000m"]

        if h3_0100 > 0:
            ratio = h2_0100 / h3_0100
        else:
            ratio = np.nan

        if h3_0100 > 0:
            difference = h2_0100 - h3_0100
        else:
            difference = np.nan

        rows.append({
            "time_index": t,
            "time_utc": TIMES[t],
            "mine": mine,

            "H_3L_0_10m": h3_010,
            "H_3L_10_100m": h3_10100,
            "H_3L_0_100m": h3_0100,

            "H_2L_0_100m": h2_0100,

            "H_3L_100_20000m": h3_upper,
            "H_2L_100_20000m": h2_upper,

            "H_2L_minus_3L_0_100m": difference,
            "H_2L_to_3L_0_100_ratio": ratio,
        })


df = pd.DataFrame(rows)


# ============================================================
# SAVE COMPLETE TABLE
# ============================================================

outfile = OUTDIR / "mine_weighted_2L_vs_3L.csv"

df.to_csv(outfile, index=False)

print("\nComplete results written to:")
print(outfile)


# ============================================================
# PRINT RESULTS
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)
pd.set_option("display.float_format", lambda x: f"{x:.8e}")


for t in range(2):

    sub = df[df["time_index"] == t].copy()

    print("\n")
    print("=" * 120)
    print(f"TIME {t}: {TIMES[t]}")
    print("=" * 120)

    print(
        sub[
            [
                "mine",
                "H_3L_0_10m",
                "H_3L_10_100m",
                "H_3L_0_100m",
                "H_2L_0_100m",
                "H_2L_to_3L_0_100_ratio",
            ]
        ].to_string(index=False)
    )

    print("\nTotals:")

    total3_010 = sub["H_3L_0_10m"].sum()
    total3_10100 = sub["H_3L_10_100m"].sum()
    total3_0100 = sub["H_3L_0_100m"].sum()
    total2_0100 = sub["H_2L_0_100m"].sum()

    total3_upper = sub["H_3L_100_20000m"].sum()
    total2_upper = sub["H_2L_100_20000m"].sum()

    print(f"3L 0-10 m:        {total3_010:.12e}")
    print(f"3L 10-100 m:      {total3_10100:.12e}")
    print(f"3L 0-100 m:       {total3_0100:.12e}")
    print(f"2L 0-100 m:       {total2_0100:.12e}")
    print(f"3L 100-20,000 m:  {total3_upper:.12e}")
    print(f"2L 100-20,000 m:  {total2_upper:.12e}")

    print(
        f"2L / 3L 0-100:    "
        f"{total2_0100 / total3_0100:.8f}"
    )


# ============================================================
# SUMMARY BY MINE ACROSS THE TWO HOURS
# ============================================================

summary = (
    df
    .groupby("mine")
    .agg(
        H3_0_10_mean=("H_3L_0_10m", "mean"),
        H3_10_100_mean=("H_3L_10_100m", "mean"),
        H3_0_100_mean=("H_3L_0_100m", "mean"),
        H2_0_100_mean=("H_2L_0_100m", "mean"),
        H2_to_H3_mean=("H_2L_to_3L_0_100_ratio", "mean"),
    )
    .reset_index()
)

summary_file = OUTDIR / "mine_weighted_summary.csv"

summary.to_csv(summary_file, index=False)

print("\n")
print("=" * 120)
print("MEAN OVER THE TWO HOURS")
print("=" * 120)

print(summary.to_string(index=False))

print("\nSummary written to:")
print(summary_file)

print("\nDone.")
