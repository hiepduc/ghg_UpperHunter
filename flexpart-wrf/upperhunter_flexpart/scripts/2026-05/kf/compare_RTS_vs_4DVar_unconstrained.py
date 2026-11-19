#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd


KF_DIR = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "H_matrix_flexpart/KF_13mine_daily_Jan2026"
)

VAR_DIR = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "H_matrix_flexpart/"
    "bayesian_4dvar_daily_scaling_comparison"
)

RTS_FILE = os.path.join(
    KF_DIR,
    "RTS_daily_scaling_factors.csv"
)

VAR_FILE = os.path.join(
    VAR_DIR,
    "daily_scaling_factors_wide.csv"
)

OUTPUT = os.path.join(
    KF_DIR,
    "RTS_vs_4DVar_unconstrained.csv"
)

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

PRIOR = {
    "OC1": 1786.45,
    "OC2": 242.86,
    "OC3": 342.53,
    "OC4": 40.60,
    "OC5": 0.01,
    "OC6": 7.85,
    "OC7": 82.69,
    "OC8": 5.95,
    "OC9": 44.92,
    "OC10": 101.93,
    "OC11": 1.87,
    "OC12": 331.49,
    "OC13": 798.08,
}

prior_vector = np.array([
    PRIOR[m] for m in MINES
])

prior_total = prior_vector.sum()


# ============================================================
# RTS
# ============================================================

rts = pd.read_csv(RTS_FILE)

rts["date"] = pd.to_datetime(
    rts["date"]
)

rts = rts.sort_values(
    "date"
).reset_index(drop=True)


# ============================================================
# Read 4D-Var file
# ============================================================

raw = pd.read_csv(
    VAR_FILE,
    header=None
)

# The first three rows contain:
#
# row 0: headers
# row 1: mine names
# row 2: dates
# rows 3+: values
#
print("4D-Var raw shape:", raw.shape)

var_mine_names = raw.iloc[1, 1:].tolist()

# Determine unconstrained columns.
#
# The original file has:
#
# col 0 = first mine/name field
# then 13 unconstrained values
# then 13 constrained values.
#
# We identify the first block by position.
#
unconstrained_raw = raw.iloc[
    3:,
    1:14
].copy()

var_dates = pd.to_datetime(
    raw.iloc[3:, 0]
)

unconstrained_raw.columns = (
    var_mine_names[:13]
)

unconstrained_raw.index = var_dates

unconstrained_raw = (
    unconstrained_raw
    .apply(pd.to_numeric)
)

# Reorder to standard mine ordering.
unconstrained = (
    unconstrained_raw[
        [
            "OC1",
            "OC10",
            "OC11",
            "OC12",
            "OC13",
            "OC2",
            "OC3",
            "OC4",
            "OC5",
            "OC6",
            "OC7",
            "OC8",
            "OC9",
        ]
    ]
)

unconstrained = (
    unconstrained
    .rename(columns={
        "OC1": "OC1",
        "OC10": "OC10",
        "OC11": "OC11",
        "OC12": "OC12",
        "OC13": "OC13",
        "OC2": "OC2",
        "OC3": "OC3",
        "OC4": "OC4",
        "OC5": "OC5",
        "OC6": "OC6",
        "OC7": "OC7",
        "OC8": "OC8",
        "OC9": "OC9",
    })
)

unconstrained = unconstrained[
    MINES
]

unconstrained["date"] = (
    unconstrained.index
)

unconstrained = (
    unconstrained
    .reset_index(drop=True)
)


# ============================================================
# Restrict to common dates
# ============================================================

common = rts.merge(
    unconstrained,
    on="date",
    suffixes=("_RTS", "_4DVar")
)

print()
print("RTS days:", len(rts))
print("4D-Var days:", len(unconstrained))
print("Common days:", len(common))


# ============================================================
# Compute totals
# ============================================================

rts_matrix = common[
    [
        f"{m}_RTS"
        for m in MINES
    ]
].values

var_matrix = common[
    [
        f"{m}_4DVar"
        for m in MINES
    ]
].values

rts_total = (
    rts_matrix @ prior_vector
)

var_total = (
    var_matrix @ prior_vector
)

rts_scaling = (
    rts_total / prior_total
)

var_scaling = (
    var_total / prior_total
)


# ============================================================
# Output
# ============================================================

out = pd.DataFrame({
    "date": common["date"],
    "prior_total_gCH4_s": prior_total,
    "RTS_total_gCH4_s": rts_total,
    "4DVar_unconstrained_total_gCH4_s":
        var_total,
    "RTS_scaling": rts_scaling,
    "4DVar_unconstrained_scaling":
        var_scaling,
})

out["RTS_minus_4DVar_gCH4_s"] = (
    out["RTS_total_gCH4_s"]
    - out["4DVar_unconstrained_total_gCH4_s"]
)

out["absolute_difference_gCH4_s"] = (
    np.abs(
        out["RTS_minus_4DVar_gCH4_s"]
    )
)

out["percentage_difference_relative_to_4DVar"] = (
    100.0
    * out["RTS_minus_4DVar_gCH4_s"]
    / out["4DVar_unconstrained_total_gCH4_s"]
)


# ============================================================
# Statistics
# ============================================================

mean_rts = out[
    "RTS_total_gCH4_s"
].mean()

mean_var = out[
    "4DVar_unconstrained_total_gCH4_s"
].mean()

mean_abs_diff = out[
    "absolute_difference_gCH4_s"
].mean()

mean_pct = out[
    "percentage_difference_relative_to_4DVar"
].mean()

rmse = np.sqrt(
    np.mean(
        (
            out["RTS_total_gCH4_s"]
            -
            out["4DVar_unconstrained_total_gCH4_s"]
        ) ** 2
    )
)


# ============================================================
# Print
# ============================================================

print()
print("=" * 90)
print("RTS vs UNCONSTRAINED 4D-VAR")
print("=" * 90)

print()
print(
    f"Prior total          : {prior_total:.2f} g CH4/s"
)

print(
    f"Mean RTS total       : {mean_rts:.2f} g CH4/s"
)

print(
    f"Mean 4D-Var total    : {mean_var:.2f} g CH4/s"
)

print(
    f"Mean absolute diff   : {mean_abs_diff:.2f} g CH4/s"
)

print(
    f"RMSE                 : {rmse:.2f} g CH4/s"
)

print(
    f"Mean % difference    : {mean_pct:.2f}%"
)

print()
print(
    out.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

out.to_csv(
    OUTPUT,
    index=False
)

print()
print("Saved:")
print(OUTPUT)

print()
print("=" * 90)
print("DONE")
print("=" * 90)
