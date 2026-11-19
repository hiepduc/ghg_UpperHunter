#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# FILES
# ============================================================

PRIOR_FILE = Path(
    "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter/"
    "UpperHuntercoalmines.csv"
)

KF_FILE = Path(
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "H_matrix_flexpart/KF_13mine_daily_Jan2026/"
    "daily_KF_scaling_factors.csv"
)

VAR_FILE = Path(
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "H_matrix_flexpart/bayesian_4dvar_daily_scaling_comparison/"
    "daily_scaling_factors_wide.csv"
)

OUT_FILE = Path(
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "H_matrix_flexpart/KF_13mine_daily_Jan2026/"
    "KF_vs_4DVar_total_emissions.csv"
)

MINES = [f"OC{i}" for i in range(1, 14)]

# ============================================================
# READ PRIOR
# ============================================================

prior = pd.read_csv(PRIOR_FILE)

prior_emission = {}

for mine in MINES:
    mask = (
        prior["Facility_code_name"]
        .astype(str)
        .str.upper()
        .str.startswith(mine)
    )

    if not mask.any():
        raise ValueError(f"Missing prior emission for {mine}")

    prior_emission[mine] = prior.loc[mask, "gCH4/s"].astype(float).sum()

prior_emission = pd.Series(prior_emission)

E_PRIOR = prior_emission.sum()

print("=" * 90)
print("PRIOR MINE EMISSIONS")
print("=" * 90)

print(prior_emission.to_string())
print(f"\nTOTAL PRIOR = {E_PRIOR:.6f} g CH4/s")


# ============================================================
# READ KF
# ============================================================

kf = pd.read_csv(KF_FILE)

kf["date"] = pd.to_datetime(kf["date"])

# Analysis states only
kf = kf[
    kf["state_type"].astype(str).str.lower() == "analysis"
].copy()

kf = kf[
    (kf["date"] >= "2026-01-05") &
    (kf["date"] <= "2026-01-31")
].copy()

kf = kf.sort_values("date").reset_index(drop=True)

print("\nKF days:", len(kf))


# ============================================================
# READ 4D-VAR
#
# Structure:
#
# row 0 = mine names
# row 1 = date
# rows 2 onward = daily values
#
# columns 1:14  = unconstrained
# columns 14:27 = constrained
# ============================================================

var_raw = pd.read_csv(VAR_FILE, header=None)

mine_order = var_raw.iloc[1, 1:14].astype(str).tolist()

constrained_order = var_raw.iloc[1, 14:27].astype(str).tolist()

print("\n4D-Var mine order:")
print(constrained_order)

if mine_order != constrained_order:
    raise ValueError(
        "Unconstrained and constrained mine ordering differ."
    )

# Build dataframe explicitly
var = pd.DataFrame()

var["date"] = pd.to_datetime(
    var_raw.iloc[2:, 0],
    errors="coerce"
)

for j, mine in enumerate(mine_order):

    var[mine] = pd.to_numeric(
        var_raw.iloc[2:, 14 + j],
        errors="coerce"
    )

var = var.dropna(subset=["date"])

var = var[
    (var["date"] >= "2026-01-05") &
    (var["date"] <= "2026-01-31")
].copy()

var = var.sort_values("date").reset_index(drop=True)

print("4D-Var days:", len(var))


# ============================================================
# ALIGN DATES
# ============================================================

dates = sorted(
    set(kf["date"]).intersection(set(var["date"]))
)

print("\nCommon days:", len(dates))

kf = kf[kf["date"].isin(dates)].copy()
var = var[var["date"].isin(dates)].copy()

kf = kf.set_index("date").loc[dates]
var = var.set_index("date").loc[dates]


# ============================================================
# EMISSION-WEIGHTED TOTALS
# ============================================================

result = pd.DataFrame(index=dates)

result.index.name = "date"

for mine in MINES:

    E = prior_emission[mine]

    result[f"{mine}_prior_gCH4s"] = E

    result[f"{mine}_KF_gCH4s"] = (
        kf[mine] * E
    )

    result[f"{mine}_4DVar_gCH4s"] = (
        var[mine] * E
    )


# ============================================================
# TOTAL EMISSIONS
# ============================================================

result["prior_total_gCH4s"] = E_PRIOR

result["KF_total_gCH4s"] = sum(
    result[f"{mine}_KF_gCH4s"]
    for mine in MINES
)

result["4DVar_total_gCH4s"] = sum(
    result[f"{mine}_4DVar_gCH4s"]
    for mine in MINES
)

# Total scaling factor
result["KF_total_scaling"] = (
    result["KF_total_gCH4s"] / E_PRIOR
)

result["4DVar_total_scaling"] = (
    result["4DVar_total_gCH4s"] / E_PRIOR
)

# Differences
result["KF_minus_prior_gCH4s"] = (
    result["KF_total_gCH4s"] - E_PRIOR
)

result["4DVar_minus_prior_gCH4s"] = (
    result["4DVar_total_gCH4s"] - E_PRIOR
)

result["KF_minus_4DVar_gCH4s"] = (
    result["KF_total_gCH4s"]
    - result["4DVar_total_gCH4s"]
)

result["KF_minus_4DVar_percent"] = (
    100.0
    * result["KF_minus_4DVar_gCH4s"]
    / result["4DVar_total_gCH4s"]
)

# ============================================================
# SAVE
# ============================================================

result.reset_index().to_csv(
    OUT_FILE,
    index=False,
    float_format="%.8f"
)


# ============================================================
# PRINT MAIN RESULT
# ============================================================

print("\n" + "=" * 90)
print("TOTAL DAILY MINE EMISSIONS")
print("=" * 90)

print(
    result[
        [
            "prior_total_gCH4s",
            "KF_total_gCH4s",
            "4DVar_total_gCH4s",
            "KF_total_scaling",
            "4DVar_total_scaling",
            "KF_minus_4DVar_gCH4s",
            "KF_minus_4DVar_percent",
        ]
    ].to_string()
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

print(
    f"Prior mean total     : "
    f"{result['prior_total_gCH4s'].mean():.4f} g CH4/s"
)

print(
    f"KF mean total        : "
    f"{result['KF_total_gCH4s'].mean():.4f} g CH4/s"
)

print(
    f"4D-Var mean total    : "
    f"{result['4DVar_total_gCH4s'].mean():.4f} g CH4/s"
)

print(
    f"\nKF mean scaling      : "
    f"{result['KF_total_scaling'].mean():.6f}"
)

print(
    f"4D-Var mean scaling  : "
    f"{result['4DVar_total_scaling'].mean():.6f}"
)

print(
    f"\nMean KF - 4DVar      : "
    f"{result['KF_minus_4DVar_gCH4s'].mean():.4f} g CH4/s"
)

print(
    f"Mean absolute diff   : "
    f"{result['KF_minus_4DVar_gCH4s'].abs().mean():.4f} g CH4/s"
)

print(
    f"Mean abs % diff      : "
    f"{result['KF_minus_4DVar_percent'].abs().mean():.4f}%"
)

print("\nSaved:")
print(OUT_FILE)

