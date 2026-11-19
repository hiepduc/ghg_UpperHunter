#!/usr/bin/env python3
"""
===============================================================================
STILT + WRF-GHG ENSEMBLE DAILY MINE EMISSION FACTOR
Upper Hunter CH4 - January 2026

Purpose
-------
Combine the two independent inverse-model estimates:

    1. STILT-WRF:
       13 individual mine scaling factors, converted to an
       emission-weighted aggregate mine scaling factor.

    2. WRF-GHG:
       Combined mining-sector daily scaling factor.

The ensemble mean is:

    S_ens = 0.5 * (S_STILT + S_WRFGHG)

The model ensemble spread is:

    spread = |S_STILT - S_WRFGHG| / 2

Posterior uncertainty propagated from the two model uncertainties
(assuming independent posterior errors) is:

    sigma_ens = 0.5 * sqrt(sigma_STILT^2 + sigma_WRFGHG^2)

IMPORTANT
---------
The ensemble spread is NOT the same thing as posterior uncertainty.
The spread represents disagreement between the two inverse models.

The propagated ensemble uncertainty assumes independent STILT and
WRF-GHG posterior errors. This assumption should be stated when
reporting the result.

Outputs
-------
1. CSV:
   STILT_WRFGHG_ensemble_daily_Jan2026.csv

2. CSV:
   STILT_WRFGHG_ensemble_summary_Jan2026.csv

3. PNG/PDF:
   01_ensemble_daily_factor_Jan2026
   02_ensemble_spread_Jan2026
   03_ensemble_mine_emission_Jan2026

===============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

START_DATE = "2026-01-05"
END_DATE   = "2026-01-31"

# -------------------------------------------------------------------------
# STILT posterior
# -------------------------------------------------------------------------
STILT_FILE = (
    "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter/"
    "stilt/Bayesian_daily_inversion_Jan2026/"
    "daily_scaling_factors_Jan2026.csv"
)

# -------------------------------------------------------------------------
# WRF-GHG posterior
# -------------------------------------------------------------------------
WRF_FILE = (
    "/mnt/scratch_lustre/duch/runnetzero/"
    "bayesian_daily_MH_5_31Jan2026/"
    "bayesian_daily_posterior.csv"
)

# -------------------------------------------------------------------------
# Prior mine emissions
# -------------------------------------------------------------------------
PRIOR_FILE = (
    "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter/"
    "UpperHuntercoalmines.csv"
)

# -------------------------------------------------------------------------
# Output directory
# -------------------------------------------------------------------------
OUTPUT_DIR = (
    "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter/"
    "stilt/Bayesian_daily_inversion_Jan2026/"
    "STILT_WRFGHG_ensemble_Jan2026"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_column(df, candidates):
    """
    Find the first matching column from a list of possible column names.
    """
    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError(
        f"Could not find any of these columns: {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )

# =============================================================================
# READ PRIOR MINE EMISSIONS
# =============================================================================

print("\n" + "=" * 80)
print("READING PRIOR MINE EMISSIONS")
print("=" * 80)

prior = pd.read_csv(PRIOR_FILE)

print("\nPrior file columns:")
print(list(prior.columns))


# -------------------------------------------------------------------------
# Your actual Upper Hunter prior file uses:
#
#   Facility_code_name  -> mine identifier
#   gCH4/s              -> prior CH4 emission
# -------------------------------------------------------------------------

mine_col = "Facility_code_name"
emission_col = "gCH4/s"


if mine_col not in prior.columns:
    raise KeyError(
        f"Required column '{mine_col}' not found.\n"
        f"Available columns:\n{list(prior.columns)}"
    )

if emission_col not in prior.columns:
    raise KeyError(
        f"Required column '{emission_col}' not found.\n"
        f"Available columns:\n{list(prior.columns)}"
    )


# =============================================================================
# CREATE PRIOR EMISSION DICTIONARY
# =============================================================================

mine_emissions = {}

for i in range(1, 14):

    mine = f"OC{i}"

    # Exact match first
    rows = prior[
        prior[mine_col]
        .astype(str)
        .str.strip()
        .str.upper()
        == mine
    ]

    # If exact match is not found, try partial match
    if len(rows) == 0:

        rows = prior[
            prior[mine_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.contains(mine, regex=False)
        ]

    if len(rows) == 0:
        raise ValueError(
            f"Could not find prior emission for {mine} "
            f"in column '{mine_col}'"
        )

    if len(rows) > 1:
        print(
            f"WARNING: {mine} has {len(rows)} matching rows. "
            f"Using the first row."
        )

    emission = float(
        rows.iloc[0][emission_col]
    )

    mine_emissions[mine] = emission


# =============================================================================
# CALCULATE TOTAL AND WEIGHTS
# =============================================================================

total_prior_emission = sum(
    mine_emissions.values()
)

weights = {
    mine: emission / total_prior_emission
    for mine, emission in mine_emissions.items()
}


print("\nPrior mine emissions:")
print("-" * 70)

for mine in mine_emissions:

    print(
        f"{mine:>4s} : "
        f"{mine_emissions[mine]:10.3f} g CH4/s   "
        f"weight = {weights[mine] * 100:6.2f}%"
    )

print("-" * 70)

print(
    f"TOTAL : "
    f"{total_prior_emission:10.3f} g CH4/s"
)

print(
    f"\nSum of weights = "
    f"{sum(weights.values()):.6f}"
)


# =============================================================================
# READ STILT POSTERIOR
# =============================================================================

print("\n" + "=" * 80)
print("READING STILT POSTERIOR")
print("=" * 80)

stilt = pd.read_csv(STILT_FILE)

print("\nSTILT columns:")
print(list(stilt.columns))


# -------------------------------------------------------------------------
# Date column
# -------------------------------------------------------------------------

stilt_date_col = find_column(
    stilt,
    ["date", "Date", "DATE"]
)

stilt["date"] = pd.to_datetime(
    stilt[stilt_date_col]
).dt.normalize()


# =============================================================================
# CALCULATE EMISSION-WEIGHTED STILT FACTOR
# =============================================================================

mine_factor_columns = []
mine_std_columns = []

for i in range(1, 14):

    mine = f"OC{i}"

    if mine not in stilt.columns:
        raise KeyError(
            f"STILT posterior does not contain {mine}"
        )

    mine_factor_columns.append(mine)

    std_col = f"{mine}_std"

    if std_col in stilt.columns:
        mine_std_columns.append(std_col)
    else:
        mine_std_columns.append(None)


# -------------------------------------------------------------------------
# Weighted STILT factor
# -------------------------------------------------------------------------

stilt["STILT_factor"] = 0.0

for mine in mine_factor_columns:

    stilt["STILT_factor"] += (
        weights[mine] * stilt[mine]
    )


# =============================================================================
# STILT COMBINED POSTERIOR UNCERTAINTY
# =============================================================================
#
# Approximation:
#
# Var(S_STILT_combined)
#       = sum(w_m^2 * sigma_m^2)
#
# This assumes individual mine posterior errors are independent.
#
# Exact calculation would require the full posterior covariance matrix.
# =============================================================================

stilt["STILT_std"] = 0.0

for i in range(1, 14):

    mine = f"OC{i}"
    std_col = f"{mine}_std"

    if std_col in stilt.columns:

        stilt["STILT_std"] += (
            weights[mine] ** 2 *
            stilt[std_col] ** 2
        )

    else:

        print(
            f"WARNING: {std_col} not found. "
            f"Cannot include {mine} uncertainty."
        )

stilt["STILT_std"] = np.sqrt(
    stilt["STILT_std"]
)


# =============================================================================
# FILTER STILT DATE RANGE
# =============================================================================

stilt = stilt[
    (stilt["date"] >= START_DATE) &
    (stilt["date"] <= END_DATE)
].copy()


print(
    f"\nSTILT dates: "
    f"{stilt['date'].min().date()} "
    f"to "
    f"{stilt['date'].max().date()}"
)

print(
    f"STILT number of daily records = {len(stilt)}"
)


# =============================================================================
# READ WRF-GHG POSTERIOR
# =============================================================================

print("\n" + "=" * 80)
print("READING WRF-GHG POSTERIOR")
print("=" * 80)

wrf = pd.read_csv(WRF_FILE)

print("\nWRF-GHG columns:")
print(list(wrf.columns))


# -------------------------------------------------------------------------
# Date
# -------------------------------------------------------------------------

wrf_date_col = find_column(
    wrf,
    ["date", "Date", "DATE"]
)

wrf["date"] = pd.to_datetime(
    wrf[wrf_date_col]
).dt.normalize()


# =============================================================================
# SELECT MINING SECTOR
# =============================================================================

sector_col = find_column(
    wrf,
    ["sector", "Sector", "SECTOR"]
)

wrf["sector_clean"] = (
    wrf[sector_col]
    .astype(str)
    .str.strip()
    .str.lower()
)


print("\nWRF-GHG sectors:")
print(
    wrf[sector_col]
    .drop_duplicates()
    .to_string(index=False)
)


# -------------------------------------------------------------------------
# Select mining sector
# -------------------------------------------------------------------------

possible_mining_names = [
    "mining",
    "mine",
    "mines",
    "mining_ch4",
    "coal mining",
    "coal_mining"
]

wrf_mining = wrf[
    wrf["sector_clean"].isin(possible_mining_names)
].copy()


if len(wrf_mining) == 0:

    # More flexible search
    wrf_mining = wrf[
        wrf["sector_clean"].str.contains(
            "min",
            na=False
        )
    ].copy()


if len(wrf_mining) == 0:

    raise ValueError(
        "Could not identify WRF-GHG mining sector.\n"
        f"Available sectors: "
        f"{wrf[sector_col].drop_duplicates().tolist()}"
    )


# =============================================================================
# WRF FACTOR + UNCERTAINTY
# =============================================================================

wrf_factor_col = find_column(
    wrf_mining,
    [
        "posterior_mean",
        "Posterior_mean",
        "posterior_mean_factor"
    ]
)

wrf_std_col = find_column(
    wrf_mining,
    [
        "posterior_std",
        "Posterior_std",
        "posterior_std_factor"
    ]
)


wrf_mining = wrf_mining[
    (wrf_mining["date"] >= START_DATE) &
    (wrf_mining["date"] <= END_DATE)
].copy()


wrf_mining["WRF_factor"] = (
    wrf_mining[wrf_factor_col].astype(float)
)

wrf_mining["WRF_std"] = (
    wrf_mining[wrf_std_col].astype(float)
)


# -------------------------------------------------------------------------
# Keep only required columns
# -------------------------------------------------------------------------

wrf_mining = wrf_mining[
    [
        "date",
        "WRF_factor",
        "WRF_std"
    ]
].copy()


print(
    f"\nWRF-GHG mining dates: "
    f"{wrf_mining['date'].min().date()} "
    f"to "
    f"{wrf_mining['date'].max().date()}"
)

print(
    f"WRF-GHG number of daily records = "
    f"{len(wrf_mining)}"
)


# =============================================================================
# MERGE STILT + WRF-GHG
# =============================================================================

print("\n" + "=" * 80)
print("CREATING ENSEMBLE")
print("=" * 80)

ensemble = pd.merge(
    stilt[
        [
            "date",
            "STILT_factor",
            "STILT_std"
        ]
    ],
    wrf_mining,
    on="date",
    how="inner"
)


ensemble = ensemble.sort_values(
    "date"
).reset_index(drop=True)


# =============================================================================
# ENSEMBLE CALCULATIONS
# =============================================================================

# -------------------------------------------------------------------------
# Equal-weight ensemble mean
# -------------------------------------------------------------------------

ensemble["ensemble_mean"] = (
    ensemble["STILT_factor"] +
    ensemble["WRF_factor"]
) / 2.0


# -------------------------------------------------------------------------
# Model ensemble spread
#
# For two members:
#
# spread = sqrt[
#    ((S1-Sens)^2 + (S2-Sens)^2)/2
# ]
#
# which is exactly:
#
# spread = |S1-S2|/2
# -------------------------------------------------------------------------

ensemble["ensemble_spread"] = (
    np.abs(
        ensemble["STILT_factor"] -
        ensemble["WRF_factor"]
    ) / 2.0
)


# -------------------------------------------------------------------------
# Propagated posterior uncertainty
#
# Assumes STILT and WRF posterior errors are independent.
# -------------------------------------------------------------------------

ensemble["ensemble_posterior_std"] = (
    0.5 *
    np.sqrt(
        ensemble["STILT_std"] ** 2 +
        ensemble["WRF_std"] ** 2
    )
)


# -------------------------------------------------------------------------
# Approximate 95% interval from propagated posterior uncertainty
# -------------------------------------------------------------------------

ensemble["ensemble_lower95"] = (
    ensemble["ensemble_mean"] -
    1.96 * ensemble["ensemble_posterior_std"]
)

ensemble["ensemble_upper95"] = (
    ensemble["ensemble_mean"] +
    1.96 * ensemble["ensemble_posterior_std"]
)


# -------------------------------------------------------------------------
# Optional physically constrained lower limit
#
# Emission scaling factors cannot normally be negative.
# Keep the unconstrained Gaussian interval above as the formal result,
# but also provide a zero-truncated reporting interval.
# -------------------------------------------------------------------------

ensemble["ensemble_lower95_nonnegative"] = (
    ensemble["ensemble_lower95"].clip(lower=0.0)
)


# -------------------------------------------------------------------------
# Difference
# -------------------------------------------------------------------------

ensemble["STILT_minus_WRF"] = (
    ensemble["STILT_factor"] -
    ensemble["WRF_factor"]
)


# =============================================================================
# CONVERT FACTORS TO TOTAL MINE EMISSIONS
# =============================================================================

ensemble["STILT_mine_emission_g_s"] = (
    total_prior_emission *
    ensemble["STILT_factor"]
)

ensemble["WRF_mine_emission_g_s"] = (
    total_prior_emission *
    ensemble["WRF_factor"]
)

ensemble["ensemble_mine_emission_g_s"] = (
    total_prior_emission *
    ensemble["ensemble_mean"]
)

ensemble["ensemble_mine_emission_std_g_s"] = (
    total_prior_emission *
    ensemble["ensemble_posterior_std"]
)

ensemble["ensemble_mine_emission_spread_g_s"] = (
    total_prior_emission *
    ensemble["ensemble_spread"]
)


# =============================================================================
# SAVE DAILY RESULTS
# =============================================================================

daily_csv = os.path.join(
    OUTPUT_DIR,
    "STILT_WRFGHG_ensemble_daily_Jan2026.csv"
)

ensemble.to_csv(
    daily_csv,
    index=False,
    float_format="%.8f"
)


# =============================================================================
# PRINT DAILY TABLE
# =============================================================================

print("\n" + "=" * 110)
print("DAILY ENSEMBLE RESULTS")
print("=" * 110)

print(
    f"{'Date':<12}"
    f"{'STILT':>10}"
    f"{'WRF-GHG':>10}"
    f"{'ENSEMBLE':>12}"
    f"{'STD':>10}"
    f"{'SPREAD':>10}"
    f"{'ENS g/s':>12}"
)

print("-" * 110)

for _, row in ensemble.iterrows():

    print(
        f"{row['date'].strftime('%Y-%m-%d'):<12}"
        f"{row['STILT_factor']:>10.4f}"
        f"{row['WRF_factor']:>10.4f}"
        f"{row['ensemble_mean']:>12.4f}"
        f"{row['ensemble_posterior_std']:>10.4f}"
        f"{row['ensemble_spread']:>10.4f}"
        f"{row['ensemble_mine_emission_g_s']:>12.2f}"
    )


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

summary = {
    "N": len(ensemble),

    "STILT_mean_factor":
        ensemble["STILT_factor"].mean(),

    "WRF_GHG_mean_factor":
        ensemble["WRF_factor"].mean(),

    "Ensemble_mean_factor":
        ensemble["ensemble_mean"].mean(),

    "STILT_median_factor":
        ensemble["STILT_factor"].median(),

    "WRF_GHG_median_factor":
        ensemble["WRF_factor"].median(),

    "Ensemble_median_factor":
        ensemble["ensemble_mean"].median(),

    "Mean_ensemble_posterior_std":
        ensemble["ensemble_posterior_std"].mean(),

    "Mean_model_spread":
        ensemble["ensemble_spread"].mean(),

    "Max_model_spread":
        ensemble["ensemble_spread"].max(),

    "STILT_WRF_correlation":
        ensemble["STILT_factor"].corr(
            ensemble["WRF_factor"]
        ),

    "STILT_minus_WRF_mean":
        ensemble["STILT_minus_WRF"].mean(),

    "STILT_minus_WRF_RMSE":
        np.sqrt(
            np.mean(
                ensemble["STILT_minus_WRF"] ** 2
            )
        ),

    "Prior_total_mine_emission_g_s":
        total_prior_emission,

    "Mean_STILT_mine_emission_g_s":
        ensemble["STILT_mine_emission_g_s"].mean(),

    "Mean_WRF_mine_emission_g_s":
        ensemble["WRF_mine_emission_g_s"].mean(),

    "Mean_ensemble_mine_emission_g_s":
        ensemble["ensemble_mine_emission_g_s"].mean(),

    "Mean_ensemble_mine_emission_std_g_s":
        ensemble["ensemble_mine_emission_std_g_s"].mean(),

    "Mean_ensemble_mine_emission_spread_g_s":
        ensemble["ensemble_mine_emission_spread_g_s"].mean()
}


summary_df = pd.DataFrame(
    [summary]
)


summary_csv = os.path.join(
    OUTPUT_DIR,
    "STILT_WRFGHG_ensemble_summary_Jan2026.csv"
)

summary_df.to_csv(
    summary_csv,
    index=False,
    float_format="%.8f"
)


# =============================================================================
# PRINT SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("ENSEMBLE SUMMARY")
print("=" * 80)

print(
    f"\nN days                         : "
    f"{summary['N']}"
)

print(
    f"STILT mean factor              : "
    f"{summary['STILT_mean_factor']:.5f}"
)

print(
    f"WRF-GHG mean factor            : "
    f"{summary['WRF_GHG_mean_factor']:.5f}"
)

print(
    f"ENSEMBLE mean factor           : "
    f"{summary['Ensemble_mean_factor']:.5f}"
)

print(
    f"\nSTILT median factor            : "
    f"{summary['STILT_median_factor']:.5f}"
)

print(
    f"WRF-GHG median factor          : "
    f"{summary['WRF_GHG_median_factor']:.5f}"
)

print(
    f"ENSEMBLE median factor         : "
    f"{summary['Ensemble_median_factor']:.5f}"
)

print(
    f"\nMean posterior std             : "
    f"{summary['Mean_ensemble_posterior_std']:.5f}"
)

print(
    f"Mean model spread              : "
    f"{summary['Mean_model_spread']:.5f}"
)

print(
    f"Maximum model spread           : "
    f"{summary['Max_model_spread']:.5f}"
)

print(
    f"\nSTILT-WRF correlation          : "
    f"{summary['STILT_WRF_correlation']:.5f}"
)

print(
    f"Mean STILT-WRF difference      : "
    f"{summary['STILT_minus_WRF_mean']:.5f}"
)

print(
    f"STILT-WRF difference RMSE      : "
    f"{summary['STILT_minus_WRF_RMSE']:.5f}"
)

print(
    f"\nPrior total mine emission      : "
    f"{total_prior_emission:.2f} g CH4/s"
)

print(
    f"Mean STILT mine emission       : "
    f"{summary['Mean_STILT_mine_emission_g_s']:.2f} g CH4/s"
)

print(
    f"Mean WRF-GHG mine emission     : "
    f"{summary['Mean_WRF_mine_emission_g_s']:.2f} g CH4/s"
)

print(
    f"Mean ENSEMBLE mine emission    : "
    f"{summary['Mean_ensemble_mine_emission_g_s']:.2f} g CH4/s"
)

print(
    f"Mean ensemble posterior std    : "
    f"{summary['Mean_ensemble_mine_emission_std_g_s']:.2f} g CH4/s"
)

print(
    f"Mean ensemble model spread     : "
    f"{summary['Mean_ensemble_mine_emission_spread_g_s']:.2f} g CH4/s"
)


# =============================================================================
# FIGURE 1 - DAILY FACTORS
# =============================================================================

fig, ax = plt.subplots(
    figsize=(13, 6)
)

ax.plot(
    ensemble["date"],
    ensemble["STILT_factor"],
    marker="o",
    linewidth=1.5,
    label="STILT"
)

ax.plot(
    ensemble["date"],
    ensemble["WRF_factor"],
    marker="s",
    linewidth=1.5,
    label="WRF-GHG"
)

ax.plot(
    ensemble["date"],
    ensemble["ensemble_mean"],
    marker="D",
    linewidth=2.5,
    label="Ensemble mean"
)

# Ensemble posterior uncertainty
ax.fill_between(
    ensemble["date"],
    ensemble["ensemble_mean"]
        - ensemble["ensemble_posterior_std"],
    ensemble["ensemble_mean"]
        + ensemble["ensemble_posterior_std"],
    alpha=0.20,
    label="Ensemble ±1σ"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Mine emission scaling factor"
)

ax.set_title(
    "STILT + WRF-GHG ensemble mine emission factor\n"
    "Upper Hunter, January 2026"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()

fig.autofmt_xdate()

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_ensemble_daily_factor_Jan2026.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_ensemble_daily_factor_Jan2026.pdf"
    ),
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# =============================================================================
# FIGURE 2 - MODEL SPREAD
# =============================================================================

fig, ax = plt.subplots(
    figsize=(13, 5)
)

ax.bar(
    ensemble["date"],
    ensemble["ensemble_spread"],
    width=0.8
)

ax.axhline(
    0,
    linewidth=0.8
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Model ensemble spread"
)

ax.set_title(
    "STILT–WRF-GHG model spread\n"
    "Upper Hunter, January 2026"
)

ax.grid(
    True,
    axis="y",
    alpha=0.3
)

fig.autofmt_xdate()

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_ensemble_spread_Jan2026.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_ensemble_spread_Jan2026.pdf"
    ),
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# =============================================================================
# FIGURE 3 - TOTAL MINE EMISSION
# =============================================================================

fig, ax = plt.subplots(
    figsize=(13, 6)
)

ax.plot(
    ensemble["date"],
    ensemble["STILT_mine_emission_g_s"],
    linewidth=1.5,
    label="STILT"
)

ax.plot(
    ensemble["date"],
    ensemble["WRF_mine_emission_g_s"],
    linewidth=1.5,
    label="WRF-GHG"
)

ax.plot(
    ensemble["date"],
    ensemble["ensemble_mine_emission_g_s"],
    linewidth=2.5,
    label="Ensemble mean"
)

ax.fill_between(
    ensemble["date"],
    ensemble["ensemble_mine_emission_g_s"]
        - ensemble["ensemble_mine_emission_spread_g_s"],
    ensemble["ensemble_mine_emission_g_s"]
        + ensemble["ensemble_mine_emission_spread_g_s"],
    alpha=0.20,
    label="Model spread"
)

ax.axhline(
    total_prior_emission,
    linestyle="--",
    linewidth=1.2,
    label="Prior"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Total mine CH$_4$ emission (g CH$_4$ s$^{-1}$)"
)

ax.set_title(
    "STILT + WRF-GHG ensemble mine emissions\n"
    "Upper Hunter, January 2026"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()

fig.autofmt_xdate()

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_ensemble_mine_emission_Jan2026.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_ensemble_mine_emission_Jan2026.pdf"
    ),
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# =============================================================================
# FINISH
# =============================================================================

print("\n" + "=" * 80)
print("FILES WRITTEN")
print("=" * 80)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print(f"\n1. {daily_csv}")
print(f"2. {summary_csv}")

print(
    "\n3. 01_ensemble_daily_factor_Jan2026.png/.pdf"
)

print(
    "4. 02_ensemble_spread_Jan2026.png/.pdf"
)

print(
    "5. 03_ensemble_mine_emission_Jan2026.png/.pdf"
)

print("\nDone.")

