#!/usr/bin/env python3
"""
===============================================================================
STILT + WRF-GHG + FLEXPART-WRF ENSEMBLE
Upper Hunter CH4 - January 2026

Purpose
-------
Combine three independent inverse-model estimates of the daily total
Upper Hunter coal-mine CH4 emission scaling factor:

    1. STILT-WRF
       13 mine-specific posterior scaling factors.
       Converted to an emission-weighted aggregate mine factor.

    2. WRF-GHG
       Combined mining-sector daily posterior scaling factor.

    3. FLEXPART-WRF
       13 mine-specific daily posterior scaling factors from the
       blocked Metropolis-Hastings inversion.
       Converted to an emission-weighted aggregate mine factor.

Three-model equal-weight ensemble:

    S_ensemble =
        (S_STILT + S_WRFGHG + S_FLEXPART) / 3

Model disagreement:
    Standard deviation of the three model estimates.

For three models:

    spread =
        sqrt(
            [(S1-Sens)^2 +
             (S2-Sens)^2 +
             (S3-Sens)^2] / 3
        )

FLEXPART-WRF uncertainty
-------------------------
The FLEXPART-WRF MCMC contains the full 351-dimensional posterior:

    13 mines x 27 days = 351 parameters

Therefore the uncertainty of the aggregate FLEXPART daily factor is
calculated directly from the MCMC samples, rather than assuming the
13 mine posterior errors are independent.

This is important because the FLEXPART H matrix has substantial
source-source correlation.

For each MCMC sample k:

    S_FLEX_k =
        sum_m E0_m * s_k,m
        ------------------
             sum_m E0_m

where s_k,m is the daily scaling factor for mine m.

The FLEXPART daily posterior standard deviation is then:

    std(S_FLEX) = std(S_FLEX_k)

Ensemble posterior uncertainty
------------------------------
STILT, WRF-GHG and FLEXPART posterior errors are assumed independent:

    sigma_ensemble =
        1/3 * sqrt(
            sigma_STILT^2 +
            sigma_WRFGHG^2 +
            sigma_FLEX^2
        )

This assumption should be stated when reporting the result.

IMPORTANT
---------
The model spread is NOT the same as posterior uncertainty.

    model spread:
        disagreement between the three inverse models

    posterior uncertainty:
        propagated statistical uncertainty assuming independent
        posterior errors

The two quantities should be reported separately.

Outputs
-------
Daily:
    STILT_WRFGHG_FLEXPART_ensemble_daily_Jan2026.csv

Summary:
    STILT_WRFGHG_FLEXPART_ensemble_summary_Jan2026.csv

Plots:
    01_three_model_ensemble_daily_factor_Jan2026.png/.pdf
    02_three_model_ensemble_spread_Jan2026.png/.pdf
    03_three_model_mine_emission_Jan2026.png/.pdf
    04_three_model_comparison_Jan2026.png/.pdf

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

N_MINES = 13
N_DAYS = 27
N_PARAMETERS = 351


# =============================================================================
# STILT POSTERIOR
# =============================================================================

STILT_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/bayesian/2026-01/"
    "Bayesian_daily_inversion_Jan2026/"
    "daily_scaling_factors_Jan2026.csv"
)


# =============================================================================
# WRF-GHG POSTERIOR
# =============================================================================

WRF_FILE = (
    "/mnt/scratch_lustre/duch/runnetzero/"
    "bayesian_daily_MH_5_31Jan2026/"
    "bayesian_daily_posterior.csv"
)


# =============================================================================
# FLEXPART-WRF POSTERIOR DAILY MEAN
# =============================================================================

FLEXPART_DAILY_FILE = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "H_matrix_flexpart/MCMC_13mine_daily_Jan2026/"
    "FLEXPART_MCMC_mine_daily_scaling_factors_Jan2026.csv"
)


# =============================================================================
# FLEXPART-WRF FULL MCMC SAMPLES
# =============================================================================

FLEXPART_SAMPLES_FILE = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "hmatrix/daily_20260105_20260131/"
    "bayesian_analytical/mcmc_blocked/"
    "mcmc_samples.npy"
)


# =============================================================================
# PRIOR MINE EMISSIONS
# =============================================================================

PRIOR_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/emission_location/"
    "UpperHuntercoalmines.csv"
)


# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/bayesian/2026-01/"
    "Bayesian_daily_inversion_Jan2026/"
    "STILT_WRFGHG_FLEXPART_ensemble_Jan2026"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =============================================================================
# HELPER
# =============================================================================

def find_column(df, candidates):

    for c in candidates:

        if c in df.columns:
            return c

    raise KeyError(
        f"Could not find any of:\n"
        f"{candidates}\n\n"
        f"Available columns:\n"
        f"{list(df.columns)}"
    )


# =============================================================================
# START
# =============================================================================

print("\n" + "=" * 90)
print("STILT + WRF-GHG + FLEXPART-WRF ENSEMBLE")
print("Upper Hunter CH4 - January 2026")
print("=" * 90)


# =============================================================================
# READ PRIOR EMISSIONS
# =============================================================================

print("\n" + "=" * 90)
print("READING PRIOR MINE EMISSIONS")
print("=" * 90)

prior = pd.read_csv(
    PRIOR_FILE
)

print("\nPrior columns:")
print(
    list(prior.columns)
)

mine_col = "Facility_code_name"
emission_col = "gCH4/s"

if mine_col not in prior.columns:
    raise KeyError(
        f"Missing {mine_col}"
    )

if emission_col not in prior.columns:
    raise KeyError(
        f"Missing {emission_col}"
    )


mine_emissions = {}

for mine in MINES:

    rows = prior[
        prior[mine_col]
        .astype(str)
        .str.strip()
        .str.upper()
        == mine
    ]

    if len(rows) == 0:

        rows = prior[
            prior[mine_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.contains(
                mine,
                regex=False
            )
        ]

    if len(rows) == 0:

        raise ValueError(
            f"Could not find prior emission for {mine}"
        )

    emission = float(
        rows.iloc[0][emission_col]
    )

    mine_emissions[mine] = emission


total_prior_emission = sum(
    mine_emissions.values()
)


weights = {
    mine:
        mine_emissions[mine]
        / total_prior_emission
    for mine in MINES
}


print("\nPrior mine emissions:")
print("-" * 75)

for mine in MINES:

    print(
        f"{mine:>5s} : "
        f"{mine_emissions[mine]:10.3f} g/s   "
        f"weight = {weights[mine] * 100:7.3f}%"
    )

print("-" * 75)

print(
    f"TOTAL : "
    f"{total_prior_emission:10.3f} g/s"
)

print(
    f"\nSum weights = "
    f"{sum(weights.values()):.8f}"
)


# =============================================================================
# READ STILT
# =============================================================================

print("\n" + "=" * 90)
print("READING STILT POSTERIOR")
print("=" * 90)

stilt = pd.read_csv(
    STILT_FILE
)

print("\nSTILT columns:")
print(
    list(stilt.columns)
)


stilt_date_col = find_column(
    stilt,
    [
        "date",
        "Date",
        "DATE"
    ]
)

stilt["date"] = pd.to_datetime(
    stilt[stilt_date_col]
).dt.normalize()


# =============================================================================
# STILT AGGREGATE FACTOR
# =============================================================================

stilt["STILT_factor"] = 0.0

for mine in MINES:

    if mine not in stilt.columns:

        raise KeyError(
            f"STILT posterior does not contain {mine}"
        )

    stilt["STILT_factor"] += (
        weights[mine]
        * pd.to_numeric(
            stilt[mine],
            errors="coerce"
        )
    )


# =============================================================================
# STILT UNCERTAINTY
# =============================================================================

stilt["STILT_variance"] = 0.0

for mine in MINES:

    std_col = f"{mine}_std"

    if std_col in stilt.columns:

        stilt["STILT_variance"] += (
            weights[mine] ** 2
            * pd.to_numeric(
                stilt[std_col],
                errors="coerce"
            ) ** 2
        )

    else:

        print(
            f"WARNING: {std_col} missing"
        )


stilt["STILT_std"] = np.sqrt(
    stilt["STILT_variance"]
)


# =============================================================================
# FILTER STILT
# =============================================================================

stilt = stilt[
    (stilt["date"] >= pd.Timestamp(START_DATE))
    &
    (stilt["date"] <= pd.Timestamp(END_DATE))
].copy()


print(
    f"\nSTILT records = {len(stilt)}"
)

print(
    f"STILT dates = "
    f"{stilt['date'].min().date()} "
    f"to "
    f"{stilt['date'].max().date()}"
)


# =============================================================================
# READ WRF-GHG
# =============================================================================

print("\n" + "=" * 90)
print("READING WRF-GHG POSTERIOR")
print("=" * 90)

wrf = pd.read_csv(
    WRF_FILE
)

print("\nWRF-GHG columns:")
print(
    list(wrf.columns)
)


wrf_date_col = find_column(
    wrf,
    [
        "date",
        "Date",
        "DATE"
    ]
)

wrf["date"] = pd.to_datetime(
    wrf[wrf_date_col]
).dt.normalize()


# =============================================================================
# IDENTIFY MINING SECTOR
# =============================================================================

sector_col = find_column(
    wrf,
    [
        "sector",
        "Sector",
        "SECTOR"
    ]
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


possible_mining_names = [
    "mining",
    "mine",
    "mines",
    "mining_ch4",
    "coal mining",
    "coal_mining",
]


wrf_mining = wrf[
    wrf["sector_clean"]
    .isin(possible_mining_names)
].copy()


if len(wrf_mining) == 0:

    wrf_mining = wrf[
        wrf["sector_clean"]
        .str.contains(
            "min",
            na=False
        )
    ].copy()


if len(wrf_mining) == 0:

    raise ValueError(
        "Could not identify WRF-GHG mining sector."
    )


# =============================================================================
# WRF FACTOR
# =============================================================================

wrf_factor_col = find_column(
    wrf_mining,
    [
        "posterior_mean",
        "Posterior_mean",
        "posterior_mean_factor",
    ]
)

wrf_std_col = find_column(
    wrf_mining,
    [
        "posterior_std",
        "Posterior_std",
        "posterior_std_factor",
    ]
)


wrf_mining["WRF_factor"] = (
    pd.to_numeric(
        wrf_mining[wrf_factor_col],
        errors="coerce"
    )
)

wrf_mining["WRF_std"] = (
    pd.to_numeric(
        wrf_mining[wrf_std_col],
        errors="coerce"
    )
)


wrf_mining = wrf_mining[
    [
        "date",
        "WRF_factor",
        "WRF_std"
    ]
].copy()


wrf_mining = wrf_mining[
    (wrf_mining["date"] >= pd.Timestamp(START_DATE))
    &
    (wrf_mining["date"] <= pd.Timestamp(END_DATE))
].copy()


print(
    f"\nWRF-GHG records = "
    f"{len(wrf_mining)}"
)

print(
    f"WRF-GHG dates = "
    f"{wrf_mining['date'].min().date()} "
    f"to "
    f"{wrf_mining['date'].max().date()}"
)


# =============================================================================
# READ FLEXPART DAILY POSTERIOR
# =============================================================================

print("\n" + "=" * 90)
print("READING FLEXPART-WRF POSTERIOR")
print("=" * 90)

flex = pd.read_csv(
    FLEXPART_DAILY_FILE
)

print("\nFLEXPART columns:")
print(
    list(flex.columns)
)


flex_date_col = find_column(
    flex,
    [
        "date",
        "Date",
        "DATE"
    ]
)

flex["date"] = pd.to_datetime(
    flex[flex_date_col]
).dt.normalize()


# =============================================================================
# CHECK FLEXPART MINE COLUMNS
# =============================================================================

for mine in MINES:

    if mine not in flex.columns:

        raise KeyError(
            f"FLEXPART posterior does not contain {mine}"
        )


# =============================================================================
# FILTER FLEXPART
# =============================================================================

flex = flex[
    (flex["date"] >= pd.Timestamp(START_DATE))
    &
    (flex["date"] <= pd.Timestamp(END_DATE))
].copy()


flex = flex.sort_values(
    "date"
).reset_index(
    drop=True
)


if len(flex) != N_DAYS:

    raise RuntimeError(
        f"Expected {N_DAYS} FLEXPART daily records, "
        f"found {len(flex)}"
    )


# =============================================================================
# CALCULATE FLEXPART EMISSION-WEIGHTED FACTOR
# =============================================================================

flex["FLEXPART_factor"] = 0.0

for mine in MINES:

    flex["FLEXPART_factor"] += (
        weights[mine]
        * pd.to_numeric(
            flex[mine],
            errors="coerce"
        )
    )


print(
    "\nFLEXPART-WRF daily factor:"
)

print(
    flex[
        [
            "date",
            "FLEXPART_factor"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =============================================================================
# READ FULL FLEXPART MCMC
# =============================================================================

print("\n" + "=" * 90)
print("READING FULL FLEXPART-WRF MCMC")
print("=" * 90)

samples = np.load(
    FLEXPART_SAMPLES_FILE
)

print(
    f"\nMCMC sample shape = "
    f"{samples.shape}"
)


if samples.ndim != 2:

    raise RuntimeError(
        "FLEXPART MCMC samples must be 2-dimensional."
    )


n_samples, n_params = samples.shape


if n_params != N_PARAMETERS:

    raise RuntimeError(
        f"Expected {N_PARAMETERS} parameters, "
        f"found {n_params}"
    )


print(
    f"Number of MCMC samples = "
    f"{n_samples}"
)

print(
    f"Number of parameters = "
    f"{n_params}"
)


# =============================================================================
# CALCULATE FLEXPART UNCERTAINTY DIRECTLY FROM MCMC
# =============================================================================
#
# Parameter ordering is:
#
#   OC1_20260105 ... OC13_20260105
#   OC1_20260106 ... OC13_20260106
#   ...
#   OC1_20260131 ... OC13_20260131
#
# Therefore for each day:
#
#   start = day_index * 13
#   end   = start + 13
#
# For every MCMC sample we calculate:
#
#   weighted_factor =
#       sum(E_m * scaling_m)
#       --------------------
#             sum(E_m)
#
# The standard deviation of these 6000 values is the
# FLEXPART posterior uncertainty of the aggregate factor.
# =============================================================================

flex_mcmc_mean = []
flex_mcmc_median = []
flex_mcmc_std = []
flex_mcmc_lower95 = []
flex_mcmc_upper95 = []


for day_index in range(N_DAYS):

    start = (
        day_index
        * N_MINES
    )

    end = (
        start
        + N_MINES
    )

    day_samples = samples[
        :,
        start:end
    ]

    weighted_samples = (
        day_samples
        * np.asarray(
            [
                mine_emissions[m]
                for m in MINES
            ]
        )[None, :]
    )

    aggregate_samples = (
        weighted_samples.sum(
            axis=1
        )
        / total_prior_emission
    )

    flex_mcmc_mean.append(
        np.mean(
            aggregate_samples
        )
    )

    flex_mcmc_median.append(
        np.median(
            aggregate_samples
        )
    )

    flex_mcmc_std.append(
        np.std(
            aggregate_samples,
            ddof=1
        )
    )

    flex_mcmc_lower95.append(
        np.percentile(
            aggregate_samples,
            2.5
        )
    )

    flex_mcmc_upper95.append(
        np.percentile(
            aggregate_samples,
            97.5
        )
    )


flex_uncertainty = pd.DataFrame(
    {
        "date": pd.date_range(
            START_DATE,
            END_DATE,
            freq="D"
        ),

        "FLEXPART_mcmc_mean":
            flex_mcmc_mean,

        "FLEXPART_mcmc_median":
            flex_mcmc_median,

        "FLEXPART_std":
            flex_mcmc_std,

        "FLEXPART_lower95":
            flex_mcmc_lower95,

        "FLEXPART_upper95":
            flex_mcmc_upper95,
    }
)


# =============================================================================
# MERGE FLEX DAILY MEAN WITH MCMC UNCERTAINTY
# =============================================================================

flex = pd.merge(
    flex,
    flex_uncertainty,
    on="date",
    how="left"
)


# =============================================================================
# CHECK DAILY MEAN AGAINST MCMC MEAN
# =============================================================================

difference = (
    flex["FLEXPART_factor"]
    - flex["FLEXPART_mcmc_mean"]
)

print(
    "\nMaximum difference between "
    "mcmc_daily_scaling weighted factor "
    "and full MCMC aggregate mean:"
)

print(
    f"{np.nanmax(np.abs(difference)):.8e}"
)


# =============================================================================
# MERGE THREE MODELS
# =============================================================================

print("\n" + "=" * 90)
print("MERGING THREE INVERSE MODELS")
print("=" * 90)

ensemble = pd.merge(
    stilt[
        [
            "date",
            "STILT_factor",
            "STILT_std",
        ]
    ],
    wrf_mining[
        [
            "date",
            "WRF_factor",
            "WRF_std",
        ]
    ],
    on="date",
    how="inner"
)


ensemble = pd.merge(
    ensemble,
    flex[
        [
            "date",
            "FLEXPART_factor",
            "FLEXPART_std",
            "FLEXPART_mcmc_mean",
            "FLEXPART_mcmc_median",
            "FLEXPART_lower95",
            "FLEXPART_upper95",
        ]
    ],
    on="date",
    how="inner"
)


ensemble = ensemble.sort_values(
    "date"
).reset_index(
    drop=True
)


print(
    f"\nNumber of common days = "
    f"{len(ensemble)}"
)


if len(ensemble) != N_DAYS:

    print(
        f"WARNING: Expected {N_DAYS} days."
    )


# =============================================================================
# THREE-MODEL EQUAL-WEIGHT ENSEMBLE
# =============================================================================

ensemble["ensemble_mean"] = (
    ensemble["STILT_factor"]
    +
    ensemble["WRF_factor"]
    +
    ensemble["FLEXPART_factor"]
) / 3.0


# =============================================================================
# MODEL SPREAD
# =============================================================================

ensemble["ensemble_spread"] = np.sqrt(
    (
        (
            ensemble["STILT_factor"]
            - ensemble["ensemble_mean"]
        ) ** 2
        +
        (
            ensemble["WRF_factor"]
            - ensemble["ensemble_mean"]
        ) ** 2
        +
        (
            ensemble["FLEXPART_factor"]
            - ensemble["ensemble_mean"]
        ) ** 2
    )
    / 3.0
)


# =============================================================================
# PROPAGATED POSTERIOR UNCERTAINTY
# =============================================================================
#
# Assumes independent posterior errors:
#
# sigma_ens =
#   1/3 sqrt(
#       sigma_STILT^2 +
#       sigma_WRFGHG^2 +
#       sigma_FLEXPART^2
#   )
# =============================================================================

ensemble["ensemble_posterior_std"] = (
    1.0 / 3.0
    *
    np.sqrt(
        ensemble["STILT_std"] ** 2
        +
        ensemble["WRF_std"] ** 2
        +
        ensemble["FLEXPART_std"] ** 2
    )
)


# =============================================================================
# ENSEMBLE 95% PROPAGATED INTERVAL
# =============================================================================

ensemble["ensemble_lower95"] = (
    ensemble["ensemble_mean"]
    -
    1.96
    *
    ensemble["ensemble_posterior_std"]
)


ensemble["ensemble_upper95"] = (
    ensemble["ensemble_mean"]
    +
    1.96
    *
    ensemble["ensemble_posterior_std"]
)


ensemble["ensemble_lower95_nonnegative"] = (
    ensemble["ensemble_lower95"]
    .clip(lower=0.0)
)


# =============================================================================
# MODEL DIFFERENCES
# =============================================================================

ensemble["STILT_minus_WRFGHG"] = (
    ensemble["STILT_factor"]
    -
    ensemble["WRF_factor"]
)

ensemble["STILT_minus_FLEXPART"] = (
    ensemble["STILT_factor"]
    -
    ensemble["FLEXPART_factor"]
)

ensemble["WRFGHG_minus_FLEXPART"] = (
    ensemble["WRF_factor"]
    -
    ensemble["FLEXPART_factor"]
)


# =============================================================================
# CONVERT TO TOTAL MINE EMISSIONS
# =============================================================================

ensemble["STILT_mine_emission_g_s"] = (
    total_prior_emission
    *
    ensemble["STILT_factor"]
)

ensemble["WRF_mine_emission_g_s"] = (
    total_prior_emission
    *
    ensemble["WRF_factor"]
)

ensemble["FLEXPART_mine_emission_g_s"] = (
    total_prior_emission
    *
    ensemble["FLEXPART_factor"]
)

ensemble["ensemble_mine_emission_g_s"] = (
    total_prior_emission
    *
    ensemble["ensemble_mean"]
)

ensemble["ensemble_mine_emission_std_g_s"] = (
    total_prior_emission
    *
    ensemble["ensemble_posterior_std"]
)

ensemble["ensemble_mine_emission_spread_g_s"] = (
    total_prior_emission
    *
    ensemble["ensemble_spread"]
)


# =============================================================================
# SAVE DAILY RESULTS
# =============================================================================

daily_csv = os.path.join(
    OUTPUT_DIR,
    "STILT_WRFGHG_FLEXPART_ensemble_daily_Jan2026.csv"
)

ensemble.to_csv(
    daily_csv,
    index=False,
    float_format="%.8f"
)


# =============================================================================
# PRINT DAILY RESULTS
# =============================================================================

print("\n" + "=" * 120)
print("DAILY THREE-MODEL ENSEMBLE")
print("=" * 120)

print(
    f"{'Date':<12}"
    f"{'STILT':>10}"
    f"{'WRF-GHG':>10}"
    f"{'FLEXPART':>12}"
    f"{'ENSEMBLE':>12}"
    f"{'STD':>10}"
    f"{'SPREAD':>10}"
    f"{'ENS g/s':>12}"
)

print("-" * 120)

for _, row in ensemble.iterrows():

    print(
        f"{row['date'].strftime('%Y-%m-%d'):<12}"
        f"{row['STILT_factor']:>10.4f}"
        f"{row['WRF_factor']:>10.4f}"
        f"{row['FLEXPART_factor']:>12.4f}"
        f"{row['ensemble_mean']:>12.4f}"
        f"{row['ensemble_posterior_std']:>10.4f}"
        f"{row['ensemble_spread']:>10.4f}"
        f"{row['ensemble_mine_emission_g_s']:>12.2f}"
    )


# =============================================================================
# SUMMARY
# =============================================================================

summary = {

    "N_days":
        len(ensemble),

    "STILT_mean_factor":
        ensemble["STILT_factor"].mean(),

    "WRFGHG_mean_factor":
        ensemble["WRF_factor"].mean(),

    "FLEXPART_mean_factor":
        ensemble["FLEXPART_factor"].mean(),

    "Ensemble_mean_factor":
        ensemble["ensemble_mean"].mean(),

    "STILT_median_factor":
        ensemble["STILT_factor"].median(),

    "WRFGHG_median_factor":
        ensemble["WRF_factor"].median(),

    "FLEXPART_median_factor":
        ensemble["FLEXPART_factor"].median(),

    "Ensemble_median_factor":
        ensemble["ensemble_mean"].median(),

    "Mean_STILT_std":
        ensemble["STILT_std"].mean(),

    "Mean_WRFGHG_std":
        ensemble["WRF_std"].mean(),

    "Mean_FLEXPART_std":
        ensemble["FLEXPART_std"].mean(),

    "Mean_ensemble_posterior_std":
        ensemble["ensemble_posterior_std"].mean(),

    "Mean_model_spread":
        ensemble["ensemble_spread"].mean(),

    "Maximum_model_spread":
        ensemble["ensemble_spread"].max(),

    "STILT_WRFGHG_correlation":
        ensemble[
            "STILT_factor"
        ].corr(
            ensemble["WRF_factor"]
        ),

    "STILT_FLEXPART_correlation":
        ensemble[
            "STILT_factor"
        ].corr(
            ensemble["FLEXPART_factor"]
        ),

    "WRFGHG_FLEXPART_correlation":
        ensemble[
            "WRF_factor"
        ].corr(
            ensemble["FLEXPART_factor"]
        ),

    "Prior_total_mine_emission_g_s":
        total_prior_emission,

    "Mean_STILT_mine_emission_g_s":
        ensemble[
            "STILT_mine_emission_g_s"
        ].mean(),

    "Mean_WRFGHG_mine_emission_g_s":
        ensemble[
            "WRF_mine_emission_g_s"
        ].mean(),

    "Mean_FLEXPART_mine_emission_g_s":
        ensemble[
            "FLEXPART_mine_emission_g_s"
        ].mean(),

    "Mean_ensemble_mine_emission_g_s":
        ensemble[
            "ensemble_mine_emission_g_s"
        ].mean(),

    "Mean_ensemble_posterior_std_g_s":
        ensemble[
            "ensemble_mine_emission_std_g_s"
        ].mean(),

    "Mean_ensemble_model_spread_g_s":
        ensemble[
            "ensemble_mine_emission_spread_g_s"
        ].mean(),
}


summary_df = pd.DataFrame(
    [summary]
)


summary_csv = os.path.join(
    OUTPUT_DIR,
    "STILT_WRFGHG_FLEXPART_ensemble_summary_Jan2026.csv"
)

summary_df.to_csv(
    summary_csv,
    index=False,
    float_format="%.8f"
)


# =============================================================================
# PRINT SUMMARY
# =============================================================================

print("\n" + "=" * 90)
print("THREE-MODEL ENSEMBLE SUMMARY")
print("=" * 90)

print(
    f"\nNumber of days                  : "
    f"{summary['N_days']}"
)

print(
    f"\nSTILT mean factor               : "
    f"{summary['STILT_mean_factor']:.6f}"
)

print(
    f"WRF-GHG mean factor             : "
    f"{summary['WRFGHG_mean_factor']:.6f}"
)

print(
    f"FLEXPART-WRF mean factor        : "
    f"{summary['FLEXPART_mean_factor']:.6f}"
)

print(
    f"ENSEMBLE mean factor            : "
    f"{summary['Ensemble_mean_factor']:.6f}"
)

print(
    f"\nSTILT median factor             : "
    f"{summary['STILT_median_factor']:.6f}"
)

print(
    f"WRF-GHG median factor           : "
    f"{summary['WRFGHG_median_factor']:.6f}"
)

print(
    f"FLEXPART-WRF median factor      : "
    f"{summary['FLEXPART_median_factor']:.6f}"
)

print(
    f"ENSEMBLE median factor          : "
    f"{summary['Ensemble_median_factor']:.6f}"
)

print(
    "\nPosterior uncertainty:"
)

print(
    f"  Mean STILT std                : "
    f"{summary['Mean_STILT_std']:.6f}"
)

print(
    f"  Mean WRF-GHG std              : "
    f"{summary['Mean_WRFGHG_std']:.6f}"
)

print(
    f"  Mean FLEXPART-WRF std         : "
    f"{summary['Mean_FLEXPART_std']:.6f}"
)

print(
    f"  Mean ensemble posterior std   : "
    f"{summary['Mean_ensemble_posterior_std']:.6f}"
)

print(
    "\nModel disagreement:"
)

print(
    f"  Mean model spread             : "
    f"{summary['Mean_model_spread']:.6f}"
)

print(
    f"  Maximum model spread          : "
    f"{summary['Maximum_model_spread']:.6f}"
)

print(
    "\nModel correlations:"
)

print(
    f"  STILT vs WRF-GHG              : "
    f"{summary['STILT_WRFGHG_correlation']:.6f}"
)

print(
    f"  STILT vs FLEXPART-WRF         : "
    f"{summary['STILT_FLEXPART_correlation']:.6f}"
)

print(
    f"  WRF-GHG vs FLEXPART-WRF      : "
    f"{summary['WRFGHG_FLEXPART_correlation']:.6f}"
)

print(
    "\nMine emissions:"
)

print(
    f"  Prior total                  : "
    f"{summary['Prior_total_mine_emission_g_s']:.2f} g/s"
)

print(
    f"  Mean STILT                   : "
    f"{summary['Mean_STILT_mine_emission_g_s']:.2f} g/s"
)

print(
    f"  Mean WRF-GHG                 : "
    f"{summary['Mean_WRFGHG_mine_emission_g_s']:.2f} g/s"
)

print(
    f"  Mean FLEXPART-WRF            : "
    f"{summary['Mean_FLEXPART_mine_emission_g_s']:.2f} g/s"
)

print(
    f"  Mean ensemble                : "
    f"{summary['Mean_ensemble_mine_emission_g_s']:.2f} g/s"
)

print(
    f"  Ensemble posterior std       : "
    f"{summary['Mean_ensemble_posterior_std_g_s']:.2f} g/s"
)

print(
    f"  Ensemble model spread        : "
    f"{summary['Mean_ensemble_model_spread_g_s']:.2f} g/s"
)


# =============================================================================
# FIGURE 1
# DAILY THREE-MODEL FACTORS
# =============================================================================

fig, ax = plt.subplots(
    figsize=(14, 6)
)

ax.plot(
    ensemble["date"],
    ensemble["STILT_factor"],
    marker="o",
    linewidth=1.5,
    label="STILT-WRF"
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
    ensemble["FLEXPART_factor"],
    marker="^",
    linewidth=1.5,
    label="FLEXPART-WRF"
)

ax.plot(
    ensemble["date"],
    ensemble["ensemble_mean"],
    marker="D",
    linewidth=2.5,
    label="Three-model ensemble"
)

ax.fill_between(
    ensemble["date"],
    ensemble["ensemble_mean"]
    -
    ensemble["ensemble_posterior_std"],
    ensemble["ensemble_mean"]
    +
    ensemble["ensemble_posterior_std"],
    alpha=0.20,
    label="Ensemble ±1σ"
)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.0,
    label="Prior factor = 1"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Mine emission scaling factor"
)

ax.set_title(
    "STILT-WRF + WRF-GHG + FLEXPART-WRF\n"
    "Upper Hunter mine emission scaling factor"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend(
    loc="best"
)

fig.autofmt_xdate()

fig.tight_layout()

outfile_png = os.path.join(
    OUTPUT_DIR,
    "01_three_model_ensemble_daily_factor_Jan2026.png"
)

outfile_pdf = os.path.join(
    OUTPUT_DIR,
    "01_three_model_ensemble_daily_factor_Jan2026.pdf"
)

fig.savefig(
    outfile_png,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    outfile_pdf,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print(
    f"\nSaved: {outfile_png}"
)

print(
    f"Saved: {outfile_pdf}"
)


# =============================================================================
# FIGURE 2
# MODEL SPREAD
# =============================================================================

fig, ax = plt.subplots(
    figsize=(14, 5)
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
    "Model spread"
)

ax.set_title(
    "STILT-WRF + WRF-GHG + FLEXPART-WRF\n"
    "Daily model disagreement"
)

ax.grid(
    True,
    axis="y",
    alpha=0.3
)

fig.autofmt_xdate()

fig.tight_layout()

outfile_png = os.path.join(
    OUTPUT_DIR,
    "02_three_model_ensemble_spread_Jan2026.png"
)

outfile_pdf = os.path.join(
    OUTPUT_DIR,
    "02_three_model_ensemble_spread_Jan2026.pdf"
)

fig.savefig(
    outfile_png,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    outfile_pdf,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print(
    f"Saved: {outfile_png}"
)

print(
    f"Saved: {outfile_pdf}"
)


# =============================================================================
# FIGURE 3
# TOTAL MINE EMISSION
# =============================================================================

fig, ax = plt.subplots(
    figsize=(14, 6)
)

ax.plot(
    ensemble["date"],
    ensemble["STILT_mine_emission_g_s"],
    linewidth=1.5,
    label="STILT-WRF"
)

ax.plot(
    ensemble["date"],
    ensemble["WRF_mine_emission_g_s"],
    linewidth=1.5,
    label="WRF-GHG"
)

ax.plot(
    ensemble["date"],
    ensemble["FLEXPART_mine_emission_g_s"],
    linewidth=1.5,
    label="FLEXPART-WRF"
)

ax.plot(
    ensemble["date"],
    ensemble["ensemble_mine_emission_g_s"],
    linewidth=2.5,
    label="Three-model ensemble"
)

ax.fill_between(
    ensemble["date"],
    ensemble["ensemble_mine_emission_g_s"]
    -
    ensemble["ensemble_mine_emission_spread_g_s"],
    ensemble["ensemble_mine_emission_g_s"]
    +
    ensemble["ensemble_mine_emission_spread_g_s"],
    alpha=0.20,
    label="Model spread"
)

ax.axhline(
    total_prior_emission,
    linestyle="--",
    linewidth=1.2,
    label="Prior mine emission"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Total mine CH$_4$ emission (g CH$_4$ s$^{-1}$)"
)

ax.set_title(
    "STILT-WRF + WRF-GHG + FLEXPART-WRF\n"
    "Upper Hunter total mine CH₄ emission"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend(
    loc="best"
)

fig.autofmt_xdate()

fig.tight_layout()

outfile_png = os.path.join(
    OUTPUT_DIR,
    "03_three_model_mine_emission_Jan2026.png"
)

outfile_pdf = os.path.join(
    OUTPUT_DIR,
    "03_three_model_mine_emission_Jan2026.pdf"
)

fig.savefig(
    outfile_png,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    outfile_pdf,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print(
    f"Saved: {outfile_png}"
)

print(
    f"Saved: {outfile_pdf}"
)


# =============================================================================
# FIGURE 4
# THREE-MODEL DIRECT COMPARISON
# =============================================================================

fig, ax = plt.subplots(
    figsize=(14, 6)
)

ax.plot(
    ensemble["date"],
    ensemble["STILT_factor"],
    marker="o",
    linewidth=1.5,
    label="STILT-WRF"
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
    ensemble["FLEXPART_factor"],
    marker="^",
    linewidth=1.5,
    label="FLEXPART-WRF"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Emission scaling factor"
)

ax.set_title(
    "Comparison of three inverse-model mine emission factors"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend(
    loc="best"
)

fig.autofmt_xdate()

fig.tight_layout()

outfile_png = os.path.join(
    OUTPUT_DIR,
    "04_three_model_comparison_Jan2026.png"
)

outfile_pdf = os.path.join(
    OUTPUT_DIR,
    "04_three_model_comparison_Jan2026.pdf"
)

fig.savefig(
    outfile_png,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    outfile_pdf,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print(
    f"Saved: {outfile_png}"
)

print(
    f"Saved: {outfile_pdf}"
)


# =============================================================================
# FINAL
# =============================================================================

print("\n" + "=" * 90)
print("THREE-MODEL ENSEMBLE COMPLETE")
print("=" * 90)

print(
    f"\nOutput directory:\n"
    f"{OUTPUT_DIR}"
)

print(
    f"\nDaily results:\n"
    f"{daily_csv}"
)

print(
    f"\nSummary:\n"
    f"{summary_csv}"
)

print(
    "\nImportant:"
)

print(
    "  FLEXPART-WRF uncertainty was calculated directly "
    "from the full MCMC samples."
)

print(
    "  Ensemble posterior uncertainty assumes "
    "independent STILT, WRF-GHG and FLEXPART-WRF "
    "posterior errors."
)

print(
    "  Model spread represents disagreement between "
    "the three inverse models and is separate from "
    "posterior uncertainty."
)

print("\nDone.")

