#!/usr/bin/env python3

"""
===============================================================================
STILT + WRF-GHG + FLEXPART-WRF ENSEMBLE
Upper Hunter CH4 - January 2026

UPDATED VERSION
---------------

FLEXPART-WRF now uses the new 351-parameter Gibbs/MCMC inversion:

    13 mines x 27 days = 351 parameters

The FLEXPART posterior is:

    y = background + H @ scaling_factors

with posterior samples from:

    FLEXPART_MCMC_posterior_samples_Jan2026.npz

The daily mine-specific posterior means are read from:

    FLEXPART_MCMC_daily_factor_matrix_Jan2026.csv

For each day:

    S_FLEXPART,d =
        sum_m E0_m * s_m,d
        --------------------
             sum_m E0_m

where:

    E0_m = prior emission of mine m
    s_m,d = posterior scaling factor for mine m on day d

FLEXPART uncertainty
--------------------

For every posterior MCMC sample k:

    S_FLEXPART,d,k =
        sum_m E0_m * s_m,d,k
        ----------------------
             sum_m E0_m

The daily FLEXPART posterior standard deviation is:

    std(S_FLEXPART,d,k)

This preserves the covariance between mine scaling factors.

Three-model equal-weight ensemble
----------------------------------

    S_ensemble =
        (S_STILT + S_WRFGHG + S_FLEXPART) / 3

Model spread
------------

    spread =
        sqrt(
            [(S1-Sens)^2 +
             (S2-Sens)^2 +
             (S3-Sens)^2] / 3
        )

Ensemble posterior uncertainty
------------------------------

Assuming independent posterior errors between the three inverse models:

    sigma_ensemble =
        1/3 * sqrt(
            sigma_STILT^2 +
            sigma_WRFGHG^2 +
            sigma_FLEXPART^2
        )

IMPORTANT
---------

Model spread and posterior uncertainty are different quantities.

Model spread:
    disagreement between the three inverse models.

Posterior uncertainty:
    propagated statistical uncertainty assuming independent
    posterior errors between the three models.

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
# NEW FLEXPART-WRF GIBBS POSTERIOR
# =============================================================================

FLEXPART_BASE = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "H_matrix_flexpart/MCMC_13mine_daily_Jan2026"
)


# Daily posterior mean scaling factors
FLEXPART_DAILY_FILE = os.path.join(
    FLEXPART_BASE,
    "FLEXPART_MCMC_daily_factor_matrix_Jan2026.csv"
)


# Full 20,000 posterior samples
FLEXPART_SAMPLES_FILE = os.path.join(
    FLEXPART_BASE,
    "FLEXPART_MCMC_posterior_samples_Jan2026.npz"
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
    "STILT_WRFGHG_FLEXPART_Gibbs_ensemble_Jan2026"
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

print("\n" + "=" * 100)
print("STILT + WRF-GHG + FLEXPART-WRF GIBBS ENSEMBLE")
print("Upper Hunter CH4 - January 2026")
print("=" * 100)


# =============================================================================
# CHECK FILES
# =============================================================================

print("\n" + "=" * 100)
print("CHECKING INPUT FILES")
print("=" * 100)

input_files = {
    "STILT": STILT_FILE,
    "WRF-GHG": WRF_FILE,
    "FLEXPART daily posterior": FLEXPART_DAILY_FILE,
    "FLEXPART full MCMC": FLEXPART_SAMPLES_FILE,
    "Prior emissions": PRIOR_FILE,
}

for name, path in input_files.items():

    exists = os.path.exists(path)

    print(
        f"{name:35s}: "
        f"{'FOUND' if exists else 'MISSING'}"
    )

    if not exists:
        raise FileNotFoundError(
            f"\n{name} file not found:\n{path}"
        )


# =============================================================================
# READ PRIOR EMISSIONS
# =============================================================================

print("\n" + "=" * 100)
print("READING PRIOR MINE EMISSIONS")
print("=" * 100)

prior = pd.read_csv(
    PRIOR_FILE
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

    # Exact matching first
    rows = prior[
        prior[mine_col]
        .astype(str)
        .str.strip()
        .str.upper()
        == mine.upper()
    ]

    # Fallback only if exact match fails
    if len(rows) == 0:

        rows = prior[
            prior[mine_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.contains(
                mine.upper(),
                regex=False
            )
        ]

    if len(rows) == 0:

        raise ValueError(
            f"Could not find prior emission for {mine}"
        )

    if len(rows) > 1:

        print(
            f"WARNING: {mine} matched {len(rows)} rows. "
            f"Using first exact/fallback match."
        )

    mine_emissions[mine] = float(
        rows.iloc[0][emission_col]
    )


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
print("-" * 80)

for mine in MINES:

    print(
        f"{mine:>5s} : "
        f"{mine_emissions[mine]:10.3f} g/s   "
        f"weight = {weights[mine] * 100:8.3f}%"
    )

print("-" * 80)

print(
    f"TOTAL : "
    f"{total_prior_emission:10.3f} g/s"
)

print(
    f"\nSum weights = "
    f"{sum(weights.values()):.10f}"
)


# =============================================================================
# READ STILT
# =============================================================================

print("\n" + "=" * 100)
print("READING STILT POSTERIOR")
print("=" * 100)

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
    stilt[stilt_date_col],
    errors="coerce"
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
        *
        pd.to_numeric(
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
            *
            pd.to_numeric(
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

if len(stilt) > 0:

    print(
        f"STILT dates = "
        f"{stilt['date'].min().date()} "
        f"to "
        f"{stilt['date'].max().date()}"
    )


# =============================================================================
# READ WRF-GHG
# =============================================================================

print("\n" + "=" * 100)
print("READING WRF-GHG POSTERIOR")
print("=" * 100)

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
    wrf[wrf_date_col],
    errors="coerce"
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

wrf_mining["WRF_factor"] = pd.to_numeric(
    wrf_mining[wrf_factor_col],
    errors="coerce"
)

wrf_mining["WRF_std"] = pd.to_numeric(
    wrf_mining[wrf_std_col],
    errors="coerce"
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

if len(wrf_mining) > 0:

    print(
        f"WRF-GHG dates = "
        f"{wrf_mining['date'].min().date()} "
        f"to "
        f"{wrf_mining['date'].max().date()}"
    )


# =============================================================================
# READ NEW FLEXPART-WRF DAILY POSTERIOR
# =============================================================================

print("\n" + "=" * 100)
print("READING NEW FLEXPART-WRF GIBBS DAILY POSTERIOR")
print("=" * 100)

flex_daily = pd.read_csv(
    FLEXPART_DAILY_FILE
)

print("\nFLEXPART daily factor columns:")
print(
    list(flex_daily.columns)
)


# =============================================================================
# IDENTIFY DATE COLUMN
# =============================================================================

flex_date_col = find_column(
    flex_daily,
    [
        "date",
        "Date",
        "DATE"
    ]
)

flex_daily["date"] = pd.to_datetime(
    flex_daily[flex_date_col],
    errors="coerce"
).dt.normalize()


# =============================================================================
# CHECK MINE COLUMNS
# =============================================================================

print("\nChecking FLEXPART mine columns:")

for mine in MINES:

    if mine not in flex_daily.columns:

        raise KeyError(
            f"FLEXPART daily posterior does not contain "
            f"mine column {mine}"
        )

    print(
        f"  {mine}: OK"
    )


# =============================================================================
# FILTER DATES
# =============================================================================

flex_daily = flex_daily[
    (flex_daily["date"] >= pd.Timestamp(START_DATE))
    &
    (flex_daily["date"] <= pd.Timestamp(END_DATE))
].copy()


flex_daily = flex_daily.sort_values(
    "date"
).reset_index(
    drop=True
)


if len(flex_daily) != N_DAYS:

    raise RuntimeError(
        f"Expected {N_DAYS} FLEXPART days, "
        f"found {len(flex_daily)}"
    )


print(
    f"\nFLEXPART daily records = "
    f"{len(flex_daily)}"
)

print(
    f"FLEXPART dates = "
    f"{flex_daily['date'].min().date()} "
    f"to "
    f"{flex_daily['date'].max().date()}"
)


# =============================================================================
# CALCULATE FLEXPART AGGREGATE DAILY FACTOR
# =============================================================================

print("\n" + "=" * 100)
print("CALCULATING FLEXPART-WRF AGGREGATE POSTERIOR FACTOR")
print("=" * 100)


flex_daily["FLEXPART_factor"] = 0.0


for mine in MINES:

    flex_daily["FLEXPART_factor"] += (
        weights[mine]
        *
        pd.to_numeric(
            flex_daily[mine],
            errors="coerce"
        )
    )


print("\nFLEXPART-WRF daily aggregate posterior:")
print("-" * 70)

print(
    flex_daily[
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
# READ FULL FLEXPART GIBBS MCMC
# =============================================================================

print("\n" + "=" * 100)
print("READING FULL FLEXPART-WRF GIBBS POSTERIOR SAMPLES")
print("=" * 100)

npz = np.load(
    FLEXPART_SAMPLES_FILE
)

print(
    "\nArrays in NPZ:"
)

print(
    npz.files
)


if "samples" not in npz.files:

    raise RuntimeError(
        "Expected array named 'samples' in:\n"
        f"{FLEXPART_SAMPLES_FILE}"
    )


samples = np.asarray(
    npz["samples"],
    dtype=float
)


print(
    f"\nMCMC sample shape = "
    f"{samples.shape}"
)


if samples.ndim != 2:

    raise RuntimeError(
        "Expected FLEXPART posterior samples to have "
        "shape (n_samples, n_parameters)."
    )


n_samples, n_parameters = samples.shape


if n_parameters != N_PARAMETERS:

    raise RuntimeError(
        f"Expected {N_PARAMETERS} parameters, "
        f"found {n_parameters}"
    )


print(
    f"Number of posterior samples = "
    f"{n_samples:,}"
)

print(
    f"Number of parameters = "
    f"{n_parameters}"
)


# =============================================================================
# CALCULATE FLEXPART AGGREGATE POSTERIOR DISTRIBUTION
# =============================================================================
#
# Parameter ordering:
#
#   OC1_20260105 ... OC13_20260105
#   OC1_20260106 ... OC13_20260106
#   ...
#   OC1_20260131 ... OC13_20260131
#
# Thus each day occupies 13 consecutive parameters.
# =============================================================================

prior_emission_array = np.asarray(
    [
        mine_emissions[mine]
        for mine in MINES
    ],
    dtype=float
)


flex_mcmc_mean = []
flex_mcmc_median = []
flex_mcmc_std = []
flex_mcmc_lower95 = []
flex_mcmc_upper95 = []


# Store aggregate posterior samples for optional later analysis
aggregate_all_days = np.zeros(
    (n_samples, N_DAYS),
    dtype=float
)


print("\nCalculating daily aggregate posterior distributions...")


for day_index in range(N_DAYS):

    start = (
        day_index
        *
        N_MINES
    )

    end = (
        start
        +
        N_MINES
    )

    day_samples = samples[
        :,
        start:end
    ]

    # Weighted mine emissions for every posterior sample
    weighted_samples = (
        day_samples
        *
        prior_emission_array[None, :]
    )

    aggregate_samples = (
        weighted_samples.sum(
            axis=1
        )
        /
        total_prior_emission
    )

    aggregate_all_days[
        :,
        day_index
    ] = aggregate_samples

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
        "date":
            pd.date_range(
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
# CHECK DAILY MATRIX AGAINST FULL MCMC
# =============================================================================

print("\n" + "=" * 100)
print("CHECKING DAILY POSTERIOR MATRIX AGAINST FULL MCMC")
print("=" * 100)


difference = (
    flex_daily["FLEXPART_factor"].to_numpy()
    -
    flex_uncertainty["FLEXPART_mcmc_mean"].to_numpy()
)


print(
    "Maximum absolute difference between:"
)

print(
    "  daily factor matrix aggregate"
)

print(
    "and"
)

print(
    "  full MCMC aggregate posterior mean"
)

print(
    f"\nMaximum difference = "
    f"{np.max(np.abs(difference)):.10e}"
)


if np.max(np.abs(difference)) > 1e-5:

    print(
        "\nWARNING:"
    )

    print(
        "The daily posterior factor matrix and full MCMC "
        "posterior mean differ by more than 1e-5."
    )

    print(
        "Check parameter ordering and the daily factor matrix."
    )

else:

    print(
        "\nPASS:"
    )

    print(
        "Daily posterior factor matrix agrees with "
        "full MCMC aggregate posterior mean."
    )


# =============================================================================
# MERGE FLEXPART UNCERTAINTY
# =============================================================================

flex = pd.merge(
    flex_daily,
    flex_uncertainty,
    on="date",
    how="left"
)


# =============================================================================
# MERGE THREE MODELS
# =============================================================================

print("\n" + "=" * 100)
print("MERGING THREE INVERSE MODELS")
print("=" * 100)


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
        f"WARNING: Expected {N_DAYS} common days."
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
            -
            ensemble["ensemble_mean"]
        ) ** 2

        +

        (
            ensemble["WRF_factor"]
            -
            ensemble["ensemble_mean"]
        ) ** 2

        +

        (
            ensemble["FLEXPART_factor"]
            -
            ensemble["ensemble_mean"]
        ) ** 2
    )
    /
    3.0
)


# =============================================================================
# PROPAGATED ENSEMBLE POSTERIOR UNCERTAINTY
# =============================================================================

ensemble["ensemble_posterior_std"] = (
    1.0
    /
    3.0
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
# ENSEMBLE 95% INTERVAL
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
    .clip(
        lower=0.0
    )
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
# CONVERT FACTORS TO TOTAL MINE EMISSIONS
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
# FLEXPART EMISSION UNCERTAINTY
# =============================================================================

ensemble["FLEXPART_mine_emission_std_g_s"] = (
    total_prior_emission
    *
    ensemble["FLEXPART_std"]
)


ensemble["FLEXPART_mine_emission_lower95_g_s"] = (
    total_prior_emission
    *
    ensemble["FLEXPART_lower95"]
)


ensemble["FLEXPART_mine_emission_upper95_g_s"] = (
    total_prior_emission
    *
    ensemble["FLEXPART_upper95"]
)


# =============================================================================
# SAVE DAILY RESULTS
# =============================================================================

daily_csv = os.path.join(
    OUTPUT_DIR,
    "STILT_WRFGHG_FLEXPART_Gibbs_ensemble_daily_Jan2026.csv"
)


ensemble.to_csv(
    daily_csv,
    index=False,
    float_format="%.8f"
)


# =============================================================================
# PRINT DAILY RESULTS
# =============================================================================

print("\n" + "=" * 125)
print("DAILY THREE-MODEL ENSEMBLE")
print("=" * 125)


print(
    f"{'Date':<12}"
    f"{'STILT':>10}"
    f"{'WRF-GHG':>10}"
    f"{'FLEXPART':>12}"
    f"{'ENSEMBLE':>12}"
    f"{'ENS STD':>10}"
    f"{'SPREAD':>10}"
    f"{'ENS g/s':>12}"
)


print("-" * 125)


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

    "Mean_FLEXPART_posterior_std_g_s":
        ensemble[
            "FLEXPART_mine_emission_std_g_s"
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
    "STILT_WRFGHG_FLEXPART_Gibbs_ensemble_summary_Jan2026.csv"
)


summary_df.to_csv(
    summary_csv,
    index=False,
    float_format="%.8f"
)


# =============================================================================
# PRINT SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("THREE-MODEL ENSEMBLE SUMMARY")
print("=" * 100)


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
    f"  Mean FLEXPART posterior std  : "
    f"{summary['Mean_FLEXPART_posterior_std_g_s']:.2f} g/s"
)

print(
    f"  Mean ensemble posterior std  : "
    f"{summary['Mean_ensemble_posterior_std_g_s']:.2f} g/s"
)

print(
    f"  Mean ensemble model spread   : "
    f"{summary['Mean_ensemble_model_spread_g_s']:.2f} g/s"
)


# =============================================================================
# FIGURE 1
# DAILY THREE-MODEL FACTORS
# =============================================================================

fig, ax = plt.subplots(
    figsize=(15, 7)
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
    label="FLEXPART-WRF Gibbs"
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
    "STILT-WRF + WRF-GHG + FLEXPART-WRF Gibbs\n"
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
    figsize=(15, 6)
)


ax.bar(
    ensemble["date"],
    ensemble["ensemble_spread"],
    width=0.8
)


ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Model spread"
)

ax.set_title(
    "STILT-WRF + WRF-GHG + FLEXPART-WRF Gibbs\n"
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
    figsize=(15, 7)
)


ax.plot(
    ensemble["date"],
    ensemble["STILT_mine_emission_g_s"],
    marker="o",
    linewidth=1.5,
    label="STILT-WRF"
)


ax.plot(
    ensemble["date"],
    ensemble["WRF_mine_emission_g_s"],
    marker="s",
    linewidth=1.5,
    label="WRF-GHG"
)


ax.plot(
    ensemble["date"],
    ensemble["FLEXPART_mine_emission_g_s"],
    marker="^",
    linewidth=1.5,
    label="FLEXPART-WRF Gibbs"
)


ax.plot(
    ensemble["date"],
    ensemble["ensemble_mine_emission_g_s"],
    marker="D",
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
    "STILT-WRF + WRF-GHG + FLEXPART-WRF Gibbs\n"
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
# DIRECT THREE-MODEL COMPARISON
# =============================================================================

fig, ax = plt.subplots(
    figsize=(15, 7)
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
    label="FLEXPART-WRF Gibbs"
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
# SAVE FLEXPART DAILY POSTERIOR SUMMARY
# =============================================================================

flex_summary_csv = os.path.join(
    OUTPUT_DIR,
    "FLEXPART_Gibbs_daily_aggregate_posterior_Jan2026.csv"
)


flex[
    [
        "date",
        "FLEXPART_factor",
        "FLEXPART_mcmc_mean",
        "FLEXPART_mcmc_median",
        "FLEXPART_std",
        "FLEXPART_lower95",
        "FLEXPART_upper95",
    ]
].to_csv(
    flex_summary_csv,
    index=False,
    float_format="%.8f"
)


print(
    f"Saved: {flex_summary_csv}"
)


# =============================================================================
# FINAL
# =============================================================================

print("\n" + "=" * 100)
print("THREE-MODEL GIBBS ENSEMBLE COMPLETE")
print("=" * 100)

print(
    f"\nOutput directory:\n"
    f"{OUTPUT_DIR}"
)

print(
    f"\nDaily ensemble:\n"
    f"{daily_csv}"
)

print(
    f"\nSummary:\n"
    f"{summary_csv}"
)

print(
    "\nFLEXPART-WRF:"
)

print(
    "  Daily factors = new Gibbs posterior"
)

print(
    "  FLEXPART uncertainty = full 20,000-sample "
    "posterior distribution"
)

print(
    "  Source-source covariance is retained."
)

print(
    "\nEnsemble:"
)

print(
    "  Equal model weights = 1/3 each."
)

print(
    "  Ensemble posterior uncertainty assumes "
    "independent model posterior errors."
)

print(
    "  Model spread is reported separately."
)

print("\nDone.")

