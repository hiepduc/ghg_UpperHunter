#!/usr/bin/env python3
"""
===============================================================================
Bayesian daily CH4 emission scaling factors from STILT H matrix
Upper Hunter - January 2026

13 mines x 31 days = 403 daily emission scaling factors.

Observation model:

    y_i - background = sum_j H_daily[i,j] * s_j + epsilon_i

where:

    y_i              = hourly Picarro CH4 observation [ppm]
    background       = 1.85 ppm
    H_daily          = STILT source-receptor sensitivity [ppm]
    s_j              = daily emission scaling factor
    epsilon          = observation/model error

For each day d and mine m:

    s[m,d] = 1

means the original inventory is correct.

Bayesian prior:

    s ~ N(s_prior, B)

with:

    s_prior = 1

and B containing:
    - prior variance for each daily factor
    - temporal correlation between adjacent days

Posterior:

    x_post = x_prior + K (y - H x_prior)

where

    K = B H^T (H B H^T + R)^-1

Equivalent information-form solution:

    C_post = (H^T R^-1 H + B^-1)^-1

    x_post = C_post [H^T R^-1 y_anom + B^-1 x_prior]

The code uses a numerically stable formulation.

===============================================================================
"""

import os
import re
import warnings

import numpy as np
import pandas as pd


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# ---------------------------------------------------------------------------
# Input H matrix
# ---------------------------------------------------------------------------

H_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/bayesian/2026-01/H-matrix/"
    "STILT_H_matrix_mines_Jan2026.csv"
)

# ---------------------------------------------------------------------------
# Picarro observation file
# ---------------------------------------------------------------------------

OBS_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/piccaro_obs/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUTPUT_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/bayesian/2026-01/"
    "Bayesian_daily_inversion_Jan2026"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# INVERSION SETTINGS
# =============================================================================

# Number of mines
N_MINES = 13

# Number of days
N_DAYS = 31

# Background CH4 [ppm]
BACKGROUND_PPM = 1.85

# ---------------------------------------------------------------------------
# Prior uncertainty of daily scaling factor.
#
# Example:
#
# PRIOR_STD = 0.50
#
# means:
#
#     s ~ N(1, 0.5^2)
#
# so before seeing observations, approximately 95% of the prior probability
# lies between 0 and 2.
#
# This is deliberately fairly broad.
# ---------------------------------------------------------------------------

PRIOR_STD = 0.50

# ---------------------------------------------------------------------------
# Temporal correlation of the prior.
#
# Adjacent days are correlated according to:
#
#     B[d1,d2] = sigma^2 * rho^abs(d1-d2)
#
# rho = 0 means independent daily factors.
# rho = 0.90 gives strong temporal smoothness.
#
# Start with 0.90.
# ---------------------------------------------------------------------------

TEMPORAL_RHO = 0.90

# ---------------------------------------------------------------------------
# Observation uncertainty [ppm].
#
# IMPORTANT:
#
# This is the TOTAL effective uncertainty assigned to each hourly observation,
# including representativeness/model/transport error, not simply the
# instrument precision.
#
# Start with 0.10 ppm as a reasonable configurable test value.
#
# You should later test 0.05, 0.10, 0.15, 0.20 etc.
# ---------------------------------------------------------------------------

OBS_ERROR_PPM = 0.10

# ---------------------------------------------------------------------------
# Physical constraint:
#
# If True, posterior factors are clipped to >= 0 AFTER the unconstrained
# Gaussian posterior calculation.
#
# IMPORTANT:
# This is NOT a mathematically exact non-negative Bayesian posterior.
# For the first diagnostic inversion keep it False.
#
# Later we can implement a proper truncated/positive MCMC or optimization.
# ---------------------------------------------------------------------------

ENFORCE_NONNEGATIVE = False


# =============================================================================
# SITE INFORMATION
# =============================================================================

SITE_NAMES = [
    "Singleton",
    "Muswellbrook",
    "Merriwa",
]


# Mapping from site names used in H matrix to Picarro columns
OBS_COLUMNS = {
    "Singleton": "Singleton cleaned",
    "Muswellbrook": "Muswellbrook cleaned",
    "Merriwa": "Merriwa cleaned",
}


# =============================================================================
# MINE NAMES
# =============================================================================

MINE_NAMES = [
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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_column(df, candidates):
    """
    Find the first matching column from a list of candidate names.
    Matching is case-insensitive after stripping whitespace.
    """

    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    return None


def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def rmse(obs, pred):
    return np.sqrt(np.mean((obs - pred) ** 2))


def mae(obs, pred):
    return np.mean(np.abs(obs - pred))


def bias(obs, pred):
    return np.mean(pred - obs)


def r_squared(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)

    if ss_tot == 0:
        return np.nan

    return 1.0 - ss_res / ss_tot


# =============================================================================
# LOAD H MATRIX
# =============================================================================

print_section("READING STILT H MATRIX")

print("H file:")
print(H_FILE)

Hdf = pd.read_csv(H_FILE)

print()
print("H matrix columns:")
print(list(Hdf.columns))

print()
print("H matrix shape:")
print(Hdf.shape)


# =============================================================================
# IDENTIFY REQUIRED H COLUMNS
# =============================================================================

missing_mines = [
    mine
    for mine in MINE_NAMES
    if mine not in Hdf.columns
]

if missing_mines:
    raise RuntimeError(
        "The following mine columns are missing from the H matrix:\n"
        + str(missing_mines)
    )


# =============================================================================
# IDENTIFY SITE COLUMN
# =============================================================================

site_col = find_column(
    Hdf,
    [
        "site",
        "Site",
        "receptor",
        "Receptor",
    ],
)

if site_col is None:
    raise RuntimeError(
        "Could not identify the site/receptor column in the H matrix."
    )

print()
print("Site column:", site_col)


# =============================================================================
# IDENTIFY DATETIME COLUMN
# =============================================================================

datetime_col = find_column(
    Hdf,
    [
        "datetime_aest",
        "Datetime_AEST",
        "datetime",
        "Datetime",
        "time_aest",
        "receptor_datetime_aest",
    ],
)

if datetime_col is None:
    raise RuntimeError(
        "Could not identify an AEST datetime column in the H matrix.\n"
        "Please inspect the CSV header."
    )

print("Datetime column:", datetime_col)


# =============================================================================
# PREPARE H MATRIX
# =============================================================================

Hdf[datetime_col] = pd.to_datetime(
    Hdf[datetime_col],
    errors="coerce",
)

Hdf[site_col] = Hdf[site_col].astype(str).str.strip()

# Keep only January 2026
Hdf = Hdf[
    (Hdf[datetime_col] >= "2026-01-01 00:00:00")
    &
    (Hdf[datetime_col] < "2026-02-01 00:00:00")
].copy()

print()
print("January H rows:", len(Hdf))

print()
print("H rows by site:")

print(
    Hdf.groupby(site_col).size()
)


# =============================================================================
# LOAD PICARRO DATA
# =============================================================================

print_section("READING PICARRO OBSERVATIONS")

print("Observation file:")
print(OBS_FILE)

obsdf = pd.read_csv(
    OBS_FILE,
    skipinitialspace=True,
)

print()
print("Observation columns:")
print(list(obsdf.columns))


# =============================================================================
# PARSE OBSERVATION DATETIME
# =============================================================================

obs_datetime_col = find_column(
    obsdf,
    [
        "Datetime",
        "datetime",
        "DATE TIME",
    ],
)

if obs_datetime_col is None:
    raise RuntimeError(
        "Could not find the Datetime column in the Picarro file."
    )

# Explicit format avoids pandas-version-dependent date inference.
# Expected examples: 1/01/2026 1:00, 13/01/2026 14:00
obsdf[obs_datetime_col] = pd.to_datetime(
    obsdf[obs_datetime_col].astype(str).str.strip(),
    format="%d/%m/%Y %H:%M",
    errors="coerce",
)

print()
print("Observation datetime column:", obs_datetime_col)
print(
    "Picarro timestamp parse failures:",
    obsdf[obs_datetime_col].isna().sum(),
)
print(
    "Picarro timestamp range:",
    obsdf[obs_datetime_col].min(),
    "to",
    obsdf[obs_datetime_col].max(),
)


# Analysis period: 5–31 January 2026.
# Exclude 1–4 January as spin-up days.
analysis_start = pd.Timestamp("2026-01-05 00:00:00")
analysis_end = pd.Timestamp("2026-02-01 00:00:00")

obsdf = obsdf[
    (obsdf[obs_datetime_col] >= analysis_start)
    &
    (obsdf[obs_datetime_col] < analysis_end)
].copy()

print()
print("Analysis period:", analysis_start, "to", analysis_end)
print("Observation rows (5–31 January):", len(obsdf))


# =============================================================================
# CREATE LONG-FORM OBSERVATION TABLE
# =============================================================================

print_section("MATCHING OBSERVATIONS TO STILT H ROWS")

obs_records = []

for site in SITE_NAMES:

    col = OBS_COLUMNS[site]

    if col not in obsdf.columns:
        raise RuntimeError(
            "Observation column not found for "
            + site
            + ": "
            + col
        )

    tmp = obsdf[
        [
            obs_datetime_col,
            col,
        ]
    ].copy()

    tmp.columns = [
        "datetime_aest",
        "observation",
    ]

    tmp["site"] = site

    # Convert to numeric
    tmp["observation"] = pd.to_numeric(
        tmp["observation"],
        errors="coerce",
    )

    obs_records.append(tmp)


obs_long = pd.concat(
    obs_records,
    ignore_index=True,
)

# Remove missing values
obs_long = obs_long[
    np.isfinite(obs_long["observation"])
].copy()

print()
print("Valid Picarro observations:")
print(len(obs_long))

print()
print("Valid observations by site:")
print(
    obs_long.groupby("site").size()
)


# =============================================================================
# NORMALIZE H DATETIME
# =============================================================================

Hdf["datetime_aest"] = Hdf[datetime_col]

# Use the same site naming convention
Hdf["site"] = Hdf[site_col].str.strip()


# =============================================================================
# MATCH H AND OBSERVATIONS
# =============================================================================

merged = pd.merge(
    Hdf,
    obs_long,
    on=["site", "datetime_aest"],
    how="inner",
)

print()
print("Matched H + observation rows:", len(merged))

print()
print("Matched rows by site:")
print(
    merged.groupby("site").size()
)


if len(merged) == 0:
    raise RuntimeError(
        "No H matrix rows matched the Picarro observations."
    )


# =============================================================================
# CHECK DUPLICATES
# =============================================================================

duplicates = merged.duplicated(
    subset=["site", "datetime_aest"],
    keep=False,
)

if duplicates.any():

    print()
    print("WARNING: duplicate site/time combinations detected.")

    print(
        merged.loc[
            duplicates,
            ["site", "datetime_aest"]
        ].head(20)
    )

    raise RuntimeError(
        "Duplicate H/observation site-time combinations found."
    )


# =============================================================================
# SORT
# =============================================================================

merged = merged.sort_values(
    ["datetime_aest", "site"]
).reset_index(drop=True)


# =============================================================================
# CREATE DAILY INDEX
# =============================================================================

merged["date"] = (
    merged["datetime_aest"]
    .dt.normalize()
)

merged["day_index"] = (
    merged["date"]
    - pd.Timestamp("2026-01-01")
).dt.days.astype(int)


# Check range
if merged["day_index"].min() < 0:
    raise RuntimeError("Observation before January 1 detected.")

if merged["day_index"].max() >= N_DAYS:
    raise RuntimeError("Observation outside January detected.")


# =============================================================================
# ORIGINAL H MATRIX
# =============================================================================

H_source = merged[MINE_NAMES].to_numpy(
    dtype=float
)

y = merged["observation"].to_numpy(
    dtype=float
)

n_obs = len(merged)


# =============================================================================
# BUILD DAILY H MATRIX
# =============================================================================

print_section("BUILDING DAILY H MATRIX")

n_parameters = N_MINES * N_DAYS

print("Number of observations :", n_obs)
print("Number of mines        :", N_MINES)
print("Number of days         :", N_DAYS)
print("Number of parameters   :", n_parameters)

H_daily = np.zeros(
    (n_obs, n_parameters),
    dtype=float,
)

for i in range(n_obs):

    day = int(
        merged.loc[i, "day_index"]
    )

    start = day * N_MINES
    end = start + N_MINES

    H_daily[i, start:end] = H_source[i, :]


print()
print("Daily H shape:")
print(H_daily.shape)


# =============================================================================
# PRIOR VECTOR
# =============================================================================

x_prior = np.ones(
    n_parameters,
    dtype=float,
)


# =============================================================================
# PRIOR COVARIANCE B
# =============================================================================

print_section("BUILDING BAYESIAN PRIOR")

sigma2 = PRIOR_STD ** 2

# Build covariance in mine/day order:
#
# parameter index:
#
#     day * 13 + mine
#
# Temporal covariance:
#
#     B(d1,d2) = sigma2 * rho^abs(d1-d2)
#
# No cross-correlation between different mines.

B = np.zeros(
    (n_parameters, n_parameters),
    dtype=float,
)

for mine in range(N_MINES):

    for d1 in range(N_DAYS):

        i = d1 * N_MINES + mine

        for d2 in range(N_DAYS):

            j = d2 * N_MINES + mine

            B[i, j] = (
                sigma2
                * TEMPORAL_RHO ** abs(d1 - d2)
            )


print("Prior mean:", 1.0)
print("Prior standard deviation:", PRIOR_STD)
print("Temporal correlation:", TEMPORAL_RHO)


# =============================================================================
# OBSERVATION COVARIANCE
# =============================================================================

print_section("OBSERVATION ERROR")

print(
    "Effective hourly observation/model error [ppm]:",
    OBS_ERROR_PPM,
)

R_diag = np.full(
    n_obs,
    OBS_ERROR_PPM ** 2,
)


# =============================================================================
# BAYESIAN POSTERIOR
# =============================================================================
#
# Instead of explicitly calculating:
#
#       inv(H^T R^-1 H + B^-1)
#
# we use the equivalent observation-space form:
#
#       S = H B H^T + R
#
#       K = B H^T S^-1
#
#       x_post = x_prior + K(y-H x_prior)
#
#       C_post = B - K H B
#
# This is more stable for this problem because n_obs is large but
# the matrices have exploitable structure.
#
# However, H B H^T is 3767 x 3767, which can still be expensive.
#
# We therefore use a Woodbury/information formulation based on
# Cholesky solves.
# =============================================================================

print_section("CALCULATING BAYESIAN POSTERIOR")

print("This may take some time for 3767 observations and 403 parameters.")


# We need:
#
# A = H^T R^-1 H + B^-1
#
# b = H^T R^-1 y_anom + B^-1 x_prior
#
# posterior covariance = inv(A)
#
# posterior mean = solve(A,b)
#
# Since R is diagonal, H^T R^-1 H is straightforward.

Rinv = 1.0 / R_diag

y_anom = y - BACKGROUND_PPM


print()
print("Calculating H^T R^-1 H ...")

Ht_Rinv_H = (
    H_daily.T
    @
    (H_daily * Rinv[:, None])
)


print("Calculating B^-1 ...")

# Solve B X = I rather than np.linalg.inv(B)
B_inv = np.linalg.solve(
    B,
    np.eye(n_parameters),
)


print("Constructing posterior precision matrix ...")

A = Ht_Rinv_H + B_inv


print("Calculating right-hand side ...")

b = (
    H_daily.T
    @
    (Rinv * y_anom)
    +
    B_inv @ x_prior
)


# Small numerical symmetrization
A = 0.5 * (A + A.T)


print("Solving posterior system ...")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    x_post = np.linalg.solve(
        A,
        b,
    )


# Posterior covariance
print("Calculating posterior covariance ...")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    C_post = np.linalg.inv(A)

C_post = 0.5 * (C_post + C_post.T)


# =============================================================================
# POSTERIOR STANDARD DEVIATIONS
# =============================================================================

posterior_std = np.sqrt(
    np.maximum(
        np.diag(C_post),
        0.0,
    )
)


# =============================================================================
# OPTIONAL NON-NEGATIVE CLIPPING
# =============================================================================

if ENFORCE_NONNEGATIVE:

    print()
    print(
        "WARNING: applying non-negative clipping to posterior mean."
    )

    x_post = np.maximum(
        x_post,
        0.0,
    )


# =============================================================================
# RESHAPE DAILY FACTORS
# =============================================================================

scaling = x_post.reshape(
    N_DAYS,
    N_MINES,
)

scaling_std = posterior_std.reshape(
    N_DAYS,
    N_MINES,
)


# =============================================================================
# CREATE DAILY OUTPUT TABLE
# =============================================================================

dates = pd.date_range(
    "2026-01-01",
    periods=N_DAYS,
    freq="D",
)

daily_rows = []

for d, date in enumerate(dates):

    row = {
        "date": date.strftime("%Y-%m-%d"),
    }

    for m, mine in enumerate(MINE_NAMES):

        row[mine] = scaling[d, m]

        row[mine + "_std"] = scaling_std[d, m]

        row[mine + "_lower95"] = (
            scaling[d, m]
            -
            1.96 * scaling_std[d, m]
        )

        row[mine + "_upper95"] = (
            scaling[d, m]
            +
            1.96 * scaling_std[d, m]
        )

    daily_rows.append(row)


daily_df = pd.DataFrame(
    daily_rows
)


# =============================================================================
# SAVE DAILY SCALING FACTORS
# =============================================================================

daily_file = os.path.join(
    OUTPUT_DIR,
    "daily_scaling_factors_Jan2026.csv",
)

daily_df.to_csv(
    daily_file,
    index=False,
)

print()
print("Daily scaling factors written:")
print(daily_file)


# =============================================================================
# PRIOR VS POSTERIOR UNCERTAINTY
# =============================================================================

uncertainty_rows = []

for d, date in enumerate(dates):

    for m, mine in enumerate(MINE_NAMES):

        idx = d * N_MINES + m

        prior_sd = PRIOR_STD
        post_sd = posterior_std[idx]

        if prior_sd > 0:
            reduction = (
                1.0
                -
                post_sd / prior_sd
            )
        else:
            reduction = np.nan

        uncertainty_rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "mine": mine,
                "prior_mean": 1.0,
                "prior_std": prior_sd,
                "posterior_mean": x_post[idx],
                "posterior_std": post_sd,
                "uncertainty_reduction": reduction,
            }
        )


uncertainty_df = pd.DataFrame(
    uncertainty_rows
)

uncertainty_file = os.path.join(
    OUTPUT_DIR,
    "posterior_uncertainty_Jan2026.csv",
)

uncertainty_df.to_csv(
    uncertainty_file,
    index=False,
)

print()
print("Posterior uncertainty written:")
print(uncertainty_file)


# =============================================================================
# RECONSTRUCT HOURLY OBSERVATIONS
# =============================================================================

print_section("RECONSTRUCTING HOURLY CH4")

model_anomaly = (
    H_daily @ x_post
)

model_ch4 = (
    BACKGROUND_PPM
    +
    model_anomaly
)

merged["CH4_observed_ppm"] = y

merged["CH4_model_prior_ppm"] = (
    BACKGROUND_PPM
    +
    H_daily @ x_prior
)

merged["CH4_model_posterior_ppm"] = (
    model_ch4
)

merged["residual_posterior_ppm"] = (
    model_ch4
    -
    y
)

merged["residual_prior_ppm"] = (
    merged["CH4_model_prior_ppm"]
    -
    y
)


# =============================================================================
# SAVE HOURLY RESULTS
# =============================================================================

hourly_file = os.path.join(
    OUTPUT_DIR,
    "hourly_posterior_reconstruction_Jan2026.csv",
)

merged.to_csv(
    hourly_file,
    index=False,
)

print()
print("Hourly reconstruction written:")
print(hourly_file)


# =============================================================================
# OVERALL METRICS
# =============================================================================

print_section("OVERALL INVERSION PERFORMANCE")

obs = y
pred_prior = merged[
    "CH4_model_prior_ppm"
].to_numpy()

pred_post = merged[
    "CH4_model_posterior_ppm"
].to_numpy()


print()
print("Number of observations:", len(obs))

print()
print("Prior model:")
print("  Bias [ppm] :", bias(obs, pred_prior))
print("  RMSE [ppm] :", rmse(obs, pred_prior))
print("  MAE [ppm]  :", mae(obs, pred_prior))
print("  R2         :", r_squared(obs, pred_prior))

print()
print("Posterior model:")
print("  Bias [ppm] :", bias(obs, pred_post))
print("  RMSE [ppm] :", rmse(obs, pred_post))
print("  MAE [ppm]  :", mae(obs, pred_post))
print("  R2         :", r_squared(obs, pred_post))


# =============================================================================
# SITE METRICS
# =============================================================================

metric_rows = []

for site in SITE_NAMES:

    mask = (
        merged["site"] == site
    )

    o = merged.loc[
        mask,
        "CH4_observed_ppm"
    ].to_numpy()

    pp = merged.loc[
        mask,
        "CH4_model_prior_ppm"
    ].to_numpy()

    po = merged.loc[
        mask,
        "CH4_model_posterior_ppm"
    ].to_numpy()

    metric_rows.append(
        {
            "site": site,
            "N": len(o),

            "prior_bias_ppm":
                bias(o, pp),

            "prior_rmse_ppm":
                rmse(o, pp),

            "prior_mae_ppm":
                mae(o, pp),

            "prior_R2":
                r_squared(o, pp),

            "posterior_bias_ppm":
                bias(o, po),

            "posterior_rmse_ppm":
                rmse(o, po),

            "posterior_mae_ppm":
                mae(o, po),

            "posterior_R2":
                r_squared(o, po),
        }
    )


metrics_df = pd.DataFrame(
    metric_rows
)

metrics_file = os.path.join(
    OUTPUT_DIR,
    "inversion_metrics_Jan2026.csv",
)

metrics_df.to_csv(
    metrics_file,
    index=False,
)

print()
print(metrics_df.to_string(index=False))


# =============================================================================
# POSTERIOR CORRELATION MATRIX
# =============================================================================

print_section("POSTERIOR SOURCE CORRELATION")

post_sd_safe = posterior_std.copy()

denominator = np.outer(
    post_sd_safe,
    post_sd_safe,
)

posterior_corr = np.zeros_like(
    C_post
)

valid = denominator > 0

posterior_corr[valid] = (
    C_post[valid]
    /
    denominator[valid]
)

np.fill_diagonal(
    posterior_corr,
    1.0,
)


# Save full correlation matrix
parameter_names = []

for d in range(N_DAYS):

    date_str = dates[d].strftime(
        "%Y%m%d"
    )

    for mine in MINE_NAMES:

        parameter_names.append(
            f"{mine}_{date_str}"
        )


posterior_corr_df = pd.DataFrame(
    posterior_corr,
    index=parameter_names,
    columns=parameter_names,
)

corr_file = os.path.join(
    OUTPUT_DIR,
    "posterior_correlation_Jan2026.csv",
)

posterior_corr_df.to_csv(
    corr_file
)

print()
print("Posterior correlation matrix written:")
print(corr_file)


# =============================================================================
# DAILY TOTAL MINE EMISSION SCALING
# =============================================================================
#
# Calculate emission-weighted total scaling factor.
#
# Inventory:
#
# OC1  553.08
# OC2  242.86
# ...
# OC13 798.08 g/s
#
# These are the annual-average inventory rates.
# =============================================================================

inventory_gs = np.array(
    [
        553.08,
        242.86,
        342.53,
        40.60,
        0.01,
        7.85,
        82.69,
        5.95,
        44.92,
        101.93,
        1.87,
        331.49,
        798.08,
    ],
    dtype=float,
)

if len(inventory_gs) != N_MINES:
    raise RuntimeError(
        "Inventory length does not match number of mines."
    )


total_inventory = np.sum(
    inventory_gs
)


daily_total_rows = []

for d, date in enumerate(dates):

    scaled_emission = np.sum(
        inventory_gs
        *
        scaling[d, :]
    )

    total_factor = (
        scaled_emission
        /
        total_inventory
    )

    # Uncertainty propagation assuming posterior covariance
    # between mines for the same day.
    idx = [
        d * N_MINES + m
        for m in range(N_MINES)
    ]

    cov_day = C_post[
        np.ix_(idx, idx)
    ]

    total_factor_variance = (
        inventory_gs
        @
        cov_day
        @
        inventory_gs
        /
        total_inventory ** 2
    )

    total_factor_std = np.sqrt(
        max(
            total_factor_variance,
            0.0,
        )
    )

    daily_total_rows.append(
        {
            "date":
                date.strftime("%Y-%m-%d"),

            "total_inventory_gCH4_s":
                total_inventory,

            "posterior_total_mine_emission_gCH4_s":
                scaled_emission,

            "posterior_total_scaling_factor":
                total_factor,

            "posterior_total_scaling_std":
                total_factor_std,

            "lower95":
                total_factor
                -
                1.96 * total_factor_std,

            "upper95":
                total_factor
                +
                1.96 * total_factor_std,
        }
    )


daily_total_df = pd.DataFrame(
    daily_total_rows
)

daily_total_file = os.path.join(
    OUTPUT_DIR,
    "daily_total_mine_emission_Jan2026.csv",
)

daily_total_df.to_csv(
    daily_total_file,
    index=False
)

print()
print("Daily total mine emission written:")
print(daily_total_file)


# =============================================================================
# DAILY DIAGNOSTIC SUMMARY
# =============================================================================

print_section("DAILY POSTERIOR SUMMARY")

summary_rows = []

for d, date in enumerate(dates):

    idx = slice(
        d * N_MINES,
        (d + 1) * N_MINES
    )

    vals = x_post[idx]
    sds = posterior_std[idx]

    summary_rows.append(
        {
            "date":
                date.strftime("%Y-%m-%d"),

            "mean_factor":
                np.mean(vals),

            "median_factor":
                np.median(vals),

            "min_factor":
                np.min(vals),

            "max_factor":
                np.max(vals),

            "mean_posterior_std":
                np.mean(sds),

            "number_factor_lt_0":
                np.sum(vals < 0),

            "number_factor_gt_2":
                np.sum(vals > 2),
        }
    )


daily_summary_df = pd.DataFrame(
    summary_rows
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "daily_posterior_summary_Jan2026.csv",
)

daily_summary_df.to_csv(
    summary_file,
    index=False
)

print()
print(
    daily_summary_df.to_string(
        index=False
    )
)


# =============================================================================
# MINE SUMMARY
# =============================================================================

print_section("MINE POSTERIOR SUMMARY")

mine_rows = []

for m, mine in enumerate(MINE_NAMES):

    idx = np.arange(
        m,
        n_parameters,
        N_MINES,
    )

    vals = x_post[idx]
    sds = posterior_std[idx]

    mine_rows.append(
        {
            "mine":
                mine,

            "inventory_gCH4_s":
                inventory_gs[m],

            "prior_mean":
                1.0,

            "prior_std":
                PRIOR_STD,

            "posterior_mean_Jan":
                np.mean(vals),

            "posterior_median_Jan":
                np.median(vals),

            "posterior_min":
                np.min(vals),

            "posterior_max":
                np.max(vals),

            "mean_posterior_std":
                np.mean(sds),

            "median_posterior_std":
                np.median(sds),

            "days_factor_lt_0":
                np.sum(vals < 0),

            "days_factor_gt_2":
                np.sum(vals > 2),
        }
    )


mine_summary_df = pd.DataFrame(
    mine_rows
)

mine_summary_file = os.path.join(
    OUTPUT_DIR,
    "mine_posterior_summary_Jan2026.csv",
)

mine_summary_df.to_csv(
    mine_summary_file,
    index=False
)

print()
print(
    mine_summary_df.to_string(
        index=False
    )
)


# =============================================================================
# SAVE NUMERICAL MATRICES
# =============================================================================

print_section("SAVING NUMERICAL MATRICES")

np.save(
    os.path.join(
        OUTPUT_DIR,
        "H_daily_Jan2026.npy",
    ),
    H_daily,
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "posterior_mean_Jan2026.npy",
    ),
    x_post,
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "posterior_covariance_Jan2026.npy",
    ),
    C_post,
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "prior_covariance_Jan2026.npy",
    ),
    B,
)


# =============================================================================
# WRITE RUN INFORMATION
# =============================================================================

info_file = os.path.join(
    OUTPUT_DIR,
    "inversion_configuration.txt",
)

with open(
    info_file,
    "w",
) as f:

    f.write(
        "Upper Hunter CH4 Bayesian daily inversion\n"
    )

    f.write(
        "==========================================\n\n"
    )

    f.write(
        f"H matrix: {H_FILE}\n"
    )

    f.write(
        f"Observation file: {OBS_FILE}\n"
    )

    f.write(
        f"Output directory: {OUTPUT_DIR}\n\n"
    )

    f.write(
        f"Period: 2026-01-01 to 2026-01-31\n"
    )

    f.write(
        f"Sites: {', '.join(SITE_NAMES)}\n"
    )

    f.write(
        f"Number of mines: {N_MINES}\n"
    )

    f.write(
        f"Number of days: {N_DAYS}\n"
    )

    f.write(
        f"Number of parameters: {n_parameters}\n"
    )

    f.write(
        f"Number of matched observations: {n_obs}\n\n"
    )

    f.write(
        f"Background CH4 [ppm]: {BACKGROUND_PPM}\n"
    )

    f.write(
        f"Prior mean: 1.0\n"
    )

    f.write(
        f"Prior standard deviation: {PRIOR_STD}\n"
    )

    f.write(
        f"Temporal rho: {TEMPORAL_RHO}\n"
    )

    f.write(
        f"Observation error [ppm]: {OBS_ERROR_PPM}\n"
    )

    f.write(
        f"Non-negative clipping: {ENFORCE_NONNEGATIVE}\n\n"
    )

    f.write(
        "Model:\n"
    )

    f.write(
        "    y - background = H_daily s + epsilon\n\n"
    )

    f.write(
        "Prior:\n"
    )

    f.write(
        "    s ~ N(1, B)\n\n"
    )

    f.write(
        "Posterior covariance:\n"
    )

    f.write(
        "    C_post = (H^T R^-1 H + B^-1)^-1\n\n"
    )

    f.write(
        "Posterior mean:\n"
    )

    f.write(
        "    s_post = C_post [H^T R^-1(y-background) + B^-1 s_prior]\n"
    )


# =============================================================================
# FINISHED
# =============================================================================

print_section("INVERSION COMPLETE")

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Main outputs:")
print(
    "  daily_scaling_factors_Jan2026.csv"
)

print(
    "  posterior_uncertainty_Jan2026.csv"
)

print(
    "  hourly_posterior_reconstruction_Jan2026.csv"
)

print(
    "  inversion_metrics_Jan2026.csv"
)

print(
    "  daily_total_mine_emission_Jan2026.csv"
)

print(
    "  mine_posterior_summary_Jan2026.csv"
)

print()
print("Done.")

