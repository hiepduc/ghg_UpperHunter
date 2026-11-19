#!/usr/bin/env python3
"""
FLEXPART-WRF Bayesian MCMC inversion
====================================

Upper Hunter CH4
13 mines x 28 daily emission scaling factors

Parameters:
    s[ mine, day ]

where:
    s = 1.0  -> prior emission
    s = 2.0  -> twice prior emission
    s = 0.5  -> half prior emission

Observation model:

    y_i = BACKGROUND + sum_j H_ij * s_j + epsilon_i

where:
    H_ij = FLEXPART prior contribution in ppmv for source/day j
    s_j  = daily emission scaling factor
    epsilon_i ~ N(0, sigma_obs^2)

IMPORTANT:
    The H matrix already contains the prior mine emissions.
    Therefore DO NOT multiply H by the prior emission again.

This script deliberately does NOT use the FLEXPART validation CSV.

It reads:
    1. H matrix CSV
    2. Picarro observation CSV

and aligns them directly using:
    receptor + AEST hourly timestamp

The H matrix contains:
    1589 FLEXPART receptor/time rows
    364 mine/day columns

The Picarro observations are converted to the same hourly AEST
time basis.

MCMC:
    Component-wise Gibbs sampling from the exact conditional
    truncated Gaussian posterior.

Outputs:
    H_matrix_flexpart/MCMC_13mine_daily_Feb2026/

Author: Upper Hunter FLEXPART inversion
"""

import os
import re
import sys
import math
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.special import ndtri, ndtr


# =============================================================================
# USER SETTINGS
# =============================================================================

BASE_DIR = Path(
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart"
)

H_FILE = (
    BASE_DIR
    / "H_matrix_flexpart"
    / "2026-02"
    / "H_flexpart_0_10m_13mine_Feb2026.csv"
)

OBS_FILE = (
    Path(
        "/home/duch/flexpart-wrf/emission_UH/"
        "RH_upperHunter/tapm_output/"
        "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
    )
)

OUTPUT_DIR = (
    BASE_DIR
    / "H_matrix_flexpart"
    / "2026-02"
    / "MCMC_13mine_daily_Feb2026"
)

# ---------------------------------------------------------------------
# Period
# ---------------------------------------------------------------------

START_DATE = pd.Timestamp("2026-02-01")
END_DATE = pd.Timestamp("2026-02-28")

# ---------------------------------------------------------------------
# Receptors
# ---------------------------------------------------------------------

RECEPTOR_OBS_COLUMN = {
    "singleton": "Singleton cleaned",
    "muswellbrook": "Muswellbrook cleaned",
    "merriwa": "Merriwa cleaned",
}

# ---------------------------------------------------------------------
# Methane background
# ---------------------------------------------------------------------

BACKGROUND_PPM = 1.85

# ---------------------------------------------------------------------
# Observation uncertainty
#
# This is deliberately configurable.
#
# sigma = 0.05 ppm means each hourly observation has a Gaussian
# uncertainty of 0.05 ppm.
# ---------------------------------------------------------------------

OBS_SIGMA_PPM = {
    "singleton": 0.05,
    "muswellbrook": 0.05,
    "merriwa": 0.05,
}

# ---------------------------------------------------------------------
# Prior on scaling factors
#
# s_j ~ Normal(1, PRIOR_SIGMA_FACTOR^2), truncated at s_j >= 0
#
# PRIOR_SIGMA_FACTOR = 1.0 is deliberately fairly weak.
#
# Try sensitivity tests later with:
#       0.25
#       0.50
#       1.00
#       2.00
# ---------------------------------------------------------------------

PRIOR_MEAN_FACTOR = 1.0
PRIOR_SIGMA_FACTOR = 1.0

# ---------------------------------------------------------------------
# MCMC settings
# ---------------------------------------------------------------------

N_CHAINS = 4

BURN_IN = 5000
N_SAMPLES = 5000
THIN = 5

RANDOM_SEED = 12345

# Save complete posterior samples?
SAVE_SAMPLES = True


# =============================================================================
# MINE / DATE DEFINITIONS
# =============================================================================

MINES = [f"OC{i}" for i in range(1, 14)]

DATES = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D",
)

EXPECTED_H_COLUMNS = [
    f"{date.strftime('%Y%m%d')}"
    for date in DATES
    for mine in MINES
]

# Actually H columns are OCx_YYYYMMDD
EXPECTED_H_COLUMNS = [
    f"{mine}_{date.strftime('%Y%m%d')}"
    for date in DATES
    for mine in MINES
]

N_PARAMS = len(EXPECTED_H_COLUMNS)


# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# PRINT HEADER
# =============================================================================

print()
print("=" * 100)
print("FLEXPART-WRF BAYESIAN MCMC INVERSION")
print("=" * 100)
print("Upper Hunter — 13 mines × 28 daily factors")
print()
print(
    f"Period: {START_DATE.strftime('%Y-%m-%d')} "
    f"to {END_DATE.strftime('%Y-%m-%d')}"
)
print(f"Number of mines : {len(MINES)}")
print(f"Number of days  : {len(DATES)}")
print(f"Number of parameters : {N_PARAMS}")
print()
print(f"Background CH4 : {BACKGROUND_PPM:.3f} ppm")
print(
    "Observation sigma : "
    + ", ".join(
        f"{k}={v:.4f}"
        for k, v in OBS_SIGMA_PPM.items()
    )
    + " ppm"
)
print(f"Prior mean factor : {PRIOR_MEAN_FACTOR:.3f}")
print(f"Prior sigma       : {PRIOR_SIGMA_FACTOR:.3f}")
print()
print(f"MCMC chains       : {N_CHAINS}")
print(f"Burn-in           : {BURN_IN}")
print(f"Posterior samples : {N_SAMPLES}")
print(f"Thin              : {THIN}")
print(f"Random seed       : {RANDOM_SEED}")
print()
print("=" * 100)


# =============================================================================
# CHECK FILES
# =============================================================================

print("CHECKING INPUT FILES")
print("=" * 100)

if not H_FILE.exists():
    raise FileNotFoundError(
        f"H matrix not found:\n{H_FILE}"
    )

if not OBS_FILE.exists():
    raise FileNotFoundError(
        f"Picarro observation file not found:\n{OBS_FILE}"
    )

print(f"H matrix:")
print(H_FILE)
print()
print("Picarro observations:")
print(OBS_FILE)


# =============================================================================
# READ H MATRIX
# =============================================================================

print()
print("=" * 100)
print("READING H MATRIX")
print("=" * 100)

H_df = pd.read_csv(H_FILE)

print(f"H dataframe shape: {H_df.shape}")

missing_cols = [
    c for c in EXPECTED_H_COLUMNS
    if c not in H_df.columns
]

if missing_cols:
    print()
    print("ERROR: Missing H columns:")
    for c in missing_cols:
        print("   ", c)

    raise RuntimeError(
        f"{len(missing_cols)} expected H columns are missing."
    )

# Use EXACT expected order
H_df = H_df.copy()

H_values_all = (
    H_df[EXPECTED_H_COLUMNS]
    .apply(pd.to_numeric, errors="coerce")
    .to_numpy(dtype=float)
)

if np.isnan(H_values_all).any():
    raise RuntimeError(
        "H matrix contains NaN values."
    )

print(
    f"H numerical matrix shape: "
    f"{H_values_all.shape}"
)

print(
    f"H min : {np.min(H_values_all):.6e}"
)
print(
    f"H max : {np.max(H_values_all):.6e}"
)
print(
    f"H non-zero values: "
    f"{np.count_nonzero(H_values_all):,}"
)


# =============================================================================
# IDENTIFY H METADATA
# =============================================================================

print()
print("=" * 100)
print("IDENTIFYING FLEXPART ROW METADATA")
print("=" * 100)

print("H metadata columns:")
print(list(H_df.columns[:10]))

# ---------------------------------------------------------------------
# The H builder should contain receptor/time information.
#
# We support the common names:
#   receptor
#   time_utc
#   time_aest
#
# If time_aest exists, use it directly.
# Otherwise convert time_utc -> AEST.
# ---------------------------------------------------------------------

if "receptor" not in H_df.columns:
    raise RuntimeError(
        "H matrix does not contain a 'receptor' column. "
        "The H builder should retain receptor metadata."
    )

H_receptor = (
    H_df["receptor"]
    .astype(str)
    .str.lower()
    .str.strip()
)

# ---------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# H matrix timestamp
#
# The H matrix was constructed with:
#
#     obs_time_aest
#
# This is the AEST timestamp represented by each FLEXPART receptor
# observation. Use it directly.
# ---------------------------------------------------------------------

if "obs_time_aest" not in H_df.columns:
    raise RuntimeError(
        "H matrix does not contain 'obs_time_aest'. "
        f"Available columns: {list(H_df.columns[:10])}"
    )

print(
    "Using H matrix 'obs_time_aest' directly."
)

H_time = pd.to_datetime(
    H_df["obs_time_aest"],
    errors="coerce",
)

if H_time.isna().any():

    n_bad = H_time.isna().sum()

    raise RuntimeError(
        f"{n_bad} H matrix obs_time_aest values "
        "could not be parsed."
    )

print(
    "H AEST time range:"
)

print(
    H_time.min(),
    "to",
    H_time.max(),
)

# =============================================================================
# NORMALIZE RECEPTOR NAMES
# =============================================================================

def normalize_receptor(x):

    x = str(x).lower().strip()

    if "singleton" in x:
        return "singleton"

    if "muswellbrook" in x:
        return "muswellbrook"

    if "merriwa" in x:
        return "merriwa"

    return x


H_receptor = H_receptor.map(
    normalize_receptor
)


unknown_receptors = sorted(
    set(H_receptor)
    - set(RECEPTOR_OBS_COLUMN)
)

if unknown_receptors:
    raise RuntimeError(
        "Unknown receptors in H matrix: "
        + str(unknown_receptors)
    )


# =============================================================================
# CREATE H HOURLY KEY
# =============================================================================

H_key = pd.DataFrame(
    {
        "receptor": H_receptor,
        "time_aest": H_time.dt.floor("h"),
    }
)

# Check duplicate H keys
duplicates = H_key.duplicated(
    subset=["receptor", "time_aest"],
    keep=False,
)

if duplicates.any():

    dup_df = H_key.loc[
        duplicates
    ].sort_values(
        ["receptor", "time_aest"]
    )

    print()
    print("ERROR: Duplicate FLEXPART receptor/time rows detected.")
    print(dup_df.head(20))

    raise RuntimeError(
        "H matrix contains duplicate receptor/time keys."
    )

print(
    f"FLEXPART rows: {len(H_df)}"
)

print(
    "FLEXPART receptors:"
)

for r in sorted(H_receptor.unique()):
    n = np.sum(H_receptor == r)
    print(f"   {r:15s} {n:6d}")


# =============================================================================
# READ PICARRO OBSERVATIONS DIRECTLY
# =============================================================================

print()
print("=" * 100)
print("READING PICARRO OBSERVATIONS DIRECTLY")
print("=" * 100)

obs = pd.read_csv(
    OBS_FILE
)

print(
    f"Observation dataframe shape: {obs.shape}"
)

print()
print("Observation columns:")
for c in obs.columns:
    print("   ", c)


# =============================================================================
# USE THE DEDICATED DATETIME COLUMN
# =============================================================================

if "Datetime" not in obs.columns:
    raise RuntimeError(
        "The Picarro CSV does not contain the expected "
        "'Datetime' column."
    )

print()
print(
    "Parsing Picarro Datetime as "
    "DD/MM/YYYY HH:MM (AEST)."
)

obs["time_aest"] = pd.to_datetime(
    obs["Datetime"],
    format="%d/%m/%Y %H:%M",
    errors="coerce",
)

if obs["time_aest"].isna().all():
    raise RuntimeError(
        "No Picarro Datetime values could be parsed."
    )

print(
    "Observation time range:"
)
print(
    obs["time_aest"].min(),
    "to",
    obs["time_aest"].max(),
)


# =============================================================================
# EXTRACT OBSERVATIONS INTO LONG FORMAT
# =============================================================================

print()
print("=" * 100)
print("BUILDING LONG-FORM OBSERVATION TABLE")
print("=" * 100)

obs_long_list = []

for receptor, column in RECEPTOR_OBS_COLUMN.items():

    if column not in obs.columns:
        raise RuntimeError(
            f"Observation column missing for {receptor}: "
            f"{column}"
        )

    tmp = pd.DataFrame(
        {
            "receptor": receptor,
            "time_aest": obs["time_aest"],
            "obs_ch4_ppm": pd.to_numeric(
                obs[column],
                errors="coerce",
            ),
        }
    )

    # Keep only actual observations
    tmp = tmp.dropna(
        subset=["time_aest", "obs_ch4_ppm"]
    )

    # Restrict to inversion period
    tmp = tmp[
        (tmp["time_aest"] >= START_DATE)
        &
        (
            tmp["time_aest"]
            < END_DATE + pd.Timedelta(days=1)
        )
    ]

    # Ensure hourly key
    tmp["time_aest"] = (
        tmp["time_aest"]
        .dt.floor("h")
    )

    obs_long_list.append(tmp)

obs_long = pd.concat(
    obs_long_list,
    ignore_index=True,
)

# ---------------------------------------------------------------------
# Check duplicate receptor/hour observations
# ---------------------------------------------------------------------

dup_obs = obs_long.duplicated(
    subset=["receptor", "time_aest"],
    keep=False,
)

if dup_obs.any():

    print()
    print(
        "WARNING: duplicate Picarro receptor/hour observations found."
    )

    print(
        obs_long.loc[dup_obs]
        .sort_values(
            ["receptor", "time_aest"]
        )
        .head(20)
    )

    # Average duplicates
    obs_long = (
        obs_long
        .groupby(
            ["receptor", "time_aest"],
            as_index=False,
        )["obs_ch4_ppm"]
        .mean()
    )

print()
print(
    f"Usable Picarro observations: "
    f"{len(obs_long)}"
)

for r in RECEPTOR_OBS_COLUMN:
    n = np.sum(
        obs_long["receptor"] == r
    )
    print(
        f"   {r:15s} {n:6d}"
    )


# =============================================================================
# ALIGN OBSERVATIONS DIRECTLY WITH H
# =============================================================================

print()
print("=" * 100)
print("ALIGNING PICARRO OBSERVATIONS WITH FLEXPART H")
print("=" * 100)

obs_key = obs_long[
    ["receptor", "time_aest"]
].copy()

obs_key["obs_ch4_ppm"] = obs_long[
    "obs_ch4_ppm"
].to_numpy()

# ---------------------------------------------------------------------
# Find matching H rows.
#
# We do NOT use the validation CSV.
# ---------------------------------------------------------------------

match = H_key.merge(
    obs_key,
    on=["receptor", "time_aest"],
    how="inner",
)

print(
    f"Matched FLEXPART/Picarro observations: "
    f"{len(match)}"
)

if len(match) == 0:
    print()
    print("ERROR: No observations matched the H matrix.")
    print()
    print("Example H times:")
    print(H_key.head(10))
    print()
    print("Example observation times:")
    print(obs_long.head(10))

    raise RuntimeError(
        "No FLEXPART/Picarro observations could be aligned."
    )


# =============================================================================
# GET ORIGINAL H ROW INDICES
# =============================================================================

# Merge back an explicit row index
H_key_indexed = H_key.copy()

H_key_indexed["H_row"] = np.arange(
    len(H_key_indexed)
)

match = H_key_indexed.merge(
    obs_key,
    on=["receptor", "time_aest"],
    how="inner",
)

H_row_indices = match["H_row"].to_numpy(
    dtype=int
)

H_obs = H_values_all[
    H_row_indices,
    :
]

y = match[
    "obs_ch4_ppm"
].to_numpy(
    dtype=float
)

receptors_obs = match[
    "receptor"
].to_numpy()
    
times_obs = match[
    "time_aest"
].to_numpy()


# =============================================================================
# OBSERVATION UNCERTAINTY VECTOR
# =============================================================================

sigma_obs = np.array(
    [
        OBS_SIGMA_PPM[r]
        for r in receptors_obs
    ],
    dtype=float,
)

if np.any(sigma_obs <= 0):
    raise RuntimeError(
        "Observation sigma must be > 0."
    )

invvar = 1.0 / (
    sigma_obs ** 2
)


# =============================================================================
# PRINT ALIGNMENT SUMMARY
# =============================================================================

print()
print(
    f"Final likelihood observations: "
    f"{len(y)}"
)

for r in RECEPTOR_OBS_COLUMN:

    n = np.sum(
        receptors_obs == r
    )

    if n:
        print(
            f"   {r:15s} {n:6d}"
        )

print()
print(
    "Matched time range:"
)

print(
    pd.Timestamp(times_obs.min()),
    "to",
    pd.Timestamp(times_obs.max()),
)


# =============================================================================
# COMPUTE PRIOR MODEL
# =============================================================================

print()
print("=" * 100)
print("CHECKING PRIOR MODEL")
print("=" * 100)

s_prior = np.ones(
    N_PARAMS,
    dtype=float,
)

prior_enhancement = (
    H_obs @ s_prior
)

prior_total = (
    BACKGROUND_PPM
    + prior_enhancement
)

print(
    f"Prior enhancement range: "
    f"{prior_enhancement.min():.6f} "
    f"to "
    f"{prior_enhancement.max():.6f} ppm"
)

print(
    f"Prior total range: "
    f"{prior_total.min():.6f} "
    f"to "
    f"{prior_total.max():.6f} ppm"
)

print(
    f"Observed range: "
    f"{y.min():.6f} "
    f"to "
    f"{y.max():.6f} ppm"
)


# =============================================================================
# SOURCE-DAY SENSITIVITY
# =============================================================================

print()
print("=" * 100)
print("SOURCE-DAY SENSITIVITY")
print("=" * 100)

nonzero_count = np.count_nonzero(
    H_obs,
    axis=0,
)

sensitivity_sum = np.sum(
    np.abs(H_obs),
    axis=0,
)

for j, name in enumerate(EXPECTED_H_COLUMNS):

    if nonzero_count[j] == 0:
        print(
            f"WARNING: {name} has ZERO sensitivity "
            f"to all matched observations."
        )


# =============================================================================
# PRE-COMPUTE NONZERO ROWS
# =============================================================================

# This is important for performance.
#
# H is sparse. Instead of scanning all observations for every parameter,
# only use rows where H[:,j] != 0.

nonzero_rows = []

for j in range(N_PARAMS):

    idx = np.flatnonzero(
        H_obs[:, j]
    )

    nonzero_rows.append(idx)


print()
print(
    f"Total non-zero H elements in likelihood: "
    f"{np.count_nonzero(H_obs):,}"
)


# =============================================================================
# TRUNCATED NORMAL SAMPLER
# =============================================================================

def sample_normal_truncated_zero(
    rng,
    mu,
    sd,
):
    """
    Draw X ~ N(mu, sd^2), conditional on X >= 0.

    Uses inverse survival-function sampling.

    For Z ~ N(0,1):

        P(Z > a)

    is used to avoid numerical problems when the truncation
    point is far into the upper tail.
    """

    if not np.isfinite(mu):
        raise RuntimeError(
            f"Non-finite conditional mean: {mu}"
        )

    if not np.isfinite(sd) or sd <= 0:

        # Degenerate case
        return max(
            0.0,
            mu,
        )

    a = -mu / sd

    # Survival probability at a
    sf_a = ndtr(-a)

    # Extremely unlikely but protect against underflow
    if sf_a <= 0:

        # Very deep tail.
        # Exponential rejection-style approximation.
        x = 0.0

        for _ in range(10000):

            x = rng.exponential(
                scale=sd
            )

            # density ratio
            z = (
                (x - mu) / sd
            )

            # exponential proposal ratio is sufficient
            # for a stable positive draw here.
            if np.isfinite(z):
                return x

        return max(
            0.0,
            mu,
        )

    u = rng.random()

    q = u * sf_a

    # Protect against q == 0
    q = max(
        q,
        np.finfo(float).tiny,
    )

    z = -ndtri(q)

    x = mu + sd * z

    # Numerical protection
    if x < 0:
        x = 0.0

    return x


# =============================================================================
# LOG POSTERIOR
# =============================================================================

prior_precision = (
    1.0
    / PRIOR_SIGMA_FACTOR ** 2
)


def log_posterior(
    s,
):
    """
    Log posterior up to an additive constant.
    """

    prediction = (
        BACKGROUND_PPM
        + H_obs @ s
    )

    residual = (
        y - prediction
    )

    log_likelihood = -0.5 * np.sum(
        residual ** 2
        * invvar
    )

    log_prior = -0.5 * np.sum(
        (
            (s - PRIOR_MEAN_FACTOR)
            / PRIOR_SIGMA_FACTOR
        ) ** 2
    )

    return (
        log_likelihood
        + log_prior
    )


# =============================================================================
# ONE GIBBS SWEEP
# =============================================================================

def gibbs_sweep(
    s,
    prediction_enhancement,
    rng,
):
    """
    One complete Gibbs sweep.

    prediction_enhancement = H_obs @ s

    Each factor has an exact conditional Gaussian posterior,
    truncated at zero.
    """

    for j in range(N_PARAMS):

        idx = nonzero_rows[j]

        old_value = s[j]

        # -------------------------------------------------------------
        # If this source/day has no sensitivity, its posterior is
        # simply its prior.
        # -------------------------------------------------------------

        if len(idx) == 0:

            mu = PRIOR_MEAN_FACTOR
            sd = PRIOR_SIGMA_FACTOR

            new_value = sample_normal_truncated_zero(
                rng,
                mu,
                sd,
            )

            s[j] = new_value

            continue

        hj = H_obs[
            idx,
            j
        ]

        # -------------------------------------------------------------
        # Prediction excluding current parameter
        # -------------------------------------------------------------

        residual_without_j = (
            y[idx]
            - BACKGROUND_PPM
            - prediction_enhancement[idx]
            + hj * old_value
        )

        # -------------------------------------------------------------
        # Conditional precision
        # -------------------------------------------------------------

        likelihood_precision = np.sum(
            hj ** 2
            * invvar[idx]
        )

        conditional_precision = (
            likelihood_precision
            + prior_precision
        )

        conditional_sd = (
            1.0
            / np.sqrt(
                conditional_precision
            )
        )

        # -------------------------------------------------------------
        # Conditional mean
        # -------------------------------------------------------------

        likelihood_linear = np.sum(
            hj
            * residual_without_j
            * invvar[idx]
        )

        conditional_linear = (
            likelihood_linear
            + prior_precision
            * PRIOR_MEAN_FACTOR
        )

        conditional_mu = (
            conditional_linear
            / conditional_precision
        )

        # -------------------------------------------------------------
        # Draw new value
        # -------------------------------------------------------------

        new_value = sample_normal_truncated_zero(
            rng,
            conditional_mu,
            conditional_sd,
        )

        # -------------------------------------------------------------
        # Update prediction efficiently
        # -------------------------------------------------------------

        delta = (
            new_value
            - old_value
        )

        if delta != 0:

            prediction_enhancement[idx] += (
                hj * delta
            )

        s[j] = new_value

    return s


# =============================================================================
# INITIAL STATES
# =============================================================================

def make_initial_state(
    chain_id,
    rng,
):
    """
    Different starting points for the chains.
    """

    if chain_id == 0:

        s = np.ones(
            N_PARAMS,
            dtype=float,
        )

    elif chain_id == 1:

        s = rng.normal(
            PRIOR_MEAN_FACTOR,
            0.25,
            size=N_PARAMS,
        )

        s = np.clip(
            s,
            0.01,
            None,
        )

    elif chain_id == 2:

        s = rng.normal(
            0.5,
            0.25,
            size=N_PARAMS,
        )

        s = np.clip(
            s,
            0.01,
            None,
        )

    else:

        s = rng.normal(
            1.5,
            0.50,
            size=N_PARAMS,
        )

        s = np.clip(
            s,
            0.01,
            None,
        )

    return s


# =============================================================================
# RUN ONE CHAIN
# =============================================================================

def run_chain(
    chain_id,
):
    """
    Run one Gibbs chain.
    """

    rng = np.random.default_rng(
        RANDOM_SEED + chain_id * 100000
    )

    s = make_initial_state(
        chain_id,
        rng,
    )

    prediction_enhancement = (
        H_obs @ s
    )

    total_iterations = (
        BURN_IN
        + N_SAMPLES * THIN
    )

    saved = []

    logpost_trace = []

    print()
    print(
        f"Starting chain {chain_id + 1}/{N_CHAINS}"
    )

    for iteration in range(
        total_iterations
    ):

        s = gibbs_sweep(
            s,
            prediction_enhancement,
            rng,
        )

        # -------------------------------------------------------------
        # Save posterior sample
        # -------------------------------------------------------------

        if (
            iteration >= BURN_IN
            and
            (
                (iteration - BURN_IN)
                % THIN
                == 0
            )
        ):

            saved.append(
                s.copy()
            )

            logpost_trace.append(
                log_posterior(s)
            )

        # -------------------------------------------------------------
        # Progress
        # -------------------------------------------------------------

        if (
            (iteration + 1) % 1000 == 0
            or iteration == 0
        ):

            print(
                f"   Chain {chain_id + 1}: "
                f"{iteration + 1:7d}/"
                f"{total_iterations}"
            )

    return (
        np.asarray(saved),
        np.asarray(logpost_trace),
    )


# =============================================================================
# RUN MCMC
# =============================================================================

print()
print("=" * 100)
print("RUNNING GIBBS MCMC")
print("=" * 100)

chains = []
logpost_chains = []

for chain_id in range(
    N_CHAINS
):

    samples, logpost = run_chain(
        chain_id
    )

    chains.append(
        samples
    )

    logpost_chains.append(
        logpost
    )

chains = np.asarray(
    chains
)

logpost_chains = np.asarray(
    logpost_chains
)

print()
print(
    "MCMC array shape:"
)
print(
    chains.shape
)

print(
    "(chains, samples, parameters)"
)


# =============================================================================
# R-HAT
# =============================================================================

print()
print("=" * 100)
print("CALCULATING GELMAN-RUBIN R-HAT")
print("=" * 100)


def calculate_rhat(
    chain_array,
):
    """
    Basic split-chain-independent Gelman-Rubin R-hat.

    chain_array:
        shape = (chains, samples, parameters)
    """

    C, N, P = chain_array.shape

    if C < 2:
        return np.full(
            P,
            np.nan,
        )

    chain_means = np.mean(
        chain_array,
        axis=1,
    )

    chain_vars = np.var(
        chain_array,
        axis=1,
        ddof=1,
    )

    W = np.mean(
        chain_vars,
        axis=0,
    )

    B = N * np.var(
        chain_means,
        axis=0,
        ddof=1,
    )

    var_hat = (
        ((N - 1) / N)
        * W
        +
        (1 / N)
        * B
    )

    rhat = np.sqrt(
        var_hat / W
    )

    return rhat


rhat = calculate_rhat(
    chains
)

print(
    f"R-hat min    : {np.nanmin(rhat):.4f}"
)

print(
    f"R-hat median : {np.nanmedian(rhat):.4f}"
)

print(
    f"R-hat max    : {np.nanmax(rhat):.4f}"
)

print()

bad_rhat = np.where(
    rhat > 1.10
)[0]

if len(bad_rhat):

    print(
        f"WARNING: {len(bad_rhat)} "
        "parameters have R-hat > 1.10"
    )

    for j in bad_rhat[:30]:

        print(
            f"   {EXPECTED_H_COLUMNS[j]:20s} "
            f"R-hat={rhat[j]:.4f}"
        )

else:

    print(
        "All parameters have R-hat <= 1.10"
    )


# =============================================================================
# COMBINE CHAINS
# =============================================================================

posterior_samples = chains.reshape(
    -1,
    N_PARAMS,
)

print()
print(
    f"Combined posterior samples: "
    f"{len(posterior_samples)}"
)


# =============================================================================
# POSTERIOR SUMMARY
# =============================================================================

print()
print("=" * 100)
print("BUILDING POSTERIOR DAILY FACTOR SUMMARY")
print("=" * 100)

summary_rows = []

for j, parameter in enumerate(
    EXPECTED_H_COLUMNS
):

    samples_j = posterior_samples[
        :, j
    ]

    mine, date_str = parameter.split(
        "_"
    )

    date = pd.Timestamp(
        date_str
    )

    summary_rows.append(
        {
            "parameter": parameter,
            "mine": mine,
            "date": date.strftime(
                "%Y-%m-%d"
            ),
            "prior_factor": 1.0,

            "posterior_mean": np.mean(
                samples_j
            ),

            "posterior_median": np.median(
                samples_j
            ),

            "posterior_sd": np.std(
                samples_j,
                ddof=1,
            ),

            "q025": np.percentile(
                samples_j,
                2.5,
            ),

            "q975": np.percentile(
                samples_j,
                97.5,
            ),

            "q05": np.percentile(
                samples_j,
                5,
            ),

            "q95": np.percentile(
                samples_j,
                95,
            ),

            "prob_factor_lt_1": np.mean(
                samples_j < 1.0
            ),

            "prob_factor_gt_1": np.mean(
                samples_j > 1.0
            ),

            "H_nonzero_observations":
                nonzero_count[j],

            "H_sensitivity_sum":
                sensitivity_sum[j],

            "rhat":
                rhat[j],
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_file = (
    OUTPUT_DIR
    / "FLEXPART_MCMC_posterior_daily_factors_Feb2026.csv"
)

summary_df.to_csv(
    summary_file,
    index=False,
)

print()
print(
    "Posterior summary:"
)
print(
    summary_file
)


# =============================================================================
# POSTERIOR PREDICTION
# =============================================================================

print()
print("=" * 100)
print("CALCULATING POSTERIOR PREDICTIONS")
print("=" * 100)

posterior_mean_factor = np.mean(
    posterior_samples,
    axis=0,
)

posterior_median_factor = np.median(
    posterior_samples,
    axis=0,
)

posterior_pred_mean = (
    BACKGROUND_PPM
    +
    H_obs @ posterior_mean_factor
)

posterior_pred_median = (
    BACKGROUND_PPM
    +
    H_obs @ posterior_median_factor
)

prior_pred = (
    BACKGROUND_PPM
    +
    H_obs @ np.ones(
        N_PARAMS
    )
)

prediction_df = pd.DataFrame(
    {
        "receptor":
            receptors_obs,

        "time_aest":
            pd.to_datetime(
                times_obs
            ),

        "obs_ch4_ppm":
            y,

        "prior_model_ppm":
            prior_pred,

        "posterior_mean_model_ppm":
            posterior_pred_mean,

        "posterior_median_model_ppm":
            posterior_pred_median,

        "prior_residual_ppm":
            prior_pred - y,

        "posterior_mean_residual_ppm":
            posterior_pred_mean - y,

        "posterior_median_residual_ppm":
            posterior_pred_median - y,
    }
)

prediction_df = prediction_df.sort_values(
    ["receptor", "time_aest"]
)

prediction_file = (
    OUTPUT_DIR
    / "FLEXPART_MCMC_predictions_Feb2026.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False,
)

print(
    prediction_file
)


# =============================================================================
# METRICS
# =============================================================================

print()
print("=" * 100)
print("PRIOR VS POSTERIOR METRICS")
print("=" * 100)


def calculate_metrics(
    observed,
    predicted,
):

    residual = (
        predicted - observed
    )

    n = len(observed)

    rmse = np.sqrt(
        np.mean(
            residual ** 2
        )
    )

    mae = np.mean(
        np.abs(residual)
    )

    bias = np.mean(
        residual
    )

    if n > 1:
        r = np.corrcoef(
            observed,
            predicted
        )[0, 1]
    else:
        r = np.nan

    r2 = r ** 2

    # FAC2
    valid = (
        observed > 0
    ) & (
        predicted > 0
    )

    ratio = (
        predicted[valid]
        /
        observed[valid]
    )

    fac2 = np.mean(
        (ratio >= 0.5)
        &
        (ratio <= 2.0)
    )

    return {
        "N": n,
        "R": r,
        "R2": r2,
        "RMSE_ppm": rmse,
        "MAE_ppm": mae,
        "Bias_ppm": bias,
        "FAC2": fac2,
    }


metric_rows = []

# ---------------------------------------------------------------------
# Overall
# ---------------------------------------------------------------------

m_prior = calculate_metrics(
    y,
    prior_pred,
)

m_post = calculate_metrics(
    y,
    posterior_pred_mean,
)

metric_rows.append(
    {
        "receptor": "ALL",
        "model": "Prior",
        **m_prior,
    }
)

metric_rows.append(
    {
        "receptor": "ALL",
        "model": "Posterior_mean",
        **m_post,
    }
)

# ---------------------------------------------------------------------
# By receptor
# ---------------------------------------------------------------------

for receptor in RECEPTOR_OBS_COLUMN:

    mask = (
        receptors_obs
        == receptor
    )

    if not np.any(mask):
        continue

    mp = calculate_metrics(
        y[mask],
        prior_pred[mask],
    )

    mpost = calculate_metrics(
        y[mask],
        posterior_pred_mean[mask],
    )

    metric_rows.append(
        {
            "receptor": receptor,
            "model": "Prior",
            **mp,
        }
    )

    metric_rows.append(
        {
            "receptor": receptor,
            "model": "Posterior_mean",
            **mpost,
        }
    )

metrics_df = pd.DataFrame(
    metric_rows
)

metrics_file = (
    OUTPUT_DIR
    / "FLEXPART_MCMC_metrics_Feb2026.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False,
)

print(
    metrics_df.to_string(
        index=False
    )
)

print()
print(
    metrics_file
)


# =============================================================================
# SAVE R-HAT
# =============================================================================

rhat_df = pd.DataFrame(
    {
        "parameter":
            EXPECTED_H_COLUMNS,
        "rhat":
            rhat,
    }
)

rhat_file = (
    OUTPUT_DIR
    / "FLEXPART_MCMC_Rhat_Feb2026.csv"
)

rhat_df.to_csv(
    rhat_file,
    index=False,
)


# =============================================================================
# SAVE LOG-POSTERIOR TRACE
# =============================================================================

logpost_df = pd.DataFrame(
    logpost_chains.T,
    columns=[
        f"chain_{i+1}"
        for i in range(N_CHAINS)
    ],
)

logpost_df.insert(
    0,
    "sample",
    np.arange(
        len(logpost_df)
    ),
)

logpost_file = (
    OUTPUT_DIR
    / "FLEXPART_MCMC_logposterior_Feb2026.csv"
)

logpost_df.to_csv(
    logpost_file,
    index=False,
)


# =============================================================================
# SAVE POSTERIOR SAMPLES
# =============================================================================

if SAVE_SAMPLES:

    samples_file = (
        OUTPUT_DIR
        / "FLEXPART_MCMC_posterior_samples_Feb2026.npz"
    )

    np.savez_compressed(
        samples_file,
        samples=posterior_samples,
        chains=chains,
        parameters=np.array(
            EXPECTED_H_COLUMNS
        ),
    )

    print()
    print(
        "Posterior samples:"
    )
    print(
        samples_file
    )


# =============================================================================
# DAILY MINE TABLE
# =============================================================================

# Convenient wide table:
#
# rows    = dates
# columns = OC1 ... OC13

daily_factor_df = (
    summary_df
    .pivot(
        index="date",
        columns="mine",
        values="posterior_mean",
    )
    .reset_index()
)

daily_factor_file = (
    OUTPUT_DIR
    / "FLEXPART_MCMC_daily_factor_matrix_Feb2026.csv"
)

daily_factor_df.to_csv(
    daily_factor_file,
    index=False,
)

print()
print(
    "Daily factor matrix:"
)
print(
    daily_factor_file
)


# =============================================================================
# MINE SUMMARY
# =============================================================================

mine_summary = (
    summary_df
    .groupby("mine")
    .agg(
        mean_factor=(
            "posterior_mean",
            "mean",
        ),

        median_factor=(
            "posterior_median",
            "mean",
        ),

        mean_sd=(
            "posterior_sd",
            "mean",
        ),

        mean_q025=(
            "q025",
            "mean",
        ),

        mean_q975=(
            "q975",
            "mean",
        ),

        mean_sensitivity=(
            "H_sensitivity_sum",
            "mean",
        ),
    )
    .reset_index()
)

mine_summary_file = (
    OUTPUT_DIR
    / "FLEXPART_MCMC_mine_summary_Feb2026.csv"
)

mine_summary.to_csv(
    mine_summary_file,
    index=False,
)


# =============================================================================
# FINAL INFORMATION
# =============================================================================

print()
print("=" * 100)
print("MCMC INVERSION COMPLETE")
print("=" * 100)

print()
print("Output directory:")
print(
    OUTPUT_DIR
)

print()
print("Main outputs:")
print(
    f"  1. {summary_file.name}"
)
print(
    f"  2. {prediction_file.name}"
)
print(
    f"  3. {metrics_file.name}"
)
print(
    f"  4. {rhat_file.name}"
)
print(
    f"  5. {daily_factor_file.name}"
)

if SAVE_SAMPLES:
    print(
        f"  6. {samples_file.name}"
    )

print()
print("Posterior factor summary:")
print()

display_columns = [
    "parameter",
    "posterior_mean",
    "posterior_sd",
    "q025",
    "q975",
    "H_nonzero_observations",
    "rhat",
]

print(
    summary_df[
        display_columns
    ].to_string(
        index=False,
        max_rows=351,
    )
)

print()
print("=" * 100)
print("DONE")
print("=" * 100)

