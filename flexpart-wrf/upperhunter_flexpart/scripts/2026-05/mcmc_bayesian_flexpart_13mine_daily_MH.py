#!/usr/bin/env python3
"""
FLEXPART-WRF Bayesian MCMC inversion
====================================

Upper Hunter CH4
13 mines x 31 daily emission scaling factors

This script is intentionally structured to match:

    mcmc_bayesian_flexpart_13mine_daily.py

The ONLY substantive difference is the MCMC algorithm:

    Gibbs version:
        exact component-wise truncated-Gaussian conditional sampling

    This version:
        component-wise random-walk Metropolis-Hastings

Therefore both algorithms use exactly the same:

    - H matrix
    - Picarro observations
    - receptor/time matching
    - 1052 matched observations
    - observation uncertainties
    - prior
    - background
    - 351 parameters
    - parameter ordering
    - initial states
    - burn-in
    - thinning

Observation model:

    y_i = BACKGROUND
          + sum_j H_ij * s_j
          + epsilon_i

where:

    H_ij = FLEXPART prior contribution in ppmv
    s_j  = daily emission scaling factor
    epsilon_i ~ N(0, sigma_obs^2)

Prior:

    s_j ~ Normal(1, 1^2), truncated at s_j >= 0

IMPORTANT:

    H already contains the prior mine emissions.
    DO NOT multiply H by prior emissions again.

MCMC:

    Component-wise random-walk Metropolis-Hastings.

For parameter j:

    s_j* = s_j + Normal(0, proposal_sd_j)

Negative proposals are rejected.

Proposal widths are adapted during burn-in toward a target
acceptance rate of approximately 0.30.

Outputs are designed to be directly comparable with the
Gibbs sampler outputs.
"""

import os
import sys
import math
import numpy as np
import pandas as pd

from pathlib import Path


# =============================================================================
# USER SETTINGS
# =============================================================================

BASE_DIR = Path(
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart"
)

H_FILE = (
    BASE_DIR
    / "H_matrix_flexpart"
    / "2026-05"
    / "H_flexpart_0_10m_13mine_May2026.csv"
)

OBS_FILE = (
    Path(
        "/mnt/scratch_lustre/nzem_meim_scratch/data/"
        "piccaro_obs/"
        "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
    )
)

OUTPUT_DIR = (
    BASE_DIR
    / "H_matrix_flexpart"
    / "2026-05"
    / "MCMC_13mine_daily_May2026_MH"
)


# =============================================================================
# PERIOD
# =============================================================================

START_DATE = pd.Timestamp(
    "2026-05-01"
)

END_DATE = pd.Timestamp(
    "2026-05-31"
)


# =============================================================================
# RECEPTORS
# =============================================================================

RECEPTOR_OBS_COLUMN = {

    "singleton":
        "Singleton cleaned",

    "muswellbrook":
        "Muswellbrook cleaned",

    "merriwa":
        "Merriwa cleaned",
}


# =============================================================================
# METHANE BACKGROUND
# =============================================================================

BACKGROUND_PPM = 1.85


# =============================================================================
# OBSERVATION UNCERTAINTY
# =============================================================================

OBS_SIGMA_PPM = {

    "singleton":
        0.05,

    "muswellbrook":
        0.05,

    "merriwa":
        0.05,
}


# =============================================================================
# PRIOR
# =============================================================================

PRIOR_MEAN_FACTOR = 1.0

PRIOR_SIGMA_FACTOR = 1.0


# =============================================================================
# MCMC SETTINGS
# =============================================================================

N_CHAINS = 4

BURN_IN = 5000

N_SAMPLES = 5000

THIN = 5

RANDOM_SEED = 12345

SAVE_SAMPLES = True


# -----------------------------------------------------------------------------
# Metropolis-Hastings proposal
# -----------------------------------------------------------------------------

INITIAL_PROPOSAL_SD = 0.10

TARGET_ACCEPTANCE = 0.45

ADAPT_EVERY = 100

MIN_PROPOSAL_SD = 0.005

MAX_PROPOSAL_SD = 1.0


# =============================================================================
# MINE / DATE DEFINITIONS
# =============================================================================

MINES = [
    f"OC{i}"
    for i in range(1, 14)
]

DATES = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D",
)


EXPECTED_H_COLUMNS = [

    f"{mine}_{date.strftime('%Y%m%d')}"

    for date in DATES

    for mine in MINES
]


N_PARAMS = len(
    EXPECTED_H_COLUMNS
)


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
print("COMPONENT-WISE METROPOLIS-HASTINGS")
print("=" * 100)

print(
    "Upper Hunter — "
    "13 mines × 27 daily factors"
)

print()

print(
    f"Period: "
    f"{START_DATE.strftime('%Y-%m-%d')} "
    f"to "
    f"{END_DATE.strftime('%Y-%m-%d')}"
)

print(
    f"Number of mines       : {len(MINES)}"
)

print(
    f"Number of days        : {len(DATES)}"
)

print(
    f"Number of parameters  : {N_PARAMS}"
)

print()

print(
    f"Background CH4        : "
    f"{BACKGROUND_PPM:.3f} ppm"
)

print(
    "Observation sigma     : "
    +
    ", ".join(
        f"{k}={v:.4f}"
        for k, v in OBS_SIGMA_PPM.items()
    )
    +
    " ppm"
)

print(
    f"Prior mean factor     : "
    f"{PRIOR_MEAN_FACTOR:.3f}"
)

print(
    f"Prior sigma            : "
    f"{PRIOR_SIGMA_FACTOR:.3f}"
)

print()

print(
    f"MCMC chains            : "
    f"{N_CHAINS}"
)

print(
    f"Burn-in                : "
    f"{BURN_IN}"
)

print(
    f"Posterior samples      : "
    f"{N_SAMPLES}"
)

print(
    f"Thin                   : "
    f"{THIN}"
)

print(
    f"Total iterations/chain : "
    f"{BURN_IN + N_SAMPLES * THIN}"
)

print()

print(
    f"Initial proposal SD    : "
    f"{INITIAL_PROPOSAL_SD}"
)

print(
    f"Target acceptance      : "
    f"{TARGET_ACCEPTANCE}"
)

print()

print(
    f"Random seed            : "
    f"{RANDOM_SEED}"
)

print()

print("=" * 100)


# =============================================================================
# CHECK INPUT FILES
# =============================================================================

print()
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

print()
print("H matrix:")
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

H_df = pd.read_csv(
    H_FILE
)

print(
    f"H dataframe shape: "
    f"{H_df.shape}"
)


# =============================================================================
# CHECK EXPECTED H COLUMNS
# =============================================================================

missing_cols = [

    c

    for c in EXPECTED_H_COLUMNS

    if c not in H_df.columns
]


if missing_cols:

    print()
    print(
        "ERROR: Missing H columns:"
    )

    for c in missing_cols:

        print(
            "   ",
            c
        )

    raise RuntimeError(
        f"{len(missing_cols)} "
        "expected H columns are missing."
    )


# =============================================================================
# EXACT H COLUMN ORDER
# =============================================================================

H_values_all = (

    H_df[
        EXPECTED_H_COLUMNS
    ]

    .apply(
        pd.to_numeric,
        errors="coerce",
    )

    .to_numpy(
        dtype=float
    )
)


if np.isnan(
    H_values_all
).any():

    raise RuntimeError(
        "H matrix contains NaN values."
    )


print(
    f"H numerical matrix shape: "
    f"{H_values_all.shape}"
)

print(
    f"H min : "
    f"{np.min(H_values_all):.6e}"
)

print(
    f"H max : "
    f"{np.max(H_values_all):.6e}"
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

print(
    "H metadata columns:"
)

print(
    list(H_df.columns[:10])
)


# =============================================================================
# RECEPTOR
# =============================================================================

if "receptor" not in H_df.columns:

    raise RuntimeError(
        "H matrix does not contain "
        "'receptor'."
    )


H_receptor = (

    H_df[
        "receptor"
    ]

    .astype(str)

    .str.lower()

    .str.strip()
)


# =============================================================================
# H TIMESTAMP
# =============================================================================

if "obs_time_aest" not in H_df.columns:

    raise RuntimeError(
        "H matrix does not contain "
        "'obs_time_aest'. "
        f"Available columns: "
        f"{list(H_df.columns[:10])}"
    )


print(
    "Using H matrix "
    "'obs_time_aest' directly."
)


H_time = pd.to_datetime(

    H_df[
        "obs_time_aest"
    ],

    errors="coerce",
)


if H_time.isna().any():

    n_bad = H_time.isna().sum()

    raise RuntimeError(
        f"{n_bad} H matrix "
        "obs_time_aest values "
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

    x = (
        str(x)
        .lower()
        .strip()
    )

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

    -
    set(RECEPTOR_OBS_COLUMN)
)


if unknown_receptors:

    raise RuntimeError(
        "Unknown receptors in H matrix: "
        +
        str(unknown_receptors)
    )


# =============================================================================
# CREATE H HOURLY KEY
# =============================================================================

H_key = pd.DataFrame(
    {
        "receptor":
            H_receptor,

        "time_aest":
            H_time.dt.floor("h"),
    }
)


# =============================================================================
# CHECK DUPLICATE H KEYS
# =============================================================================

duplicates = H_key.duplicated(

    subset=[
        "receptor",
        "time_aest",
    ],

    keep=False,
)


if duplicates.any():

    dup_df = (

        H_key.loc[
            duplicates
        ]

        .sort_values(
            [
                "receptor",
                "time_aest",
            ]
        )
    )

    print()
    print(
        "ERROR: Duplicate FLEXPART "
        "receptor/time rows detected."
    )

    print(
        dup_df.head(20)
    )

    raise RuntimeError(
        "H matrix contains duplicate "
        "receptor/time keys."
    )


print(
    f"FLEXPART rows: "
    f"{len(H_df)}"
)


print(
    "FLEXPART receptors:"
)


for r in sorted(
    H_receptor.unique()
):

    n = np.sum(
        H_receptor == r
    )

    print(
        f"   {r:15s} {n:6d}"
    )


# =============================================================================
# READ PICARRO OBSERVATIONS
# =============================================================================

print()
print("=" * 100)
print("READING PICARRO OBSERVATIONS DIRECTLY")
print("=" * 100)

obs = pd.read_csv(
    OBS_FILE
)


print(
    f"Observation dataframe shape: "
    f"{obs.shape}"
)


print()
print(
    "Observation columns:"
)

for c in obs.columns:

    print(
        "   ",
        c
    )


# =============================================================================
# PARSE DATETIME
# =============================================================================

if "Datetime" not in obs.columns:

    raise RuntimeError(
        "The Picarro CSV does not contain "
        "the expected 'Datetime' column."
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
        "No Picarro Datetime values "
        "could be parsed."
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
# BUILD LONG-FORM OBSERVATIONS
# =============================================================================

print()
print("=" * 100)
print("BUILDING LONG-FORM OBSERVATION TABLE")
print("=" * 100)


obs_long_list = []


for receptor, column in (
    RECEPTOR_OBS_COLUMN.items()
):

    if column not in obs.columns:

        raise RuntimeError(
            f"Observation column missing "
            f"for {receptor}: {column}"
        )


    tmp = pd.DataFrame(

        {
            "receptor":
                receptor,

            "time_aest":
                obs[
                    "time_aest"
                ],

            "obs_ch4_ppm":
                pd.to_numeric(
                    obs[column],
                    errors="coerce",
                ),
        }
    )


    tmp = tmp.dropna(

        subset=[
            "time_aest",
            "obs_ch4_ppm",
        ]
    )


    tmp = tmp[
        (
            tmp["time_aest"]
            >= START_DATE
        )
        &
        (
            tmp["time_aest"]
            <
            END_DATE
            +
            pd.Timedelta(
                days=1
            )
        )
    ]


    tmp["time_aest"] = (

        tmp[
            "time_aest"
        ]

        .dt.floor("h")
    )


    obs_long_list.append(
        tmp
    )


obs_long = pd.concat(

    obs_long_list,

    ignore_index=True,
)


# =============================================================================
# DUPLICATE OBSERVATIONS
# =============================================================================

dup_obs = obs_long.duplicated(

    subset=[
        "receptor",
        "time_aest",
    ],

    keep=False,
)


if dup_obs.any():

    print()
    print(
        "WARNING: duplicate Picarro "
        "receptor/hour observations found."
    )

    print(

        obs_long.loc[
            dup_obs
        ]

        .sort_values(
            [
                "receptor",
                "time_aest",
            ]
        )

        .head(20)
    )


    obs_long = (

        obs_long

        .groupby(
            [
                "receptor",
                "time_aest",
            ],

            as_index=False,
        )

        [
            "obs_ch4_ppm"
        ]

        .mean()
    )


print()
print(
    f"Usable Picarro observations: "
    f"{len(obs_long)}"
)


for r in RECEPTOR_OBS_COLUMN:

    n = np.sum(

        obs_long[
            "receptor"
        ]

        == r
    )

    print(
        f"   {r:15s} {n:6d}"
    )


# =============================================================================
# ALIGN OBSERVATIONS WITH H
# =============================================================================

print()
print("=" * 100)
print("ALIGNING PICARRO OBSERVATIONS WITH FLEXPART H")
print("=" * 100)


obs_key = obs_long[
    [
        "receptor",
        "time_aest",
    ]
].copy()


obs_key[
    "obs_ch4_ppm"
] = obs_long[
    "obs_ch4_ppm"
].to_numpy()


match = H_key.merge(

    obs_key,

    on=[
        "receptor",
        "time_aest",
    ],

    how="inner",
)


print(
    f"Matched FLEXPART/Picarro "
    f"observations: {len(match)}"
)


if len(match) == 0:

    raise RuntimeError(
        "No FLEXPART/Picarro observations "
        "could be aligned."
    )


# =============================================================================
# GET ORIGINAL H ROW INDICES
# =============================================================================

H_key_indexed = H_key.copy()


H_key_indexed[
    "H_row"
] = np.arange(
    len(H_key_indexed)
)


match = H_key_indexed.merge(

    obs_key,

    on=[
        "receptor",
        "time_aest",
    ],

    how="inner",
)


H_row_indices = (

    match[
        "H_row"
    ]

    .to_numpy(
        dtype=int
    )
)


H_obs = H_values_all[
    H_row_indices,
    :
]


y = (

    match[
        "obs_ch4_ppm"
    ]

    .to_numpy(
        dtype=float
    )
)


receptors_obs = (

    match[
        "receptor"
    ]

    .to_numpy()
)


times_obs = (

    match[
        "time_aest"
    ]

    .to_numpy()
)


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


if np.any(
    sigma_obs <= 0
):

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

        receptors_obs
        == r
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
    pd.Timestamp(
        times_obs.min()
    ),

    "to",

    pd.Timestamp(
        times_obs.max()
    ),
)


# =============================================================================
# PRIOR MODEL
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
    +
    prior_enhancement
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


for j, name in enumerate(
    EXPECTED_H_COLUMNS
):

    if nonzero_count[j] == 0:

        print(
            f"WARNING: {name} has ZERO "
            "sensitivity to all matched "
            "observations."
        )


# =============================================================================
# PRE-COMPUTE NONZERO ROWS
# =============================================================================

nonzero_rows = []


for j in range(
    N_PARAMS
):

    idx = np.flatnonzero(
        H_obs[:, j]
    )

    nonzero_rows.append(
        idx
    )


print()
print(
    f"Total non-zero H elements "
    f"in likelihood: "
    f"{np.count_nonzero(H_obs):,}"
)


# =============================================================================
# LOG POSTERIOR
# =============================================================================

prior_precision = (

    1.0
    /
    PRIOR_SIGMA_FACTOR ** 2
)


def log_posterior(
    s,
    prediction_enhancement=None,
):
    """
    Log posterior up to a constant.

    Uses the same likelihood and prior
    as the Gibbs sampler.
    """

    if np.any(
        s < 0
    ):

        return -np.inf


    if prediction_enhancement is None:

        prediction_enhancement = (
            H_obs @ s
        )


    residual = (

        y
        -
        BACKGROUND_PPM
        -
        prediction_enhancement
    )


    log_likelihood = -0.5 * np.sum(

        residual ** 2
        *
        invvar
    )


    log_prior = -0.5 * np.sum(

        (
            (
                s
                -
                PRIOR_MEAN_FACTOR
            )
            /
            PRIOR_SIGMA_FACTOR
        )
        ** 2
    )


    return (
        log_likelihood
        +
        log_prior
    )


# =============================================================================
# INITIAL STATES
# =============================================================================

def make_initial_state(
    chain_id,
    rng,
):

    if chain_id == 0:

        s = np.ones(
            N_PARAMS,
            dtype=float
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
# ONE METROPOLIS-HASTINGS SWEEP
# =============================================================================

def mh_sweep(
    s,
    prediction_enhancement,
    current_logpost,
    proposal_sd,
    rng,
    accepted,
    proposed,
):
    """
    One complete component-wise Metropolis-Hastings sweep.

    For each parameter:

        s_j* = s_j + Normal(0, proposal_sd_j)

    Negative proposals are rejected.

    Because the proposal is symmetric:

        log(alpha)
          =
        log posterior(new)
        -
        log posterior(old)

    The prediction is updated efficiently:

        H @ s_new
          =
        H @ s_old
        + H[:,j] * delta
    """

    for j in range(
        N_PARAMS
    ):

        proposed[j] += 1


        old_value = s[j]


        # -------------------------------------------------------------
        # Random-walk proposal
        # -------------------------------------------------------------

        new_value = (

            old_value

            +

            rng.normal(
                0.0,
                proposal_sd[j],
            )
        )


        # -------------------------------------------------------------
        # Truncated support:
        # s >= 0
        #
        # Proposal outside support is immediately rejected.
        # -------------------------------------------------------------

        if new_value < 0.0:

            continue


        delta = (
            new_value
            -
            old_value
        )


        # -------------------------------------------------------------
        # Efficient prediction update
        # -------------------------------------------------------------

        idx = nonzero_rows[j]


        if len(idx) == 0:

            # No likelihood information.
            #
            # Only the prior changes.

            old_prior = -0.5 * (

                (
                    (
                        old_value
                        -
                        PRIOR_MEAN_FACTOR
                    )
                    /
                    PRIOR_SIGMA_FACTOR
                )
                ** 2
            )


            new_prior = -0.5 * (

                (
                    (
                        new_value
                        -
                        PRIOR_MEAN_FACTOR
                    )
                    /
                    PRIOR_SIGMA_FACTOR
                )
                ** 2
            )


            log_alpha = (
                new_prior
                -
                old_prior
            )


            if (
                np.log(
                    rng.random()
                )
                <
                min(
                    0.0,
                    log_alpha,
                )
            ):

                s[j] = new_value

                current_logpost += (
                    new_prior
                    -
                    old_prior
                )

                accepted[j] += 1


            continue


        # -------------------------------------------------------------
        # Current residual for affected observations
        # -------------------------------------------------------------

        residual_old = (

            y[idx]
            -
            BACKGROUND_PPM
            -
            prediction_enhancement[idx]
        )


        hj = H_obs[
            idx,
            j
        ]


        # -------------------------------------------------------------
        # Proposed residual
        # -------------------------------------------------------------

        residual_new = (

            residual_old
            -
            hj * delta
        )


        # -------------------------------------------------------------
        # Likelihood change
        # -------------------------------------------------------------

        loglike_old = -0.5 * np.sum(

            residual_old ** 2
            *
            invvar[idx]
        )


        loglike_new = -0.5 * np.sum(

            residual_new ** 2
            *
            invvar[idx]
        )


        # -------------------------------------------------------------
        # Prior change
        # -------------------------------------------------------------

        old_prior = -0.5 * (

            (
                (
                    old_value
                    -
                    PRIOR_MEAN_FACTOR
                )
                /
                PRIOR_SIGMA_FACTOR
            )
            ** 2
        )


        new_prior = -0.5 * (

            (
                (
                    new_value
                    -
                    PRIOR_MEAN_FACTOR
                )
                /
                PRIOR_SIGMA_FACTOR
            )
            ** 2
        )


        # -------------------------------------------------------------
        # MH acceptance ratio
        # -------------------------------------------------------------

        log_alpha = (

            loglike_new
            +
            new_prior

            -

            loglike_old
            -
            old_prior
        )


        # -------------------------------------------------------------
        # Accept/reject
        # -------------------------------------------------------------

        if (

            np.log(
                rng.random()
            )

            <

            min(
                0.0,
                log_alpha,
            )
        ):

            s[j] = new_value


            prediction_enhancement[
                idx
            ] += hj * delta


            current_logpost += (

                loglike_new
                -
                loglike_old

                +

                new_prior
                -
                old_prior
            )


            accepted[j] += 1


    return (
        s,
        prediction_enhancement,
        current_logpost,
    )


# =============================================================================
# RUN ONE MH CHAIN
# =============================================================================

def run_chain(
    chain_id,
):

    rng = np.random.default_rng(

        RANDOM_SEED
        +
        chain_id * 100000
    )


    s = make_initial_state(
        chain_id,
        rng,
    )


    prediction_enhancement = (
        H_obs @ s
    )


    current_logpost = log_posterior(

        s,

        prediction_enhancement
    )


    proposal_sd = np.full(

        N_PARAMS,

        INITIAL_PROPOSAL_SD,

        dtype=float,
    )


    accepted_total = np.zeros(

        N_PARAMS,

        dtype=np.int64,
    )


    proposed_total = np.zeros(

        N_PARAMS,

        dtype=np.int64,
    )


    accepted_window = np.zeros(

        N_PARAMS,

        dtype=np.int64,
    )


    proposed_window = np.zeros(

        N_PARAMS,

        dtype=np.int64,
    )


    total_iterations = (

        BURN_IN
        +
        N_SAMPLES * THIN
    )


    saved = []

    logpost_trace = []


    print()
    print(
        f"Starting MH chain "
        f"{chain_id + 1}/{N_CHAINS}"
    )


    for iteration in range(
        total_iterations
    ):


        # -------------------------------------------------------------

        # MH sweep

        # -------------------------------------------------------------


        # Save counters before this sweep

        accepted_before = accepted_window.copy()

        proposed_before = proposed_window.copy()

        (

            s,

            prediction_enhancement,

            current_logpost,

        ) = mh_sweep(

            s,
    
            prediction_enhancement,

            current_logpost,

            proposal_sd,

            rng,

            accepted_window,

            proposed_window,
        )

        # Add only the new counts from this sweep
        accepted_total += (
            accepted_window - accepted_before
        )

        proposed_total += (
            proposed_window - proposed_before
        )

        # -------------------------------------------------------------
        # Adapt proposal SD during burn-in
        # -------------------------------------------------------------

        if (

            iteration < BURN_IN

            and

            (iteration + 1)
            % ADAPT_EVERY
            == 0
        ):


            window_acceptance = (

                accepted_window
                /
                np.maximum(
                    proposed_window,
                    1,
                )
            )


            # ---------------------------------------------------------
            # Robbins-Monro-like multiplicative adjustment
            #
            # acceptance > target -> increase proposal
            # acceptance < target -> decrease proposal
            # ---------------------------------------------------------

            adjustment = np.exp(

                0.5
                *
                (
                    window_acceptance
                    -
                    TARGET_ACCEPTANCE
                )
            )


            proposal_sd *= adjustment


            proposal_sd = np.clip(

                proposal_sd,

                MIN_PROPOSAL_SD,

                MAX_PROPOSAL_SD,
            )


            accepted_window[:] = 0

            proposed_window[:] = 0


        # -------------------------------------------------------------
        # Save posterior
        # -------------------------------------------------------------

        if (

            iteration >= BURN_IN

            and

            (
                (
                    iteration
                    -
                    BURN_IN
                )
                %
                THIN
                ==
                0
            )
        ):


            saved.append(
                s.copy()
            )


            logpost_trace.append(
                current_logpost
            )


        # -------------------------------------------------------------
        # Progress
        # -------------------------------------------------------------

        if (

            (iteration + 1)
            % 1000
            == 0

            or

            iteration == 0
        ):


            overall_acc = (

                np.sum(
                    accepted_total
                )

                /

                max(
                    np.sum(
                        proposed_total
                    ),
                    1,
                )
            )


            print(

                f"   Chain {chain_id + 1}: "

                f"{iteration + 1:7d}/"
                f"{total_iterations} "

                f"acceptance="
                f"{overall_acc:.4f} "

                f"logpost="
                f"{current_logpost:.2f}"
            )


    saved = np.asarray(
        saved
    )


    logpost_trace = np.asarray(
        logpost_trace
    )


    final_acceptance = (

        accepted_total
        /
        np.maximum(
            proposed_total,
            1,
        )
    )


    print()
    print(
        f"Chain {chain_id + 1} finished."
    )


    print(
        f"Mean acceptance   : "
        f"{np.mean(final_acceptance):.4f}"
    )


    print(
        f"Median acceptance : "
        f"{np.median(final_acceptance):.4f}"
    )


    print(
        f"Minimum acceptance: "
        f"{np.min(final_acceptance):.4f}"
    )


    print(
        f"Maximum acceptance: "
        f"{np.max(final_acceptance):.4f}"
    )


    return (

        saved,

        logpost_trace,

        final_acceptance,

        proposal_sd,
    )


# =============================================================================
# RUN MCMC
# =============================================================================

print()
print("=" * 100)
print("RUNNING METROPOLIS-HASTINGS MCMC")
print("=" * 100)


chains = []

logpost_chains = []

acceptance_chains = []

proposal_sd_chains = []


for chain_id in range(
    N_CHAINS
):


    (

        samples,

        logpost,

        acceptance,

        final_proposal_sd,

    ) = run_chain(
        chain_id
    )


    chains.append(
        samples
    )


    logpost_chains.append(
        logpost
    )


    acceptance_chains.append(
        acceptance
    )


    proposal_sd_chains.append(
        final_proposal_sd
    )


chains = np.asarray(
    chains
)


logpost_chains = np.asarray(
    logpost_chains
)


acceptance_chains = np.asarray(
    acceptance_chains
)


proposal_sd_chains = np.asarray(
    proposal_sd_chains
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


expected_shape = (

    N_CHAINS,

    N_SAMPLES,

    N_PARAMS,
)


if chains.shape != expected_shape:

    raise RuntimeError(

        f"Unexpected MCMC shape: "
        f"{chains.shape}; "
        f"expected {expected_shape}"
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

    C, N, P = (
        chain_array.shape
    )


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


    B = (

        N
        *
        np.var(

            chain_means,

            axis=0,

            ddof=1,
        )
    )


    var_hat = (

        (
            (N - 1)
            /
            N
        )
        *
        W

        +

        (
            1
            /
            N
        )
        *
        B
    )


    rhat = np.sqrt(

        var_hat
        /
        W
    )


    return rhat


rhat = calculate_rhat(
    chains
)


print(
    f"R-hat min    : "
    f"{np.nanmin(rhat):.4f}"
)


print(
    f"R-hat median : "
    f"{np.nanmedian(rhat):.4f}"
)


print(
    f"R-hat max    : "
    f"{np.nanmax(rhat):.4f}"
)


bad_rhat = np.where(
    rhat > 1.10
)[0]


print()


if len(bad_rhat):

    print(
        f"WARNING: "
        f"{len(bad_rhat)} parameters "
        "have R-hat > 1.10"
    )


    for j in bad_rhat[:30]:

        print(

            f"   "
            f"{EXPECTED_H_COLUMNS[j]:20s} "
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


    samples_j = (
        posterior_samples[:, j]
    )


    mine, date_str = (
        parameter.split("_")
    )


    date = pd.Timestamp(
        date_str
    )


    summary_rows.append(

        {
            "parameter":
                parameter,

            "mine":
                mine,

            "date":
                date.strftime(
                    "%Y-%m-%d"
                ),

            "prior_factor":
                1.0,

            "posterior_mean":
                np.mean(
                    samples_j
                ),

            "posterior_median":
                np.median(
                    samples_j
                ),

            "posterior_sd":
                np.std(
                    samples_j,
                    ddof=1,
                ),

            "q025":
                np.percentile(
                    samples_j,
                    2.5,
                ),

            "q975":
                np.percentile(
                    samples_j,
                    97.5,
                ),

            "q05":
                np.percentile(
                    samples_j,
                    5,
                ),

            "q95":
                np.percentile(
                    samples_j,
                    95,
                ),

            "prob_factor_lt_1":
                np.mean(
                    samples_j < 1.0
                ),

            "prob_factor_gt_1":
                np.mean(
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
    /
    "FLEXPART_MH_posterior_daily_factors_May2026.csv"
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


posterior_mean_factor = (

    np.mean(

        posterior_samples,

        axis=0,
    )
)


posterior_median_factor = (

    np.median(

        posterior_samples,

        axis=0,
    )
)


posterior_pred_mean = (

    BACKGROUND_PPM

    +

    H_obs
    @
    posterior_mean_factor
)


posterior_pred_median = (

    BACKGROUND_PPM

    +

    H_obs
    @
    posterior_median_factor
)


prior_pred = (

    BACKGROUND_PPM

    +

    H_obs
    @
    np.ones(
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


prediction_df = (

    prediction_df

    .sort_values(
        [
            "receptor",
            "time_aest",
        ]
    )
)


prediction_file = (

    OUTPUT_DIR
    /
    "FLEXPART_MH_predictions_May2026.csv"
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
        predicted
        -
        observed
    )


    n = len(
        observed
    )


    rmse = np.sqrt(
        np.mean(
            residual ** 2
        )
    )


    mae = np.mean(
        np.abs(
            residual
        )
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

        (
            ratio >= 0.5
        )
        &
        (
            ratio <= 2.0
        )
    )


    return {

        "N":
            n,

        "R":
            r,

        "R2":
            r2,

        "RMSE_ppm":
            rmse,

        "MAE_ppm":
            mae,

        "Bias_ppm":
            bias,

        "FAC2":
            fac2,
    }


metric_rows = []


# -----------------------------------------------------------------------------
# Overall
# -----------------------------------------------------------------------------

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
        "receptor":
            "ALL",

        "model":
            "Prior",

        **m_prior,
    }
)


metric_rows.append(

    {
        "receptor":
            "ALL",

        "model":
            "Posterior_mean",

        **m_post,
    }
)


# -----------------------------------------------------------------------------
# By receptor
# -----------------------------------------------------------------------------

for receptor in RECEPTOR_OBS_COLUMN:

    mask = (

        receptors_obs
        ==
        receptor
    )


    if not np.any(
        mask
    ):

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
            "receptor":
                receptor,

            "model":
                "Prior",

            **mp,
        }
    )


    metric_rows.append(

        {
            "receptor":
                receptor,

            "model":
                "Posterior_mean",

            **mpost,
        }
    )


metrics_df = pd.DataFrame(
    metric_rows
)


metrics_file = (

    OUTPUT_DIR
    /
    "FLEXPART_MH_metrics_May2026.csv"
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
    /
    "FLEXPART_MH_Rhat_May2026.csv"
)


rhat_df.to_csv(

    rhat_file,

    index=False,
)


# =============================================================================
# SAVE ACCEPTANCE RATES
# =============================================================================

acceptance_df = pd.DataFrame(

    {
        "parameter":
            EXPECTED_H_COLUMNS,

        "acceptance_rate_mean":
            np.mean(
                acceptance_chains,
                axis=0,
            ),

        "acceptance_rate_min":
            np.min(
                acceptance_chains,
                axis=0,
            ),

        "acceptance_rate_max":
            np.max(
                acceptance_chains,
                axis=0,
            ),

        "proposal_sd_final":
            np.mean(
                proposal_sd_chains,
                axis=0,
            ),
    }
)


acceptance_file = (

    OUTPUT_DIR
    /
    "FLEXPART_MH_acceptance_May2026.csv"
)


acceptance_df.to_csv(

    acceptance_file,

    index=False,
)


# =============================================================================
# SAVE LOG-POSTERIOR TRACE
# =============================================================================

logpost_df = pd.DataFrame(

    logpost_chains.T,

    columns=[

        f"chain_{i+1}"

        for i in range(
            N_CHAINS
        )
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
    /
    "FLEXPART_MH_logposterior_May2026.csv"
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
        /
        "FLEXPART_MH_posterior_samples_May2026.npz"
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
# DAILY MINE FACTOR MATRIX
# =============================================================================

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
    /
    "FLEXPART_MH_daily_factor_matrix_May2026.csv"
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
    /
    "FLEXPART_MH_mine_summary_May2026.csv"
)


mine_summary.to_csv(

    mine_summary_file,

    index=False,
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 100)
print("METROPOLIS-HASTINGS INVERSION COMPLETE")
print("=" * 100)


overall_acceptance = (

    np.sum(
        acceptance_chains
    )

    /
    (
        N_CHAINS
        *
        N_PARAMS
    )
)


print()
print(
    f"Overall mean acceptance: "
    f"{overall_acceptance:.4f}"
)


print()
print(
    f"R-hat min    : "
    f"{np.nanmin(rhat):.4f}"
)

print(
    f"R-hat median : "
    f"{np.nanmedian(rhat):.4f}"
)

print(
    f"R-hat max    : "
    f"{np.nanmax(rhat):.4f}"
)


print()
print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)


print()
print(
    "Main outputs:"
)

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

print(
    f"  6. {acceptance_file.name}"
)

if SAVE_SAMPLES:

    print(
        f"  7. {samples_file.name}"
    )


print()
print("=" * 100)
print("DONE")
print("=" * 100)

