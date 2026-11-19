#!/usr/bin/env python3

"""
===============================================================================
UPPER HUNTER CH4 BAYESIAN INVERSION
BLOCKED METROPOLIS-HASTINGS MCMC
===============================================================================

Inverse problem
---------------

13 mines x 27 days = 351 daily scaling factors.

    y = G s + epsilon

where

    G = H * E0

and

    y = observed CH4 - background

All emission scaling factors are strictly positive:

    s_j > 0

Sampling is performed in log space:

    z_j = log(s_j)

    s_j = exp(z_j)

Prior
-----

A correlated log-normal prior is used:

    z ~ N(0, Bz)

with

    sigma_log = log(1.5)

and temporal correlation

    Corr(d1,d2) =
        exp(-|d1-d2| / 3 days)

Only factors belonging to the same mine are temporally correlated.

Different mines have zero prior covariance.

MCMC
----

Blocked Metropolis-Hastings.

Each MCMC sweep updates:

    OC1: 27 daily factors
    OC2: 27 daily factors
    ...
    OC13: 27 daily factors

Proposal for each mine:

    z_new = z_old + alpha * L * epsilon

where

    L L^T = B_day
    epsilon ~ N(0,I)

The proposal is symmetric, therefore:

    acceptance =
        min(1, exp(logposterior_new - logposterior_old))

The proposal scale is adaptively tuned during burn-in.

Production configuration
------------------------

    Burn-in       = 10,000 sweeps
    Production    = 30,000 sweeps
    Thin          = 5

This gives:

    6,000 retained posterior samples

Outputs
-------

    mcmc_samples.npy
    mcmc_logposterior.npy
    mcmc_scaling_summary.csv
    mcmc_daily_scaling.csv
    mcmc_daily_scaling_lower95.csv
    mcmc_daily_scaling_upper95.csv
    mcmc_mine_summary.csv
    mcmc_diagnostics.csv
    mcmc_block_acceptance.csv
    mcmc_trace_logposterior.csv
    mcmc_checkpoint.npz

===============================================================================
"""

from pathlib import Path
import sys
import time
import os

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE = (
    Path.home()
    / "flexpart-wrf"
    / "upperhunter_flexpart"
)

HMATRIX_DIR = (
    BASE
    / "hmatrix"
    / "daily_20260105_20260131"
)

BAYES_DIR = (
    HMATRIX_DIR
    / "bayesian_analytical"
)

H_FILE = (
    HMATRIX_DIR
    / "H_daily_13mine_1124x351_ppm.npy"
)

OBS_FILE = (
    HMATRIX_DIR
    / "observation_vector.csv"
)

PRIOR_FILE = (
    Path.home()
    / "flexpart-wrf"
    / "emission_UH"
    / "RH_upperHunter"
    / "UpperHuntercoalmines.csv"
)

ANALYTICAL_FILE = (
    BAYES_DIR
    / "posterior_scaling_factors.csv"
)

OUTPUT = (
    BAYES_DIR
    / "mcmc_blocked"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# PRODUCTION MCMC SETTINGS
# =============================================================================

NDIM = 351

N_MINES = 13

N_DAYS = 27

BLOCK_SIZE = 27

BURN_IN = 10000

N_PRODUCTION = 30000

THIN = 5

# Number of retained samples:
N_RETAIN = N_PRODUCTION // THIN

# Initial proposal scale.
#
# This is multiplied by the Cholesky factor of the 27-day prior covariance.
#
# The scale is automatically tuned during burn-in.
INITIAL_PROPOSAL_SCALE = 0.10

# Target acceptance for block MH.
TARGET_ACCEPTANCE = 0.30

# Adapt proposal scale every N burn-in sweeps.
ADAPT_INTERVAL = 250

# Checkpoint every N sweeps.
CHECKPOINT_INTERVAL = 1000

# Reproducible random seed.
SEED = 20260918


# =============================================================================
# PHYSICAL / BAYESIAN SETTINGS
# =============================================================================

BACKGROUND_PPM = 1.85

OBS_SIGMA_PPM = 0.05

# Multiplicative prior uncertainty.
#
# median(s) = 1
#
# sigma_log = ln(1.5)
#
# Thus approximately one prior standard deviation corresponds to
# multiplying/dividing the scaling by 1.5.

SIGMA_LOG = np.log(1.5)

TEMPORAL_CORR_LENGTH_DAYS = 3.0


# =============================================================================
# SOURCE NAMES
# =============================================================================

mine_names = [
    f"OC{i}"
    for i in range(1, 14)
]

dates = pd.date_range(
    "2026-01-05",
    "2026-01-31",
    freq="D"
)

if len(dates) != N_DAYS:
    raise RuntimeError(
        f"Expected {N_DAYS} days, got {len(dates)}"
    )


# =============================================================================
# HEADER
# =============================================================================

print("=" * 90)
print("UPPER HUNTER CH4")
print("BLOCKED METROPOLIS-HASTINGS MCMC")
print("=" * 90)

print()
print("Production configuration:")
print(f"  Parameters              : {NDIM}")
print(f"  Mines                   : {N_MINES}")
print(f"  Days                    : {N_DAYS}")
print(f"  Block size              : {BLOCK_SIZE}")
print(f"  Burn-in sweeps          : {BURN_IN}")
print(f"  Production sweeps      : {N_PRODUCTION}")
print(f"  Thinning                : {THIN}")
print(f"  Retained samples        : {N_RETAIN}")
print(f"  Initial proposal scale  : {INITIAL_PROPOSAL_SCALE}")
print(f"  Target acceptance       : {TARGET_ACCEPTANCE}")
print(f"  Observation sigma       : {OBS_SIGMA_PPM} ppm")
print(f"  Background              : {BACKGROUND_PPM} ppm")
print(f"  Log prior sigma         : {SIGMA_LOG}")
print(f"  Temporal correlation L  : {TEMPORAL_CORR_LENGTH_DAYS} days")
print(f"  Random seed             : {SEED}")


# =============================================================================
# LOAD H
# =============================================================================

print()
print("=" * 90)
print("READING H MATRIX")
print("=" * 90)

print(H_FILE)

if not H_FILE.exists():
    raise FileNotFoundError(
        f"H matrix not found:\n{H_FILE}"
    )

H = np.load(
    H_FILE
)

if H.shape != (1124, 351):
    raise ValueError(
        f"Expected H shape (1124,351), got {H.shape}"
    )

print(
    "H shape:",
    H.shape
)

print(
    "H min/max:",
    H.min(),
    H.max()
)


# =============================================================================
# LOAD OBSERVATIONS
# =============================================================================

print()
print("=" * 90)
print("READING OBSERVATIONS")
print("=" * 90)

obs = pd.read_csv(
    OBS_FILE
)

if "observation_ppm" not in obs.columns:
    raise ValueError(
        "observation_vector.csv does not contain observation_ppm"
    )

y_obs = (
    obs["observation_ppm"]
    .to_numpy(dtype=float)
)

if len(y_obs) != H.shape[0]:
    raise ValueError(
        f"Observation count {len(y_obs)} "
        f"does not match H rows {H.shape[0]}"
    )

y = (
    y_obs
    -
    BACKGROUND_PPM
)

print(
    "Observations:",
    len(y_obs)
)

print(
    "Observed CH4:",
    y_obs.min(),
    "to",
    y_obs.max(),
    "ppm"
)

print(
    "Enhancement:",
    y.min(),
    "to",
    y.max(),
    "ppm"
)


# =============================================================================
# LOAD PRIOR EMISSIONS
# =============================================================================

print()
print("=" * 90)
print("READING PRIOR MINE EMISSIONS")
print("=" * 90)

prior_df = pd.read_csv(
    PRIOR_FILE
)

prior_g_s = {}

for mine in mine_names:

    rows = prior_df[
        prior_df[
            "Facility_code_name"
        ]
        .astype(str)
        .str.strip()
        == mine
    ]

    if len(rows) == 0:

        raise ValueError(
            f"Prior emission for {mine} not found."
        )

    prior_g_s[mine] = float(
        rows.iloc[0]["gCH4/s"]
    )


for mine in mine_names:

    print(
        f"{mine:>5s}: "
        f"{prior_g_s[mine]:12.4f} g CH4/s"
    )

total_prior_g_s = sum(
    prior_g_s.values()
)

print(
    f"{'TOTAL':>5s}: "
    f"{total_prior_g_s:12.4f} g CH4/s"
)


# =============================================================================
# PARAMETER ORDER
# =============================================================================
#
# Same order as H:
#
# Jan 5:
#   OC1 ... OC13
# Jan 6:
#   OC1 ... OC13
# ...
# Jan 31:
#   OC1 ... OC13
#
# =============================================================================

parameters = []

for date in dates:

    datestr = date.strftime(
        "%Y%m%d"
    )

    for mine in mine_names:

        parameters.append(
            f"{mine}_{datestr}"
        )


if len(parameters) != NDIM:

    raise ValueError(
        f"Expected {NDIM} parameters, "
        f"got {len(parameters)}"
    )


# =============================================================================
# PRIOR EMISSION VECTOR
# =============================================================================

E0_g_s = np.array(
    [
        prior_g_s[mine]
        for date in dates
        for mine in mine_names
    ],
    dtype=float
)

# g/s -> kg/s
E0_kg_s = (
    E0_g_s
    /
    1000.0
)


# =============================================================================
# BUILD G
# =============================================================================

print()
print("=" * 90)
print("BUILDING FORWARD SENSITIVITY G")
print("=" * 90)

G = (
    H
    *
    E0_kg_s[
        np.newaxis,
        :
    ]
)

print(
    "G shape:",
    G.shape
)

print(
    "G min/max:",
    G.min(),
    G.max()
)


# =============================================================================
# BUILD TEMPORAL PRIOR COVARIANCE
# =============================================================================

print()
print("=" * 90)
print("BUILDING TEMPORAL PRIOR")
print("=" * 90)

day_index = np.arange(
    N_DAYS
)

distance = np.abs(
    day_index[:, None]
    -
    day_index[None, :]
)

Corr_day = np.exp(
    -
    distance
    /
    TEMPORAL_CORR_LENGTH_DAYS
)

B_day = (
    SIGMA_LOG ** 2
    *
    Corr_day
)

print(
    "B_day shape:",
    B_day.shape
)

print(
    "B_day diagonal:",
    np.diag(B_day)[0]
)

print(
    "Correlation day 1:",
    Corr_day[0, 1]
)

print(
    "Correlation day 3:",
    Corr_day[0, 3]
)


# =============================================================================
# CHOLESKY OF BLOCK PRIOR
# =============================================================================

print()
print(
    "Computing Cholesky factor of 27-day prior..."
)

L_day = np.linalg.cholesky(
    B_day
    +
    1e-10
    *
    np.eye(N_DAYS)
)

print(
    "Cholesky successful."
)


# =============================================================================
# FULL PRIOR PRECISION MATRIX
# =============================================================================
#
# Since different mines are independent:
#
# Bz is block diagonal.
#
# Instead of explicitly inverting the full 351x351 matrix using
# a generic inverse, invert the 27x27 block once and repeat it.
#
# =============================================================================

print()
print(
    "Computing prior precision..."
)

B_day_inv = np.linalg.solve(
    B_day
    +
    1e-10
    *
    np.eye(N_DAYS),
    np.eye(N_DAYS)
)

B_day_inv = (
    0.5
    *
    (
        B_day_inv
        +
        B_day_inv.T
    )
)

print(
    "27x27 prior precision ready."
)


# =============================================================================
# OBSERVATION ERROR
# =============================================================================

RINV = (
    1.0
    /
    OBS_SIGMA_PPM ** 2
)


# =============================================================================
# LIKELIHOOD INITIALIZATION
# =============================================================================
#
# We maintain:
#
#     s
#     model = Gs
#     residual = y - model
#
# This lets us update one mine block cheaply.
#
# =============================================================================

# Start with prior scaling = 1.
s_current = np.ones(
    NDIM,
    dtype=float
)


# =============================================================================
# OPTIONAL ANALYTICAL INITIALIZATION
# =============================================================================
#
# The unconstrained analytical posterior can contain negative factors.
#
# We DO NOT use those negative values.
#
# If the analytical posterior file exists, use only positive information
# from it as an initial point, with a conservative lower floor.
#
# This affects initialization only.
#
# =============================================================================

INITIAL_FLOOR = 0.05

if ANALYTICAL_FILE.exists():

    print()
    print(
        "Analytical posterior file found:"
    )
    print(
        ANALYTICAL_FILE
    )

    analytical = pd.read_csv(
        ANALYTICAL_FILE
    )

    if (
        "parameter" in analytical.columns
        and
        "posterior_mean" in analytical.columns
    ):

        analytical_map = dict(
            zip(
                analytical["parameter"],
                analytical["posterior_mean"]
            )
        )

        positive_count = 0

        for j, p in enumerate(
            parameters
        ):

            value = analytical_map.get(
                p,
                1.0
            )

            if (
                np.isfinite(value)
                and
                value > INITIAL_FLOOR
            ):

                s_current[j] = value
                positive_count += 1

            else:

                s_current[j] = INITIAL_FLOOR

        print(
            "Positive analytical initial values:",
            positive_count,
            "/",
            NDIM
        )

        print(
            "Initial scaling range:",
            s_current.min(),
            "to",
            s_current.max()
        )

    else:

        print(
            "Analytical file does not contain expected columns."
        )

else:

    print(
        "Analytical posterior file not found."
    )

    print(
        "Starting all factors at 1."
    )


# =============================================================================
# LOG PARAMETERS
# =============================================================================

z_current = np.log(
    s_current
)


# =============================================================================
# INITIAL MODEL / RESIDUAL
# =============================================================================

model_current = (
    G
    @
    s_current
)

residual_current = (
    y
    -
    model_current
)


# =============================================================================
# PRIOR PRECISION VECTOR
# =============================================================================
#
# Because Bz is block diagonal, calculate Pz block-by-block.
#
# =============================================================================

prior_precision_z = np.zeros(
    NDIM,
    dtype=float
)

for m in range(
    N_MINES
):

    start = (
        m
        *
        BLOCK_SIZE
    )

    stop = (
        start
        +
        BLOCK_SIZE
    )

    prior_precision_z[
        start:stop
    ] = (
        B_day_inv
        @
        z_current[
            start:stop
        ]
    )


# =============================================================================
# LOG POSTERIOR
# =============================================================================

def calculate_log_posterior_from_current():

    # Likelihood
    chi2 = (
        RINV
        *
        np.dot(
            residual_current,
            residual_current
        )
    )

    log_like = (
        -0.5
        *
        chi2
    )

    # Prior
    prior_quad = 0.0

    for m in range(
        N_MINES
    ):

        start = (
            m
            *
            BLOCK_SIZE
        )

        stop = (
            start
            +
            BLOCK_SIZE
        )

        z_block = (
            z_current[
                start:stop
            ]
        )

        prior_quad += np.dot(
            z_block,
            B_day_inv @ z_block
        )

    log_prior = (
        -0.5
        *
        prior_quad
    )

    return (
        log_like
        +
        log_prior
    )


logp_current = (
    calculate_log_posterior_from_current()
)


print()
print(
    "Initial model CH4 range:"
)

model_total_initial = (
    model_current
    +
    BACKGROUND_PPM
)

print(
    model_total_initial.min(),
    "to",
    model_total_initial.max(),
    "ppm"
)

print(
    "Initial log posterior:",
    logp_current
)


# =============================================================================
# RANDOM NUMBER GENERATOR
# =============================================================================

rng = np.random.default_rng(
    SEED
)


# =============================================================================
# BLOCK INDICES
# =============================================================================

block_indices = []

for m in range(
    N_MINES
):

    start = (
        m
        *
        BLOCK_SIZE
    )

    stop = (
        start
        +
        BLOCK_SIZE
    )

    block_indices.append(
        np.arange(
            start,
            stop
        )
    )


# =============================================================================
# PROPOSAL SCALES
# =============================================================================

proposal_scales = np.full(
    N_MINES,
    INITIAL_PROPOSAL_SCALE,
    dtype=float
)


# =============================================================================
# STORAGE
# =============================================================================

samples = np.zeros(
    (
        N_RETAIN,
        NDIM
    ),
    dtype=np.float64
)

sample_logp = np.zeros(
    N_RETAIN,
    dtype=np.float64
)

trace_rows = []


# Acceptance counters
accepted_total = np.zeros(
    N_MINES,
    dtype=np.int64
)

proposed_total = np.zeros(
    N_MINES,
    dtype=np.int64
)

accepted_burn = np.zeros(
    N_MINES,
    dtype=np.int64
)

proposed_burn = np.zeros(
    N_MINES,
    dtype=np.int64
)

accepted_prod = np.zeros(
    N_MINES,
    dtype=np.int64
)

proposed_prod = np.zeros(
    N_MINES,
    dtype=np.int64
)


# =============================================================================
# CHECKPOINT FILE
# =============================================================================

checkpoint_file = (
    OUTPUT
    /
    "mcmc_checkpoint.npz"
)


# =============================================================================
# PRODUCTION CHAIN FILE
# =============================================================================

samples_file = (
    OUTPUT
    /
    "mcmc_samples.npy"
)


# =============================================================================
# BLOCK UPDATE FUNCTION
# =============================================================================

def update_block(
    mine_index,
    logp_current
):

    """
    Perform one blocked MH update for one mine.

    Updates the 27 daily log scaling factors for that mine.

    Returns:

        accepted,
        new_logp
    """

    indices = block_indices[
        mine_index
    ]

    # Current block
    z_old = (
        z_current[
            indices
        ]
        .copy()
    )

    s_old = (
        s_current[
            indices
        ]
        .copy()
    )

    # -------------------------------------------------------------------------
    # Correlated proposal
    # -------------------------------------------------------------------------

    epsilon = rng.normal(
        0.0,
        1.0,
        BLOCK_SIZE
    )

    delta_z = (
        proposal_scales[
            mine_index
        ]
        *
        (
            L_day
            @
            epsilon
        )
    )

    z_new = (
        z_old
        +
        delta_z
    )

    # Numerical safeguard.
    #
    # This corresponds to scaling factors between approximately
    #
    #     exp(-8) = 0.00034
    #
    # and
    #
    #     exp(+8) = 2981
    #
    if np.any(
        z_new < -8.0
    ):

        return (
            False,
            logp_current
        )

    if np.any(
        z_new > 8.0
    ):

        return (
            False,
            logp_current
        )

    s_new = np.exp(
        z_new
    )

    # -------------------------------------------------------------------------
    # Likelihood update
    # -------------------------------------------------------------------------
    #
    # Only the contribution from this mine changes.
    #
    # delta_model = G_block @ (s_new - s_old)
    #
    # -------------------------------------------------------------------------

    delta_s = (
        s_new
        -
        s_old
    )

    G_block = (
        G[
            :,
            indices
        ]
    )

    delta_model = (
        G_block
        @
        delta_s
    )

    residual_new = (
        residual_current
        -
        delta_model
    )

    # Difference in likelihood quadratic:
    #
    # ||r_new||² - ||r_old||²

    old_r2 = np.dot(
        residual_current,
        residual_current
    )

    new_r2 = np.dot(
        residual_new,
        residual_new
    )

    delta_log_like = (
        -0.5
        *
        RINV
        *
        (
            new_r2
            -
            old_r2
        )
    )

    # -------------------------------------------------------------------------
    # Prior update
    # -------------------------------------------------------------------------

    old_prior_block = (
        B_day_inv
        @
        z_old
    )

    new_prior_block = (
        B_day_inv
        @
        z_new
    )

    old_quad = np.dot(
        z_old,
        old_prior_block
    )

    new_quad = np.dot(
        z_new,
        new_prior_block
    )

    delta_log_prior = (
        -0.5
        *
        (
            new_quad
            -
            old_quad
        )
    )

    # -------------------------------------------------------------------------
    # Posterior ratio
    # -------------------------------------------------------------------------

    delta_logp = (
        delta_log_like
        +
        delta_log_prior
    )

    # -------------------------------------------------------------------------
    # MH decision
    # -------------------------------------------------------------------------

    accepted = False

    if (
        delta_logp >= 0.0
        or
        np.log(
            rng.random()
        )
        <
        delta_logp
    ):

        accepted = True

        # Update z
        z_current[
            indices
        ] = z_new

        # Update s
        s_current[
            indices
        ] = s_new

        # Update residual
        residual_current[:] = (
            residual_new
        )

        # Update model
        model_current[:] = (
            model_current
            +
            delta_model
        )

        # Update prior precision vector
        prior_precision_z[
            indices
        ] = (
            new_prior_block
        )

        # Update posterior
        logp_current = (
            logp_current
            +
            delta_logp
        )

    return (
        accepted,
        logp_current
    )


# =============================================================================
# START MCMC
# =============================================================================

print()
print("=" * 90)
print("STARTING BLOCKED METROPOLIS-HASTINGS")
print("=" * 90)

start_time = time.time()

total_sweeps = (
    BURN_IN
    +
    N_PRODUCTION
)

sample_index = 0


# =============================================================================
# MCMC LOOP
# =============================================================================

for sweep in range(
    1,
    total_sweeps + 1
):

    in_burn = (
        sweep
        <=
        BURN_IN
    )

    # -------------------------------------------------------------------------
    # Randomize block order each sweep.
    # -------------------------------------------------------------------------

    block_order = rng.permutation(
        N_MINES
    )

    for mine_index in block_order:

        proposed_total[
            mine_index
        ] += 1

        if in_burn:

            proposed_burn[
                mine_index
            ] += 1

        else:

            proposed_prod[
                mine_index
            ] += 1

        accepted, logp_current = (
            update_block(
                mine_index,
                logp_current
            )
        )

        if accepted:

            accepted_total[
                mine_index
            ] += 1

            if in_burn:

                accepted_burn[
                    mine_index
                ] += 1

            else:

                accepted_prod[
                    mine_index
                ] += 1

    # -------------------------------------------------------------------------
    # Adaptive proposal tuning during burn-in.
    # -------------------------------------------------------------------------

    if (
        in_burn
        and
        sweep % ADAPT_INTERVAL == 0
    ):

        for m in range(
            N_MINES
        ):

            if proposed_burn[m] > 0:

                acc = (
                    accepted_burn[m]
                    /
                    proposed_burn[m]
                )

                # Multiplicative Robbins-Monro-like adjustment.
                #
                # Conservative factor prevents violent changes.

                if acc < 0.15:

                    proposal_scales[m] *= 0.80

                elif acc < 0.22:

                    proposal_scales[m] *= 0.90

                elif acc > 0.50:

                    proposal_scales[m] *= 1.20

                elif acc > 0.40:

                    proposal_scales[m] *= 1.10

                # Keep proposal in sensible numerical range.
                proposal_scales[m] = np.clip(
                    proposal_scales[m],
                    0.005,
                    2.0
                )

        # Reset burn-in counters after each adaptation window.
        accepted_burn[:] = 0
        proposed_burn[:] = 0

    # -------------------------------------------------------------------------
    # Save production sample.
    # -------------------------------------------------------------------------

    if (
        sweep > BURN_IN
        and
        (
            (sweep - BURN_IN)
            %
            THIN
            ==
            0
        )
    ):

        samples[
            sample_index,
            :
        ] = s_current

        sample_logp[
            sample_index
        ] = logp_current

        sample_index += 1

    # -------------------------------------------------------------------------
    # Progress output.
    # -------------------------------------------------------------------------

    if (
        sweep == 1
        or
        sweep % 250 == 0
        or
        sweep == total_sweeps
    ):

        overall_acceptance = (
            accepted_total.sum()
            /
            proposed_total.sum()
        )

        elapsed = (
            time.time()
            -
            start_time
        )

        phase = (
            "BURN-IN"
            if in_burn
            else "PRODUCTION"
        )

        print(
            f"Sweep {sweep:6d}/{total_sweeps} "
            f"{phase:10s} "
            f"| logP = {logp_current:14.3f} "
            f"| accept = {overall_acceptance:7.4f} "
            f"| elapsed = {elapsed/60.0:8.2f} min"
        )

        trace_rows.append(
            {
                "sweep": sweep,
                "phase": phase,
                "log_posterior": logp_current,
                "overall_acceptance":
                    overall_acceptance,
                "mean_proposal_scale":
                    proposal_scales.mean(),
            }
        )

    # -------------------------------------------------------------------------
    # Checkpoint.
    # -------------------------------------------------------------------------

    if (
        sweep % CHECKPOINT_INTERVAL
        == 0
    ):

        np.savez_compressed(
            checkpoint_file,

            sweep=sweep,

            z_current=z_current,

            s_current=s_current,

            model_current=model_current,

            residual_current=residual_current,

            prior_precision_z=prior_precision_z,

            logp_current=logp_current,

            proposal_scales=proposal_scales,

            accepted_total=accepted_total,

            proposed_total=proposed_total,

            sample_index=sample_index,

            samples=samples,

            sample_logp=sample_logp,

            seed=SEED,
        )


# =============================================================================
# FINALIZE SAMPLES
# =============================================================================

samples = samples[
    :sample_index,
    :
]

sample_logp = sample_logp[
    :sample_index
]


# =============================================================================
# SAVE RAW SAMPLES
# =============================================================================

np.save(
    samples_file,
    samples
)

np.save(
    OUTPUT / "mcmc_logposterior.npy",
    sample_logp
)

print()
print(
    "Saved raw posterior samples:"
)

print(
    samples_file
)

print(
    "Sample shape:",
    samples.shape
)


# =============================================================================
# BASIC MCMC DIAGNOSTICS
# =============================================================================

overall_acceptance = (
    accepted_total.sum()
    /
    proposed_total.sum()
)

block_acceptance = (
    accepted_total
    /
    np.maximum(
        proposed_total,
        1
    )
)

production_block_acceptance = (
    accepted_prod
    /
    np.maximum(
        proposed_prod,
        1
    )
)


print()
print("=" * 90)
print("MCMC ACCEPTANCE DIAGNOSTICS")
print("=" * 90)

print()
print(
    "Overall acceptance:",
    overall_acceptance
)

print()
print(
    "Production block acceptance:"
)

for i, mine in enumerate(
    mine_names
):

    print(
        f"  {mine:>5s}: "
        f"{production_block_acceptance[i]:.4f} "
        f"proposal={proposal_scales[i]:.5f}"
    )


# =============================================================================
# POSTERIOR STATISTICS
# =============================================================================

posterior_mean = np.mean(
    samples,
    axis=0
)

posterior_median = np.median(
    samples,
    axis=0
)

posterior_sd = np.std(
    samples,
    axis=0
)

posterior_lower95 = np.percentile(
    samples,
    2.5,
    axis=0
)

posterior_upper95 = np.percentile(
    samples,
    97.5,
    axis=0
)


# =============================================================================
# VERIFY POSITIVITY
# =============================================================================

minimum_sample = samples.min()

maximum_sample = samples.max()

all_positive = bool(
    np.all(
        samples > 0.0
    )
)

print()
print("=" * 90)
print("POSITIVITY CHECK")
print("=" * 90)

print(
    "Minimum sampled scaling:",
    minimum_sample
)

print(
    "Maximum sampled scaling:",
    maximum_sample
)

print(
    "All samples positive:",
    all_positive
)

if not all_positive:

    raise RuntimeError(
        "ERROR: Negative/non-positive MCMC scaling detected."
    )


# =============================================================================
# EMISSION POSTERIOR
# =============================================================================

emission_samples = (
    samples
    *
    E0_kg_s[
        np.newaxis,
        :
    ]
)

posterior_emission_mean = (
    emission_samples.mean(
        axis=0
    )
)

posterior_emission_median = (
    np.median(
        emission_samples,
        axis=0
    )
)

posterior_emission_lower95 = (
    np.percentile(
        emission_samples,
        2.5,
        axis=0
    )
)

posterior_emission_upper95 = (
    np.percentile(
        emission_samples,
        97.5,
        axis=0
    )
)


# =============================================================================
# PARAMETER SUMMARY
# =============================================================================

summary = pd.DataFrame(
    {
        "parameter": parameters,

        "mine": [
            p.split("_")[0]
            for p in parameters
        ],

        "date": pd.to_datetime(
            [
                p.split("_")[1]
                for p in parameters
            ],
            format="%Y%m%d"
        ),

        "prior_scaling": 1.0,

        "posterior_mean":
            posterior_mean,

        "posterior_median":
            posterior_median,

        "posterior_sd":
            posterior_sd,

        "posterior_lower_95":
            posterior_lower95,

        "posterior_upper_95":
            posterior_upper95,

        "prior_emission_kg_s":
            E0_kg_s,

        "posterior_emission_kg_s":
            posterior_emission_mean,

        "posterior_emission_median_kg_s":
            posterior_emission_median,

        "posterior_emission_lower_kg_s":
            posterior_emission_lower95,

        "posterior_emission_upper_kg_s":
            posterior_emission_upper95,
    }
)

summary_file = (
    OUTPUT
    /
    "mcmc_scaling_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print()
print(
    "Saved:"
)
print(
    summary_file
)


# =============================================================================
# DAILY MEAN TABLE
# =============================================================================

daily_mean = summary.pivot(
    index="date",
    columns="mine",
    values="posterior_mean"
)

daily_mean = daily_mean.reindex(
    columns=mine_names
)

daily_mean_file = (
    OUTPUT
    /
    "mcmc_daily_scaling.csv"
)

daily_mean.to_csv(
    daily_mean_file
)


# =============================================================================
# DAILY LOWER 95%
# =============================================================================

daily_lower = summary.pivot(
    index="date",
    columns="mine",
    values="posterior_lower_95"
)

daily_lower = daily_lower.reindex(
    columns=mine_names
)

daily_lower_file = (
    OUTPUT
    /
    "mcmc_daily_scaling_lower95.csv"
)

daily_lower.to_csv(
    daily_lower_file
)


# =============================================================================
# DAILY UPPER 95%
# =============================================================================

daily_upper = summary.pivot(
    index="date",
    columns="mine",
    values="posterior_upper_95"
)

daily_upper = daily_upper.reindex(
    columns=mine_names
)

daily_upper_file = (
    OUTPUT
    /
    "mcmc_daily_scaling_upper95.csv"
)

daily_upper.to_csv(
    daily_upper_file
)


# =============================================================================
# MINE SUMMARY
# =============================================================================

mine_rows = []

for mine in mine_names:

    sub = summary[
        summary["mine"] == mine
    ]

    mine_rows.append(
        {
            "mine": mine,

            "prior_emission_kg_s":
                sub[
                    "prior_emission_kg_s"
                ].iloc[0],

            "mean_daily_scaling":
                sub[
                    "posterior_mean"
                ].mean(),

            "median_daily_scaling":
                sub[
                    "posterior_median"
                ].median(),

            "mean_daily_sd":
                sub[
                    "posterior_sd"
                ].mean(),

            "minimum_daily_mean":
                sub[
                    "posterior_mean"
                ].min(),

            "maximum_daily_mean":
                sub[
                    "posterior_mean"
                ].max(),

            "minimum_lower95":
                sub[
                    "posterior_lower_95"
                ].min(),

            "maximum_upper95":
                sub[
                    "posterior_upper_95"
                ].max(),

            "mean_posterior_emission_kg_s":
                sub[
                    "posterior_emission_kg_s"
                ].mean(),

            "mean_posterior_emission_lower_kg_s":
                sub[
                    "posterior_emission_lower_kg_s"
                ].mean(),

            "mean_posterior_emission_upper_kg_s":
                sub[
                    "posterior_emission_upper_kg_s"
                ].mean(),
        }
    )


mine_summary = pd.DataFrame(
    mine_rows
)

mine_summary_file = (
    OUTPUT
    /
    "mcmc_mine_summary.csv"
)

mine_summary.to_csv(
    mine_summary_file,
    index=False
)


# =============================================================================
# BLOCK ACCEPTANCE FILE
# =============================================================================

block_acceptance = pd.DataFrame(
    {
        "mine": mine_names,

        "total_proposals":
            proposed_total,

        "total_accepted":
            accepted_total,

        "total_acceptance":
            block_acceptance,

        "production_proposals":
            proposed_prod,

        "production_accepted":
            accepted_prod,

        "production_acceptance":
            production_block_acceptance,

        "final_proposal_scale":
            proposal_scales,
    }
)

block_acceptance_file = (
    OUTPUT
    /
    "mcmc_block_acceptance.csv"
)

block_acceptance.to_csv(
    block_acceptance_file,
    index=False
)


# =============================================================================
# TRACE FILE
# =============================================================================

trace_df = pd.DataFrame(
    trace_rows
)

trace_file = (
    OUTPUT
    /
    "mcmc_trace_logposterior.csv"
)

trace_df.to_csv(
    trace_file,
    index=False
)


# =============================================================================
# DIAGNOSTICS
# =============================================================================

diagnostic_rows = [
    (
        "ndim",
        NDIM
    ),

    (
        "n_mines",
        N_MINES
    ),

    (
        "n_days",
        N_DAYS
    ),

    (
        "burn_in_sweeps",
        BURN_IN
    ),

    (
        "production_sweeps",
        N_PRODUCTION
    ),

    (
        "thinning",
        THIN
    ),

    (
        "retained_samples",
        len(samples)
    ),

    (
        "initial_proposal_scale",
        INITIAL_PROPOSAL_SCALE
    ),

    (
        "mean_final_proposal_scale",
        proposal_scales.mean()
    ),

    (
        "overall_acceptance",
        overall_acceptance
    ),

    (
        "minimum_scaling",
        minimum_sample
    ),

    (
        "maximum_scaling",
        maximum_sample
    ),

    (
        "all_scalings_positive",
        all_positive
    ),

    (
        "final_log_posterior",
        logp_current
    ),

    (
        "mean_retained_log_posterior",
        sample_logp.mean()
    ),

    (
        "median_retained_log_posterior",
        np.median(sample_logp)
    ),
]

diagnostics_df = pd.DataFrame(
    diagnostic_rows,
    columns=[
        "metric",
        "value"
    ]
)

diagnostics_file = (
    OUTPUT
    /
    "mcmc_diagnostics.csv"
)

diagnostics_df.to_csv(
    diagnostics_file,
    index=False
)


# =============================================================================
# PRINT DAILY RESULTS
# =============================================================================

print()
print("=" * 90)
print("MCMC DAILY POSTERIOR MEAN SCALING FACTORS")
print("=" * 90)

print(
    daily_mean.to_string(
        float_format=lambda x:
        f"{x:10.4f}"
    )
)


# =============================================================================
# PRINT MINE SUMMARY
# =============================================================================

print()
print("=" * 90)
print("MCMC MINE SUMMARY")
print("=" * 90)

print(
    mine_summary.to_string(
        index=False,
        float_format=lambda x:
        f"{x:12.5f}"
    )
)


# =============================================================================
# SAVE FINAL CHECKPOINT
# =============================================================================

np.savez_compressed(
    checkpoint_file,

    sweep=total_sweeps,

    z_current=z_current,

    s_current=s_current,

    model_current=model_current,

    residual_current=residual_current,

    prior_precision_z=prior_precision_z,

    logp_current=logp_current,

    proposal_scales=proposal_scales,

    accepted_total=accepted_total,

    proposed_total=proposed_total,

    sample_index=sample_index,

    samples=samples,

    sample_logp=sample_logp,

    seed=SEED,
)


# =============================================================================
# FINISHED
# =============================================================================

elapsed_total = (
    time.time()
    -
    start_time
)

print()
print("=" * 90)
print("BLOCKED MCMC COMPLETE")
print("=" * 90)

print()
print(
    f"Runtime: "
    f"{elapsed_total/60.0:.2f} minutes"
)

print()
print(
    "Retained samples:",
    len(samples)
)

print(
    "Sample dimensions:",
    samples.shape
)

print(
    "All scaling factors positive:",
    all_positive
)

print()
print(
    "Output directory:"
)

print(
    OUTPUT
)

print()
print(
    "Output files:"
)

for f in sorted(
    OUTPUT.iterdir()
):

    print(
        " ",
        f.name
    )

