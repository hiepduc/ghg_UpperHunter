#!/usr/bin/env python3

"""
BOUNDED FULL-FLEXPART-FOOTPRINT AUGMENTED KALMAN FILTER
=======================================================

Upper Hunter CH4 inverse modelling
13 open-cut mines x daily scaling factors
January 2026

This is the BOUNDED counterpart of:

    run_kf_upperhunter_full_footprint_revised.py

State formulation
-----------------

The FLEXPART footprint matrix has sensitivity to the current day
and, in some cases, the immediately preceding day.

Therefore the augmented state is

    X_d = [ x_{d-1} ]
          [ x_d     ]

where x_d contains the 13 daily mine emission scaling factors.

Observation equation
--------------------

    y_t = BACKGROUND_PPM + h_aug @ X_d + eps_t

with

    h_aug = [H_{t,d-1}, H_{t,d}]

Daily transition
----------------

    x_{d+1}
        =
        rho*x_d
        +
        (1-rho)*PRIOR_MEAN
        +
        eta

where

    rho = exp(-1 / L_CORR)

and

    Q = PRIOR_VARIANCE * (1-rho^2) I

BOUNDING
--------

Every physical mine scaling factor is constrained to

    LOWER_BOUND <= x <= UPPER_BOUND

Default:

    LOWER_BOUND = 0.0
    UPPER_BOUND = 5.0

The bounds apply to both halves of the augmented state:

    previous-day scaling factors
    current-day scaling factors

The observation update first computes the ordinary Gaussian
posterior. If the unconstrained posterior lies inside the bounds,
the ordinary KF result is retained.

If one or more state components violate a bound, the posterior
mean is obtained from the corresponding box-constrained quadratic
MAP problem.

The covariance is then constructed using the active-set
approximation:

    active parameters:
        fixed at the bound
        zero posterior variance

    free parameters:
        covariance from the inverse posterior precision
        conditional on the active parameters

This gives a consistent constrained-Gaussian/Laplace approximation
for propagation through the subsequent KF steps.

IMPORTANT
---------

This is NOT equivalent to simply clipping the output after an
ordinary KF run.

The bounds participate in the sequential inference.

The script also computes the unconstrained full 351-parameter
batch MAP as a diagnostic reference. The unconstrained batch MAP
is NOT expected to equal the bounded KF when the bounds become
active.

Outputs
-------

The script saves:

    daily_bounded_full_footprint_KF_scaling_factors.csv

    hourly_bounded_full_footprint_KF_results.csv

    daily_bounded_full_footprint_KF_covariance_diagonal.csv

    bounded_full_footprint_KF_daily_states.npz

    bounded_full_footprint_KF_vs_unbounded_KF.csv

    bounded_full_footprint_KF_vs_batch_MAP.csv

    bounded_full_footprint_KF_summary.txt

It also records which parameters are active at the lower or upper
bound at every observation update.
"""

import os
import warnings

import numpy as np
import pandas as pd

from scipy.linalg import solve_triangular
from scipy.optimize import minimize


# ============================================================
# CONFIGURATION
# ============================================================

START_DATE = pd.Timestamp("2026-01-05")
END_DATE = pd.Timestamp("2026-01-31")

N_MINE = 13

MINES = [
    f"OC{i}"
    for i in range(1, N_MINE + 1)
]

# ------------------------------------------------------------
# Atmospheric background
# ------------------------------------------------------------

BACKGROUND_PPM = 1.85

# ------------------------------------------------------------
# Prior
# ------------------------------------------------------------

PRIOR_MEAN = 1.0
PRIOR_SIGMA = 0.5
PRIOR_VARIANCE = PRIOR_SIGMA ** 2

# ------------------------------------------------------------
# Observation uncertainty
# ------------------------------------------------------------

OBS_SIGMA = 0.05
OBS_VARIANCE = OBS_SIGMA ** 2

# ------------------------------------------------------------
# Temporal correlation
# ------------------------------------------------------------

L_CORR = 3.0

# ------------------------------------------------------------
# HARD EMISSION-FACTOR BOUNDS
#
# Every mine/day scaling factor must satisfy:
#
#     LOWER_BOUND <= x <= UPPER_BOUND
#
# Change these two values if a different physical range is
# required.
# ------------------------------------------------------------

LOWER_BOUND = 0.0
UPPER_BOUND = 5.0

if LOWER_BOUND >= UPPER_BOUND:

    raise RuntimeError(
        "LOWER_BOUND must be smaller than UPPER_BOUND."
    )

if PRIOR_MEAN < LOWER_BOUND or PRIOR_MEAN > UPPER_BOUND:

    raise RuntimeError(
        "PRIOR_MEAN lies outside the specified bounds."
    )


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    "/home/duch/flexpart-wrf/"
    "upperhunter_flexpart"
)

H_FILE = os.path.join(
    BASE_DIR,
    "H_matrix_flexpart",
    "H_flexpart_0_10m_13mine_Jan2026.csv",
)

PICARRO_FILE = (
    "/home/duch/flexpart-wrf/"
    "emission_UH/RH_upperHunter/"
    "tapm_output/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

# ------------------------------------------------------------
# Existing unbounded KF directory
#
# Used ONLY for optional comparison.
# ------------------------------------------------------------

UNBOUNDED_KF_DIR = os.path.join(
    BASE_DIR,
    "H_matrix_flexpart",
    "KF_full_footprint_13mine_Jan2026",
)

UNBOUNDED_KF_NPZ = os.path.join(
    UNBOUNDED_KF_DIR,
    "full_footprint_KF_daily_states.npz",
)

# ------------------------------------------------------------
# BOUNDED KF output directory
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "H_matrix_flexpart",
    "KF_bounded_full_footprint_13mine_Jan2026",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parameter_name(
    mine,
    date
):
    return (
        f"{mine}_"
        f"{pd.Timestamp(date).strftime('%Y%m%d')}"
    )


def get_day_index(
    timestamp
):
    timestamp = (
        pd.Timestamp(timestamp)
        .normalize()
    )

    delta = (
        timestamp - START_DATE
    ).days

    if delta < 0 or delta > (
        END_DATE - START_DATE
    ).days:

        return None

    return int(delta)


def get_day_block(
    day_index
):
    """
    Return the 13 parameter indices corresponding
    to one calendar day.
    """

    start = (
        day_index * N_MINE
    )

    return np.arange(
        start,
        start + N_MINE,
        dtype=int
    )


def symmetrize(
    A
):
    return (
        0.5
        * (A + A.T)
    )


# ============================================================
# BOUNDED POSTERIOR UPDATE
# ============================================================

def bounded_gaussian_update(
    x,
    P,
    h,
    observation
):
    """
    Perform a scalar-observation bounded Bayesian update.

    Prior:

        x ~ N(x, P)

    Observation:

        y = BACKGROUND_PPM + h x + eps

    with

        eps ~ N(0, OBS_VARIANCE)

    subject to

        LOWER_BOUND <= x_i <= UPPER_BOUND

    The unconstrained KF posterior is first calculated.

    If all posterior means satisfy the bounds, the ordinary KF
    posterior is returned.

    Otherwise the posterior MAP is obtained from the convex
    box-constrained quadratic objective:

        0.5 (x_new-x)^T P^-1 (x_new-x)
        +
        0.5 (y-bg-h@x_new)^2 / R

    The covariance of free variables is obtained from the
    posterior precision conditional on active variables.

    Returns
    -------

    x_new
    P_new
    prediction
    innovation
    innovation_variance
    NIS
    active_lower
    active_upper
    """

    x = np.asarray(
        x,
        dtype=np.float64
    )

    P = np.asarray(
        P,
        dtype=np.float64
    )

    h = np.asarray(
        h,
        dtype=np.float64
    )

    n = len(x)

    # --------------------------------------------------------
    # Numerical covariance cleanup
    # --------------------------------------------------------

    P = symmetrize(P)

    # Small regularisation only if necessary.
    #
    # The covariance should normally remain positive
    # semidefinite, but bounded updates can produce very small
    # numerical eigenvalues.
    # --------------------------------------------------------

    eigenvalues = np.linalg.eigvalsh(P)

    minimum_eigenvalue = np.min(
        eigenvalues
    )

    if minimum_eigenvalue < 1.0e-12:

        P = (
            P
            + (
                1.0e-12
                - minimum_eigenvalue
                + 1.0e-14
            )
            * np.eye(n)
        )

        P = symmetrize(P)

    # --------------------------------------------------------
    # Observation prediction from PRIOR state
    # --------------------------------------------------------

    prediction = (
        BACKGROUND_PPM
        + np.dot(h, x)
    )

    innovation = (
        observation
        - prediction
    )

    innovation_variance = (
        np.dot(
            h,
            np.dot(P, h)
        )
        + OBS_VARIANCE
    )

    if (
        not np.isfinite(
            innovation_variance
        )
        or innovation_variance <= 0.0
    ):

        raise RuntimeError(
            "Invalid innovation variance: "
            f"{innovation_variance}"
        )

    # --------------------------------------------------------
    # Ordinary unconstrained KF posterior
    # --------------------------------------------------------

    PHt = (
        P @ h
    )

    K = (
        PHt
        / innovation_variance
    )

    x_unconstrained = (
        x
        + K * innovation
    )

    # --------------------------------------------------------
    # Ordinary Joseph covariance
    # --------------------------------------------------------

    I = np.eye(
        n,
        dtype=np.float64
    )

    KH = np.outer(
        K,
        h
    )

    IKH = (
        I - KH
    )

    P_unconstrained = (
        IKH
        @ P
        @ IKH.T
        +
        OBS_VARIANCE
        * np.outer(K, K)
    )

    P_unconstrained = symmetrize(
        P_unconstrained
    )

    # --------------------------------------------------------
    # Determine whether bounds are active
    # --------------------------------------------------------

    lower_violation = (
        x_unconstrained
        < LOWER_BOUND
    )

    upper_violation = (
        x_unconstrained
        > UPPER_BOUND
    )

    active = (
        lower_violation
        | upper_violation
    )

    # --------------------------------------------------------
    # FAST PATH:
    #
    # No active bounds -> ordinary KF
    # --------------------------------------------------------

    if not np.any(active):

        nis = (
            innovation ** 2
            / innovation_variance
        )

        return (
            x_unconstrained,
            P_unconstrained,
            prediction,
            innovation,
            innovation_variance,
            nis,
            lower_violation,
            upper_violation,
            False,
        )

    # ========================================================
    # CONSTRAINED MAP
    # ========================================================

    # --------------------------------------------------------
    # Precision of prior
    # --------------------------------------------------------

    P_inv = np.linalg.inv(
        P
    )

    P_inv = symmetrize(
        P_inv
    )

    # --------------------------------------------------------
    # Objective
    #
    # f(z) =
    #   0.5(z-x)^T P^-1(z-x)
    #   +
    #   0.5(y-bg-hz)^2/R
    #
    # Gradient:
    #
    # P^-1(z-x)
    # -
    # h*(y-bg-hz)/R
    # --------------------------------------------------------

    def objective(z):

        dz = (
            z - x
        )

        residual = (
            observation
            - BACKGROUND_PPM
            - np.dot(h, z)
        )

        return (
            0.5
            * np.dot(
                dz,
                P_inv @ dz
            )
            +
            0.5
            * residual ** 2
            / OBS_VARIANCE
        )

    def gradient(z):

        dz = (
            z - x
        )

        residual = (
            observation
            - BACKGROUND_PPM
            - np.dot(h, z)
        )

        return (
            P_inv @ dz
            -
            h
            * residual
            / OBS_VARIANCE
        )

    # --------------------------------------------------------
    # Use the clipped unconstrained solution as the starting
    # point. This is already close to the constrained optimum.
    # --------------------------------------------------------

    x_start = np.clip(
        x_unconstrained,
        LOWER_BOUND,
        UPPER_BOUND
    )

    bounds = [
        (
            LOWER_BOUND,
            UPPER_BOUND
        )
        for _ in range(n)
    ]

    result = minimize(
        objective,
        x_start,
        jac=gradient,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "ftol": 1.0e-14,
            "gtol": 1.0e-10,
            "maxiter": 2000,
            "maxls": 50,
        },
    )

    if not result.success:

        # L-BFGS-B can occasionally report a line-search
        # warning even when the solution is effectively
        # converged. Check the gradient before failing.

        grad_norm = np.linalg.norm(
            gradient(result.x)
        )

        if grad_norm > 1.0e-7:

            raise RuntimeError(
                "Bounded posterior optimisation failed:\n"
                + str(result.message)
                + f"\nGradient norm = {grad_norm}"
            )

    x_new = np.asarray(
        result.x,
        dtype=np.float64
    )

    # Final hard numerical projection
    x_new = np.clip(
        x_new,
        LOWER_BOUND,
        UPPER_BOUND
    )

    # --------------------------------------------------------
    # Identify truly active parameters
    # --------------------------------------------------------

    tolerance = 1.0e-9

    active_lower = (
        x_new
        <= (
            LOWER_BOUND
            + tolerance
        )
    )

    active_upper = (
        x_new
        >= (
            UPPER_BOUND
            - tolerance
        )
    )

    active = (
        active_lower
        | active_upper
    )

    # ========================================================
    # CONSTRAINED POSTERIOR COVARIANCE
    # ========================================================

    # Posterior precision:
    #
    # Lambda =
    # P^-1 + h^T R^-1 h
    #
    # where h^T R^-1 h is rank one.
    # --------------------------------------------------------

    Lambda = (
        P_inv
        +
        np.outer(h, h)
        / OBS_VARIANCE
    )

    Lambda = symmetrize(
        Lambda
    )

    free = np.flatnonzero(
        ~active
    )

    P_new = np.zeros(
        (
            n,
            n
        ),
        dtype=np.float64
    )

    if len(free) > 0:

        Lambda_ff = Lambda[
            np.ix_(
                free,
                free
            )
        ]

        Lambda_ff = symmetrize(
            Lambda_ff
        )

        P_free = np.linalg.inv(
            Lambda_ff
        )

        P_free = symmetrize(
            P_free
        )

        P_new[
            np.ix_(
                free,
                free
            )
        ] = P_free

    P_new = symmetrize(
        P_new
    )

    # --------------------------------------------------------
    # NIS
    #
    # Retain the innovation based on the prior state, exactly
    # as in the ordinary KF diagnostic.
    # --------------------------------------------------------

    nis = (
        innovation ** 2
        / innovation_variance
    )

    return (
        x_new,
        P_new,
        prediction,
        innovation,
        innovation_variance,
        nis,
        active_lower,
        active_upper,
        True,
    )


# ============================================================
# DAILY TRANSITION
# ============================================================

def transition_to_next_day(
    x_aug,
    P_aug
):
    """
    Transition

        X_d = [x_{d-1}]
              [x_d    ]

    to

        X_{d+1} = [x_d    ]
                  [x_{d+1}]

    with

        x_{d+1}
          =
          rho*x_d
          +
          (1-rho)*PRIOR_MEAN
          +
          eta

    and

        Cov(eta) = Q_daily.

    The transition itself is not optimised. The hard bounds are
    enforced at every observation update.
    """

    F_aug = np.zeros(
        (
            2 * N_MINE,
            2 * N_MINE
        ),
        dtype=np.float64
    )

    # New previous state = old current state

    F_aug[
        :N_MINE,
        N_MINE:
    ] = np.eye(
        N_MINE
    )

    # New current state = rho * old current state

    F_aug[
        N_MINE:,
        N_MINE:
    ] = (
        rho
        * np.eye(
            N_MINE
        )
    )

    c_aug = np.zeros(
        2 * N_MINE,
        dtype=np.float64
    )

    c_aug[
        N_MINE:
    ] = (
        (1.0 - rho)
        * PRIOR_MEAN
    )

    Q_aug = np.zeros(
        (
            2 * N_MINE,
            2 * N_MINE
        ),
        dtype=np.float64
    )

    Q_aug[
        N_MINE:,
        N_MINE:
    ] = Q_daily

    x_new = (
        F_aug
        @ x_aug
        +
        c_aug
    )

    P_new = (
        F_aug
        @ P_aug
        @ F_aug.T
        +
        Q_aug
    )

    P_new = symmetrize(
        P_new
    )

    # --------------------------------------------------------
    # Keep the forecast state inside the physical domain.
    #
    # Normally the AR(1) forecast should remain inside the
    # bounds when the previous state is inside them and the
    # prior mean is inside them. The stochastic covariance does
    # not alter the mean.
    # --------------------------------------------------------

    x_new[
        :N_MINE
    ] = np.clip(
        x_new[
            :N_MINE
        ],
        LOWER_BOUND,
        UPPER_BOUND
    )

    x_new[
        N_MINE:
    ] = np.clip(
        x_new[
            N_MINE:
        ],
        LOWER_BOUND,
        UPPER_BOUND
    )

    return (
        x_new,
        P_new
    )


# ============================================================
# HEADER
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "BOUNDED FULL-FLEXPART-FOOTPRINT AUGMENTED KALMAN FILTER"
)

print(
    "=" * 90
)

print(
    "Start date:",
    START_DATE.strftime("%Y-%m-%d")
)

print(
    "End date:",
    END_DATE.strftime("%Y-%m-%d")
)

print(
    "Number of mines:",
    N_MINE
)

N_DAY = (
    END_DATE
    - START_DATE
).days + 1

N_PARAM = (
    N_MINE
    * N_DAY
)

print(
    "Number of days:",
    N_DAY
)

print(
    "Number of parameters:",
    N_PARAM
)

print(
    "Prior mean:",
    PRIOR_MEAN
)

print(
    "Prior sigma:",
    PRIOR_SIGMA
)

print(
    "Observation sigma:",
    OBS_SIGMA
)

print(
    "Correlation length:",
    L_CORR,
    "days"
)

print(
    "Lower emission-factor bound:",
    LOWER_BOUND
)

print(
    "Upper emission-factor bound:",
    UPPER_BOUND
)


# ============================================================
# READ H MATRIX
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "READING FLEXPART H MATRIX"
)

print(
    "=" * 90
)

H_df = pd.read_csv(
    H_FILE
)

print(
    "H shape:",
    H_df.shape
)

metadata_columns = [
    "receptor",
    "release_utc",
    "obs_time_aest",
    "flexpart_file",
]

for column in metadata_columns:

    if column not in H_df.columns:

        raise RuntimeError(
            f"Missing H metadata column: {column}"
        )

parameter_columns = [
    c
    for c in H_df.columns
    if c not in metadata_columns
]

expected_parameter_count = N_PARAM

if len(parameter_columns) != (
    expected_parameter_count
):

    raise RuntimeError(
        "Unexpected number of H parameters: "
        f"{len(parameter_columns)} "
        f"expected {expected_parameter_count}"
    )

print(
    "Number of parameters:",
    len(parameter_columns)
)


# ============================================================
# SORT H PARAMETERS INTO DAY-MAJOR ORDER
# ============================================================

expected_parameter_names = []

for day_offset in range(
    N_DAY
):

    day = (
        START_DATE
        +
        pd.Timedelta(
            days=day_offset
        )
    )

    for mine in MINES:

        expected_parameter_names.append(
            parameter_name(
                mine,
                day
            )
        )

missing_parameters = [
    p
    for p in expected_parameter_names
    if p not in H_df.columns
]

if missing_parameters:

    raise RuntimeError(
        "Missing expected H parameters:\n"
        +
        "\n".join(
            missing_parameters
        )
    )

H_full = H_df[
    expected_parameter_names
].to_numpy(
    dtype=np.float64
)

print(
    "Reordered H shape:",
    H_full.shape
)


# ============================================================
# H TIME INFORMATION
# ============================================================

H_df["time_aest"] = pd.to_datetime(
    H_df["obs_time_aest"],
    errors="coerce"
)

if H_df["time_aest"].isna().any():

    raise RuntimeError(
        "H contains invalid obs_time_aest values."
    )

H_df["time_aest"] = (
    H_df["time_aest"]
    .dt.floor("h")
)

receptors = (
    H_df["receptor"]
    .astype(str)
    .to_numpy()
)


# ============================================================
# DETERMINE MAXIMUM BACKWARD H LAG
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "CHECKING FULL FLEXPART TEMPORAL SUPPORT"
)

print(
    "=" * 90
)

parameter_dates = []

for day_offset in range(
    N_DAY
):

    day = (
        START_DATE
        +
        pd.Timedelta(
            days=day_offset
        )
    )

    parameter_dates.extend(
        [day] * N_MINE
    )

parameter_dates = pd.DatetimeIndex(
    parameter_dates
)

max_backward_lag = 0

for row_index in range(
    H_full.shape[0]
):

    nonzero = np.flatnonzero(
        np.abs(
            H_full[row_index]
        )
        > 0.0
    )

    if len(nonzero) == 0:

        continue

    obs_day = (
        H_df.iloc[
            row_index
        ]["time_aest"]
        .normalize()
    )

    earliest_source_day = (
        parameter_dates[
            nonzero.min()
        ]
    )

    lag = (
        obs_day
        - earliest_source_day
    ).days

    if lag > max_backward_lag:

        max_backward_lag = lag

print(
    "Maximum backward lag:",
    max_backward_lag,
    "day(s)"
)

if max_backward_lag > 1:

    warnings.warn(
        "H contains sensitivity more than one day backward. "
        "The current 26-state formulation only supports a "
        "one-day backward lag."
    )


# ============================================================
# READ PICARRO OBSERVATIONS
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "READING PICARRO OBSERVATIONS"
)

print(
    "=" * 90
)

obs_df = pd.read_csv(
    PICARRO_FILE
)

if "Datetime" not in obs_df.columns:

    raise RuntimeError(
        "Picarro file does not contain 'Datetime'."
    )

obs_df["Datetime"] = pd.to_datetime(
    obs_df["Datetime"],
    format="%d/%m/%Y %H:%M",
    errors="coerce"
)

if obs_df["Datetime"].isna().any():

    raise RuntimeError(
        "Invalid Picarro Datetime values."
    )

receptor_columns = {
    "singleton": "Singleton cleaned",
    "muswellbrook": "Muswellbrook cleaned",
    "merriwa": "Merriwa cleaned",
}

missing_obs_columns = [
    column
    for column in receptor_columns.values()
    if column not in obs_df.columns
]

if missing_obs_columns:

    raise RuntimeError(
        "Missing Picarro columns:\n"
        +
        "\n".join(
            missing_obs_columns
        )
    )


# ============================================================
# WIDE -> LONG OBSERVATIONS
# ============================================================

obs_long = []

for receptor, column in (
    receptor_columns.items()
):

    temp = obs_df[
        [
            "Datetime",
            column,
        ]
    ].copy()

    temp = temp.rename(
        columns={
            "Datetime": "time_aest",
            column: "observation",
        }
    )

    temp["receptor"] = receptor

    temp = temp[
        [
            "time_aest",
            "receptor",
            "observation",
        ]
    ]

    obs_long.append(
        temp
    )

obs_long = pd.concat(
    obs_long,
    ignore_index=True
)

obs_long["time_aest"] = (
    obs_long["time_aest"]
    .dt.floor("h")
)

obs_long["observation"] = pd.to_numeric(
    obs_long["observation"],
    errors="coerce"
)

obs_long = obs_long.dropna(
    subset=["observation"]
)

obs_long = obs_long[
    (
        obs_long["time_aest"]
        >= START_DATE
    )
    &
    (
        obs_long["time_aest"]
        <
        (
            END_DATE
            +
            pd.Timedelta(
                days=1
            )
        )
    )
]

# Average duplicate receptor/hour

obs_long = (
    obs_long
    .groupby(
        [
            "time_aest",
            "receptor",
        ],
        as_index=False
    )["observation"]
    .mean()
)

obs_long = (
    obs_long
    .sort_values(
        [
            "time_aest",
            "receptor",
        ]
    )
    .reset_index(
        drop=True
    )
)

print(
    "Observation rows:",
    len(obs_long)
)


# ============================================================
# MATCH H TO OBSERVATIONS
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "MATCHING FLEXPART H TO PICARRO OBSERVATIONS"
)

print(
    "=" * 90
)

H_meta = H_df[
    [
        "receptor",
        "time_aest",
    ]
].copy()

H_meta["H_row"] = np.arange(
    len(H_meta)
)

merged = pd.merge(
    obs_long,
    H_meta,
    on=[
        "receptor",
        "time_aest",
    ],
    how="inner",
)

merged = (
    merged
    .sort_values(
        "time_aest"
    )
    .reset_index(
        drop=True
    )
)

print(
    "Matched observations:",
    len(merged)
)

if len(merged) == 0:

    raise RuntimeError(
        "No observations matched H."
    )

matched_h_rows = (
    merged["H_row"]
    .to_numpy(
        dtype=int
    )
)

y = (
    merged["observation"]
    .to_numpy(
        dtype=np.float64
    )
)

timestamps = (
    merged["time_aest"]
    .to_numpy()
)

matched_receptors = (
    merged["receptor"]
    .astype(str)
    .to_numpy()
)


# ============================================================
# PRIOR COVARIANCE
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "CONSTRUCTING TEMPORAL PRIOR"
)

print(
    "=" * 90
)

rho = np.exp(
    -1.0 / L_CORR
)

Q_daily = (
    PRIOR_VARIANCE
    *
    (
        1.0
        -
        rho ** 2
    )
    *
    np.eye(
        N_MINE
    )
)

print(
    "rho:",
    rho
)

print(
    "Q diagonal:",
    Q_daily[0, 0]
)

B = np.zeros(
    (
        N_PARAM,
        N_PARAM
    ),
    dtype=np.float64
)

for d1 in range(
    N_DAY
):

    for d2 in range(
        N_DAY
    ):

        temporal_covariance = (
            PRIOR_VARIANCE
            *
            np.exp(
                -abs(d1 - d2)
                /
                L_CORR
            )
        )

        for mine in range(
            N_MINE
        ):

            i = (
                d1 * N_MINE
                + mine
            )

            j = (
                d2 * N_MINE
                + mine
            )

            B[i, j] = (
                temporal_covariance
            )

print(
    "B shape:",
    B.shape
)

print(
    "B[OC1 Jan5, OC1 Jan5] =",
    B[0, 0]
)

print(
    "B[OC1 Jan5, OC1 Jan6] =",
    B[0, N_MINE]
)

print(
    "B[OC1 Jan5, OC2 Jan5] =",
    B[0, 1]
)


# ============================================================
# CHOLESKY PRIOR WHITENING
# ============================================================

L_B = np.linalg.cholesky(
    B
)

B_reconstructed = (
    L_B
    @
    L_B.T
)

print(
    "Cholesky reconstruction error:",
    np.max(
        np.abs(
            B_reconstructed
            -
            B
        )
    )
)


# ============================================================
# UNCONSTRAINED BATCH MAP REFERENCE
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "COMPUTING UNCONSTRAINED 351-PARAMETER BATCH MAP REFERENCE"
)

print(
    "=" * 90
)

H_matched = H_full[
    matched_h_rows,
    :
]

x_b = np.full(
    N_PARAM,
    PRIOR_MEAN,
    dtype=np.float64
)

A_prior = solve_triangular(
    L_B,
    np.eye(
        N_PARAM
    ),
    lower=True,
)

d_obs = (
    y
    -
    BACKGROUND_PPM
)

observation_rhs = (
    d_obs
    -
    H_matched @ x_b
)

A_obs = (
    H_matched
    /
    OBS_SIGMA
)

b_obs = (
    observation_rhs
    /
    OBS_SIGMA
)

A_aug_batch = np.vstack(
    [
        A_prior,
        A_obs,
    ]
)

b_aug_batch = np.concatenate(
    [
        np.zeros(
            N_PARAM
        ),
        b_obs,
    ]
)

z_map, residuals, rank, singular_values = (
    np.linalg.lstsq(
        A_aug_batch,
        b_aug_batch,
        rcond=None,
    )
)

x_map = (
    x_b
    +
    z_map
)

print(
    "Whitened system shape:",
    A_aug_batch.shape
)

print(
    "MAP rank:",
    rank
)

print(
    "Least-squares residual norm:",
    residuals
)

print(
    "Unconstrained batch MAP mean:",
    np.mean(x_map)
)

print(
    "Unconstrained batch MAP minimum:",
    np.min(x_map)
)

print(
    "Unconstrained batch MAP maximum:",
    np.max(x_map)
)

print(
    "Number below lower bound:",
    np.sum(
        x_map
        <
        LOWER_BOUND
    )
)

print(
    "Number above upper bound:",
    np.sum(
        x_map
        >
        UPPER_BOUND
    )
)


# ============================================================
# AUGMENTED KF INITIALIZATION
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "INITIALISING BOUNDED AUGMENTED KF"
)

print(
    "=" * 90
)

# ------------------------------------------------------------
# Initial previous and current states
# ------------------------------------------------------------

x_prev = np.full(
    N_MINE,
    PRIOR_MEAN,
    dtype=np.float64
)

x_current = np.full(
    N_MINE,
    PRIOR_MEAN,
    dtype=np.float64
)

P_aug = np.zeros(
    (
        2 * N_MINE,
        2 * N_MINE
    ),
    dtype=np.float64
)

# Previous state covariance

P_aug[
    :N_MINE,
    :N_MINE
] = (
    PRIOR_VARIANCE
    *
    np.eye(
        N_MINE
    )
)

# Current state covariance

P_aug[
    N_MINE:,
    N_MINE:
] = (
    PRIOR_VARIANCE
    *
    np.eye(
        N_MINE
    )
)

# Cross covariance

P_aug[
    :N_MINE,
    N_MINE:
] = (
    PRIOR_VARIANCE
    *
    rho
    *
    np.eye(
        N_MINE
    )
)

P_aug[
    N_MINE:,
    :N_MINE
] = (
    PRIOR_VARIANCE
    *
    rho
    *
    np.eye(
        N_MINE
    )
)

x_aug = np.concatenate(
    [
        x_prev,
        x_current,
    ]
)

x_aug = np.clip(
    x_aug,
    LOWER_BOUND,
    UPPER_BOUND
)


# ============================================================
# STORAGE
# ============================================================

hourly_records = []

daily_records = []

daily_states = []

daily_covariances = []

daily_augmented_states = []

daily_augmented_covariances = []

current_day = None

current_day_state = None

current_day_covariance = None

current_day_nobs = 0


# ============================================================
# KF LOOP
# ============================================================

for row_index in range(
    len(merged)
):

    timestamp = pd.Timestamp(
        timestamps[row_index]
    )

    day = timestamp.normalize()

    observation = y[
        row_index
    ]

    # --------------------------------------------------------
    # INITIAL DAY
    # --------------------------------------------------------

    if current_day is None:

        current_day = day

        current_day_state = (
            x_aug[
                N_MINE:
            ].copy()
        )

        current_day_covariance = (
            P_aug[
                N_MINE:,
                N_MINE:
            ].copy()
        )

        current_day_nobs = 0

    # --------------------------------------------------------
    # NEW DAY
    # --------------------------------------------------------

    elif day != current_day:

        # ----------------------------------------------------
        # Save previous day's final analysis
        # ----------------------------------------------------

        daily_records.append(
            {
                "date": current_day,
                "state_type": "analysis",
                "n_observations": (
                    current_day_nobs
                ),
                **{
                    mine:
                    current_day_state[i]
                    for i, mine
                    in enumerate(
                        MINES
                    )
                }
            }
        )

        daily_states.append(
            current_day_state.copy()
        )

        daily_covariances.append(
            current_day_covariance.copy()
        )

        daily_augmented_states.append(
            x_aug.copy()
        )

        daily_augmented_covariances.append(
            P_aug.copy()
        )

        # ----------------------------------------------------
        # Fill missing calendar days
        # ----------------------------------------------------

        next_day = (
            current_day
            +
            pd.Timedelta(
                days=1
            )
        )

        while next_day < day:

            (
                x_aug,
                P_aug
            ) = transition_to_next_day(
                x_aug,
                P_aug
            )

            forecast_state = (
                x_aug[
                    N_MINE:
                ].copy()
            )

            forecast_covariance = (
                P_aug[
                    N_MINE:,
                    N_MINE:
                ].copy()
            )

            daily_records.append(
                {
                    "date": next_day,
                    "state_type": "forecast",
                    "n_observations": 0,
                    **{
                        mine:
                        forecast_state[i]
                        for i, mine
                        in enumerate(
                            MINES
                        )
                    }
                }
            )

            daily_states.append(
                forecast_state.copy()
            )

            daily_covariances.append(
                forecast_covariance.copy()
            )

            daily_augmented_states.append(
                x_aug.copy()
            )

            daily_augmented_covariances.append(
                P_aug.copy()
            )

            next_day += (
                pd.Timedelta(
                    days=1
                )
            )

        # ----------------------------------------------------
        # Transition into current observation day
        # ----------------------------------------------------

        (
            x_aug,
            P_aug
        ) = transition_to_next_day(
            x_aug,
            P_aug
        )

        current_day = day

        current_day_nobs = 0

    # --------------------------------------------------------
    # DETERMINE CURRENT DAY BLOCK
    # --------------------------------------------------------

    day_index = get_day_index(
        timestamp
    )

    if day_index is None:

        raise RuntimeError(
            "Observation outside inversion window: "
            f"{timestamp}"
        )

    current_indices = get_day_block(
        day_index
    )

    if day_index > 0:

        previous_indices = get_day_block(
            day_index - 1
        )

    else:

        previous_indices = np.array(
            [],
            dtype=int
        )

    # --------------------------------------------------------
    # FULL AUGMENTED OBSERVATION VECTOR
    #
    # h_aug = [H_previous, H_current]
    # --------------------------------------------------------

    h_aug = np.zeros(
        2 * N_MINE,
        dtype=np.float64
    )

    if len(previous_indices):

        h_aug[
            :N_MINE
        ] = H_full[
            matched_h_rows[
                row_index
            ],
            previous_indices
        ]

    h_aug[
        N_MINE:
    ] = H_full[
        matched_h_rows[
            row_index
        ],
        current_indices
    ]

    # --------------------------------------------------------
    # BOUNDED KF UPDATE
    # --------------------------------------------------------

    (
        x_aug,
        P_aug,
        prediction,
        innovation,
        innovation_variance,
        nis,
        active_lower,
        active_upper,
        bounds_active,
    ) = bounded_gaussian_update(
        x_aug,
        P_aug,
        h_aug,
        observation
    )

    current_day_nobs += 1

    # --------------------------------------------------------
    # Current-day posterior
    # --------------------------------------------------------

    current_day_state = (
        x_aug[
            N_MINE:
        ].copy()
    )

    current_day_covariance = (
        P_aug[
            N_MINE:,
            N_MINE:
        ].copy()
    )

    # --------------------------------------------------------
    # Count active bounds
    # --------------------------------------------------------

    n_lower = int(
        np.sum(
            active_lower
        )
    )

    n_upper = int(
        np.sum(
            active_upper
        )
    )

    # --------------------------------------------------------
    # HOURLY RECORD
    # --------------------------------------------------------

    hourly_record = {
        "time_aest": timestamp,
        "receptor": matched_receptors[
            row_index
        ],
        "observation": observation,
        "prediction": prediction,
        "innovation": innovation,
        "innovation_variance": (
            innovation_variance
        ),
        "NIS": nis,
        "bounds_active": bounds_active,
        "n_lower_active": n_lower,
        "n_upper_active": n_upper,
    }

    for i, mine in enumerate(
        MINES
    ):

        hourly_record[
            f"previous_{mine}"
        ] = x_aug[i]

        hourly_record[
            f"current_{mine}"
        ] = x_aug[
            N_MINE + i
        ]

        hourly_record[
            f"previous_{mine}_lower_active"
        ] = bool(
            active_lower[i]
        )

        hourly_record[
            f"previous_{mine}_upper_active"
        ] = bool(
            active_upper[i]
        )

        hourly_record[
            f"current_{mine}_lower_active"
        ] = bool(
            active_lower[
                N_MINE + i
            ]
        )

        hourly_record[
            f"current_{mine}_upper_active"
        ] = bool(
            active_upper[
                N_MINE + i
            ]
        )

    hourly_records.append(
        hourly_record
    )


# ============================================================
# SAVE FINAL DAY
# ============================================================

if current_day is not None:

    daily_records.append(
        {
            "date": current_day,
            "state_type": "analysis",
            "n_observations": (
                current_day_nobs
            ),
            **{
                mine:
                current_day_state[i]
                for i, mine
                in enumerate(
                    MINES
                )
            }
        }
    )

    daily_states.append(
        current_day_state.copy()
    )

    daily_covariances.append(
        current_day_covariance.copy()
    )

    daily_augmented_states.append(
        x_aug.copy()
    )

    daily_augmented_covariances.append(
        P_aug.copy()
    )


# ============================================================
# CONVERT STORAGE TO ARRAYS
# ============================================================

daily_states = np.asarray(
    daily_states,
    dtype=np.float64
)

daily_covariances = np.asarray(
    daily_covariances,
    dtype=np.float64
)

daily_augmented_states = np.asarray(
    daily_augmented_states,
    dtype=np.float64
)

daily_augmented_covariances = np.asarray(
    daily_augmented_covariances,
    dtype=np.float64
)


# ============================================================
# DAILY DATAFRAME
# ============================================================

daily_df = pd.DataFrame(
    daily_records
)

daily_df["date"] = pd.to_datetime(
    daily_df["date"]
)

daily_df = (
    daily_df
    .sort_values(
        "date"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# HOURLY DATAFRAME
# ============================================================

hourly_df = pd.DataFrame(
    hourly_records
)

hourly_df = (
    hourly_df
    .sort_values(
        "time_aest"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# STORAGE VALIDATION
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "VALIDATING BOUNDED KF STORAGE"
)

print(
    "=" * 90
)

print(
    "Expected days:",
    N_DAY
)

print(
    "Daily dataframe:",
    daily_df.shape
)

print(
    "13D states:",
    daily_states.shape
)

print(
    "13D covariances:",
    daily_covariances.shape
)

print(
    "26D augmented states:",
    daily_augmented_states.shape
)

print(
    "26D augmented covariances:",
    daily_augmented_covariances.shape
)

if daily_states.shape != (
    N_DAY,
    N_MINE
):

    raise RuntimeError(
        "Unexpected daily_states shape: "
        f"{daily_states.shape}"
    )

if daily_covariances.shape != (
    N_DAY,
    N_MINE,
    N_MINE
):

    raise RuntimeError(
        "Unexpected daily_covariances shape: "
        f"{daily_covariances.shape}"
    )

if daily_augmented_states.shape != (
    N_DAY,
    2 * N_MINE
):

    raise RuntimeError(
        "Unexpected augmented state shape: "
        f"{daily_augmented_states.shape}"
    )

if daily_augmented_covariances.shape != (
    N_DAY,
    2 * N_MINE,
    2 * N_MINE
):

    raise RuntimeError(
        "Unexpected augmented covariance shape: "
        f"{daily_augmented_covariances.shape}"
    )


# ============================================================
# BOUND VALIDATION
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "CHECKING EMISSION-FACTOR BOUNDS"
)

print(
    "=" * 90
)

minimum_state = np.min(
    daily_augmented_states
)

maximum_state = np.max(
    daily_augmented_states
)

print(
    "Minimum stored state:",
    minimum_state
)

print(
    "Maximum stored state:",
    maximum_state
)

print(
    "Lower bound:",
    LOWER_BOUND
)

print(
    "Upper bound:",
    UPPER_BOUND
)

if minimum_state < (
    LOWER_BOUND - 1.0e-8
):

    raise RuntimeError(
        "Bound violation below LOWER_BOUND."
    )

if maximum_state > (
    UPPER_BOUND + 1.0e-8
):

    raise RuntimeError(
        "Bound violation above UPPER_BOUND."
    )


# ============================================================
# SAVE DAILY SCALING FACTORS
# ============================================================

daily_scaling_file = os.path.join(
    OUTPUT_DIR,
    "daily_bounded_full_footprint_KF_scaling_factors.csv"
)

daily_df.to_csv(
    daily_scaling_file,
    index=False
)


# ============================================================
# SAVE HOURLY RESULTS
# ============================================================

hourly_file = os.path.join(
    OUTPUT_DIR,
    "hourly_bounded_full_footprint_KF_results.csv"
)

hourly_df.to_csv(
    hourly_file,
    index=False
)


# ============================================================
# SAVE DAILY COVARIANCE DIAGONAL
# ============================================================

covariance_records = []

for d in range(
    N_DAY
):

    record = {
        "date": daily_df.loc[
            d,
            "date"
        ]
    }

    for i, mine in enumerate(
        MINES
    ):

        record[mine] = (
            daily_covariances[
                d,
                i,
                i
            ]
        )

    covariance_records.append(
        record
    )

covariance_df = pd.DataFrame(
    covariance_records
)

covariance_file = os.path.join(
    OUTPUT_DIR,
    "daily_bounded_full_footprint_KF_covariance_diagonal.csv"
)

covariance_df.to_csv(
    covariance_file,
    index=False
)


# ============================================================
# SAVE NPZ
# ============================================================

npz_file = os.path.join(
    OUTPUT_DIR,
    "bounded_full_footprint_KF_daily_states.npz"
)

np.savez_compressed(
    npz_file,

    states=daily_states,

    covariances=daily_covariances,

    augmented_states=daily_augmented_states,

    augmented_covariances=daily_augmented_covariances,

    dates=daily_df[
        "date"
    ].astype(str).to_numpy(),

    mines=np.asarray(
        MINES
    ),

    lower_bound=np.asarray(
        LOWER_BOUND
    ),

    upper_bound=np.asarray(
        UPPER_BOUND
    ),

    prior_mean=np.asarray(
        PRIOR_MEAN
    ),

    prior_sigma=np.asarray(
        PRIOR_SIGMA
    ),

    observation_sigma=np.asarray(
        OBS_SIGMA
    ),

    correlation_length=np.asarray(
        L_CORR
    ),
)


# ============================================================
# NPZ VERIFICATION
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "SAVED BOUNDED KF NPZ"
)

print(
    "=" * 90
)

print(
    "File:",
    npz_file
)

print(
    "states:",
    daily_states.shape
)

print(
    "covariances:",
    daily_covariances.shape
)

print(
    "augmented_states:",
    daily_augmented_states.shape
)

print(
    "augmented_covariances:",
    daily_augmented_covariances.shape
)


# ============================================================
# BOUNDED KF VS UNCONSTRAINED BATCH MAP
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "BOUNDED KF VS UNCONSTRAINED BATCH MAP"
)

print(
    "=" * 90
)

bounded_vector = (
    daily_states.reshape(
        -1
    )
)

difference_map = (
    bounded_vector
    -
    x_map
)

print(
    "Maximum absolute difference:",
    np.max(
        np.abs(
            difference_map
        )
    )
)

print(
    "RMSE:",
    np.sqrt(
        np.mean(
            difference_map ** 2
        )
    )
)

print(
    "MAE:",
    np.mean(
        np.abs(
            difference_map
        )
    )
)

if (
    np.std(
        bounded_vector
    ) > 0
    and
    np.std(
        x_map
    ) > 0
):

    correlation_map = np.corrcoef(
        bounded_vector,
        x_map
    )[0, 1]

else:

    correlation_map = np.nan

print(
    "Correlation:",
    correlation_map
)


# ============================================================
# COMPARE WITH EXISTING UNBOUNDED KF
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "COMPARING BOUNDED KF WITH EXISTING UNBOUNDED KF"
)

print(
    "=" * 90
)

unbounded_difference_available = False

if os.path.exists(
    UNBOUNDED_KF_NPZ
):

    try:

        unbounded_npz = np.load(
            UNBOUNDED_KF_NPZ,
            allow_pickle=True
        )

        unbounded_states = (
            unbounded_npz[
                "states"
            ]
        )

        print(
            "Unbounded KF states:",
            unbounded_states.shape
        )

        if unbounded_states.shape == (
            N_DAY,
            N_MINE
        ):

            unbounded_vector = (
                unbounded_states.reshape(
                    -1
                )
            )

            bounded_minus_unbounded = (
                bounded_vector
                -
                unbounded_vector
            )

            print(
                "Bounded - unbounded maximum:",
                np.max(
                    np.abs(
                        bounded_minus_unbounded
                    )
                )
            )

            print(
                "Bounded - unbounded RMSE:",
                np.sqrt(
                    np.mean(
                        bounded_minus_unbounded ** 2
                    )
                )
            )

            print(
                "Bounded - unbounded MAE:",
                np.mean(
                    np.abs(
                        bounded_minus_unbounded
                    )
                )
            )

            unbounded_difference_available = True

            unbounded_comparison_records = []

            for d in range(
                N_DAY
            ):

                date = (
                    START_DATE
                    +
                    pd.Timedelta(
                        days=d
                    )
                )

                for i, mine in enumerate(
                    MINES
                ):

                    unbounded_comparison_records.append(
                        {
                            "date": date,
                            "mine": mine,
                            "bounded_KF": (
                                daily_states[
                                    d,
                                    i
                                ]
                            ),
                            "unbounded_KF": (
                                unbounded_states[
                                    d,
                                    i
                                ]
                            ),
                            "bounded_minus_unbounded": (
                                daily_states[
                                    d,
                                    i
                                ]
                                -
                                unbounded_states[
                                    d,
                                    i
                                ]
                            ),
                            "batch_MAP": (
                                x_map[
                                    d * N_MINE
                                    +
                                    i
                                ]
                            ),
                        }
                    )

            unbounded_comparison_df = pd.DataFrame(
                unbounded_comparison_records
            )

            unbounded_comparison_file = os.path.join(
                OUTPUT_DIR,
                "bounded_full_footprint_KF_vs_unbounded_KF.csv"
            )

            unbounded_comparison_df.to_csv(
                unbounded_comparison_file,
                index=False
            )

            print(
                "Saved:",
                unbounded_comparison_file
            )

        else:

            print(
                "WARNING: unbounded KF shape is unexpected."
            )

    except Exception as exc:

        print(
            "WARNING: could not load unbounded KF:"
        )

        print(
            repr(exc)
        )

else:

    print(
        "Existing unbounded KF NPZ not found."
    )


# ============================================================
# FINAL-DAY TEST
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "FINAL-DAY TEST"
)

print(
    "=" * 90
)

final_bounded = (
    daily_states[
        -1
    ]
)

final_map = (
    x_map[
        -N_MINE:
    ]
)

for i, mine in enumerate(
    MINES
):

    print(
        f"{mine:5s} "
        f"bounded_KF "
        f"{final_bounded[i]:14.8f} "
        f"MAP "
        f"{final_map[i]:14.8f} "
        f"diff "
        f"{final_bounded[i] - final_map[i]:14.8f}"
    )

final_difference = (
    final_bounded
    -
    final_map
)

print(
    "Final-day maximum difference:",
    np.max(
        np.abs(
            final_difference
        )
    )
)


# ============================================================
# BOUND ACTIVITY DIAGNOSTICS
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "BOUND ACTIVITY DIAGNOSTICS"
)

print(
    "=" * 90
)

bounds_active_rows = hourly_df[
    "bounds_active"
].to_numpy(
    dtype=bool
)

n_bound_updates = int(
    np.sum(
        bounds_active_rows
    )
)

print(
    "Total observation updates:",
    len(hourly_df)
)

print(
    "Updates with at least one active bound:",
    n_bound_updates
)

print(
    "Percentage of updates bounded:",
    (
        100.0
        *
        n_bound_updates
        /
        len(hourly_df)
    )
)

total_lower_activations = int(
    hourly_df[
        "n_lower_active"
    ].sum()
)

total_upper_activations = int(
    hourly_df[
        "n_upper_active"
    ].sum()
)

print(
    "Total lower-bound activations:",
    total_lower_activations
)

print(
    "Total upper-bound activations:",
    total_upper_activations
)


# ============================================================
# PER-MINE BOUND ACTIVITY
# ============================================================

bound_summary_records = []

for mine in MINES:

    lower_previous = (
        hourly_df[
            f"previous_{mine}_lower_active"
        ]
        .sum()
    )

    upper_previous = (
        hourly_df[
            f"previous_{mine}_upper_active"
        ]
        .sum()
    )

    lower_current = (
        hourly_df[
            f"current_{mine}_lower_active"
        ]
        .sum()
    )

    upper_current = (
        hourly_df[
            f"current_{mine}_upper_active"
        ]
        .sum()
    )

    bound_summary_records.append(
        {
            "mine": mine,
            "previous_lower_activations":
                int(lower_previous),
            "previous_upper_activations":
                int(upper_previous),
            "current_lower_activations":
                int(lower_current),
            "current_upper_activations":
                int(upper_current),
        }
    )

bound_summary_df = pd.DataFrame(
    bound_summary_records
)

bound_summary_file = os.path.join(
    OUTPUT_DIR,
    "bounded_KF_bound_activity_summary.csv"
)

bound_summary_df.to_csv(
    bound_summary_file,
    index=False
)

print(
    "\nPer-mine bound activity:"
)

print(
    bound_summary_df.to_string(
        index=False
    )
)


# ============================================================
# NIS DIAGNOSTICS
# ============================================================

nis_values = (
    hourly_df[
        "NIS"
    ].to_numpy(
        dtype=np.float64
    )
)

print(
    "\n"
    + "=" * 90
)

print(
    "NIS"
)

print(
    "=" * 90
)

print(
    "Observations:",
    len(nis_values)
)

print(
    "Mean NIS:",
    np.mean(
        nis_values
    )
)

print(
    "Median NIS:",
    np.median(
        nis_values
    )
)

print(
    "Maximum NIS:",
    np.max(
        nis_values
    )
)


# ============================================================
# SAVE BOUNDED KF VS BATCH MAP
# ============================================================

comparison_records = []

for d in range(
    N_DAY
):

    date = (
        START_DATE
        +
        pd.Timedelta(
            days=d
        )
    )

    for i, mine in enumerate(
        MINES
    ):

        parameter = (
            f"{mine}_"
            f"{date.strftime('%Y%m%d')}"
        )

        bounded_value = (
            daily_states[
                d,
                i
            ]
        )

        map_value = (
            x_map[
                d * N_MINE
                +
                i
            ]
        )

        comparison_records.append(
            {
                "date": date,
                "mine": mine,
                "parameter": parameter,
                "bounded_KF": bounded_value,
                "batch_MAP_unbounded": map_value,
                "difference": (
                    bounded_value
                    -
                    map_value
                ),
                "at_lower_bound": (
                    bounded_value
                    <=
                    LOWER_BOUND
                    +
                    1.0e-9
                ),
                "at_upper_bound": (
                    bounded_value
                    >=
                    UPPER_BOUND
                    -
                    1.0e-9
                ),
            }
        )

comparison_df = pd.DataFrame(
    comparison_records
)

comparison_file = os.path.join(
    OUTPUT_DIR,
    "bounded_full_footprint_KF_vs_batch_MAP.csv"
)

comparison_df.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "bounded_full_footprint_KF_summary.txt"
)

with open(
    summary_file,
    "w"
) as f:

    f.write(
        "BOUNDED FULL-FLEXPART-FOOTPRINT "
        "AUGMENTED KALMAN FILTER\n"
    )

    f.write(
        "=" * 80
        +
        "\n\n"
    )

    f.write(
        f"Start date: "
        f"{START_DATE:%Y-%m-%d}\n"
    )

    f.write(
        f"End date: "
        f"{END_DATE:%Y-%m-%d}\n"
    )

    f.write(
        f"Number of mines: "
        f"{N_MINE}\n"
    )

    f.write(
        f"Number of days: "
        f"{N_DAY}\n"
    )

    f.write(
        f"Number of parameters: "
        f"{N_PARAM}\n"
    )

    f.write(
        f"Matched observations: "
        f"{len(merged)}\n"
    )

    f.write(
        f"Prior mean: "
        f"{PRIOR_MEAN}\n"
    )

    f.write(
        f"Prior sigma: "
        f"{PRIOR_SIGMA}\n"
    )

    f.write(
        f"Observation sigma: "
        f"{OBS_SIGMA}\n"
    )

    f.write(
        f"Correlation length: "
        f"{L_CORR} days\n"
    )

    f.write(
        f"rho: "
        f"{rho}\n"
    )

    f.write(
        f"Lower emission-factor bound: "
        f"{LOWER_BOUND}\n"
    )

    f.write(
        f"Upper emission-factor bound: "
        f"{UPPER_BOUND}\n"
    )

    f.write(
        f"Maximum backward H lag: "
        f"{max_backward_lag} days\n"
    )

    f.write(
        "\nSTATE STORAGE\n"
    )

    f.write(
        "-" * 80
        +
        "\n"
    )

    f.write(
        f"13-D daily states: "
        f"{daily_states.shape}\n"
    )

    f.write(
        f"13-D daily covariances: "
        f"{daily_covariances.shape}\n"
    )

    f.write(
        f"26-D augmented states: "
        f"{daily_augmented_states.shape}\n"
    )

    f.write(
        f"26-D augmented covariances: "
        f"{daily_augmented_covariances.shape}\n"
    )

    f.write(
        "\nBOUNDS\n"
    )

    f.write(
        "-" * 80
        +
        "\n"
    )

    f.write(
        f"Minimum stored state: "
        f"{minimum_state}\n"
    )

    f.write(
        f"Maximum stored state: "
        f"{maximum_state}\n"
    )

    f.write(
        f"Lower bound: "
        f"{LOWER_BOUND}\n"
    )

    f.write(
        f"Upper bound: "
        f"{UPPER_BOUND}\n"
    )

    f.write(
        "\nBOUNDS ACTIVITY\n"
    )

    f.write(
        "-" * 80
        +
        "\n"
    )

    f.write(
        f"Total observation updates: "
        f"{len(hourly_df)}\n"
    )

    f.write(
        f"Updates with active bounds: "
        f"{n_bound_updates}\n"
    )

    f.write(
        f"Percentage of updates bounded: "
        f"{100.0 * n_bound_updates / len(hourly_df)}\n"
    )

    f.write(
        f"Lower-bound activations: "
        f"{total_lower_activations}\n"
    )

    f.write(
        f"Upper-bound activations: "
        f"{total_upper_activations}\n"
    )

    f.write(
        "\nBOUNDED KF VS UNCONSTRAINED BATCH MAP\n"
    )

    f.write(
        "-" * 80
        +
        "\n"
    )

    f.write(
        f"Maximum absolute difference: "
        f"{np.max(np.abs(difference_map))}\n"
    )

    f.write(
        f"RMSE: "
        f"{np.sqrt(np.mean(difference_map ** 2))}\n"
    )

    f.write(
        f"MAE: "
        f"{np.mean(np.abs(difference_map))}\n"
    )

    f.write(
        f"Correlation: "
        f"{correlation_map}\n"
    )

    if unbounded_difference_available:

        f.write(
            "\nBOUNDED KF VS UNBOUNDED KF\n"
        )

        f.write(
            "-" * 80
            +
            "\n"
        )

        f.write(
            f"Maximum absolute difference: "
            f"{np.max(np.abs(bounded_minus_unbounded))}\n"
        )

        f.write(
            f"RMSE: "
            f"{np.sqrt(np.mean(bounded_minus_unbounded ** 2))}\n"
        )

        f.write(
            f"MAE: "
            f"{np.mean(np.abs(bounded_minus_unbounded))}\n"
        )

    f.write(
        "\nFINAL DAY\n"
    )

    f.write(
        "-" * 80
        +
        "\n"
    )

    f.write(
        f"Maximum final-day difference from "
        f"unconstrained batch MAP: "
        f"{np.max(np.abs(final_difference))}\n"
    )

    f.write(
        "\nNIS\n"
    )

    f.write(
        "-" * 80
        +
        "\n"
    )

    f.write(
        f"Mean NIS: "
        f"{np.mean(nis_values)}\n"
    )

    f.write(
        f"Median NIS: "
        f"{np.median(nis_values)}\n"
    )

    f.write(
        f"Maximum NIS: "
        f"{np.max(nis_values)}\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "BOUNDED KF RUN COMPLETE"
)

print(
    "=" * 90
)

print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Daily scaling factors:"
)

print(
    daily_scaling_file
)

print()

print(
    "Hourly results:"
)

print(
    hourly_file
)

print()

print(
    "Daily covariance diagonal:"
)

print(
    covariance_file
)

print()

print(
    "NPZ:"
)

print(
    npz_file
)

print()

print(
    "Bound activity:"
)

print(
    bound_summary_file
)

print()

print(
    "Bounded KF vs batch MAP:"
)

print(
    comparison_file
)

if unbounded_difference_available:

    print()

    print(
        "Bounded KF vs unbounded KF:"
    )

    print(
        unbounded_comparison_file
    )

print()

print(
    "Summary:"
)

print(
    summary_file
)

print(
    "\n"
    + "=" * 90
)

print(
    "IMPORTANT STATE DIMENSIONS"
)

print(
    "=" * 90
)

print(
    "13-D states:",
    daily_states.shape
)

print(
    "13-D covariances:",
    daily_covariances.shape
)

print(
    "26-D augmented states:",
    daily_augmented_states.shape
)

print(
    "26-D augmented covariances:",
    daily_augmented_covariances.shape
)

print(
    "=" * 90
)

