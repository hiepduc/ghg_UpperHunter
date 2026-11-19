#!/usr/bin/env python3

"""
FULL-FLEXPART-FOOTPRINT AUGMENTED KALMAN FILTER
================================================

Upper Hunter CH4 inverse modelling
13 open-cut mines x daily scaling factors
January 2026

State formulation
-----------------

The FLEXPART footprint matrix has non-zero sensitivity to the
current day and, in some cases, the immediately preceding day.

Therefore the observation model is

    y_t = H_{t,d-1} x_{d-1} + H_{t,d} x_d + eps_t

The KF uses the augmented state

    X_d = [ x_{d-1} ]
          [ x_d     ]

with dimension 26.

The daily transition is

    x_d -> x_{d+1}

with AR(1) prior:

    x_{d+1} = rho*x_d + (1-rho)*PRIOR_MEAN + eta

where

    rho = exp(-1 / L_CORR)

and

    Q = PRIOR_VARIANCE * (1-rho^2) I

The script preserves the existing 13-dimensional daily current-state
outputs and additionally saves the complete 26-dimensional augmented
states and covariances required by the augmented RTS smoother.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

from scipy.linalg import solve_triangular


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

BACKGROUND_PPM = 1.85

PRIOR_MEAN = 1.0
PRIOR_SIGMA = 0.5
PRIOR_VARIANCE = PRIOR_SIGMA ** 2

OBS_SIGMA = 0.05
OBS_VARIANCE = OBS_SIGMA ** 2

L_CORR = 3.0


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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "H_matrix_flexpart",
    "KF_full_footprint_13mine_Jan2026",
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
    timestamp = pd.Timestamp(timestamp).normalize()

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
    Return the 13 parameter indices corresponding to
    one calendar day.

    Parameter order:

        OC1_day
        OC2_day
        ...
        OC13_day
    """

    start = (
        day_index * N_MINE
    )

    return np.arange(
        start,
        start + N_MINE,
        dtype=int
    )


def joseph_update(
    x,
    P,
    h,
    observation
):
    """
    Scalar-observation Kalman update using the Joseph
    covariance form.

    Observation model:

        y = BACKGROUND_PPM + h @ x + eps

    Returns
    -------

    x_new
    P_new
    prediction
    innovation
    innovation_variance
    NIS
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

    # --------------------------------------------------------
    # Prediction
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
    # Kalman gain
    # --------------------------------------------------------

    PHt = np.dot(
        P,
        h
    )

    K = (
        PHt
        / innovation_variance
    )

    # --------------------------------------------------------
    # State update
    # --------------------------------------------------------

    x_new = (
        x
        + K * innovation
    )

    # --------------------------------------------------------
    # Joseph covariance update
    #
    # P_new =
    # (I-KH)P(I-KH)^T + K R K^T
    # --------------------------------------------------------

    I = np.eye(
        len(x),
        dtype=np.float64
    )

    KH = np.outer(
        K,
        h
    )

    IKH = I - KH

    P_new = (
        IKH
        @ P
        @ IKH.T
        + OBS_VARIANCE
        * np.outer(K, K)
    )

    # Numerical symmetrisation
    P_new = (
        0.5
        * (P_new + P_new.T)
    )

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
    )


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

    where

        x_{d+1}
        =
        rho*x_d
        +
        (1-rho)*PRIOR_MEAN
        +
        eta

    and

        Cov(eta) = Q_daily.

    The previous/current covariance structure is propagated
    using the augmented transition matrix.
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
        * np.eye(N_MINE)
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
        F_aug @ x_aug
        + c_aug
    )

    P_new = (
        F_aug
        @ P_aug
        @ F_aug.T
        + Q_aug
    )

    P_new = (
        0.5
        * (P_new + P_new.T)
    )

    return (
        x_new,
        P_new
    )


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 90)
print("FULL-FLEXPART-FOOTPRINT AUGMENTED KALMAN FILTER")
print("=" * 90)

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

print(
    "Number of days:",
    (END_DATE - START_DATE).days + 1
)

print(
    "Number of parameters:",
    N_MINE
    * (
        (END_DATE - START_DATE).days + 1
    )
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


# ============================================================
# READ H MATRIX
# ============================================================

print("\n" + "=" * 90)
print("READING FLEXPART H MATRIX")
print("=" * 90)

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
            f"Missing H metadata column: "
            f"{column}"
        )

parameter_columns = [
    c
    for c in H_df.columns
    if c not in metadata_columns
]

expected_parameter_count = (
    N_MINE
    * (
        (END_DATE - START_DATE).days
        + 1
    )
)

if len(parameter_columns) != (
    expected_parameter_count
):

    raise RuntimeError(
        "Unexpected number of H parameters: "
        f"{len(parameter_columns)} "
        f"expected "
        f"{expected_parameter_count}"
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
    (END_DATE - START_DATE).days + 1
):

    day = (
        START_DATE
        + pd.Timedelta(days=day_offset)
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
        + "\n".join(
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

print("\n" + "=" * 90)
print("CHECKING FULL FLEXPART TEMPORAL SUPPORT")
print("=" * 90)

parameter_dates = []

for day_offset in range(
    (END_DATE - START_DATE).days + 1
):

    day = (
        START_DATE
        + pd.Timedelta(days=day_offset)
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
        ) > 0.0
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
        "H contains sensitivity more than "
        "one day backward. The current "
        "26-state formulation only supports "
        "a one-day backward lag."
    )


# ============================================================
# READ PICARRO OBSERVATIONS
# ============================================================

print("\n" + "=" * 90)
print("READING PICARRO OBSERVATIONS")
print("=" * 90)

obs_df = pd.read_csv(
    PICARRO_FILE
)

if "Datetime" not in obs_df.columns:

    raise RuntimeError(
        "Picarro file does not contain "
        "'Datetime'."
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
        + "\n".join(
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

# Restrict to inversion period
obs_long = obs_long[
    (
        obs_long["time_aest"]
        >= START_DATE
    )
    &
    (
        obs_long["time_aest"]
        < (
            END_DATE
            + pd.Timedelta(days=1)
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

obs_long = obs_long.sort_values(
    [
        "time_aest",
        "receptor",
    ]
).reset_index(
    drop=True
)

print(
    "Observation rows:",
    len(obs_long)
)


# ============================================================
# MATCH H TO OBSERVATIONS
# ============================================================

print("\n" + "=" * 90)
print("MATCHING FLEXPART H TO PICARRO OBSERVATIONS")
print("=" * 90)

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

merged = merged.sort_values(
    "time_aest"
).reset_index(
    drop=True
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
    .to_numpy(dtype=int)
)

y = (
    merged["observation"]
    .to_numpy(dtype=np.float64)
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

print("\n" + "=" * 90)
print("CONSTRUCTING TEMPORAL PRIOR")
print("=" * 90)

rho = np.exp(
    -1.0 / L_CORR
)

Q_daily = (
    PRIOR_VARIANCE
    * (
        1.0 - rho ** 2
    )
    * np.eye(N_MINE)
)

print(
    "rho:",
    rho
)

print(
    "Q diagonal:",
    Q_daily[0, 0]
)

N_DAY = (
    END_DATE
    - START_DATE
).days + 1

N_PARAM = (
    N_MINE
    * N_DAY
)

B = np.zeros(
    (
        N_PARAM,
        N_PARAM
    ),
    dtype=np.float64
)

for d1 in range(N_DAY):

    for d2 in range(N_DAY):

        temporal_covariance = (
            PRIOR_VARIANCE
            * np.exp(
                -abs(d1 - d2)
                / L_CORR
            )
        )

        for mine in range(N_MINE):

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
    L_B @ L_B.T
)

print(
    "Cholesky reconstruction error:",
    np.max(
        np.abs(
            B_reconstructed - B
        )
    )
)


# ============================================================
# BATCH MAP REFERENCE
# ============================================================

print("\n" + "=" * 90)
print("COMPUTING FULL 351-PARAMETER BATCH MAP")
print("=" * 90)

H_matched = H_full[
    matched_h_rows,
    :
]

x_b = np.full(
    N_PARAM,
    PRIOR_MEAN,
    dtype=np.float64
)

# Prior:
#
# (x-xb)^T B^-1 (x-xb)
#
# whitened with B = L L^T
#
# z = L^-1 (x-xb)

A_prior = solve_triangular(
    L_B,
    np.eye(N_PARAM),
    lower=True,
)

d_obs = (
    y
    - BACKGROUND_PPM
)

observation_rhs = (
    d_obs
    - H_matched @ x_b
)

A_obs = (
    H_matched
    / OBS_SIGMA
)

b_obs = (
    observation_rhs
    / OBS_SIGMA
)

A_aug_batch = np.vstack(
    [
        A_prior,
        A_obs,
    ]
)

b_aug_batch = np.concatenate(
    [
        np.zeros(N_PARAM),
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
    + z_map
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

# Gradient check
B_inv_z = solve_triangular(
    L_B.T,
    solve_triangular(
        L_B,
        z_map,
        lower=True,
    ),
)

gradient = (
    B_inv_z
    - (
        H_matched.T
        @ (
            observation_rhs
            - H_matched @ z_map
        )
        / OBS_VARIANCE
    )
)

# The expression above is easier and safer to verify
# directly from x_map.
prior_term = solve_triangular(
    L_B.T,
    solve_triangular(
        L_B,
        x_map - x_b,
        lower=True,
    ),
)

gradient_direct = (
    solve_triangular(
        L_B.T,
        prior_term,
        lower=True,
    )
    -
    H_matched.T
    @ (
        (
            H_matched @ x_map
            - d_obs
        )
        / OBS_VARIANCE
    )
)

print(
    "MAP gradient L2:",
    np.linalg.norm(
        gradient_direct
    )
)

print(
    "MAP gradient max abs:",
    np.max(
        np.abs(
            gradient_direct
        )
    )
)

print(
    "Batch MAP mean:",
    np.mean(x_map)
)


# ============================================================
# AUGMENTED KF INITIALIZATION
# ============================================================

print("\n" + "=" * 90)
print("RUNNING FULL-FOOTPRINT AUGMENTED KF")
print("=" * 90)

# ------------------------------------------------------------
# IMPORTANT INITIALIZATION
#
# X = [x_prev, x_current]
#
# Jan 5 has no unknown Jan 4 inversion parameter.
# The previous state is nevertheless represented as a
# stationary prior boundary state.
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
    * np.eye(N_MINE)
)

# Current state covariance
P_aug[
    N_MINE:,
    N_MINE:
] = (
    PRIOR_VARIANCE
    * np.eye(N_MINE)
)

# Cross covariance
P_aug[
    :N_MINE,
    N_MINE:
] = (
    PRIOR_VARIANCE
    * rho
    * np.eye(N_MINE)
)

P_aug[
    N_MINE:,
    :N_MINE
] = (
    PRIOR_VARIANCE
    * rho
    * np.eye(N_MINE)
)

x_aug = np.concatenate(
    [
        x_prev,
        x_current,
    ]
)


# ============================================================
# STORAGE
# ============================================================

hourly_records = []

daily_records = []

# Existing 13-D outputs
daily_states = []
daily_covariances = []

# New 26-D augmented outputs
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
                    mine: current_day_state[i]
                    for i, mine
                    in enumerate(MINES)
                }
            }
        )

        # Existing 13-D state
        daily_states.append(
            current_day_state.copy()
        )

        daily_covariances.append(
            current_day_covariance.copy()
        )

        # Complete 26-D augmented posterior
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
            + pd.Timedelta(days=1)
        )

        while next_day < day:

            x_aug, P_aug = (
                transition_to_next_day(
                    x_aug,
                    P_aug
                )
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
                        mine: forecast_state[i]
                        for i, mine
                        in enumerate(MINES)
                    }
                }
            )

            # Existing 13-D forecast
            daily_states.append(
                forecast_state.copy()
            )

            daily_covariances.append(
                forecast_covariance.copy()
            )

            # Complete 26-D augmented forecast
            daily_augmented_states.append(
                x_aug.copy()
            )

            daily_augmented_covariances.append(
                P_aug.copy()
            )

            next_day += (
                pd.Timedelta(days=1)
            )

        # ----------------------------------------------------
        # Transition into current observation day
        # ----------------------------------------------------

        x_aug, P_aug = (
            transition_to_next_day(
                x_aug,
                P_aug
            )
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
            "Observation outside inversion "
            f"window: {timestamp}"
        )

    current_indices = get_day_block(
        day_index
    )

    if day_index > 0:

        previous_indices = (
            get_day_block(
                day_index - 1
            )
        )

    else:

        previous_indices = (
            np.array(
                [],
                dtype=int
            )
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

    # Previous-day contribution
    if len(previous_indices):

        h_aug[
            :N_MINE
        ] = H_full[
            matched_h_rows[
                row_index
            ],
            previous_indices
        ]

    # Current-day contribution
    h_aug[
        N_MINE:
    ] = H_full[
        matched_h_rows[
            row_index
        ],
        current_indices
    ]

    # --------------------------------------------------------
    # KF UPDATE
    # --------------------------------------------------------

    (
        x_aug,
        P_aug,
        prediction,
        innovation,
        innovation_variance,
        nis,
    ) = joseph_update(
        x_aug,
        P_aug,
        h_aug,
        observation
    )

    current_day_nobs += 1

    # --------------------------------------------------------
    # Current-day posterior
    #
    # The LAST observation of the day is the daily analysis
    # state for that day.
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
                mine: current_day_state[i]
                for i, mine
                in enumerate(MINES)
            }
        }
    )

    daily_states.append(
        current_day_state.copy()
    )

    daily_covariances.append(
        current_day_covariance.copy()
    )

    # Complete 26-D final posterior
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

daily_df = daily_df.sort_values(
    "date"
).reset_index(
    drop=True
)


# ============================================================
# HOURLY DATAFRAME
# ============================================================

hourly_df = pd.DataFrame(
    hourly_records
)

hourly_df = hourly_df.sort_values(
    "time_aest"
).reset_index(
    drop=True
)


# ============================================================
# STORAGE VALIDATION
# ============================================================

print("\n" + "=" * 90)
print("VALIDATING KF STORAGE")
print("=" * 90)

expected_days = N_DAY

print(
    "Expected days:",
    expected_days
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
    expected_days,
    N_MINE
):

    raise RuntimeError(
        "Unexpected daily_states shape: "
        f"{daily_states.shape}"
    )

if daily_covariances.shape != (
    expected_days,
    N_MINE,
    N_MINE
):

    raise RuntimeError(
        "Unexpected daily_covariances shape: "
        f"{daily_covariances.shape}"
    )

if daily_augmented_states.shape != (
    expected_days,
    2 * N_MINE
):

    raise RuntimeError(
        "Unexpected augmented state shape: "
        f"{daily_augmented_states.shape}"
    )

if daily_augmented_covariances.shape != (
    expected_days,
    2 * N_MINE,
    2 * N_MINE
):

    raise RuntimeError(
        "Unexpected augmented covariance shape: "
        f"{daily_augmented_covariances.shape}"
    )


# ============================================================
# SAVE DAILY SCALING FACTORS
# ============================================================

daily_scaling_file = os.path.join(
    OUTPUT_DIR,
    "daily_full_footprint_KF_scaling_factors.csv"
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
    "hourly_full_footprint_KF_results.csv"
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
    expected_days
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
    "daily_full_footprint_KF_covariance_diagonal.csv"
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
    "full_footprint_KF_daily_states.npz"
)

np.savez_compressed(
    npz_file,

    # Existing 13-D outputs
    states=daily_states,
    covariances=daily_covariances,

    # New complete augmented outputs
    augmented_states=daily_augmented_states,
    augmented_covariances=daily_augmented_covariances,

    dates=daily_df[
        "date"
    ].astype(str).to_numpy(),

    mines=np.asarray(
        MINES
    ),
)


# ============================================================
# NPZ VERIFICATION
# ============================================================

print("\n" + "=" * 90)
print("SAVED NPZ")
print("=" * 90)

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
# COMPARE DAILY FORWARD KF WITH BATCH MAP
# ============================================================

print("\n" + "=" * 90)
print("FULL-FOOTPRINT KF VS BATCH MAP")
print("=" * 90)

kf_vector = (
    daily_states.reshape(-1)
)

difference = (
    kf_vector
    - x_map
)

print(
    "Maximum absolute difference:",
    np.max(
        np.abs(difference)
    )
)

print(
    "RMSE:",
    np.sqrt(
        np.mean(
            difference ** 2
        )
    )
)

print(
    "MAE:",
    np.mean(
        np.abs(difference)
    )
)

if (
    np.std(kf_vector) > 0
    and np.std(x_map) > 0
):

    correlation = np.corrcoef(
        kf_vector,
        x_map
    )[0, 1]

else:

    correlation = np.nan

print(
    "Correlation:",
    correlation
)


# ============================================================
# FINAL-DAY TEST
# ============================================================

print("\n" + "=" * 90)
print("FINAL-DAY TEST")
print("=" * 90)

final_kf = (
    daily_states[-1]
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
        f"{final_kf[i]:14.8f} "
        f"{final_map[i]:14.8f} "
        f"{final_kf[i] - final_map[i]:14.8f}"
    )

final_difference = (
    final_kf
    - final_map
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
# NIS DIAGNOSTICS
# ============================================================

nis_values = (
    hourly_df[
        "NIS"
    ].to_numpy(
        dtype=np.float64
    )
)

print("\n" + "=" * 90)
print("NIS")
print("=" * 90)

print(
    "Observations:",
    len(nis_values)
)

print(
    "Mean NIS:",
    np.mean(nis_values)
)

print(
    "Median NIS:",
    np.median(nis_values)
)

print(
    "Maximum NIS:",
    np.max(nis_values)
)


# ============================================================
# SAVE COMPARISON CSV
# ============================================================

comparison_records = []

for d in range(
    expected_days
):

    date = (
        START_DATE
        + pd.Timedelta(days=d)
    )

    for i, mine in enumerate(
        MINES
    ):

        parameter = (
            f"{mine}_"
            f"{date.strftime('%Y%m%d')}"
        )

        comparison_records.append(
            {
                "date": date,
                "mine": mine,
                "parameter": parameter,
                "KF": (
                    daily_states[
                        d,
                        i
                    ]
                ),
                "batch_MAP": (
                    x_map[
                        d * N_MINE + i
                    ]
                ),
                "difference": (
                    daily_states[
                        d,
                        i
                    ]
                    -
                    x_map[
                        d * N_MINE + i
                    ]
                ),
            }
        )

comparison_df = pd.DataFrame(
    comparison_records
)

comparison_file = os.path.join(
    OUTPUT_DIR,
    "full_footprint_KF_vs_MAP.csv"
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
    "full_footprint_KF_summary.txt"
)

with open(
    summary_file,
    "w"
) as f:

    f.write(
        "FULL-FLEXPART-FOOTPRINT AUGMENTED "
        "KALMAN FILTER\n"
    )

    f.write(
        "=" * 80
        + "\n\n"
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
        f"Maximum backward H lag: "
        f"{max_backward_lag} days\n"
    )

    f.write(
        "\nSTATE STORAGE\n"
    )

    f.write(
        "-" * 80
        + "\n"
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
        "\nFORWARD KF VS BATCH MAP\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    f.write(
        f"Maximum absolute difference: "
        f"{np.max(np.abs(difference))}\n"
    )

    f.write(
        f"RMSE: "
        f"{np.sqrt(np.mean(difference ** 2))}\n"
    )

    f.write(
        f"MAE: "
        f"{np.mean(np.abs(difference))}\n"
    )

    f.write(
        f"Correlation: "
        f"{correlation}\n"
    )

    f.write(
        "\nFINAL DAY\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    f.write(
        f"Maximum final-day difference: "
        f"{np.max(np.abs(final_difference))}\n"
    )

    f.write(
        "\nNIS\n"
    )

    f.write(
        "-" * 80
        + "\n"
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

print("\n" + "=" * 90)
print("RUN COMPLETE")
print("=" * 90)

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
    "NPZ:"
)

print(
    npz_file
)

print()
print(
    "Summary:"
)

print(
    summary_file
)

print("\n" + "=" * 90)
print("IMPORTANT STATE DIMENSIONS")
print("=" * 90)

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

print("=" * 90)

