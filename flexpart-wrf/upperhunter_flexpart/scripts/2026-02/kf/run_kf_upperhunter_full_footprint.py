#!/usr/bin/env python3

"""
UPPER HUNTER CH4
FULL-FLEXPART-FOOTPRINT AUGMENTED KALMAN FILTER

Purpose
-------
Sequentially assimilate Picarro observations using the COMPLETE
FLEXPART H observation row.

Unlike the original 13-state daily KF, this implementation retains
the previous and current daily emission scaling factors because the
FLEXPART footprints have a maximum temporal lag of one day.

Observation model:

    y_t = H_{t,d-1} x_{d-1}
        + H_{t,d}   x_d
        + epsilon_t

Augmented state:

    X_d = [x_{d-1}, x_d]^T

where each x contains 13 mine scaling factors.

The original production KF and RTS scripts are NOT modified.

The script also constructs the equivalent full batch MAP solution
using the complete 351-column H matrix and compares it against the
augmented KF.

Expected relationship:

    Full-footprint augmented KF
             =
    Full-footprint batch MAP

provided the initial/boundary treatment is identical.
"""

import os
import re
import numpy as np
import pandas as pd

from scipy.linalg import solve_triangular


# ============================================================
# CONFIGURATION
# ============================================================

H_FILE = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "H_matrix_flexpart/"
    "H_flexpart_0_10m_13mine_Jan2026.csv"
)

PICARRO_FILE = (
    "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter/"
    "tapm_output/Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

OUTPUT_DIR = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "H_matrix_flexpart/"
    "KF_full_footprint_13mine_Jan2026"
)

START_DATE = pd.Timestamp("2026-01-05")
END_DATE = pd.Timestamp("2026-01-31")

BACKGROUND_PPM = 1.85

PRIOR_MEAN = 1.0
PRIOR_SIGMA = 0.50
PRIOR_VARIANCE = PRIOR_SIGMA ** 2

OBS_SIGMA = 0.05
OBS_VARIANCE = OBS_SIGMA ** 2

L_CORR = 3.0

N_MINE = 13
N_DAY = (END_DATE - START_DATE).days + 1
N_PARAM = N_MINE * N_DAY

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


# ============================================================
# OUTPUT
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parameter_name(day_index, mine_index):
    """
    Day-major parameter ordering:

        OC1_day1 ... OC13_day1
        OC1_day2 ... OC13_day2
        ...
    """
    date = START_DATE + pd.Timedelta(days=day_index)

    return f"{MINES[mine_index]}_{date.strftime('%Y%m%d')}"


def get_day_index(timestamp):
    """
    Return inversion-window day index.
    """
    day = pd.Timestamp(timestamp).normalize()

    index = (day - START_DATE).days

    if index < 0 or index >= N_DAY:
        return None

    return int(index)


def get_day_block(day_index):
    """
    Return the 13 parameter indices for a day.
    """
    start = day_index * N_MINE
    return np.arange(start, start + N_MINE)


def build_B():
    """
    Full 351 x 351 prior covariance.

    Same mine:
        B[(d,m),(j,m)]
          = sigma^2 exp(-|d-j|/L)

    Different mines:
        covariance = 0
    """

    B = np.zeros(
        (N_PARAM, N_PARAM),
        dtype=np.float64
    )

    for day_i in range(N_DAY):

        for mine_i in range(N_MINE):

            i = day_i * N_MINE + mine_i

            for day_j in range(N_DAY):

                j = day_j * N_MINE + mine_i

                B[i, j] = (
                    PRIOR_VARIANCE
                    * np.exp(
                        -abs(day_i - day_j) / L_CORR
                    )
                )

    return B


def joseph_update(x, P, h, observation):
    """
    Scalar-observation Kalman update.

    Uses Joseph covariance form for numerical stability.
    """

    prediction = (
        BACKGROUND_PPM
        + np.dot(h, x)
    )

    innovation = observation - prediction

    HP = np.dot(h, P)

    innovation_variance = (
        np.dot(HP, h)
        + OBS_VARIANCE
    )

    K = np.dot(P, h) / innovation_variance

    x_new = x + K * innovation

    I_KH = (
        np.eye(len(x))
        - np.outer(K, h)
    )

    P_new = (
        I_KH @ P @ I_KH.T
        + np.outer(K, K) * OBS_VARIANCE
    )

    P_new = 0.5 * (
        P_new + P_new.T
    )

    # Protect diagonal against tiny negative roundoff
    diagonal = np.maximum(
        np.diag(P_new),
        0.0
    )

    for i in range(len(x)):
        P_new[i, i] = diagonal[i]

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
        nis
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("FULL-FLEXPART-FOOTPRINT AUGMENTED KALMAN FILTER")
print("=" * 90)

print("\nH file:")
print(H_FILE)

print("\nPicarro file:")
print(PICARRO_FILE)

print("\nInversion period:")
print(
    START_DATE.date(),
    "to",
    END_DATE.date()
)

print("\nNumber of days:", N_DAY)
print("Number of mines:", N_MINE)
print("Number of parameters:", N_PARAM)

print("\nObservation model:")
print(
    "y_t = H_(d-1) x_(d-1) + H_d x_d + epsilon"
)

print("\nState:")
print(
    "X_d = [x_(d-1), x_d]"
)

print("\nPrior:")
print("mean =", PRIOR_MEAN)
print("sigma =", PRIOR_SIGMA)
print("correlation length =", L_CORR, "days")

print("\nObservation error:")
print("sigma =", OBS_SIGMA, "ppm")


# ============================================================
# PARAMETER NAMES
# ============================================================

PARAMETER_NAMES = [
    parameter_name(day, mine)
    for day in range(N_DAY)
    for mine in range(N_MINE)
]

print("\nFirst parameter names:")
print(PARAMETER_NAMES[:16])


# ============================================================
# READ H
# ============================================================

print("\n" + "=" * 90)
print("READING FLEXPART H")
print("=" * 90)

H_df = pd.read_csv(H_FILE)

print("H shape:", H_df.shape)

required_columns = [
    "receptor",
    "release_utc",
    "obs_time_aest",
    "flexpart_file",
]

for column in required_columns:

    if column not in H_df.columns:
        raise RuntimeError(
            f"Missing H metadata column: {column}"
        )

for column in PARAMETER_NAMES:

    if column not in H_df.columns:
        raise RuntimeError(
            f"Missing H parameter column: {column}"
        )


# Convert time
H_df["obs_time_aest"] = pd.to_datetime(
    H_df["obs_time_aest"]
)

H_df["obs_time_aest"] = (
    H_df["obs_time_aest"]
    .dt.floor("h")
)


# ============================================================
# READ PICARRO
# ============================================================

print("\n" + "=" * 90)
print("READING PICARRO")
print("=" * 90)

picarro = pd.read_csv(
    PICARRO_FILE
)

picarro["Datetime"] = pd.to_datetime(
    picarro["Datetime"],
    format="%d/%m/%Y %H:%M",
    errors="coerce"
)

receptor_columns = {
    "Singleton cleaned": "singleton",
    "Muswellbrook cleaned": "muswellbrook",
    "Merriwa cleaned": "merriwa",
}

obs_frames = []

for column, receptor in receptor_columns.items():

    if column not in picarro.columns:
        raise RuntimeError(
            f"Missing Picarro column: {column}"
        )

    temp = picarro[
        [
            "Datetime",
            column,
        ]
    ].copy()

    temp = temp.rename(
        columns={
            "Datetime": "obs_time_aest",
            column: "observation",
        }
    )

    temp["receptor"] = receptor

    temp["obs_time_aest"] = (
        temp["obs_time_aest"]
        .dt.floor("h")
    )

    temp = temp.dropna(
        subset=["observation"]
    )

    obs_frames.append(temp)


obs = pd.concat(
    obs_frames,
    ignore_index=True
)

obs = obs[
    (obs["obs_time_aest"] >= START_DATE)
    &
    (
        obs["obs_time_aest"]
        < END_DATE + pd.Timedelta(days=1)
    )
].copy()


# Average duplicate receptor/hour
obs = (
    obs
    .groupby(
        [
            "receptor",
            "obs_time_aest",
        ],
        as_index=False
    )["observation"]
    .mean()
)

print("Observations:", len(obs))


# ============================================================
# MATCH H AND OBSERVATIONS
# ============================================================

print("\n" + "=" * 90)
print("MATCHING H AND OBSERVATIONS")
print("=" * 90)

matched = H_df.merge(
    obs,
    on=[
        "receptor",
        "obs_time_aest",
    ],
    how="inner"
)

matched = (
    matched
    .sort_values(
        [
            "obs_time_aest",
            "receptor",
        ]
    )
    .reset_index(drop=True)
)

print("Matched observations:", len(matched))

if len(matched) == 0:
    raise RuntimeError(
        "No matched observations."
    )


# ============================================================
# FULL H
# ============================================================

H_full = matched[
    PARAMETER_NAMES
].to_numpy(
    dtype=np.float64
)

y = matched[
    "observation"
].to_numpy(
    dtype=np.float64
)

d = y - BACKGROUND_PPM

times = matched[
    "obs_time_aest"
].to_numpy()

receptors = matched[
    "receptor"
].to_numpy()


print("\nFull H shape:", H_full.shape)


# ============================================================
# VERIFY TEMPORAL STRUCTURE
# ============================================================

print("\n" + "=" * 90)
print("VERIFYING FLEXPART TEMPORAL STRUCTURE")
print("=" * 90)

maximum_lag = 0

for row_index in range(len(matched)):

    timestamp = pd.Timestamp(
        matched.loc[
            row_index,
            "obs_time_aest"
        ]
    )

    day_index = get_day_index(timestamp)

    if day_index is None:
        continue

    row = H_full[row_index]

    nonzero = np.flatnonzero(
        np.abs(row) > 0
    )

    for parameter_index in nonzero:

        source_day = (
            parameter_index // N_MINE
        )

        lag = day_index - source_day

        maximum_lag = max(
            maximum_lag,
            lag
        )

print(
    "Maximum backward lag:",
    maximum_lag,
    "day(s)"
)

if maximum_lag > 1:

    raise RuntimeError(
        "Expected maximum lag <= 1 day, "
        f"but found {maximum_lag}."
    )


# ============================================================
# BUILD FULL PRIOR
# ============================================================

print("\n" + "=" * 90)
print("BUILDING FULL PRIOR COVARIANCE")
print("=" * 90)

B = build_B()

print("B shape:", B.shape)

print(
    "B[OC1 Jan5, OC1 Jan5] =",
    B[0, 0]
)

print(
    "B[OC1 Jan5, OC1 Jan6] =",
    B[0, 13]
)

print(
    "B[OC1 Jan5, OC2 Jan5] =",
    B[0, 1]
)


# ============================================================
# CHOLESKY
# ============================================================

LB = np.linalg.cholesky(B)

print(
    "Cholesky reconstruction error:",
    np.max(
        np.abs(
            B - LB @ LB.T
        )
    )
)


# ============================================================
# IMPORTANT:
# FULL BATCH MAP REFERENCE SOLUTION
# ============================================================

print("\n" + "=" * 90)
print("COMPUTING FULL BATCH MAP REFERENCE")
print("=" * 90)

xb_full = np.full(
    N_PARAM,
    PRIOR_MEAN,
    dtype=np.float64
)

# Whiten prior:
#
# B = L L^T
#
# || L^-1 (x-xb) ||^2

A_prior = solve_triangular(
    LB,
    np.eye(N_PARAM),
    lower=True
)

# Observation block
A_obs = H_full / OBS_SIGMA

# x = xb + z
#
# H(xb + z) = d
#
# H z = d - H xb

observation_rhs = (
    d
    - H_full @ xb_full
)

A_aug = np.vstack(
    [
        A_prior,
        A_obs,
    ]
)

b_aug = np.concatenate(
    [
        np.zeros(N_PARAM),
        observation_rhs / OBS_SIGMA,
    ]
)

print(
    "Whitened system shape:",
    A_aug.shape
)

z_map, residuals, rank, singular_values = (
    np.linalg.lstsq(
        A_aug,
        b_aug,
        rcond=None
    )
)

x_map = xb_full + z_map

print("MAP rank:", rank)

if len(residuals):
    print(
        "Least-squares residual norm:",
        residuals
    )


# MAP gradient check
delta_map = (
    x_map
    - xb_full
)

tmp = solve_triangular(
    LB,
    delta_map,
    lower=True
)

Binv_delta = solve_triangular(
    LB.T,
    tmp,
    lower=False
)

map_gradient = (
    Binv_delta
    + H_full.T @ (
        H_full @ x_map - d
    ) / OBS_VARIANCE
)

print(
    "\nMAP gradient L2:",
    np.linalg.norm(map_gradient)
)

print(
    "MAP gradient max abs:",
    np.max(
        np.abs(map_gradient)
    )
)

print(
    "Batch MAP mean:",
    np.mean(x_map)
)


# ============================================================
# AUGMENTED KF
# ============================================================

print("\n" + "=" * 90)
print("RUNNING FULL-FOOTPRINT AUGMENTED KF")
print("=" * 90)

rho = np.exp(
    -1.0 / L_CORR
)

Q_daily = (
    PRIOR_VARIANCE
    * (1.0 - rho ** 2)
    * np.eye(N_MINE)
)

# ------------------------------------------------------------
# IMPORTANT INITIALIZATION
#
# At Jan 5, we do not have an unknown Jan 4 state.
#
# Therefore:
#
# X = [x_prev, x_current]
#
# is initialized as:
#
# x_prev = PRIOR_MEAN
# x_current = PRIOR_MEAN
#
# with the current-state prior covariance corresponding to
# the AR(1) stationary prior.
#
# The Jan 5 observations only contain Jan 5 source terms in
# the inversion window.
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

# The augmented covariance is initially constructed from
# the stationary prior.
#
# Previous and current states:
#
# Cov(prev, prev) = sigma^2
# Cov(current,current) = sigma^2
# Cov(prev,current) = sigma^2*rho

P_aug = np.zeros(
    (2 * N_MINE, 2 * N_MINE),
    dtype=np.float64
)

P_aug[
    :N_MINE,
    :N_MINE
] = PRIOR_VARIANCE * np.eye(N_MINE)

P_aug[
    N_MINE:,
    N_MINE:
] = PRIOR_VARIANCE * np.eye(N_MINE)

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

daily_states = []
daily_covariances = []

current_day = None

# Store final current-state posterior for each day
current_day_state = None
current_day_covariance = None
current_day_nobs = 0


# ============================================================
# DAILY TRANSITION
# ============================================================

def transition_to_next_day(
    x_aug,
    P_aug
):
    """
    Shift:

        [x_d-1, x_d]

    to:

        [x_d, x_d+1]

    where

        x_d+1 = rho*x_d
                +(1-rho)*prior_mean
                + w

    """

    x_old_current = (
        x_aug[N_MINE:]
    )

    P_old = P_aug

    x_new_current = (
        rho * x_old_current
        + (1.0 - rho) * PRIOR_MEAN
    )

    x_new = np.concatenate(
        [
            x_old_current,
            x_new_current,
        ]
    )

    # New covariance
    #
    # x_new_prev = x_old_current
    # x_new_current = rho*x_old_current + w

    P_new = np.zeros_like(
        P_old
    )

    # Cov(x_d, x_d)
    P_current = P_old[
        N_MINE:,
        N_MINE:
    ]

    P_new[
        :N_MINE,
        :N_MINE
    ] = P_current

    # Cov(x_d, x_d+1)
    P_cross = (
        rho * P_current
    )

    P_new[
        :N_MINE,
        N_MINE:
    ] = P_cross

    P_new[
        N_MINE:,
        :N_MINE
    ] = P_cross.T

    # Cov(x_d+1, x_d+1)
    P_new[
        N_MINE:,
        N_MINE:
    ] = (
        rho ** 2 * P_current
        + Q_daily
    )

    P_new = 0.5 * (
        P_new + P_new.T
    )

    return (
        x_new,
        P_new
    )


# ============================================================
# ASSIMILATE OBSERVATIONS CHRONOLOGICALLY
# ============================================================

for row_index in range(
    len(matched)
):

    timestamp = pd.Timestamp(
        matched.loc[
            row_index,
            "obs_time_aest"
        ]
    )

    day = timestamp.normalize()

    observation = y[row_index]

    # --------------------------------------------------------
    # INITIAL DAY
    # --------------------------------------------------------

    if current_day is None:

        current_day = day

        current_day_state = (
            x_aug[N_MINE:].copy()
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

        # Save previous day's final current-state posterior
        daily_records.append(
            {
                "date": current_day,
                "state_type": "analysis",
                "n_observations": current_day_nobs,
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

        # Move through any missing days
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
                x_aug[N_MINE:].copy()
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

            daily_states.append(
                forecast_state.copy()
            )

            daily_covariances.append(
                forecast_covariance.copy()
            )

            next_day += pd.Timedelta(days=1)

        # Transition into current observation day
        x_aug, P_aug = (
            transition_to_next_day(
                x_aug,
                P_aug
            )
        )

        current_day = day

        current_day_nobs = 0

    # --------------------------------------------------------
    # OBSERVATION OPERATOR
    # --------------------------------------------------------

    day_index = get_day_index(
        timestamp
    )

    if day_index is None:
        raise RuntimeError(
            f"Observation outside inversion window: "
            f"{timestamp}"
        )

    current_indices = get_day_block(
        day_index
    )

    previous_indices = (
        get_day_block(day_index - 1)
        if day_index > 0
        else np.array([], dtype=int)
    )

    # Full augmented observation vector:
    #
    # [H_previous, H_current]

    h_aug = np.zeros(
        2 * N_MINE,
        dtype=np.float64
    )

    # Previous day contribution
    if len(previous_indices):

        h_aug[
            :N_MINE
        ] = H_full[
            row_index,
            previous_indices
        ]

    # Current day contribution
    h_aug[
        N_MINE:
    ] = H_full[
        row_index,
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

    # Current-day state after this observation
    current_day_state = (
        x_aug[N_MINE:].copy()
    )

    current_day_covariance = (
        P_aug[
            N_MINE:,
            N_MINE:
        ].copy()
    )

    # --------------------------------------------------------
    # STORE HOURLY RESULT
    # --------------------------------------------------------

    hourly_record = {
        "time_aest": timestamp,
        "receptor": receptors[row_index],
        "observation": observation,
        "prediction": prediction,
        "innovation": innovation,
        "innovation_variance": innovation_variance,
        "NIS": nis,
    }

    for i, mine in enumerate(MINES):

        hourly_record[
            f"previous_{mine}"
        ] = x_aug[i]

        hourly_record[
            f"current_{mine}"
        ] = x_aug[N_MINE + i]

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
            "n_observations": current_day_nobs,
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


# ============================================================
# DATAFRAMES
# ============================================================

hourly_df = pd.DataFrame(
    hourly_records
)

daily_df = pd.DataFrame(
    daily_records
)

daily_df["date"] = pd.to_datetime(
    daily_df["date"]
)

daily_df = (
    daily_df
    .sort_values("date")
    .reset_index(drop=True)
)

daily_states = np.asarray(
    daily_states,
    dtype=np.float64
)

daily_covariances = np.asarray(
    daily_covariances,
    dtype=np.float64
)


# ============================================================
# SAVE DAILY SCALING FACTORS
# ============================================================

daily_file = os.path.join(
    OUTPUT_DIR,
    "daily_full_footprint_KF_scaling_factors.csv"
)

daily_df.to_csv(
    daily_file,
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
# SAVE COVARIANCE DIAGONALS
# ============================================================

covariance_records = []

for day_index, date in enumerate(
    daily_df["date"]
):

    record = {
        "date": date
    }

    covariance = daily_covariances[
        day_index
    ]

    for mine_index, mine in enumerate(
        MINES
    ):

        record[mine] = covariance[
            mine_index,
            mine_index
        ]

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
    states=daily_states,
    covariances=daily_covariances,
    dates=daily_df["date"]
        .astype(str)
        .to_numpy(),
    mines=np.asarray(MINES),
)


# ============================================================
# COMPARE DAILY KF WITH BATCH MAP
# ============================================================

print("\n" + "=" * 90)
print("FULL-FOOTPRINT KF VS BATCH MAP")
print("=" * 90)

kf_vector = daily_states.reshape(
    -1
)

difference = (
    kf_vector
    - x_map
)

print(
    "Maximum absolute difference:",
    np.max(np.abs(difference))
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
    and
    np.std(x_map) > 0
):

    print(
        "Correlation:",
        np.corrcoef(
            kf_vector,
            x_map
        )[0, 1]
    )


# ============================================================
# DAILY MEAN COMPARISON
# ============================================================

map_daily = x_map.reshape(
    N_DAY,
    N_MINE
)

comparison_records = []

for day_index in range(N_DAY):

    date = (
        START_DATE
        + pd.Timedelta(days=day_index)
    )

    comparison_records.append(
        {
            "date": date,
            "KF_mean": np.mean(
                daily_states[
                    day_index
                ]
            ),
            "MAP_mean": np.mean(
                map_daily[
                    day_index
                ]
            ),
            "difference": np.mean(
                daily_states[
                    day_index
                ]
                -
                map_daily[
                    day_index
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
# FINAL-DAY TEST
# ============================================================

print("\n" + "=" * 90)
print("FINAL-DAY TEST")
print("=" * 90)

final_kf = daily_states[-1]
final_map = map_daily[-1]

for i, mine in enumerate(MINES):

    print(
        f"{mine:<8s}"
        f"{final_kf[i]:14.8f}"
        f"{final_map[i]:14.8f}"
        f"{final_kf[i]-final_map[i]:14.8f}"
    )

print(
    "\nFinal-day maximum difference:",
    np.max(
        np.abs(
            final_kf - final_map
        )
    )
)


# ============================================================
# NIS SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("NIS SUMMARY")
print("=" * 90)

print(
    "Number of assimilated observations:",
    len(hourly_df)
)

print(
    "Mean NIS:",
    hourly_df["NIS"].mean()
)

print(
    "Median NIS:",
    hourly_df["NIS"].median()
)

print(
    "Maximum NIS:",
    hourly_df["NIS"].max()
)


# ============================================================
# SUMMARY
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
        "UPPER HUNTER FULL-FLEXPART-FOOTPRINT "
        "AUGMENTED KF\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Start date: {START_DATE.date()}\n"
    )

    f.write(
        f"End date: {END_DATE.date()}\n"
    )

    f.write(
        f"Number of mines: {N_MINE}\n"
    )

    f.write(
        f"Number of parameters: {N_PARAM}\n"
    )

    f.write(
        f"Matched observations: {len(matched)}\n"
    )

    f.write(
        f"Maximum FLEXPART lag: "
        f"{maximum_lag} day(s)\n"
    )

    f.write(
        f"Prior mean: {PRIOR_MEAN}\n"
    )

    f.write(
        f"Prior sigma: {PRIOR_SIGMA}\n"
    )

    f.write(
        f"Observation sigma: {OBS_SIGMA}\n"
    )

    f.write(
        f"Correlation length: {L_CORR} days\n\n"
    )

    f.write(
        "FULL KF VS BATCH MAP\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "Maximum absolute difference: "
        f"{np.max(np.abs(difference))}\n"
    )

    f.write(
        "RMSE: "
        f"{np.sqrt(np.mean(difference**2))}\n"
    )

    f.write(
        "MAE: "
        f"{np.mean(np.abs(difference))}\n"
    )

    f.write(
        "\nMAP GRADIENT\n"
    )

    f.write(
        f"L2: {np.linalg.norm(map_gradient)}\n"
    )

    f.write(
        f"Maximum absolute: "
        f"{np.max(np.abs(map_gradient))}\n"
    )

    f.write(
        "\nNIS\n"
    )

    f.write(
        f"Mean: {hourly_df['NIS'].mean()}\n"
    )

    f.write(
        f"Median: {hourly_df['NIS'].median()}\n"
    )

    f.write(
        f"Maximum: {hourly_df['NIS'].max()}\n"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 90)
print("OUTPUT FILES")
print("=" * 90)

print(daily_file)
print(hourly_file)
print(covariance_file)
print(npz_file)
print(comparison_file)
print(summary_file)

print("\nDone.")

