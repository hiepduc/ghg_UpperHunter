#!/usr/bin/env python3

"""
AUGMENTED RTS SMOOTHER FOR FULL-FLEXPART-FOOTPRINT KF

Upper Hunter CH4
13 mines x 27 daily scaling factors = 351 parameters

State used by the full-footprint KF:

    z_d = [x_{d-1}, x_d]

where x_d is the 13-dimensional daily mine-scaling state.

The full FLEXPART observation equation is:

    y_t = BACKGROUND_PPM
          + H_{t,d-1} x_{d-1}
          + H_{t,d}   x_d
          + epsilon_t

The augmented state transition is:

    [x_d    ]   [0  I] [x_{d-1}]   [0]
    [x_{d+1}] = [0 rhoI][x_d    ] + [(1-rho)mu] + w

with

    Q_aug = [[0, 0],
             [0, Q]]

Because F_aug is singular, the RTS gain uses a Moore-Penrose
pseudoinverse of the forecast covariance.

This script:

1. Loads the FULL-FOOTPRINT KF augmented states/covariances.
2. Performs the augmented RTS backward smoother.
3. Extracts the smoothed 13-D daily states.
4. Independently reconstructs the 351-parameter batch MAP.
5. Correctly accounts for the prior mean in the MAP observation RHS.
6. Checks the MAP gradient.
7. Compares augmented RTS against batch MAP.
8. Checks the final-day solution.
9. Saves the smoothed results.

"""

import os
import numpy as np
import pandas as pd

from scipy.linalg import (
    cholesky,
    cho_solve,
)


# ============================================================
# CONFIGURATION
# ============================================================

H_FILE = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "H_matrix_flexpart/H_flexpart_0_10m_13mine_Jan2026.csv"
)

PICARRO_FILE = (
    "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter/"
    "tapm_output/Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

KF_DIR = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "H_matrix_flexpart/KF_full_footprint_13mine_Jan2026"
)

# Use the NPZ produced by the revised full-footprint KF.
KF_NPZ = os.path.join(
    KF_DIR,
    "full_footprint_KF_daily_states.npz"
)
OUTPUT_DIR = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "H_matrix_flexpart/"
    "KF_full_footprint_13mine_Jan2026"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MODEL SETTINGS
# ============================================================

START_DATE = pd.Timestamp("2026-01-05")
END_DATE = pd.Timestamp("2026-01-31")

N_MINE = 13
N_DAY = 27
N_PARAM = N_MINE * N_DAY

MINES = [f"OC{i}" for i in range(1, 14)]

BACKGROUND_PPM = 1.85

PRIOR_MEAN = 1.0
PRIOR_SIGMA = 0.5
PRIOR_VARIANCE = PRIOR_SIGMA ** 2

OBS_SIGMA = 0.05
OBS_VARIANCE = OBS_SIGMA ** 2

CORRELATION_LENGTH_DAYS = 3.0

RHO = np.exp(
    -1.0 / CORRELATION_LENGTH_DAYS
)

Q_DIAGONAL = (
    PRIOR_VARIANCE
    * (1.0 - RHO ** 2)
)


# ============================================================
# HELPERS
# ============================================================

def symmetrize(A):
    """Force numerical symmetry."""
    return 0.5 * (A + A.T)


def parameter_names():

    names = []

    dates = pd.date_range(
        START_DATE,
        END_DATE,
        freq="D"
    )

    for date in dates:

        datestr = date.strftime("%Y%m%d")

        for mine in MINES:

            names.append(
                f"{mine}_{datestr}"
            )

    return names


def get_day_index(timestamp):

    timestamp = pd.Timestamp(timestamp)

    day = timestamp.normalize()

    idx = (
        day - START_DATE
    ).days

    return int(idx)


def get_day_block(day_index):

    start = day_index * N_MINE

    return np.arange(
        start,
        start + N_MINE
    )


def build_prior_covariance():

    """
    B is block diagonal by mine.

    Within each mine:

        B[d1,d2] =
            sigma^2 exp(-|d1-d2| / L)

    Cross-mine covariance = 0.
    """

    B = np.zeros(
        (N_PARAM, N_PARAM),
        dtype=np.float64
    )

    for mine_index in range(N_MINE):

        for d1 in range(N_DAY):

            i = (
                d1 * N_MINE
                + mine_index
            )

            for d2 in range(N_DAY):

                j = (
                    d2 * N_MINE
                    + mine_index
                )

                B[i, j] = (
                    PRIOR_VARIANCE
                    * np.exp(
                        -abs(d1 - d2)
                        / CORRELATION_LENGTH_DAYS
                    )
                )

    return symmetrize(B)


# ============================================================
# LOAD KF RESULTS
# ============================================================

print("=" * 90)
print("AUGMENTED RTS SMOOTHER — FULL FLEXPART FOOTPRINT")
print("=" * 90)

print()
print("Loading KF NPZ:")
print(KF_NPZ)

kf = np.load(
    KF_NPZ,
    allow_pickle=True
)

print()
print("NPZ keys:")
print(list(kf.keys()))

states_13 = kf["states"]
covariances_13 = kf["covariances"]

augmented_states = kf["augmented_states"]
augmented_covariances = kf["augmented_covariances"]

dates = pd.to_datetime(
    kf["dates"]
)

mines = list(
    kf["mines"]
)

print()
print("13-D states:",
      states_13.shape)

print("13-D covariances:",
      covariances_13.shape)

print("26-D augmented states:",
      augmented_states.shape)

print("26-D augmented covariances:",
      augmented_covariances.shape)

print(
    "Dates:",
    dates[0].date(),
    "to",
    dates[-1].date()
)

print()
print("Mines:")
print(mines)


# ============================================================
# BASIC VALIDATION
# ============================================================

assert augmented_states.shape == (
    N_DAY,
    2 * N_MINE
)

assert augmented_covariances.shape == (
    N_DAY,
    2 * N_MINE,
    2 * N_MINE
)

assert states_13.shape == (
    N_DAY,
    N_MINE
)

assert covariances_13.shape == (
    N_DAY,
    N_MINE,
    N_MINE
)


# ============================================================
# AUGMENTED STATE TRANSITION
# ============================================================

I13 = np.eye(N_MINE)

F_aug = np.zeros(
    (2 * N_MINE, 2 * N_MINE),
    dtype=np.float64
)

# x_d becomes previous half of next augmented state
F_aug[
    :N_MINE,
    N_MINE:
] = I13

# AR(1) transition for new current state
F_aug[
    N_MINE:,
    N_MINE:
] = RHO * I13


c_aug = np.zeros(
    2 * N_MINE,
    dtype=np.float64
)

c_aug[N_MINE:] = (
    (1.0 - RHO)
    * PRIOR_MEAN
)


Q_aug = np.zeros(
    (2 * N_MINE, 2 * N_MINE),
    dtype=np.float64
)

Q_aug[
    N_MINE:,
    N_MINE:
] = (
    Q_DIAGONAL * I13
)


print()
print("=" * 90)
print("AUGMENTED TRANSITION MODEL")
print("=" * 90)

print()
print("rho:", RHO)
print(
    "Daily Q diagonal:",
    Q_DIAGONAL
)

print(
    "F_aug rank:",
    np.linalg.matrix_rank(F_aug)
)

print(
    "F_aug expected rank:",
    N_MINE
)


# ============================================================
# CHECK STORED AUGMENTED STATE OVERLAP
# ============================================================

print()
print("=" * 90)
print("CHECKING STORED AUGMENTED STATES")
print("=" * 90)

max_stored_overlap = 0.0

for d in range(N_DAY - 1):

    current_state = (
        augmented_states[d, N_MINE:]
    )

    next_previous_state = (
        augmented_states[d + 1, :N_MINE]
    )

    difference = np.max(
        np.abs(
            current_state
            - next_previous_state
        )
    )

    max_stored_overlap = max(
        max_stored_overlap,
        difference
    )

print(
    "Maximum stored-state overlap difference:",
    max_stored_overlap
)

print(
    "(Nonzero is expected because adjacent stored "
    "posteriors contain different observations.)"
)


# ============================================================
# AUGMENTED RTS SMOOTHER
# ============================================================

print()
print("=" * 90)
print("RUNNING AUGMENTED RTS SMOOTHER")
print("=" * 90)

n_state = 2 * N_MINE

x_a = np.asarray(
    augmented_states,
    dtype=np.float64
)

P_a = np.asarray(
    augmented_covariances,
    dtype=np.float64
)

x_s = np.zeros_like(x_a)

P_s = np.zeros_like(P_a)


# Final day
x_s[-1] = x_a[-1]

P_s[-1] = symmetrize(
    P_a[-1]
)


# ------------------------------------------------------------
# Backward pass
# ------------------------------------------------------------

for d in range(N_DAY - 2, -1, -1):

    # Forecast from day d to day d+1.
    #
    # x_f = F x_a + c
    #
    x_f = (
        F_aug @ x_a[d]
        + c_aug
    )

    # Forecast covariance.
    #
    P_f = (
        F_aug
        @ P_a[d]
        @ F_aug.T
        + Q_aug
    )

    P_f = symmetrize(P_f)

    # Because F_aug is singular, P_f can be singular
    # or numerically rank deficient.
    #
    # Use the Moore-Penrose pseudoinverse rather than
    # np.linalg.inv().
    P_f_pinv = np.linalg.pinv(
        P_f,
        rcond=1e-12
    )

    # RTS gain:
    #
    # C_d =
    # P_a[d] F^T P_f^{-1}
    #
    C = (
        P_a[d]
        @ F_aug.T
        @ P_f_pinv
    )

    # Smoothed state
    x_s[d] = (
        x_a[d]
        + C
        @ (
            x_s[d + 1]
            - x_f
        )
    )

    # Smoothed covariance
    P_s[d] = (
        P_a[d]
        + C
        @ (
            P_s[d + 1]
            - P_f
        )
        @ C.T
    )

    P_s[d] = symmetrize(
        P_s[d]
    )


print()
print("RTS smoothing complete.")


# ============================================================
# EXTRACT 13-D CURRENT-DAY STATES
# ============================================================

smoothed_states = (
    x_s[:, N_MINE:]
)

smoothed_covariances = (
    P_s[
        :,
        N_MINE:,
        N_MINE:
    ]
)


# ============================================================
# CHECK SMOOTHED OVERLAP CONSISTENCY
# ============================================================

print()
print("=" * 90)
print("CHECKING SMOOTHED STATE OVERLAP")
print("=" * 90)

max_smoothed_overlap = 0.0

for d in range(N_DAY - 1):

    current_smoothed = (
        x_s[d, N_MINE:]
    )

    next_previous_smoothed = (
        x_s[d + 1, :N_MINE]
    )

    difference = np.max(
        np.abs(
            current_smoothed
            - next_previous_smoothed
        )
    )

    max_smoothed_overlap = max(
        max_smoothed_overlap,
        difference
    )

print(
    "Maximum smoothed overlap difference:",
    max_smoothed_overlap
)


# ============================================================
# LOAD H MATRIX
# ============================================================

print()
print("=" * 90)
print("LOADING FLEXPART H MATRIX")
print("=" * 90)

H_df = pd.read_csv(
    H_FILE
)

H_df["receptor"] = (
    H_df["receptor"]
    .astype(str)
    .str.strip()
    .str.lower()
)

H_df["time_aest"] = pd.to_datetime(
    H_df["obs_time_aest"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
).dt.floor("h")

PARAMETER_NAMES = parameter_names()

assert list(
    H_df[PARAMETER_NAMES].columns
) == PARAMETER_NAMES

H_full = H_df[
    PARAMETER_NAMES
].to_numpy(
    dtype=np.float64
)

print()
print("H shape:",
      H_df.shape)

print(
    "Number of parameter columns:",
    len(PARAMETER_NAMES)
)

print(
    "Expected:",
    N_PARAM
)


# ============================================================
# LOAD PICARRO OBSERVATIONS
# ============================================================

print()
print("=" * 90)
print("LOADING PICARRO OBSERVATIONS")
print("=" * 90)

P_df = pd.read_csv(
    PICARRO_FILE
)

P_df["Datetime"] = pd.to_datetime(
    P_df["Datetime"],
    format="%d/%m/%Y %H:%M",
    errors="coerce"
).dt.floor("h")


receptor_columns = {
    "singleton": "Singleton cleaned",
    "muswellbrook": "Muswellbrook cleaned",
    "merriwa": "Merriwa cleaned",
}


obs_parts = []

for receptor, column in receptor_columns.items():

    temp = P_df[
        ["Datetime", column]
    ].copy()

    temp = temp.rename(
        columns={
            "Datetime": "time_aest",
            column: "observation",
        }
    )

    temp["receptor"] = receptor

    obs_parts.append(
        temp[
            [
                "time_aest",
                "receptor",
                "observation",
            ]
        ]
    )


obs_long = pd.concat(
    obs_parts,
    ignore_index=True
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
        <= END_DATE
        + pd.Timedelta(hours=23)
    )
].copy()


# Average duplicate receptor/hour observations
obs_long = (
    obs_long
    .groupby(
        ["receptor", "time_aest"],
        as_index=False
    )["observation"]
    .mean()
)


print()
print(
    "Observation rows:",
    len(obs_long)
)


# ============================================================
# ALIGN H AND OBSERVATIONS
# ============================================================

H_meta = H_df[
    ["receptor", "time_aest"]
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
    how="inner"
)

merged = (
    merged
    .sort_values("time_aest")
    .reset_index(drop=True)
)

H_row_indices = (
    merged["H_row"]
    .to_numpy(dtype=int)
)

H_batch = H_full[
    H_row_indices,
    :
]

d_obs = merged[
    "observation"
].to_numpy(
    dtype=np.float64
)

print()
print(
    "Matched observations:",
    len(d_obs)
)

print(
    "H_batch shape:",
    H_batch.shape
)


# ============================================================
# BUILD B
# ============================================================

print()
print("=" * 90)
print("BUILDING PRIOR COVARIANCE")
print("=" * 90)

B = build_prior_covariance()

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
# CHOLESKY FACTOR
# ============================================================

L_B = cholesky(
    B,
    lower=True,
    check_finite=True
)

B_reconstructed = (
    L_B @ L_B.T
)

print()
print(
    "Cholesky reconstruction error:",
    np.max(
        np.abs(
            B
            - B_reconstructed
        )
    )
)


# ============================================================
# CORRECT BATCH MAP
# ============================================================

print()
print("=" * 90)
print("RECONSTRUCTING 351-PARAMETER BATCH MAP")
print("=" * 90)

# ------------------------------------------------------------
# Background state
# ------------------------------------------------------------
#
# x = x_background + z
#
# x_background = 1.0 for every mine/day.
#

x_background = np.full(
    N_PARAM,
    PRIOR_MEAN,
    dtype=np.float64
)


# ------------------------------------------------------------
# IMPORTANT CORRECTION
# ------------------------------------------------------------
#
# Observation equation:
#
#     d_obs = BACKGROUND_PPM + H x + epsilon
#
# and
#
#     x = x_background + z
#
# Therefore:
#
#     d_obs
#       = BACKGROUND_PPM
#       + H x_background
#       + H z
#       + epsilon
#
# hence:
#
#     H z =
#         d_obs
#         - BACKGROUND_PPM
#         - H x_background
#
# The previous RTS diagnostic omitted:
#
#     H_batch @ x_background
#
# which produced the incorrect MAP mean of ~1.794.
#

y_rhs = (
    d_obs
    - BACKGROUND_PPM
    - H_batch @ x_background
)


# ------------------------------------------------------------
# Whitened prior system
# ------------------------------------------------------------

A_prior = np.linalg.solve(
    L_B,
    np.eye(N_PARAM)
)


# ------------------------------------------------------------
# Whitened observation system
# ------------------------------------------------------------

A_obs = (
    H_batch
    / OBS_SIGMA
)


# ------------------------------------------------------------
# Augmented least-squares system
# ------------------------------------------------------------

A_aug = np.vstack(
    [
        A_prior,
        A_obs,
    ]
)

b_aug = np.concatenate(
    [
        np.zeros(N_PARAM),
        y_rhs / OBS_SIGMA,
    ]
)


print()
print(
    "Whitened system shape:",
    A_aug.shape
)


# ============================================================
# SOLVE MAP
# ============================================================

z_map, residuals, rank, singular_values = np.linalg.lstsq(
    A_aug,
    b_aug,
    rcond=None
)

x_map = (
    x_background
    + z_map
)


print()
print(
    "MAP rank:",
    rank
)

print(
    "Least-squares residual norm:",
    residuals
)

print(
    "Batch MAP mean:",
    np.mean(x_map)
)


# ============================================================
# MAP GRADIENT CHECK
# ============================================================

print()
print("=" * 90)
print("CHECKING BATCH MAP GRADIENT")
print("=" * 90)

prior_gradient = cho_solve(
    (L_B, True),
    x_map - x_background
)

obs_gradient = (
    H_batch.T
    @ (
        H_batch @ x_map
        - (
            d_obs
            - BACKGROUND_PPM
        )
    )
    / OBS_VARIANCE
)

map_gradient = (
    prior_gradient
    + obs_gradient
)

print(
    "MAP gradient L2:",
    np.linalg.norm(map_gradient)
)

print(
    "MAP gradient max abs:",
    np.max(
        np.abs(map_gradient)
    )
)


# ============================================================
# INDEPENDENT NORMAL-EQUATION CHECK
# ============================================================

ls_residual = (
    A_aug @ z_map
    - b_aug
)

normal_gradient = (
    A_aug.T
    @ ls_residual
)

print()
print(
    "Whitened LS residual L2:",
    np.linalg.norm(ls_residual)
)

print(
    "Whitened normal-equation gradient L2:",
    np.linalg.norm(
        normal_gradient
    )
)

print(
    "Whitened normal-equation gradient max abs:",
    np.max(
        np.abs(normal_gradient)
    )
)


# ============================================================
# COMPARE RTS TO BATCH MAP
# ============================================================

print()
print("=" * 90)
print("AUGMENTED RTS VS BATCH MAP")
print("=" * 90)

difference = (
    smoothed_states
    - x_map.reshape(
        N_DAY,
        N_MINE
    )
)

max_difference = np.max(
    np.abs(difference)
)

rmse = np.sqrt(
    np.mean(
        difference ** 2
    )
)

mae = np.mean(
    np.abs(difference)
)

correlation = np.corrcoef(
    smoothed_states.ravel(),
    x_map
)[0, 1]


print(
    "Maximum absolute difference:",
    max_difference
)

print(
    "RMSE:",
    rmse
)

print(
    "MAE:",
    mae
)

print(
    "Correlation:",
    correlation
)


# ============================================================
# FINAL-DAY TEST
# ============================================================

print()
print("=" * 90)
print("FINAL-DAY TEST")
print("=" * 90)

final_day = N_DAY - 1

for i, mine in enumerate(MINES):

    rts_value = (
        smoothed_states[
            final_day,
            i
        ]
    )

    map_value = (
        x_map[
            final_day * N_MINE
            + i
        ]
    )

    diff = (
        rts_value
        - map_value
    )

    print(
        f"{mine:<5s}"
        f" RTS {rts_value:.8f}"
        f" MAP {map_value:.8f}"
        f" diff {diff:.8e}"
    )


final_day_difference = np.max(
    np.abs(
        smoothed_states[
            final_day
        ]
        -
        x_map[
            final_day * N_MINE:
            (final_day + 1) * N_MINE
        ]
    )
)

print()
print(
    "Final-day maximum difference:",
    final_day_difference
)


# ============================================================
# DAILY COMPARISON CSV
# ============================================================

print()
print("=" * 90)
print("SAVING RESULTS")
print("=" * 90)

comparison_rows = []

for d in range(N_DAY):

    row = {
        "date": dates[d]
    }

    for i, mine in enumerate(MINES):

        row[
            f"{mine}_RTS"
        ] = smoothed_states[d, i]

        row[
            f"{mine}_MAP"
        ] = x_map[
            d * N_MINE + i
        ]

        row[
            f"{mine}_DIFF"
        ] = (
            smoothed_states[d, i]
            - x_map[
                d * N_MINE + i
            ]
        )

    comparison_rows.append(row)


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_file = os.path.join(
    OUTPUT_DIR,
    "RTS_vs_batch_MAP_daily.csv"
)

comparison_df.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# SAVE RTS RESULTS
# ============================================================

rts_daily_file = os.path.join(
    OUTPUT_DIR,
    "daily_RTS_smoothed_scaling_factors.csv"
)

rts_daily_df = pd.DataFrame(
    smoothed_states,
    columns=MINES
)

rts_daily_df.insert(
    0,
    "date",
    dates
)

rts_daily_df.to_csv(
    rts_daily_file,
    index=False
)


# ============================================================
# SAVE MAP RESULTS
# ============================================================

map_daily = x_map.reshape(
    N_DAY,
    N_MINE
)

map_daily_file = os.path.join(
    OUTPUT_DIR,
    "daily_batch_MAP_scaling_factors.csv"
)

map_daily_df = pd.DataFrame(
    map_daily,
    columns=MINES
)

map_daily_df.insert(
    0,
    "date",
    dates
)

map_daily_df.to_csv(
    map_daily_file,
    index=False
)


# ============================================================
# SAVE COVARIANCES
# ============================================================

npz_file = os.path.join(
    OUTPUT_DIR,
    "RTS_full_footprint_daily_states.npz"
)

np.savez(
    npz_file,

    # 26-D augmented smoothed state
    augmented_smoothed_states=x_s,

    # 26-D augmented smoothed covariance
    augmented_smoothed_covariances=P_s,

    # 13-D current-day smoothed state
    smoothed_states=smoothed_states,

    # 13-D current-day smoothed covariance
    smoothed_covariances=smoothed_covariances,

    # Batch MAP
    batch_map=x_map,

    # Dates
    dates=np.asarray(
        dates.astype(str)
    ),

    # Mine names
    mines=np.asarray(
        MINES
    ),
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 90)
print("COMPLETE")
print("=" * 90)

print()
print(
    "RTS daily states:"
)
print(
    rts_daily_file
)

print()
print(
    "Batch MAP daily states:"
)
print(
    map_daily_file
)

print()
print(
    "RTS vs MAP comparison:"
)
print(
    comparison_file
)

print()
print(
    "NPZ:"
)
print(
    npz_file
)

print()
print("Final diagnostics:")
print(
    "  Batch MAP mean:",
    np.mean(x_map)
)

print(
    "  MAP gradient L2:",
    np.linalg.norm(map_gradient)
)

print(
    "  MAP gradient max:",
    np.max(
        np.abs(map_gradient)
    )
)

print(
    "  RTS-vs-MAP RMSE:",
    rmse
)

print(
    "  RTS-vs-MAP MAE:",
    mae
)

print(
    "  RTS-vs-MAP max difference:",
    max_difference
)

print()
print("=" * 90)

