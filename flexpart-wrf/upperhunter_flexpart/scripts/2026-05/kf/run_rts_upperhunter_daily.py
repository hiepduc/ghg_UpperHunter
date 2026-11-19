#!/usr/bin/env python3

"""
RTS smoother for the Upper Hunter 13-mine daily Kalman filter.

Uses the existing daily KF states/covariances and the same AR(1)
state-transition model:

    x[d+1] = rho*x[d] + (1-rho)*PRIOR_MEAN + w[d]

with

    Q = PRIOR_VARIANCE * (1-rho^2) I

The RTS smoothed solution is the full-window posterior estimate and
is the appropriate linear-Gaussian counterpart to an unconstrained
batch 4D-Var solution.
"""

import os
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

KF_DIR = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "H_matrix_flexpart/KF_13mine_daily_Jan2026"
)

KF_DAILY_NPZ = os.path.join(
    KF_DIR,
    "KF_daily_states.npz"
)

KF_DAILY_CSV = os.path.join(
    KF_DIR,
    "daily_KF_scaling_factors.csv"
)

OUTPUT_RTS = os.path.join(
    KF_DIR,
    "RTS_daily_scaling_factors.csv"
)

OUTPUT_RTS_COV = os.path.join(
    KF_DIR,
    "RTS_daily_covariance_diagonal.csv"
)

OUTPUT_RTS_EMISSIONS = os.path.join(
    KF_DIR,
    "RTS_daily_total_emissions.csv"
)

PRIOR_MEAN = 1.0
PRIOR_SIGMA = 0.5
PRIOR_VARIANCE = PRIOR_SIGMA ** 2

CORRELATION_LENGTH_DAYS = 3.0

rho = np.exp(
    -1.0 / CORRELATION_LENGTH_DAYS
)

N_MINE = 13

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
# Prior emissions
# ============================================================

PRIOR_EMISSIONS = {
    "OC1": 1786.45,
    "OC2": 242.86,
    "OC3": 342.53,
    "OC4": 40.60,
    "OC5": 0.01,
    "OC6": 7.85,
    "OC7": 82.69,
    "OC8": 5.95,
    "OC9": 44.92,
    "OC10": 101.93,
    "OC11": 1.87,
    "OC12": 331.49,
    "OC13": 798.08,
}


# ============================================================
# Read KF output
# ============================================================

print("=" * 80)
print("UPPER HUNTER DAILY RTS KALMAN SMOOTHER")
print("=" * 80)

print()
print("Reading:")
print(KF_DAILY_NPZ)

z = np.load(
    KF_DAILY_NPZ,
    allow_pickle=True
)

print()
print("NPZ keys:", z.files)

if "states" not in z.files:
    raise RuntimeError(
        "KF_daily_states.npz does not contain 'states'."
    )

if "covariances" not in z.files:
    raise RuntimeError(
        "KF_daily_states.npz does not contain 'covariances'."
    )

daily_states = np.asarray(
    z["states"],
    dtype=np.float64
)

daily_covariances = np.asarray(
    z["covariances"],
    dtype=np.float64
)

print("states shape:", daily_states.shape)
print("covariances shape:", daily_covariances.shape)


# ============================================================
# Read daily metadata
# ============================================================

daily_df = pd.read_csv(
    KF_DAILY_CSV
)

print()
print("Daily CSV shape:", daily_df.shape)
print("Columns:", list(daily_df.columns))


if "date" not in daily_df.columns:
    raise RuntimeError(
        "daily_KF_scaling_factors.csv has no 'date' column."
    )

daily_df["date"] = pd.to_datetime(
    daily_df["date"]
)

# Make sure the mine columns are present.
missing_mines = [
    m for m in MINES
    if m not in daily_df.columns
]

if missing_mines:
    raise RuntimeError(
        f"Missing mine columns in daily CSV: {missing_mines}"
    )


# ============================================================
# Validate dimensions
# ============================================================

n_days = len(daily_df)

if daily_states.shape != (n_days, N_MINE):
    raise RuntimeError(
        "State array does not match daily CSV.\n"
        f"states = {daily_states.shape}\n"
        f"expected = ({n_days}, {N_MINE})"
    )

if daily_covariances.shape != (
    n_days,
    N_MINE,
    N_MINE
):
    raise RuntimeError(
        "Covariance array does not match daily CSV.\n"
        f"covariances = {daily_covariances.shape}\n"
        f"expected = ({n_days}, {N_MINE}, {N_MINE})"
    )


# ============================================================
# Verify chronological order
# ============================================================

daily_df = daily_df.sort_values(
    "date"
).reset_index(drop=True)

dates = daily_df["date"].values

if not np.all(
    dates[1:] > dates[:-1]
):
    raise RuntimeError(
        "Daily dates are not strictly increasing."
    )


# ============================================================
# AR(1) transition
# ============================================================

F = rho * np.eye(N_MINE)

c = (
    (1.0 - rho)
    * PRIOR_MEAN
    * np.ones(N_MINE)
)

Q = (
    PRIOR_VARIANCE
    * (1.0 - rho ** 2)
    * np.eye(N_MINE)
)

print()
print("RTS model:")
print(f"rho                  = {rho:.10f}")
print(f"prior mean            = {PRIOR_MEAN:.6f}")
print(f"prior sigma           = {PRIOR_SIGMA:.6f}")
print(f"prior variance        = {PRIOR_VARIANCE:.6f}")
print(f"Q diagonal            = {Q[0,0]:.10f}")
print(f"number of days        = {n_days}")
print(f"number of mines       = {N_MINE}")


# ============================================================
# Treat daily KF states as posterior states
# ============================================================
#
# For days with observations:
#
#   x_a[d] = KF analysis
#   P_a[d] = KF analysis covariance
#
# For forecast-only days:
#
#   x_a[d] = x_f[d]
#   P_a[d] = P_f[d]
#
# because there was no observation update.
#
# Therefore the existing daily arrays can be used directly.
# ============================================================

x_a = daily_states.copy()
P_a = daily_covariances.copy()

# Numerical symmetrization
for d in range(n_days):
    P_a[d] = 0.5 * (
        P_a[d] + P_a[d].T
    )


# ============================================================
# Allocate smoother arrays
# ============================================================

x_s = np.zeros_like(x_a)
P_s = np.zeros_like(P_a)

# Final state:
#
# The last day has no future observations, so its smoothed
# state is exactly its filtered analysis state.
#
x_s[-1] = x_a[-1]
P_s[-1] = P_a[-1]


# ============================================================
# RTS backward recursion
# ============================================================

print()
print("Running RTS backward recursion...")

for d in range(n_days - 2, -1, -1):

    # Forecast from analysis state d to d+1.
    x_f_next = (
        F @ x_a[d]
        + c
    )

    P_f_next = (
        F @ P_a[d] @ F.T
        + Q
    )

    P_f_next = 0.5 * (
        P_f_next + P_f_next.T
    )

    # RTS smoother gain:
    #
    # C_d = P_a[d] F' inv(P_f[d+1])
    #
    # Avoid explicit matrix inverse.
    #
    A = (
        P_a[d] @ F.T
    )

    try:
        C = np.linalg.solve(
            P_f_next.T,
            A.T
        ).T
    except np.linalg.LinAlgError:
        print(
            f"WARNING: singular forecast covariance "
            f"on {pd.Timestamp(dates[d]).date()}"
        )

        C = (
            A @ np.linalg.pinv(
                P_f_next
            )
        )

    # Smoothed state
    x_s[d] = (
        x_a[d]
        + C @ (
            x_s[d + 1]
            - x_f_next
        )
    )

    # Smoothed covariance
    P_s[d] = (
        P_a[d]
        + C @ (
            P_s[d + 1]
            - P_f_next
        ) @ C.T
    )

    # Numerical cleanup
    P_s[d] = 0.5 * (
        P_s[d] + P_s[d].T
    )

    # Remove tiny negative diagonal roundoff.
    diag = np.maximum(
        np.diag(P_s[d]),
        0.0
    )

    for i in range(N_MINE):
        P_s[d, i, i] = diag[i]


# ============================================================
# Diagnostics
# ============================================================

print()
print("RTS smoothing completed.")

print()
print("Maximum absolute difference between")
print("RTS and forward KF states:")

max_difference = np.max(
    np.abs(x_s - x_a)
)

print(
    f"{max_difference:.10f}"
)

print()
print(
    "Mean absolute difference:",
    f"{np.mean(np.abs(x_s-x_a)):.10f}"
)


# ============================================================
# Create RTS scaling-factor dataframe
# ============================================================

rts_df = pd.DataFrame({
    "date": daily_df["date"],
})

for i, mine in enumerate(MINES):
    rts_df[mine] = x_s[:, i]

# Preserve state information if available.
if "state_type" in daily_df.columns:
    rts_df["KF_state_type"] = (
        daily_df["state_type"].values
    )

if "n_observations" in daily_df.columns:
    rts_df["n_observations"] = (
        daily_df["n_observations"].values
    )


# ============================================================
# Covariance output
# ============================================================

rts_cov_df = pd.DataFrame({
    "date": daily_df["date"],
})

for i, mine in enumerate(MINES):

    rts_cov_df[
        f"{mine}_variance"
    ] = P_s[:, i, i]

    rts_cov_df[
        f"{mine}_sigma"
    ] = np.sqrt(
        np.maximum(
            P_s[:, i, i],
            0.0
        )
    )


# ============================================================
# Total emissions
# ============================================================

prior_vector = np.array([
    PRIOR_EMISSIONS[m]
    for m in MINES
])

prior_total = prior_vector.sum()

rts_total = (
    x_s @ prior_vector
)

rts_scaling = (
    rts_total / prior_total
)

emissions_df = pd.DataFrame({
    "date": daily_df["date"],
    "prior_total_gCH4_s": prior_total,
    "RTS_total_gCH4_s": rts_total,
    "RTS_scaling": rts_scaling,
    "RTS_minus_prior_gCH4_s":
        rts_total - prior_total,
})

# Add individual emissions.
for i, mine in enumerate(MINES):
    emissions_df[
        f"{mine}_scaling"
    ] = x_s[:, i]

    emissions_df[
        f"{mine}_gCH4_s"
    ] = (
        x_s[:, i]
        * PRIOR_EMISSIONS[mine]
    )


# ============================================================
# Save
# ============================================================

rts_df.to_csv(
    OUTPUT_RTS,
    index=False
)

rts_cov_df.to_csv(
    OUTPUT_RTS_COV,
    index=False
)

emissions_df.to_csv(
    OUTPUT_RTS_EMISSIONS,
    index=False
)


# ============================================================
# Print summary
# ============================================================

print()
print("=" * 80)
print("RTS RESULTS")
print("=" * 80)

print()
print(
    f"Prior total emission : "
    f"{prior_total:.2f} g CH4/s"
)

print(
    f"Mean RTS total       : "
    f"{rts_total.mean():.2f} g CH4/s"
)

print(
    f"Mean RTS scaling     : "
    f"{rts_scaling.mean():.6f}"
)

print(
    f"Min RTS total        : "
    f"{rts_total.min():.2f} g CH4/s"
)

print(
    f"Max RTS total        : "
    f"{rts_total.max():.2f} g CH4/s"
)

print()
print("Daily RTS scaling factors:")
print(
    rts_df[
        ["date"] + MINES
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print("Saved:")
print(OUTPUT_RTS)
print(OUTPUT_RTS_COV)
print(OUTPUT_RTS_EMISSIONS)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
