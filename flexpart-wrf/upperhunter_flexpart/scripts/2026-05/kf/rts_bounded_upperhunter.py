#!/usr/bin/env python3

"""
FLEXPART-WRF Bayesian Constrained RTS Smoother
Upper Hunter CH4 — 13 mines × 27 daily scaling factors

This is a genuine recursive constrained RTS smoother.

Forward:
    - 26-D augmented bounded Kalman filter
    - X_d = [x_{d-1}, x_d]
    - hourly Picarro observations
    - hard bounds 0 <= x <= 5

Backward:
    - 13-D daily constrained RTS recursion
    - each smoothed daily state solves a box-constrained
      conditional quadratic MAP problem
    - 0 <= x_d <= 5

Reference:
    bounded batch MAP from:
    Bounded_batch_MAP_full_footprint_13mine_Jan2026

IMPORTANT:
    This is NOT another batch optimizer.
    The batch MAP is used only as the independent reference
    solution after the recursive RTS calculation.
"""

import os
import warnings

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.linalg import solve, inv


# ============================================================
# CONFIG
# ============================================================

BASE = "/home/duch/flexpart-wrf/upperhunter_flexpart"

H_FILE = (
    BASE
    + "/H_matrix_flexpart/"
    + "H_flexpart_0_10m_13mine_Jan2026.csv"
)

PICARRO_FILE = (
    "/home/duch/flexpart-wrf/emission_UH/"
    "RH_upperHunter/tapm_output/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

BATCH_DIR = (
    BASE
    + "/H_matrix_flexpart/"
    + "Bounded_batch_MAP_full_footprint_13mine_Jan2026"
)

OUT_DIR = (
    BASE
    + "/H_matrix_flexpart/"
    + "RTS_bounded_full_footprint_13mine_Jan2026"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

START_DATE = pd.Timestamp("2026-01-05")

N_DAYS = 27
N_MINES = 13

PRIOR_MEAN = 1.0
PRIOR_SIGMA = 0.5

OBS_SIGMA = 0.05
BACKGROUND_PPM = 1.85

RHO = np.exp(-1.0 / 3.0)

Q_DAILY = (
    PRIOR_SIGMA**2
    * (1.0 - RHO**2)
)

LOWER = 0.0
UPPER = 5.0

N_STATE = N_MINES
N_AUG = 2 * N_MINES


# ============================================================
# PARAMETER NAMES
# ============================================================

parameter_columns = []

for d in range(N_DAYS):

    date = (
        START_DATE
        + pd.Timedelta(days=d)
    )

    for m in range(N_MINES):

        parameter_columns.append(
            f"OC{m+1}_{date.strftime('%Y%m%d')}"
        )


# ============================================================
# HELPER: STABLE INVERSE
# ============================================================

def stable_inverse(A):

    """
    Symmetric positive-definite inverse.

    Small diagonal regularisation is used only if required
    numerically.
    """

    A = 0.5 * (A + A.T)

    try:
        return np.linalg.inv(A)

    except np.linalg.LinAlgError:

        scale = max(
            1.0,
            np.max(np.abs(A))
        )

        eps = 1e-10 * scale

        return np.linalg.inv(
            A
            + eps * np.eye(A.shape[0])
        )


# ============================================================
# LOAD H
# ============================================================

print("=" * 90)
print("FLEXPART-WRF BAYESIAN CONSTRAINED RTS SMOOTHER")
print("=" * 90)

print("\n")
print("=" * 90)
print("LOADING H MATRIX")
print("=" * 90)

H_df = pd.read_csv(H_FILE)

print(
    "H dataframe:",
    H_df.shape
)

H_all = H_df[
    parameter_columns
].to_numpy(float)

metadata = H_df[
    [
        "receptor",
        "release_utc",
        "obs_time_aest",
        "flexpart_file",
    ]
].copy()

metadata["obs_time_aest"] = pd.to_datetime(
    metadata["obs_time_aest"],
    errors="coerce"
)

metadata["receptor"] = (
    metadata["receptor"]
    .astype(str)
    .str.lower()
)


# ============================================================
# LOAD PICARRO
# ============================================================

print("\n")
print("=" * 90)
print("LOADING PICARRO OBSERVATIONS")
print("=" * 90)

obs = pd.read_csv(
    PICARRO_FILE
)

obs["Datetime"] = pd.to_datetime(
    obs["Datetime"],
    format="%d/%m/%Y %H:%M",
    errors="coerce"
)

receptor_columns = {
    "singleton": "Singleton cleaned",
    "muswellbrook": "Muswellbrook cleaned",
    "merriwa": "Merriwa cleaned",
}

obs_lookup = {}

for receptor, column in receptor_columns.items():

    valid = obs[
        ["Datetime", column]
    ].dropna()

    for _, row in valid.iterrows():

        timestamp = pd.Timestamp(
            row["Datetime"]
        ).floor("h")

        obs_lookup[
            (
                receptor,
                timestamp
            )
        ] = float(
            row[column]
        )


# ============================================================
# MATCH H TO OBSERVATIONS
# ============================================================

print("\n")
print("=" * 90)
print("MATCHING H TO PICARRO")
print("=" * 90)

keep = []
y = []
times = []
receptors = []

for i, row in metadata.iterrows():

    receptor = row["receptor"]

    timestamp = pd.Timestamp(
        row["obs_time_aest"]
    )

    if timestamp < START_DATE:
        continue

    if timestamp >= (
        START_DATE
        + pd.Timedelta(days=N_DAYS)
    ):
        continue

    key = (
        receptor,
        timestamp.floor("h")
    )

    if key not in obs_lookup:
        continue

    keep.append(i)

    y.append(
        obs_lookup[key]
    )

    times.append(
        timestamp.floor("h")
    )

    receptors.append(
        receptor
    )


keep = np.asarray(
    keep,
    dtype=int
)

y = np.asarray(
    y,
    dtype=float
)

times = pd.DatetimeIndex(
    times
)

receptors = np.asarray(
    receptors
)

H = H_all[keep]

metadata_matched = metadata.iloc[
    keep
].copy()

metadata_matched[
    "time_aest"
] = times

metadata_matched[
    "receptor"
] = receptors

print(
    "Matched observations:",
    len(y)
)

print(
    "H matched:",
    H.shape
)

print(
    "Matched receptors:",
    dict(
        pd.Series(receptors)
        .value_counts()
    )
)


# ============================================================
# BUILD DAY INDEX
# ============================================================

day_index = np.full(
    len(times),
    -1,
    dtype=int
)

for i, timestamp in enumerate(times):

    day = (
        timestamp.normalize()
        - START_DATE
    ).days

    if 0 <= day < N_DAYS:

        day_index[i] = day


valid = (
    day_index >= 0
)

H = H[valid]
y = y[valid]
times = times[valid]
receptors = receptors[valid]
day_index = day_index[valid]

metadata_matched = (
    metadata_matched.iloc[
        np.where(valid)[0]
    ].copy()
)


# ============================================================
# BUILD AUGMENTED OBSERVATION VECTORS
# ============================================================

"""
For observation on day d:

    h_aug = [H_{d-1}, H_d]

because the 26-D state is

    X_d = [x_{d-1}, x_d]

The first day has no previous-day parameter in the
Jan 5–31 state vector, so its previous contribution is zero.
"""

H_aug = np.zeros(
    (
        len(y),
        N_AUG
    )
)

for r in range(len(y)):

    d = day_index[r]

    current_start = (
        d * N_MINES
    )

    current_end = (
        current_start
        + N_MINES
    )

    H_aug[
        r,
        N_MINES:
    ] = H[r, current_start:current_end]

    if d > 0:

        previous_start = (
            (d - 1)
            * N_MINES
        )

        previous_end = (
            previous_start
            + N_MINES
        )

        H_aug[
            r,
            :N_MINES
        ] = H[
            r,
            previous_start:previous_end
        ]


# ============================================================
# AUGMENTED TRANSITION
# ============================================================

"""
X_d = [x_{d-1}, x_d]

X_{d+1} =
    [ x_d,
      rho*x_d + (1-rho)*xb + w ]

Therefore:

F =
    [ 0       I
      0      rho I ]

c =
    [ 0
      (1-rho)*xb ]

Q_aug =
    [ 0  0
      0  Q ]
"""

F = np.zeros(
    (
        N_AUG,
        N_AUG
    )
)

F[
    :N_MINES,
    N_MINES:
] = np.eye(N_MINES)

F[
    N_MINES:,
    N_MINES:
] = RHO * np.eye(
    N_MINES
)

c_aug = np.zeros(
    N_AUG
)

c_aug[
    N_MINES:
] = (
    1.0 - RHO
) * PRIOR_MEAN

Q_aug = np.zeros(
    (
        N_AUG,
        N_AUG
    )
)

Q_aug[
    N_MINES:,
    N_MINES:
] = (
    Q_DAILY
    * np.eye(N_MINES)
)


# ============================================================
# INITIAL AUGMENTED STATE
# ============================================================

xb = np.full(
    N_MINES,
    PRIOR_MEAN
)

x_aug = np.concatenate(
    [
        xb,
        xb
    ]
)

"""
Initial covariance:

Var(x_prev) = sigma^2
Var(x_current) = sigma^2
Cov(prev,current) = sigma^2*rho

This corresponds to the stationary AR(1) prior.
"""

P_aug = np.zeros(
    (
        N_AUG,
        N_AUG
    )
)

P0 = PRIOR_SIGMA**2

P_aug[
    :N_MINES,
    :N_MINES
] = P0 * np.eye(
    N_MINES
)

P_aug[
    N_MINES:,
    N_MINES:
] = P0 * np.eye(
    N_MINES
)

P_cross = (
    P0
    * RHO
    * np.eye(N_MINES)
)

P_aug[
    :N_MINES,
    N_MINES:
] = P_cross

P_aug[
    N_MINES:,
    :N_MINES
] = P_cross.T


# ============================================================
# STORAGE
# ============================================================

filtered_states = np.zeros(
    (
        N_DAYS,
        N_MINES
    )
)

filtered_covariances = np.zeros(
    (
        N_DAYS,
        N_MINES,
        N_MINES
    )
)

aug_filtered_states = np.zeros(
    (
        N_DAYS,
        N_AUG
    )
)

aug_filtered_covariances = np.zeros(
    (
        N_DAYS,
        N_AUG,
        N_AUG
    )
)

predicted_states = np.zeros(
    (
        N_DAYS,
        N_MINES
    )
)

predicted_covariances = np.zeros(
    (
        N_DAYS,
        N_MINES,
        N_MINES
    )
)

active_lower_count = 0
active_upper_count = 0


# ============================================================
# CONSTRAINED AUGMENTED UPDATE
# ============================================================

def constrained_update(
    x_prior,
    P_prior,
    h,
    observation
):
    """
    Solve the bounded Gaussian posterior:

        min_x
        0.5 (x-x_prior)' P^-1 (x-x_prior)
        +
        0.5 (y-background-hx)^2 / sigma^2

        subject to 0 <= x <= 5.

    L-BFGS-B is used only for the small 26-D
    constrained KF update.

    The covariance is then approximated using the
    Hessian restricted to the free variables.
    """

    P_inv = stable_inverse(
        P_prior
    )

    def objective(x):

        dx = x - x_prior

        innovation = (
            observation
            - BACKGROUND_PPM
            - h @ x
        )

        return (
            0.5
            * dx @ P_inv @ dx
            +
            0.5
            * (
                innovation
                / OBS_SIGMA
            )**2
        )

    def gradient(x):

        dx = x - x_prior

        innovation = (
            observation
            - BACKGROUND_PPM
            - h @ x
        )

        return (
            P_inv @ dx
            -
            h
            * (
                innovation
                / OBS_SIGMA**2
            )
        )

    result = minimize(
        objective,
        np.clip(
            x_prior,
            LOWER,
            UPPER
        ),
        jac=gradient,
        method="L-BFGS-B",
        bounds=[
            (
                LOWER,
                UPPER
            )
        ] * N_AUG,
        options={
            "ftol": 1e-12,
            "gtol": 1e-10,
            "maxiter": 500,
            "maxls": 50,
        }
    )

    if not result.success:

        warnings.warn(
            "Bounded KF update did not "
            "report success: "
            + result.message
        )

    x_post = np.clip(
        result.x,
        LOWER,
        UPPER
    )

    lower = (
        x_post <= LOWER + 1e-8
    )

    upper = (
        x_post >= UPPER - 1e-8
    )

    active = lower | upper

    HESS = (
        P_inv
        +
        np.outer(h, h)
        / OBS_SIGMA**2
    )

    P_post = np.zeros_like(
        HESS
    )

    free = np.where(
        ~active
    )[0]

    if len(free) > 0:

        H_free = HESS[
            np.ix_(
                free,
                free
            )
        ]

        P_free = stable_inverse(
            H_free
        )

        P_post[
            np.ix_(
                free,
                free
            )
        ] = P_free

    return (
        x_post,
        P_post,
        lower,
        upper
    )


# ============================================================
# FORWARD BOUNDED KALMAN FILTER
# ============================================================

print("\n")
print("=" * 90)
print("FORWARD BOUNDED KALMAN FILTER")
print("=" * 90)

for d in range(N_DAYS):

    day_start = (
        START_DATE
        + pd.Timedelta(days=d)
    )

    day_end = (
        day_start
        + pd.Timedelta(days=1)
    )

    # --------------------------------------------------------
    # DAY TRANSITION
    # --------------------------------------------------------

    if d > 0:

        x_aug = (
            F @ x_aug
            + c_aug
        )

        P_aug = (
            F @ P_aug @ F.T
            + Q_aug
        )

        P_aug = 0.5 * (
            P_aug
            + P_aug.T
        )

    # --------------------------------------------------------
    # SAVE DAILY PREDICTED CURRENT STATE
    # --------------------------------------------------------

    predicted_states[d] = (
        x_aug[N_MINES:].copy()
    )

    predicted_covariances[d] = (
        P_aug[
            N_MINES:,
            N_MINES:
        ].copy()
    )

    # --------------------------------------------------------
    # HOURLY OBSERVATIONS FOR THIS DAY
    # --------------------------------------------------------

    rows = np.where(
        day_index == d
    )[0]

    day_lower = 0
    day_upper = 0

    for r in rows:

        (
            x_aug,
            P_aug,
            lower,
            upper
        ) = constrained_update(
            x_aug,
            P_aug,
            H_aug[r],
            y[r]
        )

        day_lower += np.sum(
            lower
        )

        day_upper += np.sum(
            upper
        )

    active_lower_count += (
        day_lower
    )

    active_upper_count += (
        day_upper
    )

    # --------------------------------------------------------
    # STORE CURRENT-DAY MARGINAL
    # --------------------------------------------------------

    filtered_states[d] = (
        x_aug[N_MINES:].copy()
    )

    filtered_covariances[d] = (
        P_aug[
            N_MINES:,
            N_MINES:
        ].copy()
    )

    aug_filtered_states[d] = (
        x_aug.copy()
    )

    aug_filtered_covariances[d] = (
        P_aug.copy()
    )

    print(
        f"{day_start.strftime('%Y-%m-%d')} "
        f"obs={len(rows):4d} "
        f"lower={day_lower:2d} "
        f"upper={day_upper:2d}"
    )


# ============================================================
# FORWARD FILTER SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("FORWARD FILTER SUMMARY")
print("=" * 90)

print(
    "Total lower-bound activations:",
    active_lower_count
)

print(
    "Total upper-bound activations:",
    active_upper_count
)

print(
    "Filtered minimum:",
    np.min(filtered_states)
)

print(
    "Filtered maximum:",
    np.max(filtered_states)
)


# ============================================================
# CONSTRAINED RTS BACKWARD STEP
# ============================================================

def constrained_rts_step(
    x_filt,
    P_filt,
    x_next_smooth,
):
    """
    One constrained RTS backward step.

    Given:

        x_t | y_1:t  ~ N(x_filt, P_filt)

    and the transition

        x_{t+1}
          = rho*x_t
            + (1-rho)*xb
            + w

    solve:

        argmin_{0 <= x_t <= 5}

        0.5 (x_t-x_filt)' P_filt^-1
                         (x_t-x_filt)

        +
        0.5 (x_next_smooth
             - rho*x_t
             - c)^T
             Q^-1
             ( ... )

    This is the constrained analogue of the
    conditional MAP step underlying RTS.

    Because the bounds make the posterior non-Gaussian,
    the returned covariance is an active-set/Laplace
    approximation rather than an exact truncated-Gaussian
    covariance.
    """

    P_inv = stable_inverse(
        P_filt
    )

    Q_inv = (
        1.0 / Q_DAILY
    ) * np.eye(
        N_MINES
    )

    transition_offset = (
        (1.0 - RHO)
        * PRIOR_MEAN
    )

    def objective(x):

        dx = (
            x
            - x_filt
        )

        transition_error = (
            x_next_smooth
            - RHO * x
            - transition_offset
        )

        return (
            0.5
            * dx @ P_inv @ dx
            +
            0.5
            * transition_error
            @ Q_inv
            @ transition_error
        )

    def gradient(x):

        dx = (
            x
            - x_filt
        )

        transition_error = (
            x_next_smooth
            - RHO * x
            - transition_offset
        )

        return (
            P_inv @ dx
            -
            RHO
            * (
                Q_inv
                @ transition_error
            )
        )

    result = minimize(
        objective,
        np.clip(
            x_filt,
            LOWER,
            UPPER
        ),
        jac=gradient,
        method="L-BFGS-B",
        bounds=[
            (
                LOWER,
                UPPER
            )
        ] * N_MINES,
        options={
            "ftol": 1e-13,
            "gtol": 1e-11,
            "maxiter": 1000,
            "maxls": 100,
        }
    )

    if not result.success:

        warnings.warn(
            "Constrained RTS step did not "
            "report success: "
            + result.message
        )

    x_smooth = np.clip(
        result.x,
        LOWER,
        UPPER
    )

    lower = (
        x_smooth
        <= LOWER + 1e-8
    )

    upper = (
        x_smooth
        >= UPPER - 1e-8
    )

    active = (
        lower
        | upper
    )

    # --------------------------------------------------------
    # Active-set/Laplace covariance
    # --------------------------------------------------------

    Hessian = (
        P_inv
        +
        (RHO**2)
        * Q_inv
    )

    P_smooth = np.zeros(
        (
            N_MINES,
            N_MINES
        )
    )

    free = np.where(
        ~active
    )[0]

    if len(free) > 0:

        H_free = Hessian[
            np.ix_(
                free,
                free
            )
        ]

        P_free = stable_inverse(
            H_free
        )

        P_smooth[
            np.ix_(
                free,
                free
            )
        ] = P_free

    return (
        x_smooth,
        P_smooth,
        lower,
        upper,
        result
    )


# ============================================================
# BACKWARD CONSTRAINED RTS
# ============================================================

print("\n")
print("=" * 90)
print("BACKWARD CONSTRAINED RTS SMOOTHER")
print("=" * 90)

smooth_states = np.zeros(
    (
        N_DAYS,
        N_MINES
    )
)

smooth_covariances = np.zeros(
    (
        N_DAYS,
        N_MINES,
        N_MINES
    )
)

smooth_lower = np.zeros(
    (
        N_DAYS,
        N_MINES
    ),
    dtype=bool
)

smooth_upper = np.zeros(
    (
        N_DAYS,
        N_MINES
    ),
    dtype=bool
)

# ------------------------------------------------------------
# FINAL DAY
# ------------------------------------------------------------

smooth_states[-1] = (
    filtered_states[-1]
)

smooth_covariances[-1] = (
    filtered_covariances[-1]
)

smooth_lower[-1] = (
    smooth_states[-1]
    <= LOWER + 1e-8
)

smooth_upper[-1] = (
    smooth_states[-1]
    >= UPPER - 1e-8
)

# ------------------------------------------------------------
# BACKWARD RECURSION
# ------------------------------------------------------------

for d in range(
    N_DAYS - 2,
    -1,
    -1
):

    (
        x_s,
        P_s,
        lower,
        upper,
        result
    ) = constrained_rts_step(
        filtered_states[d],
        filtered_covariances[d],
        smooth_states[d + 1]
    )

    smooth_states[d] = x_s

    smooth_covariances[d] = P_s

    smooth_lower[d] = lower
    smooth_upper[d] = upper

    print(
        f"{(START_DATE + pd.Timedelta(days=d)).strftime('%Y-%m-%d')} "
        f"lower={np.sum(lower):2d} "
        f"upper={np.sum(upper):2d} "
        f"min={np.min(x_s):.6f} "
        f"max={np.max(x_s):.6f}"
    )


# ============================================================
# RTS SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("CONSTRAINED RTS SUMMARY")
print("=" * 90)

print(
    "Smoothed minimum:",
    np.min(smooth_states)
)

print(
    "Smoothed maximum:",
    np.max(smooth_states)
)

print(
    "Smoothed mean:",
    np.mean(smooth_states)
)

print(
    "Final-day smoothed mean:",
    np.mean(smooth_states[-1])
)

print(
    "Total smoothed lower activations:",
    np.sum(smooth_lower)
)

print(
    "Total smoothed upper activations:",
    np.sum(smooth_upper)
)


# ============================================================
# COMPARE FILTERED VS SMOOTHED
# ============================================================

filter_smooth_diff = (
    smooth_states
    - filtered_states
)

print("\n")
print("=" * 90)
print("FILTERED VS CONSTRAINED RTS")
print("=" * 90)

print(
    "Max absolute difference:",
    np.max(
        np.abs(
            filter_smooth_diff
        )
    )
)

print(
    "RMSE:",
    np.sqrt(
        np.mean(
            filter_smooth_diff**2
        )
    )
)

print(
    "MAE:",
    np.mean(
        np.abs(
            filter_smooth_diff
        )
    )
)


# ============================================================
# LOAD BOUNDED BATCH MAP
# ============================================================

print("\n")
print("=" * 90)
print("LOADING BOUNDED BATCH MAP REFERENCE")
print("=" * 90)

batch_file = os.path.join(
    BATCH_DIR,
    "bounded_batch_MAP_351_parameters.csv"
)

batch = pd.read_csv(
    batch_file
)

batch_x = batch[
    "bounded_batch_MAP"
].to_numpy(float)

batch_matrix = (
    batch_x
    .reshape(
        N_DAYS,
        N_MINES
    )
)


# ============================================================
# RTS VS BATCH MAP
# ============================================================

rts_flat = (
    smooth_states
    .reshape(-1)
)

diff_batch = (
    rts_flat
    - batch_x
)

print("\n")
print("=" * 90)
print("CONSTRAINED RTS VS BOUNDED BATCH MAP")
print("=" * 90)

print(
    "Number of parameters:",
    len(rts_flat)
)

print(
    "Max absolute difference:",
    np.max(
        np.abs(
            diff_batch
        )
    )
)

print(
    "RMSE:",
    np.sqrt(
        np.mean(
            diff_batch**2
        )
    )
)

print(
    "MAE:",
    np.mean(
        np.abs(
            diff_batch
        )
    )
)

print(
    "Correlation:",
    np.corrcoef(
        rts_flat,
        batch_x
    )[0, 1]
)

print(
    "|difference| > 0.1:",
    np.sum(
        np.abs(diff_batch) > 0.1
    ),
    "/",
    len(diff_batch)
)


# ============================================================
# DAILY COMPARISON
# ============================================================

daily_rows = []

for d in range(N_DAYS):

    date = (
        START_DATE
        + pd.Timedelta(days=d)
    )

    diff_d = (
        smooth_states[d]
        - batch_matrix[d]
    )

    daily_rows.append(
        {
            "date":
                date.strftime("%Y-%m-%d"),

            "RTS_mean":
                np.mean(
                    smooth_states[d]
                ),

            "MAP_mean":
                np.mean(
                    batch_matrix[d]
                ),

            "difference_mean":
                np.mean(diff_d),

            "RMSE":
                np.sqrt(
                    np.mean(
                        diff_d**2
                    )
                ),

            "MAE":
                np.mean(
                    np.abs(diff_d)
                ),

            "max_abs_difference":
                np.max(
                    np.abs(diff_d)
                ),
        }
    )

daily_comparison = pd.DataFrame(
    daily_rows
)

print("\n")
print(
    daily_comparison.to_string(
        index=False
    )
)


# ============================================================
# PARAMETER COMPARISON
# ============================================================

parameter_rows = []

for d in range(N_DAYS):

    date = (
        START_DATE
        + pd.Timedelta(days=d)
    )

    for m in range(N_MINES):

        idx = (
            d * N_MINES
            + m
        )

        parameter_rows.append(
            {
                "date":
                    date.strftime("%Y-%m-%d"),

                "mine":
                    f"OC{m+1}",

                "index":
                    idx,

                "filtered_KF":
                    filtered_states[
                        d, m
                    ],

                "bounded_RTS":
                    smooth_states[
                        d, m
                    ],

                "bounded_batch_MAP":
                    batch_matrix[
                        d, m
                    ],

                "RTS_minus_KF":
                    (
                        smooth_states[d, m]
                        - filtered_states[d, m]
                    ),

                "RTS_minus_MAP":
                    (
                        smooth_states[d, m]
                        - batch_matrix[d, m]
                    ),

                "RTS_lower_active":
                    smooth_lower[d, m],

                "RTS_upper_active":
                    smooth_upper[d, m],
            }
        )

parameter_comparison = pd.DataFrame(
    parameter_rows
)


# ============================================================
# ACTIVE RTS PARAMETERS
# ============================================================

active_rts = (
    parameter_comparison[
        parameter_comparison[
            "RTS_lower_active"
        ]
        |
        parameter_comparison[
            "RTS_upper_active"
        ]
    ]
    .copy()
)

print("\n")
print("=" * 90)
print("ACTIVE PARAMETERS IN CONSTRAINED RTS")
print("=" * 90)

if len(active_rts) == 0:

    print(
        "No smoothed parameters are exactly "
        "at a bound."
    )

else:

    print(
        active_rts.to_string(
            index=False
        )
    )


# ============================================================
# SAVE DAILY CSV
# ============================================================

daily_output = []

for d in range(N_DAYS):

    date = (
        START_DATE
        + pd.Timedelta(days=d)
    )

    row = {
        "date":
            date.strftime("%Y-%m-%d")
    }

    for m in range(N_MINES):

        row[
            f"OC{m+1}"
        ] = smooth_states[
            d, m
        ]

    daily_output.append(
        row
    )

daily_output = pd.DataFrame(
    daily_output
)

daily_output.to_csv(
    os.path.join(
        OUT_DIR,
        "daily_bounded_RTS_scaling_factors.csv"
    ),
    index=False
)


# ============================================================
# SAVE PARAMETER COMPARISON
# ============================================================

parameter_comparison.to_csv(
    os.path.join(
        OUT_DIR,
        "bounded_RTS_vs_bounded_batch_MAP_351.csv"
    ),
    index=False
)


# ============================================================
# SAVE DAILY COMPARISON
# ============================================================

daily_comparison.to_csv(
    os.path.join(
        OUT_DIR,
        "bounded_RTS_vs_bounded_batch_MAP_by_day.csv"
    ),
    index=False
)


# ============================================================
# SAVE ACTIVE SET
# ============================================================

active_rts.to_csv(
    os.path.join(
        OUT_DIR,
        "bounded_RTS_active_parameters.csv"
    ),
    index=False
)


# ============================================================
# SAVE NPZ
# ============================================================

np.savez(
    os.path.join(
        OUT_DIR,
        "bounded_RTS_solution.npz"
    ),

    states_filtered=filtered_states,

    covariances_filtered=filtered_covariances,

    states_smoothed=smooth_states,

    covariances_smoothed=smooth_covariances,

    augmented_states=aug_filtered_states,

    augmented_covariances=aug_filtered_covariances,

    predicted_states=predicted_states,

    predicted_covariances=predicted_covariances,

    lower_active=smooth_lower,

    upper_active=smooth_upper,

    dates=np.array(
        [
            (
                START_DATE
                + pd.Timedelta(days=d)
            ).strftime("%Y-%m-%d")
            for d in range(N_DAYS)
        ]
    ),

    mines=np.array(
        [
            f"OC{m+1}"
            for m in range(N_MINES)
        ]
    ),
)


# ============================================================
# SUMMARY TEXT
# ============================================================

summary_file = os.path.join(
    OUT_DIR,
    "bounded_RTS_summary.txt"
)

with open(
    summary_file,
    "w"
) as f:

    f.write(
        "FLEXPART-WRF Bayesian "
        "Constrained RTS Smoother\n"
    )

    f.write(
        "Upper Hunter CH4\n\n"
    )

    f.write(
        f"Start date: {START_DATE.date()}\n"
    )

    f.write(
        f"Number of days: {N_DAYS}\n"
    )

    f.write(
        f"Number of mines: {N_MINES}\n"
    )

    f.write(
        f"Parameters: {N_DAYS * N_MINES}\n"
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
        f"Background: {BACKGROUND_PPM}\n"
    )

    f.write(
        f"rho: {RHO}\n"
    )

    f.write(
        f"Q daily: {Q_DAILY}\n"
    )

    f.write(
        f"Bounds: [{LOWER}, {UPPER}]\n\n"
    )

    f.write(
        f"Matched observations: {len(y)}\n"
    )

    f.write(
        f"RTS minimum: {np.min(smooth_states)}\n"
    )

    f.write(
        f"RTS maximum: {np.max(smooth_states)}\n"
    )

    f.write(
        f"RTS mean: {np.mean(smooth_states)}\n\n"
    )

    f.write(
        "RTS vs bounded batch MAP\n"
    )

    f.write(
        f"Max abs difference: "
        f"{np.max(np.abs(diff_batch))}\n"
    )

    f.write(
        f"RMSE: "
        f"{np.sqrt(np.mean(diff_batch**2))}\n"
    )

    f.write(
        f"MAE: "
        f"{np.mean(np.abs(diff_batch))}\n"
    )

    f.write(
        f"Correlation: "
        f"{np.corrcoef(rts_flat, batch_x)[0,1]}\n"
    )


# ============================================================
# FINAL DAY
# ============================================================

print("\n")
print("=" * 90)
print("FINAL-DAY RESULTS")
print("=" * 90)

final_rows = []

for m in range(N_MINES):

    final_rows.append(
        {
            "mine":
                f"OC{m+1}",

            "filtered_KF":
                filtered_states[
                    -1, m
                ],

            "bounded_RTS":
                smooth_states[
                    -1, m
                ],

            "bounded_batch_MAP":
                batch_matrix[
                    -1, m
                ],

            "RTS_minus_MAP":
                (
                    smooth_states[-1, m]
                    - batch_matrix[-1, m]
                ),
        }
    )

final_df = pd.DataFrame(
    final_rows
)

print(
    final_df.to_string(
        index=False
    )
)


# ============================================================
# DONE
# ============================================================

print("\n")
print("=" * 90)
print("DONE")
print("=" * 90)

print(
    "Output directory:",
    OUT_DIR
)

