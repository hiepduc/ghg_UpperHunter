#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd

from scipy.linalg import solve_triangular


# ============================================================
# CONFIG
# ============================================================

BASE = "/home/duch/flexpart-wrf/upperhunter_flexpart"

H_FILE = (
    BASE +
    "/H_matrix_flexpart/"
    "H_flexpart_0_10m_13mine_Jan2026.csv"
)

PICARRO_FILE = (
    "/home/duch/flexpart-wrf/emission_UH/"
    "RH_upperHunter/tapm_output/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

BATCH_DIR = (
    BASE +
    "/H_matrix_flexpart/"
    "Bounded_batch_MAP_full_footprint_13mine_Jan2026"
)

OUT_DIR = (
    BASE +
    "/H_matrix_flexpart/"
    "Equality_constrained_smoother_validation"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

START_DATE = pd.Timestamp("2026-01-05")

N_DAYS = 27
N_MINES = 13
N_STATE = N_DAYS * N_MINES

PRIOR_MEAN = 1.0
PRIOR_SIGMA = 0.5

OBS_SIGMA = 0.05
BACKGROUND_PPM = 1.85

RHO = np.exp(-1.0 / 3.0)


# ============================================================
# PARAMETER NAMES
# ============================================================

parameter_columns = []

for d in range(N_DAYS):

    date = START_DATE + pd.Timedelta(days=d)

    for m in range(N_MINES):

        parameter_columns.append(
            f"OC{m+1}_{date.strftime('%Y%m%d')}"
        )


# ============================================================
# LOAD H
# ============================================================

print("=" * 90)
print("EQUALITY-CONSTRAINED SMOOTHER VALIDATION")
print("=" * 90)

H_df = pd.read_csv(H_FILE)

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


# ============================================================
# LOAD OBS
# ============================================================

obs = pd.read_csv(PICARRO_FILE)

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

for receptor, col in receptor_columns.items():

    valid = obs[
        ["Datetime", col]
    ].dropna()

    for _, row in valid.iterrows():

        t = pd.Timestamp(
            row["Datetime"]
        ).floor("h")

        obs_lookup[
            (receptor, t)
        ] = float(row[col])


# ============================================================
# MATCH
# ============================================================

keep = []
y = []

for i, row in metadata.iterrows():

    receptor = str(
        row["receptor"]
    ).lower()

    t = pd.Timestamp(
        row["obs_time_aest"]
    )

    if t < START_DATE:
        continue

    if t >= START_DATE + pd.Timedelta(days=N_DAYS):
        continue

    key = (
        receptor,
        t.floor("h")
    )

    if key not in obs_lookup:
        continue

    keep.append(i)
    y.append(obs_lookup[key])


keep = np.asarray(
    keep,
    dtype=int
)

y = np.asarray(
    y,
    dtype=float
)

H = H_all[keep]

print("H:", H.shape)
print("observations:", len(y))


# ============================================================
# PRIOR COVARIANCE
# ============================================================

B = np.zeros(
    (N_STATE, N_STATE)
)

for m in range(N_MINES):

    for d1 in range(N_DAYS):

        i = d1 * N_MINES + m

        for d2 in range(N_DAYS):

            j = d2 * N_MINES + m

            B[i, j] = (
                PRIOR_SIGMA**2
                * RHO**abs(d1-d2)
            )


xb = np.full(
    N_STATE,
    PRIOR_MEAN
)


# ============================================================
# WHITENED SYSTEM
# ============================================================

L = np.linalg.cholesky(B)

A_prior = solve_triangular(
    L,
    np.eye(N_STATE),
    lower=True
)

A_obs = H / OBS_SIGMA

b_obs = (
    y
    - BACKGROUND_PPM
    - H @ xb
) / OBS_SIGMA

A = np.vstack(
    [
        A_prior,
        A_obs
    ]
)

b = np.concatenate(
    [
        np.zeros(N_STATE),
        b_obs
    ]
)

print("Whitened system:", A.shape)


# ============================================================
# KNOWN ACTIVE SET FROM EXACT BOUNDED MAP
# ============================================================

constraints = {
    141: 0.0,   # Jan15 OC12
    142: 0.0,   # Jan15 OC13

    190: 5.0,   # Jan19 OC9
    203: 5.0,   # Jan20 OC9

    210: 0.0,   # Jan21 OC3
    216: 5.0,   # Jan21 OC9
    219: 0.0,   # Jan21 OC12

    223: 0.0,   # Jan22 OC3
}


active = np.array(
    sorted(constraints.keys()),
    dtype=int
)

free = np.array(
    [
        i
        for i in range(N_STATE)
        if i not in constraints
    ],
    dtype=int
)


c = np.array(
    [
        constraints[i]
        for i in active
    ]
)


# ============================================================
# SOLVE EQUALITY-CONSTRAINED LEAST SQUARES
# ============================================================

print("\nActive parameters:", len(active))
print("Free parameters:", len(free))

A_free = A[:, free]

A_active = A[:, active]

# IMPORTANT:
# A operates on z = x - xb, not directly on x.
#
# Therefore:
#
#     z_active = x_active - xb_active
#
# Lower bound x=0  -> z=-1
# Upper bound x=5  -> z=+4

z_active = c - PRIOR_MEAN

rhs = (
    b
    - A_active @ z_active
)

z_free = np.linalg.lstsq(
    A_free,
    rhs,
    rcond=None
)[0]


z = np.zeros(
    N_STATE
)

z[active] = z_active

z[free] = z_free

x = xb + z


# ============================================================
# GRADIENT
# ============================================================

gradient = (
    2.0
    * A.T
    @ (
        A @ z - b
    )
)


# ============================================================
# KKT CHECK
# ============================================================

lower_active = [
    i for i in active
    if constraints[i] == 0.0
]

upper_active = [
    i for i in active
    if constraints[i] == 5.0
]

free_gradient = np.max(
    np.abs(
        gradient[free]
    )
)

lower_violation = np.max(
    np.maximum(
        -gradient[
            lower_active
        ],
        0.0
    )
) if lower_active else 0.0

upper_violation = np.max(
    np.maximum(
        gradient[
            upper_active
        ],
        0.0
    )
) if upper_active else 0.0


print("\n")
print("=" * 90)
print("KKT CHECK")
print("=" * 90)

print(
    "Max free |gradient|:",
    free_gradient
)

print(
    "Max lower KKT violation:",
    lower_violation
)

print(
    "Max upper KKT violation:",
    upper_violation
)

print(
    "Solution min:",
    np.min(x)
)

print(
    "Solution max:",
    np.max(x)
)

print(
    "Solution mean:",
    np.mean(x)
)


# ============================================================
# COMPARE WITH BOUNDED BATCH MAP
# ============================================================

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

diff = x - batch_x


print("\n")
print("=" * 90)
print("COMPARISON WITH BOUNDED BATCH MAP")
print("=" * 90)

print(
    "Max absolute difference:",
    np.max(np.abs(diff))
)

print(
    "RMSE:",
    np.sqrt(
        np.mean(diff**2)
    )
)

print(
    "MAE:",
    np.mean(
        np.abs(diff)
    )
)

print(
    "Correlation:",
    np.corrcoef(
        x,
        batch_x
    )[0, 1]
)


# ============================================================
# ACTIVE PARAMETER REPORT
# ============================================================

rows = []

for i in active:

    day = i // N_MINES
    mine = i % N_MINES

    date = (
        START_DATE
        + pd.Timedelta(days=day)
    )

    rows.append(
        {
            "date":
                date.strftime("%Y-%m-%d"),

            "mine":
                f"OC{mine+1}",

            "index":
                i,

            "equality_value":
                constraints[i],

            "solution":
                x[i],

            "batch_MAP":
                batch_x[i],

            "difference":
                diff[i],

            "gradient":
                gradient[i],
        }
    )


active_df = pd.DataFrame(rows)

print("\n")
print(active_df.to_string(index=False))


# ============================================================
# SAVE
# ============================================================

comparison = batch.copy()

comparison[
    "equality_constrained_smoother"
] = x

comparison[
    "difference"
] = diff

comparison.to_csv(
    os.path.join(
        OUT_DIR,
        "equality_smoother_vs_bounded_MAP.csv"
    ),
    index=False
)

active_df.to_csv(
    os.path.join(
        OUT_DIR,
        "active_set_validation.csv"
    ),
    index=False
)

np.savez(
    os.path.join(
        OUT_DIR,
        "equality_constrained_solution.npz"
    ),
    solution=x,
    gradient=gradient,
    active=active,
)


print("\nDONE.")

