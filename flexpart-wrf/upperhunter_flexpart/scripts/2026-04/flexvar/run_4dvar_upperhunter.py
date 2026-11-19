#!/usr/bin/env python3

"""
Upper Hunter CH4 — Linear Variational / 4D-Var Inversion
=========================================================

13 open-cut mines × 27 daily scaling factors = 351 parameters.

This implementation uses the SAME Gaussian prior formulation as
the Bayesian inverse-model experiments:

    prior mean          = 1.0
    prior sigma         = 0.5
    temporal correlation length = 3 days
    observation sigma   = 0.05 ppm
    background          = 1.85 ppm

Objective:

    J(s) =
        0.5 (s-sb)^T B^-1 (s-sb)
      + 0.5 (d-Hs)^T R^-1 (d-Hs)

where:

    d = y - background

The solution is obtained through the variational normal equations,
using linear solves rather than explicit matrix inverses.

IMPORTANT:

The observation preprocessing is intentionally aligned with the
Gibbs and Metropolis-Hastings implementations:

    1. Picarro timestamps are floored to hourly resolution.
    2. Duplicate receptor/hour observations are averaged.
    3. H timestamps are floored to hourly resolution.
    4. H and observations are matched using an INNER merge on:
           receptor + time_aest
    5. Original H row indices are preserved.

This ensures that the 4D-Var calculation uses the same H/y
observation subset as the Gibbs and MH calculations.

This first version is UNCONSTRAINED.

No 0 <= scaling <= 3 truncation/bounds are applied.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE = Path(
    "/home/duch/flexpart-wrf/upperhunter_flexpart"
)

H_FILE = (
    BASE
    / "H_matrix_flexpart"
    / "H_flexpart_0_10m_13mine_Jan2026.csv"
)

PICARRO_FILE = (
    Path("/home/duch/flexpart-wrf")
    / "emission_UH"
    / "RH_upperHunter"
    / "tapm_output"
    / "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

PRIOR_FILE = (
    BASE
    / "emission_UH"
    / "RH_upperHunter"
    / "UpperHuntercoalmines.csv"
)

ANALYTICAL_DIR = (
    BASE
    / "upperhunter_flexinvert"
    / "inversion_results"
)

# Existing analytical script output.
#
# NOTE:
# These files are only directly comparable if they were generated
# using the SAME H matrix and SAME observation preprocessing.
ANALYTICAL_FILE_CANDIDATES = [
    ANALYTICAL_DIR
    / "analytical_posterior_scaling_factors.csv",

    BASE
    / "H_matrix_flexpart"
    / "bayesian_analytical"
    / "posterior_scaling_factors.csv",
]

OUT_DIR = (
    BASE
    / "H_matrix_flexpart"
    / "bayesian_4dvar"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# SETTINGS
# =============================================================================

BACKGROUND_PPM = 1.85

PRIOR_MEAN = 1.0
PRIOR_SIGMA = 0.50

CORRELATION_LENGTH_DAYS = 3.0

OBS_SIGMA_PPM = 0.05

N_MINE = 13
N_DAY = 27
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

START_DATE = pd.Timestamp("2026-01-05")
END_DATE = pd.Timestamp("2026-01-31")

RECEPTOR_COLUMNS = {
    "singleton": "Singleton cleaned",
    "muswellbrook": "Muswellbrook cleaned",
    "merriwa": "Merriwa cleaned",
}


# =============================================================================
# HEADER
# =============================================================================

print()
print("=" * 90)
print("UPPER HUNTER CH4 — VARIATIONAL / 4D-VAR INVERSION")
print("=" * 90)

print()
print("Configuration")
print("-" * 90)

print(
    f"H matrix                    : {H_FILE}"
)

print(
    f"Picarro observations       : {PICARRO_FILE}"
)

print(
    f"Prior emissions            : {PRIOR_FILE}"
)

print(
    f"Number of parameters       : {N_PARAM}"
)

print(
    f"Number of mines            : {N_MINE}"
)

print(
    f"Number of days             : {N_DAY}"
)

print(
    f"Prior mean                 : {PRIOR_MEAN:.3f}"
)

print(
    f"Prior sigma                : {PRIOR_SIGMA:.3f}"
)

print(
    f"Temporal correlation L     : "
    f"{CORRELATION_LENGTH_DAYS:.3f} days"
)

print(
    f"Observation sigma          : "
    f"{OBS_SIGMA_PPM:.4f} ppm"
)

print(
    f"Background                 : "
    f"{BACKGROUND_PPM:.3f} ppm"
)

print()


# =============================================================================
# PARAMETER NAMES
# =============================================================================

def build_parameter_names():

    dates = pd.date_range(
        START_DATE,
        END_DATE,
        freq="D",
    )

    names = []

    # MUST be day-major, matching H:
    #
    # OC1_20260105 ... OC13_20260105
    # OC1_20260106 ... OC13_20260106
    # ...
    # OC1_20260131 ... OC13_20260131

    for date in dates:

        datestr = date.strftime(
            "%Y%m%d"
        )

        for mine in MINES:

            names.append(
                f"{mine}_{datestr}"
            )

    if len(names) != N_PARAM:

        raise RuntimeError(
            f"Expected {N_PARAM} parameter names, "
            f"got {len(names)}"
        )

    return names


PARAMETER_NAMES = (
    build_parameter_names()
)


# =============================================================================
# READ H MATRIX
# =============================================================================

print("=" * 90)
print("READING FLEXPART H MATRIX")
print("=" * 90)

H_df = pd.read_csv(
    H_FILE
)

metadata_columns = [
    "receptor",
    "release_utc",
    "obs_time_aest",
    "flexpart_file",
]

missing_metadata = [
    c
    for c in metadata_columns
    if c not in H_df.columns
]

if missing_metadata:

    raise ValueError(
        f"Missing H metadata columns: "
        f"{missing_metadata}"
    )

missing_parameters = [
    p
    for p in PARAMETER_NAMES
    if p not in H_df.columns
]

if missing_parameters:

    raise ValueError(
        f"Missing {len(missing_parameters)} "
        f"parameter columns in H."
    )

print(
    f"H dataframe shape          : "
    f"{H_df.shape}"
)

print()
print("Metadata columns:")
print(metadata_columns)

print()
print(
    f"Parameter columns          : "
    f"{len(PARAMETER_NAMES)}"
)


# =============================================================================
# CONVERT H TO NUMERICAL ARRAY
# =============================================================================

H_all = (
    H_df[
        PARAMETER_NAMES
    ]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
    .to_numpy(
        dtype=np.float64
    )
)

if not np.isfinite(H_all).all():

    bad_h = (
        ~np.isfinite(H_all)
    ).sum()

    raise ValueError(
        f"H contains {bad_h} "
        f"non-finite values."
    )

print()
print(
    f"H array shape              : "
    f"{H_all.shape}"
)

print(
    f"H min/max                  : "
    f"{H_all.min():.6e}, "
    f"{H_all.max():.6e}"
)

print(
    f"H mean                     : "
    f"{H_all.mean():.6e}"
)


# =============================================================================
# NORMALISE RECEPTOR NAMES
# =============================================================================

def normalise_receptor(value):

    s = str(
        value
    ).strip().lower()

    if s in (
        "sin",
        "singleton",
    ):
        return "singleton"

    if s in (
        "mus",
        "muswellbrook",
    ):
        return "muswellbrook"

    if s in (
        "mer",
        "merriwa",
    ):
        return "merriwa"

    return s


H_df["_receptor_norm"] = (
    H_df["receptor"].map(
        normalise_receptor
    )
)


# =============================================================================
# READ PICARRO
# =============================================================================

print()
print("=" * 90)
print("READING PICARRO OBSERVATIONS")
print("=" * 90)

obs = pd.read_csv(
    PICARRO_FILE
)

print(
    f"Observation dataframe shape : "
    f"{obs.shape}"
)

print()
print("Observation columns:")
print(
    obs.columns.tolist()
)

required_obs_columns = [
    "Datetime",
    *RECEPTOR_COLUMNS.values(),
]

missing_obs = [
    c
    for c in required_obs_columns
    if c not in obs.columns
]

if missing_obs:

    raise ValueError(
        f"Missing observation columns: "
        f"{missing_obs}"
    )


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

print()
print("Observation time range:")
print(
    obs["time_aest"].min(),
    "to",
    obs["time_aest"].max(),
)

print()
print(
    "Invalid Picarro Datetime rows:",
    obs["time_aest"].isna().sum(),
)

# =============================================================================
# BUILD LONG-FORM OBSERVATION TABLE
# =============================================================================

print()
print("=" * 100)
print("BUILDING LONG-FORM OBSERVATION TABLE")
print("=" * 100)

obs_long_list = []

for receptor, column in RECEPTOR_COLUMNS.items():

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
        "WARNING: duplicate Picarro "
        "receptor/hour observations found."
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

for r in RECEPTOR_COLUMNS:
    n = np.sum(
        obs_long["receptor"] == r
    )
    print(
        f"   {r:15s} {n:6d}"
    )



# =============================================================================
# AVERAGE DUPLICATE RECEPTOR/HOUR OBSERVATIONS
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

    # EXACTLY MATCH GIBBS / MH:
    #
    # Average duplicate observations having
    # the same receptor and hourly timestamp.

    obs_long = (
        obs_long
        .groupby(
            [
                "receptor",
                "time_aest",
            ],
            as_index=False,
        )[
            "obs_ch4_ppm"
        ]
        .mean()
    )

print()
print(
    f"Usable Picarro observations: "
    f"{len(obs_long)}"
)


# =============================================================================
# PARSE H AEST DATETIME
# =============================================================================

print()
print(
    "Parsing H observation times ..."
)

H_df["obs_time_aest"] = pd.to_datetime(
    H_df["obs_time_aest"],
    errors="coerce",
)

if H_df[
    "obs_time_aest"
].isna().any():

    nbad = (
        H_df[
            "obs_time_aest"
        ]
        .isna()
        .sum()
    )

    raise ValueError(
        f"{nbad} H rows have invalid "
        f"obs_time_aest."
    )


# =============================================================================
# CREATE H HOURLY KEY
# =============================================================================

H_key = pd.DataFrame(
    {
        "receptor":
            H_df[
                "receptor"
            ],

        "time_aest":
            H_df[
                "obs_time_aest"
            ].dt.floor("h"),
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

    print()
    print(
        "WARNING: duplicate H "
        "receptor/hour keys found."
    )

    print(
        H_key.loc[
            duplicates
        ]
        .sort_values(
            [
                "receptor",
                "time_aest",
            ]
        )
        .head(20)
    )


# =============================================================================
# ALIGN OBSERVATIONS DIRECTLY WITH H
# =============================================================================

print()
print("=" * 90)
print(
    "ALIGNING PICARRO OBSERVATIONS WITH FLEXPART H"
)
print("=" * 90)


obs_key = obs_long[
    [
        "receptor",
        "time_aest",
        "obs_ch4_ppm",
    ]
].copy()


# Preserve original H row number.

H_key_indexed = H_key.copy()

H_key_indexed[
    "H_row"
] = np.arange(
    len(H_key_indexed)
)


# IMPORTANT:
#
# EXACTLY MATCH GIBBS / MH:
#
# INNER merge on:
#     receptor
#     time_aest
#
# The original H row number is retained so that
# the corresponding row can be selected from H_all.

match = H_key_indexed.merge(
    obs_key,
    on=[
        "receptor",
        "time_aest",
    ],
    how="inner",
)


print(
    f"H rows available           : "
    f"{len(H_df)}"
)

print(
    f"Picarro hourly observations : "
    f"{len(obs_long)}"
)

print(
    f"Matched FLEXPART/Picarro observations: "
    f"{len(match)}"
)


if len(match) == 0:

    print()
    print(
        "ERROR: No observations matched "
        "the H matrix."
    )

    print()
    print("Example H times:")
    print(
        H_key.head(10)
    )

    print()
    print("Example observation times:")
    print(
        obs_long.head(10)
    )

    raise RuntimeError(
        "No FLEXPART/Picarro observations "
        "could be aligned."
    )


# =============================================================================
# DIAGNOSTIC RECEPTOR COUNTS
# =============================================================================

print()
print(
    "Matched observations by receptor:"
)

counts = (
    match[
        "receptor"
    ]
    .value_counts()
    .reindex(
        [
            "singleton",
            "muswellbrook",
            "merriwa",
        ],
        fill_value=0,
    )
)

for receptor, count in counts.items():

    print(
        f"  {receptor:15s}: "
        f"{count:6d}"
    )


# =============================================================================
# EXTRACT MATCHED H ROWS AND OBSERVATION VECTOR
# =============================================================================

H_row_indices = (
    match[
        "H_row"
    ]
    .to_numpy(
        dtype=int
    )
)


H = H_all[
    H_row_indices,
    :
]


y = (
    match[
        "obs_ch4_ppm"
    ]
    .to_numpy(
        dtype=np.float64
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


n_obs = len(y)


print()
print(
    f"Final H shape               : "
    f"{H.shape}"
)

print(
    f"Observation vector shape    : "
    f"{y.shape}"
)

print()
print(
    f"Observed CH4 min/max        : "
    f"{y.min():.6f}, "
    f"{y.max():.6f} ppm"
)

print(
    f"Observed CH4 mean           : "
    f"{y.mean():.6f} ppm"
)


# =============================================================================
# VERIFY OBSERVATION VECTOR
# =============================================================================

if not np.isfinite(y).all():

    raise ValueError(
        "Observation vector contains "
        "non-finite values."
    )


# =============================================================================
# PRIOR STATE
# =============================================================================

print()
print("=" * 90)
print("CONSTRUCTING PRIOR STATE")
print("=" * 90)


s_b = np.full(
    N_PARAM,
    PRIOR_MEAN,
    dtype=np.float64,
)


print(
    f"Prior vector length         : "
    f"{len(s_b)}"
)

print(
    f"Prior mean                  : "
    f"{s_b.mean():.6f}"
)

print(
    f"Prior sigma                 : "
    f"{PRIOR_SIGMA:.6f}"
)


# =============================================================================
# CONSTRUCT B
# =============================================================================

print()
print("=" * 90)
print("CONSTRUCTING PRIOR COVARIANCE B")
print("=" * 90)


B = np.zeros(
    (
        N_PARAM,
        N_PARAM,
    ),
    dtype=np.float64,
)


sigma2 = (
    PRIOR_SIGMA ** 2
)

Lcorr = (
    CORRELATION_LENGTH_DAYS
)


# Parameter ordering is day-major:
#
# index:
#   day 0: OC1 ... OC13
#   day 1: OC1 ... OC13
#   ...
#
# Therefore the same mine occurs every 13 positions.

for mine_index in range(
    N_MINE
):

    indices = np.arange(
        mine_index,
        N_PARAM,
        N_MINE,
    )

    for i, ii in enumerate(
        indices
    ):

        for j, jj in enumerate(
            indices
        ):

            lag = abs(
                i - j
            )

            B[ii, jj] = (
                sigma2
                * np.exp(
                    -lag / Lcorr
                )
            )


print(
    f"B shape                    : "
    f"{B.shape}"
)

print(
    f"B diagonal min/max         : "
    f"{np.diag(B).min():.6f}, "
    f"{np.diag(B).max():.6f}"
)


print()
print(
    "Expected temporal correlations:"
)

for lag in range(
    0,
    7,
):

    corr = np.exp(
        -lag / Lcorr
    )

    print(
        f"  lag {lag} day(s): "
        f"{corr:.6f}"
    )


# =============================================================================
# VERIFY CROSS-MINE COVARIANCE
# =============================================================================

if not np.isclose(
    B[0, 1],
    0.0,
):

    raise RuntimeError(
        "B cross-mine covariance "
        "is not zero."
    )


# =============================================================================
# VERIFY SAME-MINE 1-DAY COVARIANCE
# =============================================================================

expected_cov_1day = (
    sigma2
    * np.exp(
        -1.0 / Lcorr
    )
)

actual_cov_1day = (
    B[0, N_MINE]
)


print()
print(
    "B validation:"
)

print(
    f"  Same mine 1-day covariance : "
    f"{actual_cov_1day:.12e}"
)

print(
    f"  Expected                   : "
    f"{expected_cov_1day:.12e}"
)


if not np.isclose(
    actual_cov_1day,
    expected_cov_1day,
):

    raise RuntimeError(
        "B temporal covariance "
        "validation failed."
    )


# =============================================================================
# OBSERVATION COVARIANCE
# =============================================================================

print()
print("=" * 90)
print("CONSTRUCTING OBSERVATION COVARIANCE R")
print("=" * 90)


obs_variance = (
    OBS_SIGMA_PPM ** 2
)


# R is diagonal.
#
# We do not need to physically construct
# the 1124 x 1124 matrix.

R_diag = np.full(
    n_obs,
    obs_variance,
    dtype=np.float64,
)

Rinv_diag = (
    1.0 / R_diag
)


print(
    f"R variance                  : "
    f"{obs_variance:.8f} ppm^2"
)

print(
    f"R precision                 : "
    f"{Rinv_diag[0]:.6f}"
)


# =============================================================================
# VARIATIONAL FORMULATION
# =============================================================================

print()
print("=" * 90)
print("VARIATIONAL 4D-VAR SOLUTION")
print("=" * 90)


# FLEXPART H represents CH4 enhancement
# above background.
#
# Observations are total atmospheric CH4:
#
#     y = background + H s
#
# Therefore:
#
#     d = y - background


d = (
    y
    - BACKGROUND_PPM
)


print()
print(
    "Observation formulation:"
)

print()
print(
    "  d = y - BACKGROUND_PPM"
)

print(
    f"  Background                 : "
    f"{BACKGROUND_PPM:.6f} ppm"
)

print(
    f"  Enhancement min/max        : "
    f"{d.min():.6f}, "
    f"{d.max():.6f} ppm"
)

print(
    f"  Enhancement mean           : "
    f"{d.mean():.6f} ppm"
)

print()
print(
    "Objective:"
)

print()
print(
    "  J(s) = 0.5 (s-sb)^T B^-1 (s-sb)"
)

print(
    "       + 0.5 (d-Hs)^T R^-1 (d-Hs)"
)

print()


# =============================================================================
# CHOLESKY FACTORISATION OF B
# =============================================================================

print(
    "Factorising B ..."
)


Lb = np.linalg.cholesky(
    B
)


def solve_B(rhs):
    """
    Solve B x = rhs using Cholesky(B).
    """

    z = np.linalg.solve(
        Lb,
        rhs,
    )

    return np.linalg.solve(
        Lb.T,
        z,
    )


print(
    "B Cholesky factorisation complete."
)


# =============================================================================
# BUILD VARIATIONAL HESSIAN
# =============================================================================

print()
print(
    "Constructing variational Hessian ..."
)


# -------------------------------------------------------------------------
# Build B^-1.
#
# For this 351-dimensional problem, explicitly forming B^-1 is practical.
# -------------------------------------------------------------------------

Binv = solve_B(
    np.eye(
        N_PARAM,
        dtype=np.float64,
    )
)


# -------------------------------------------------------------------------
# H^T R^-1 H
#
# Since R is diagonal:
# -------------------------------------------------------------------------

H_weighted = (
    H
    * Rinv_diag[:, None]
)

HT_Rinv_H = (
    H.T
    @ H_weighted
)


# -------------------------------------------------------------------------
# Hessian:
#
#     A = B^-1 + H^T R^-1 H
# -------------------------------------------------------------------------

A = (
    Binv
    + HT_Rinv_H
)


# -------------------------------------------------------------------------
# Right-hand side:
#
#     b = B^-1 sb + H^T R^-1 d
# -------------------------------------------------------------------------

Binv_sb = (
    solve_B(
        s_b
    )
)

HT_Rinv_d = (
    H.T
    @ (
        Rinv_diag
        * d
    )
)

b = (
    Binv_sb
    + HT_Rinv_d
)


print(
    f"Hessian shape               : "
    f"{A.shape}"
)

print(
    f"Hessian symmetry error     : "
    f"{np.max(np.abs(A - A.T)):.6e}"
)


# Force exact numerical symmetry.

A = (
    A
    + A.T
) / 2.0


# =============================================================================
# SOLVE VARIATIONAL SYSTEM
# =============================================================================

print()
print(
    "Solving:"
)

print()
print(
    "  (B^-1 + H^T R^-1 H) s ="
)

print(
    "  B^-1 sb + H^T R^-1 d"
)

print()


s_post = np.linalg.solve(
    A,
    b,
)


# =============================================================================
# GRADIENT CHECK
# =============================================================================

print("=" * 90)
print("GRADIENT CHECK")
print("=" * 90)


innovation = (
    H
    @ s_post
    - d
)


prior_increment = (
    s_post
    - s_b
)


gradient = (
    solve_B(
        prior_increment
    )
    + H.T
    @ (
        Rinv_diag
        * innovation
    )
)


gradient_norm = np.linalg.norm(
    gradient
)

gradient_max = np.max(
    np.abs(gradient)
)


print(
    f"Gradient L2 norm           : "
    f"{gradient_norm:.6e}"
)

print(
    f"Gradient max abs           : "
    f"{gradient_max:.6e}"
)


# =============================================================================
# COST FUNCTION
# =============================================================================

def calculate_cost(s):
    """
    Calculate total variational cost and its two components.

    The observation term is evaluated in enhancement space:

        d = y - BACKGROUND_PPM
    """

    dx = (
        s
        - s_b
    )

    jb = (
        0.5
        * np.dot(
            dx,
            solve_B(dx),
        )
    )

    residual = (
        H
        @ s
        - d
    )

    jo = (
        0.5
        * np.sum(
            Rinv_diag
            * residual**2
        )
    )

    return (
        jb,
        jo,
        jb + jo,
    )


Jb_post, Jo_post, J_total_post = (
    calculate_cost(
        s_post
    )
)

Jb_prior0, Jo_obs0, J_total0 = (
    calculate_cost(
        s_b
    )
)


print()
print("=" * 90)
print("COST FUNCTION")
print("=" * 90)


print(
    "At prior:"
)

print(
    f"  J_prior                  : "
    f"{Jb_prior0:.12f}"
)

print(
    f"  J_obs                    : "
    f"{Jo_obs0:.12f}"
)

print(
    f"  J_total                  : "
    f"{J_total0:.12f}"
)


print()
print(
    "At 4D-Var solution:"
)

print(
    f"  J_prior                  : "
    f"{Jb_post:.12f}"
)

print(
    f"  J_obs                    : "
    f"{Jo_post:.12f}"
)

print(
    f"  J_total                  : "
    f"{J_total_post:.12f}"
)


# =============================================================================
# POSTERIOR COVARIANCE
# =============================================================================

print()
print("=" * 90)
print("POSTERIOR COVARIANCE")
print("=" * 90)


print(
    "Calculating posterior covariance "
    "from the variational Hessian ..."
)


# For this 351-dimensional problem:
#
#     P_post = A^-1
#
# This is practical.

A_chol = np.linalg.cholesky(
    A
)

I = np.eye(
    N_PARAM,
    dtype=np.float64,
)

Z = np.linalg.solve(
    A_chol,
    I,
)

P_post = np.linalg.solve(
    A_chol.T,
    Z,
)


P_post = (
    P_post
    + P_post.T
) / 2.0


posterior_sd = np.sqrt(
    np.maximum(
        np.diag(P_post),
        0.0,
    )
)


eigvals = np.linalg.eigvalsh(
    P_post
)


print(
    f"Posterior covariance shape  : "
    f"{P_post.shape}"
)

print(
    f"Posterior SD min/max        : "
    f"{posterior_sd.min():.8f}, "
    f"{posterior_sd.max():.8f}"
)

print(
    f"Posterior covariance min eig: "
    f"{eigvals.min():.6e}"
)

print(
    f"Posterior covariance max eig: "
    f"{eigvals.max():.6e}"
)


# =============================================================================
# POSTERIOR SUMMARY
# =============================================================================

print()
print("=" * 90)
print("POSTERIOR SUMMARY")
print("=" * 90)


print(
    f"Mean scaling               : "
    f"{np.mean(s_post):.8f}"
)

print(
    f"Median scaling             : "
    f"{np.median(s_post):.8f}"
)

print(
    f"Minimum scaling            : "
    f"{np.min(s_post):.8f}"
)

print(
    f"Maximum scaling            : "
    f"{np.max(s_post):.8f}"
)

print(
    f"Negative factors           : "
    f"{np.sum(s_post < 0)} / {N_PARAM}"
)


# =============================================================================
# COMPARE WITH OBSERVATIONS
# =============================================================================

y_prior = (
    H
    @ s_b
    + BACKGROUND_PPM
)

y_post = (
    H
    @ s_post
    + BACKGROUND_PPM
)


prior_residual = (
    y_prior
    - y
)

post_residual = (
    y_post
    - y
)


print()
print("=" * 90)
print("OBSERVATION FIT")
print("=" * 90)


print(
    f"Prior RMSE                 : "
    f"{np.sqrt(np.mean(prior_residual**2)):.8f} ppm"
)

print(
    f"Posterior RMSE             : "
    f"{np.sqrt(np.mean(post_residual**2)):.8f} ppm"
)

print(
    f"Prior mean residual        : "
    f"{np.mean(prior_residual):.8f} ppm"
)

print(
    f"Posterior mean residual    : "
    f"{np.mean(post_residual):.8f} ppm"
)


# =============================================================================
# COMPARE WITH EXISTING ANALYTICAL RESULT
# =============================================================================

print()
print("=" * 90)
print("COMPARISON WITH EXISTING ANALYTICAL SOLUTION")
print("=" * 90)


analytical_file = None

for candidate in (
    ANALYTICAL_FILE_CANDIDATES
):

    if candidate.exists():

        analytical_file = candidate

        break


if analytical_file is None:

    print()
    print(
        "No existing analytical posterior "
        "file found."
    )

    print()
    print("Checked:")

    for candidate in (
        ANALYTICAL_FILE_CANDIDATES
    ):

        print(
            f"  {candidate}"
        )

    comparison_df = None


else:

    print(
        f"Analytical result file     : "
        f"{analytical_file}"
    )

    print()
    print(
        "WARNING:"
    )

    print(
        "The analytical result is only a valid "
        "numerical comparison if it was generated "
        "using the same H CSV, observation vector, "
        "hourly timestamp flooring, duplicate "
        "averaging, prior B, and R."
    )

    analytical_df = pd.read_csv(
        analytical_file
    )

    print(
        f"Analytical dataframe shape : "
        f"{analytical_df.shape}"
    )


    # -------------------------------------------------------------------------
    # Find posterior column.
    # -------------------------------------------------------------------------

    posterior_column = None

    for candidate_column in [
        "posterior_mean",
        "posterior_scaling",
        "posterior_mean_scaling",
    ]:

        if candidate_column in (
            analytical_df.columns
        ):

            posterior_column = (
                candidate_column
            )

            break


    if posterior_column is None:

        print()
        print(
            "WARNING: Could not identify the "
            "analytical posterior mean column."
        )

        comparison_df = None


    elif "parameter" not in analytical_df.columns:

        print()
        print(
            "WARNING: Analytical file does not "
            "contain a 'parameter' column."
        )

        comparison_df = None


    else:

        analytical_names = (
            analytical_df[
                "parameter"
            ]
            .astype(str)
            .tolist()
        )

        analytical_values = (
            analytical_df[
                posterior_column
            ]
            .to_numpy(
                dtype=float
            )
        )

        analytical_map = dict(
            zip(
                analytical_names,
                analytical_values,
            )
        )


        missing = [
            p
            for p in PARAMETER_NAMES
            if p not in analytical_map
        ]


        if missing:

            print()
            print(
                f"WARNING: {len(missing)} parameters "
                "missing from analytical result."
            )

            comparison_df = None


        else:

            analytical_post = np.asarray(
                [
                    analytical_map[p]
                    for p in PARAMETER_NAMES
                ],
                dtype=np.float64,
            )


            difference = (
                s_post
                - analytical_post
            )

            abs_difference = np.abs(
                difference
            )


            comparison_df = pd.DataFrame(
                {
                    "parameter":
                        PARAMETER_NAMES,

                    "four_dvar_posterior":
                        s_post,

                    "analytical_posterior":
                        analytical_post,

                    "difference":
                        difference,

                    "absolute_difference":
                        abs_difference,

                    "four_dvar_posterior_sd":
                        posterior_sd,
                }
            )


            print()
            print(
                f"Maximum absolute difference : "
                f"{abs_difference.max():.12e}"
            )

            print(
                f"Mean absolute difference    : "
                f"{abs_difference.mean():.12e}"
            )

            print(
                f"RMS difference              : "
                f"{np.sqrt(np.mean(difference**2)):.12e}"
            )


            if (
                np.std(s_post) > 0
                and
                np.std(analytical_post) > 0
            ):

                correlation = (
                    np.corrcoef(
                        s_post,
                        analytical_post,
                    )[0, 1]
                )

            else:

                correlation = np.nan


            print(
                f"Correlation coefficient     : "
                f"{correlation:.12f}"
            )


            # Print largest discrepancies.

            largest = (
                comparison_df
                .sort_values(
                    "absolute_difference",
                    ascending=False,
                )
                .head(10)
            )


            print()
            print(
                "Largest differences:"
            )

            print(
                largest.to_string(
                    index=False,
                    float_format=(
                        lambda x:
                        f"{x:.10e}"
                    ),
                )
            )


# =============================================================================
# SAVE RESULTS
# =============================================================================

print()
print("=" * 90)
print("SAVING 4D-VAR RESULTS")
print("=" * 90)


# =============================================================================
# MAIN PARAMETER TABLE
# =============================================================================

result_df = pd.DataFrame(
    {
        "parameter":
            PARAMETER_NAMES,

        "prior_scaling":
            s_b,

        "posterior_scaling":
            s_post,

        "posterior_sigma":
            posterior_sd,
    }
)


result_df[
    "posterior_lower_95"
] = (
    result_df[
        "posterior_scaling"
    ]
    - 1.96
    * result_df[
        "posterior_sigma"
    ]
)


result_df[
    "posterior_upper_95"
] = (
    result_df[
        "posterior_scaling"
    ]
    + 1.96
    * result_df[
        "posterior_sigma"
    ]
)


result_file = (
    OUT_DIR
    / "4dvar_posterior_scaling_factors.csv"
)


result_df.to_csv(
    result_file,
    index=False,
)


# =============================================================================
# SAVE NUMERICAL ARRAYS
# =============================================================================

np.save(
    OUT_DIR
    / "posterior_mean_scaling.npy",
    s_post,
)

np.save(
    OUT_DIR
    / "posterior_sd_scaling.npy",
    posterior_sd,
)

np.save(
    OUT_DIR
    / "B_prior.npy",
    B,
)

np.save(
    OUT_DIR
    / "B_posterior.npy",
    P_post,
)

np.save(
    OUT_DIR
    / "prior_scaling.npy",
    s_b,
)


# =============================================================================
# SAVE OBSERVATION-SPACE PREDICTIONS
# =============================================================================

# Use the NEW matched dataframe.
#
# Internally the observation is called obs_ch4_ppm,
# but retain the existing output column name obs_ppm
# for compatibility with previous files.

prediction_df = match[
    [
        "receptor",
        "time_aest",
        "obs_ch4_ppm",
    ]
].copy()


prediction_df = prediction_df.rename(
    columns={
        "time_aest":
            "obs_time_aest",

        "obs_ch4_ppm":
            "obs_ppm",
    }
)


prediction_df[
    "prior_model_ppm"
] = y_prior


prediction_df[
    "posterior_model_ppm"
] = y_post


prediction_df[
    "prior_residual_ppm"
] = prior_residual


prediction_df[
    "posterior_residual_ppm"
] = post_residual


prediction_file = (
    OUT_DIR
    / "4dvar_observation_predictions.csv"
)


prediction_df.to_csv(
    prediction_file,
    index=False,
)


# =============================================================================
# SAVE ANALYTICAL COMPARISON
# =============================================================================

if comparison_df is not None:

    comparison_file = (
        OUT_DIR
        / "4dvar_vs_analytical.csv"
    )

    comparison_df.to_csv(
        comparison_file,
        index=False,
    )


# =============================================================================
# SAVE DIAGNOSTICS
# =============================================================================

diagnostics = pd.DataFrame(
    [
        {
            "n_observations":
                n_obs,

            "n_parameters":
                N_PARAM,

            "n_h_rows":
                len(H_df),

            "n_picarro_hourly":
                len(obs_long),

            "prior_mean":
                PRIOR_MEAN,

            "prior_sigma":
                PRIOR_SIGMA,

            "prior_variance":
                PRIOR_SIGMA ** 2,

            "temporal_correlation_length_days":
                CORRELATION_LENGTH_DAYS,

            "observation_sigma_ppm":
                OBS_SIGMA_PPM,

            "observation_variance_ppm2":
                OBS_SIGMA_PPM ** 2,

            "background_ppm":
                BACKGROUND_PPM,

            "cost_prior":
                J_total0,

            "cost_posterior":
                J_total_post,

            "prior_cost_component":
                Jb_prior0,

            "prior_observation_cost":
                Jo_obs0,

            "posterior_prior_cost":
                Jb_post,

            "posterior_observation_cost":
                Jo_post,

            "gradient_l2_norm":
                gradient_norm,

            "gradient_max_abs":
                gradient_max,

            "posterior_mean_scaling":
                np.mean(s_post),

            "posterior_median_scaling":
                np.median(s_post),

            "posterior_min_scaling":
                np.min(s_post),

            "posterior_max_scaling":
                np.max(s_post),

            "negative_scaling_count":
                np.sum(s_post < 0),

            "prior_rmse_ppm":
                np.sqrt(
                    np.mean(
                        prior_residual ** 2
                    )
                ),

            "posterior_rmse_ppm":
                np.sqrt(
                    np.mean(
                        post_residual ** 2
                    )
                ),
        }
    ]
)


diagnostic_file = (
    OUT_DIR
    / "4dvar_diagnostics.csv"
)


diagnostics.to_csv(
    diagnostic_file,
    index=False,
)


# =============================================================================
# FINAL
# =============================================================================

print()
print(
    "Saved:"
)

print(
    f"  {result_file}"
)

print(
    f"  {prediction_file}"
)

print(
    f"  {diagnostic_file}"
)

print(
    f"  {OUT_DIR / 'posterior_mean_scaling.npy'}"
)

print(
    f"  {OUT_DIR / 'posterior_sd_scaling.npy'}"
)

print(
    f"  {OUT_DIR / 'B_prior.npy'}"
)

print(
    f"  {OUT_DIR / 'B_posterior.npy'}"
)

if comparison_df is not None:

    print(
        f"  {OUT_DIR / '4dvar_vs_analytical.csv'}"
    )


print()
print("=" * 90)
print("4D-VAR RUN COMPLETE")
print("=" * 90)
print()

