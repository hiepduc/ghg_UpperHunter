#!/usr/bin/env python3

"""
===============================================================================
UPPER HUNTER CH4 ANALYTICAL BAYESIAN INVERSION
===============================================================================

FLEXPART-WRF
------------

13 open-cut coal mines × 30 days
3 Picarro monitoring stations
403 daily mine scaling factors

Dates:
    2026-07-01 to 2027-07-30

State vector:
    s = 403 dimensionless daily scaling factors

Parameter ordering:
    OC1_20260701
    OC2_20260701
    ...
    OC13_20260701
    OC1_20260702
    ...
    OC13_20260730

IMPORTANT:
----------

The supplied FLEXPART H matrix ALREADY contains the prior OC mine emissions.

Therefore the forward model is:

    y = b + H s + epsilon

where:

    y       = observed CH4 [ppm]
    b       = background + agriculture + power [ppm]
    H       = FLEXPART prior-emission-weighted response [ppm]
    s       = dimensionless daily scaling factors
    epsilon = observation/model error

The prior state is:

    s_prior = 1

for all 403 parameters.

Therefore the prior mine contribution is simply:

    H @ s_prior

DO NOT multiply H by the prior mine emissions again.

Current background:
    b = 1.85 ppm

Agriculture and power are NOT yet included in b in this version.
They can be added after matching the WRF-Chem agriculture/power
time series to the same observation rows.

Analytical Gaussian posterior:

    S_post =
        (B^-1 + H^T R^-1 H)^-1

    s_post =
        S_post [
            B^-1 s_prior
            +
            H^T R^-1 (y-b)
        ]

Equivalent Kalman/Bayesian update:

    s_post =
        s_prior
        +
        B H^T
        (H B H^T + R)^-1
        [
            (y-b) - H s_prior
        ]

The two formulations are compared as a numerical validation.

This is a constrained solution with cap negative posterior values

===============================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# =============================================================================
# USER SETTINGS
# =============================================================================

# -----------------------------------------------------------------------------
# H MATRIX
# -----------------------------------------------------------------------------

H_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/"
    "upperhunter_flexpart/H_matrix_flexpart/2026-07/"
    "H_flexpart_0_10m_13mine_Jul2026.csv"
)


# -----------------------------------------------------------------------------
# PICARRO OBSERVATIONS
# -----------------------------------------------------------------------------

PICARRO_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/"
    "piccaro_obs/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)


# -----------------------------------------------------------------------------
# BACKGROUND
# -----------------------------------------------------------------------------

BACKGROUND_PPM = 1.85


# -----------------------------------------------------------------------------
# PRIOR SCALING
# -----------------------------------------------------------------------------

PRIOR_SCALING = 1.0


# -----------------------------------------------------------------------------
# PRIOR RELATIVE 1-SIGMA UNCERTAINTY
# -----------------------------------------------------------------------------

# Example:
#
#     0.50 = 50% uncertainty
#
# Therefore for prior scaling = 1:
#
#     sigma_prior = 0.5
#
# This can be changed later for sensitivity experiments.

PRIOR_RELATIVE_ERROR = 0.5


# -----------------------------------------------------------------------------
# OBSERVATION 1-SIGMA UNCERTAINTY
# -----------------------------------------------------------------------------

# ppm

OBSERVATION_ERROR_PPM = 0.05


# -----------------------------------------------------------------------------
# OUTPUT DIRECTORY
# -----------------------------------------------------------------------------

OUTPUT_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/"
    "upperhunter_flexpart/inversion_results/2026-07"
)


POSTERIOR_FILE = os.path.join(
    OUTPUT_DIR,
    "analytical_posterior_scaling_factors_constrained.csv"
)


OBS_PRED_FILE = os.path.join(
    OUTPUT_DIR,
    "analytical_forward_model_comparison_constrained.csv"
)


SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "analytical_inversion_summary_constrained.txt"
)


COVARIANCE_FILE = os.path.join(
    OUTPUT_DIR,
    "analytical_posterior_covariance_constrained.npy"
)


CORRELATION_FILE = os.path.join(
    OUTPUT_DIR,
    "analytical_posterior_correlation_constrained.npy"
)


# =============================================================================
# MINE / DATE DEFINITIONS
# =============================================================================

MINE_NAMES = [
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


DATES = pd.date_range(
    "2026-07-01",
    "2026-07-31",
    freq="D"
)


PARAMETER_NAMES = [
    f"{mine}_{date.strftime('%Y%m%d')}"
    for date in DATES
    for mine in MINE_NAMES
]


N_PARAMETERS = len(PARAMETER_NAMES)


# =============================================================================
# RECEPTOR NORMALISATION
# =============================================================================

def normalise_receptor(value):

    value = str(value).strip().lower()

    if value.startswith("sin"):
        return "singleton"

    if value.startswith("mus"):
        return "muswellbrook"

    if value.startswith("mer"):
        return "merriwa"

    return value


# =============================================================================
# FIND FULL DATETIME COLUMN
# =============================================================================

def find_time_column(df):

    """
    The Picarro file contains:

        Date
        Time
        Datetime

    We MUST use Datetime.

    The separate Time column contains time-of-day only and therefore
    cannot be used to match the FLEXPART H observation timestamps.
    """

    if "Datetime" in df.columns:

        return "Datetime"

    if "DateTime" in df.columns:

        return "DateTime"

    raise RuntimeError(
        "Could not find the full Datetime column in the Picarro CSV."
    )


# =============================================================================
# LOAD H MATRIX
# =============================================================================

def load_H():

    print("\n" + "=" * 80)
    print("LOADING FLEXPART H MATRIX")
    print("=" * 80)

    print("\nH file:")
    print(H_FILE)

    if not os.path.exists(H_FILE):

        raise FileNotFoundError(
            f"H matrix not found:\n{H_FILE}"
        )

    H_df = pd.read_csv(
        H_FILE
    )

    print(
        f"\nH dataframe shape: {H_df.shape}"
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

        raise RuntimeError(
            "Missing H metadata columns:\n"
            + "\n".join(missing_metadata)
        )


    # -------------------------------------------------------------------------
    # Identify mine/day columns
    # -------------------------------------------------------------------------

    mine_columns = [
        c
        for c in H_df.columns
        if c not in metadata_columns
    ]

    print(
        f"Number of H parameters: {len(mine_columns)}"
    )

    print(
        f"Expected parameters: {N_PARAMETERS}"
    )


    if len(mine_columns) != N_PARAMETERS:

        raise RuntimeError(
            f"H contains {len(mine_columns)} parameters, "
            f"but {N_PARAMETERS} were expected."
        )


    # -------------------------------------------------------------------------
    # Verify exact ordering
    # -------------------------------------------------------------------------

    if mine_columns != PARAMETER_NAMES:

        print("\nERROR: H parameter ordering mismatch.")

        print("\nFirst 20 H columns:")
        for x in mine_columns[:20]:
            print(" ", x)

        print("\nFirst 20 expected columns:")
        for x in PARAMETER_NAMES[:20]:
            print(" ", x)

        print("\nLast 20 H columns:")
        for x in mine_columns[-20:]:
            print(" ", x)

        print("\nLast 20 expected columns:")
        for x in PARAMETER_NAMES[-20:]:
            print(" ", x)

        raise RuntimeError(
            "H parameter ordering does not match the expected "
            "13 mines × 28 days ordering."
        )


    # -------------------------------------------------------------------------
    # Convert H to numpy
    # -------------------------------------------------------------------------

    H = H_df[
        mine_columns
    ].to_numpy(
        dtype=np.float64
    )


    print(
        f"\nH shape: {H.shape}"
    )

    print(
        f"H minimum: {np.nanmin(H):.10g}"
    )

    print(
        f"H maximum: {np.nanmax(H):.10g}"
    )

    print(
        f"H mean:    {np.nanmean(H):.10g}"
    )

    print(
        f"H non-zero fraction: "
        f"{np.count_nonzero(H) / H.size:.8f}"
    )


    # -------------------------------------------------------------------------
    # Check numerical validity
    # -------------------------------------------------------------------------

    if not np.all(
        np.isfinite(H)
    ):

        raise RuntimeError(
            "H contains NaN or infinite values."
        )


    return (
        H_df,
        H,
        mine_columns
    )


# =============================================================================
# LOAD PICARRO OBSERVATIONS
# =============================================================================

def load_observations():

    print("\n" + "=" * 80)
    print("LOADING PICARRO OBSERVATIONS")
    print("=" * 80)

    print("\nPicarro file:")
    print(PICARRO_FILE)

    if not os.path.exists(PICARRO_FILE):

        raise FileNotFoundError(
            f"Picarro file not found:\n{PICARRO_FILE}"
        )


    obs = pd.read_csv(
        PICARRO_FILE
    )


    print(
        f"\nPicarro dataframe shape: {obs.shape}"
    )


    print("\nColumns:")
    print(
        list(obs.columns)
    )


    # =========================================================================
    # USE FULL DATETIME COLUMN
    # =========================================================================

    time_col = find_time_column(
        obs
    )

    print(
        f"\nUsing full datetime column: {time_col}"
    )


    obs["datetime_aest"] = pd.to_datetime(
        obs[time_col],
        format="%d/%m/%Y %H:%M",
        errors="coerce"
    )


    n_bad_datetime = (
        obs["datetime_aest"]
        .isna()
        .sum()
    )


    print(
        f"Invalid datetime rows: "
        f"{n_bad_datetime}"
    )


    if n_bad_datetime == len(obs):

        raise RuntimeError(
            "All Picarro Datetime values failed to parse."
        )


    # =========================================================================
    # EXPLICITLY USE THE CLEANED COLUMNS
    # =========================================================================

    station_columns = {

        "singleton":
            "Singleton cleaned",

        "muswellbrook":
            "Muswellbrook cleaned",

        "merriwa":
            "Merriwa cleaned",
    }


    missing_columns = [
        column
        for column in station_columns.values()
        if column not in obs.columns
    ]


    if missing_columns:

        raise RuntimeError(
            "Required cleaned Picarro columns are missing:\n"
            + "\n".join(missing_columns)
        )


    print(
        "\nUsing cleaned Picarro columns:"
    )

    for receptor, column in station_columns.items():

        print(
            f"  {receptor:15s} -> {column}"
        )


    # =========================================================================
    # CONSTRUCT LONG-FORM DATASET
    # =========================================================================

    records = []


    for receptor, column in station_columns.items():

        tmp = obs[
            [
                "datetime_aest",
                column
            ]
        ].copy()


        tmp["receptor"] = receptor


        # ---------------------------------------------------------------------
        # Convert cleaned observations to numeric
        # ---------------------------------------------------------------------

        tmp["obs_ppm"] = pd.to_numeric(
            tmp[column],
            errors="coerce"
        )


        tmp = tmp[
            [
                "receptor",
                "datetime_aest",
                "obs_ppm"
            ]
        ]


        records.append(
            tmp
        )


    obs_long = pd.concat(
        records,
        ignore_index=True
    )


    # =========================================================================
    # REMOVE INVALID DATETIME
    # =========================================================================

    obs_long = obs_long[
        obs_long["datetime_aest"].notna()
    ].copy()


    # =========================================================================
    # REMOVE MISSING CLEANED OBSERVATIONS
    # =========================================================================

    obs_long = obs_long[
        np.isfinite(
            obs_long["obs_ppm"]
        )
    ].copy()


    # =========================================================================
    # FLOOR TO HOURLY TIMESTAMP
    # =========================================================================

    obs_long["datetime_aest"] = (
        obs_long["datetime_aest"]
        .dt.floor("h")
    )


    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    print(
        "\nValid cleaned observations by receptor:"
    )

    counts = (
        obs_long
        .groupby("receptor")
        .size()
    )

    for receptor in [
        "singleton",
        "muswellbrook",
        "merriwa"
    ]:

        print(
            f"  {receptor:15s}: "
            f"{counts.get(receptor, 0)}"
        )


    print(
        "\nCleaned observation time ranges:"
    )


    for receptor in [
        "singleton",
        "muswellbrook",
        "merriwa"
    ]:

        tmp = obs_long[
            obs_long["receptor"] == receptor
        ]


        if len(tmp) > 0:

            print(
                f"  {receptor:15s}: "
                f"{tmp['datetime_aest'].min()} "
                f"to "
                f"{tmp['datetime_aest'].max()}"
            )


    print(
        "\nFirst cleaned observations:"
    )

    print(
        obs_long
        .head(10)
        .to_string(
            index=False
        )
    )


    return obs_long


# =============================================================================
# MATCH H TO PICARRO
# =============================================================================

def match_observations(
    H_df,
    obs_long
):

    print("\n" + "=" * 80)
    print("MATCHING FLEXPART H ROWS TO PICARRO OBSERVATIONS")
    print("=" * 80)


    H_work = H_df.copy()


    # =========================================================================
    # H TIMESTAMP
    # =========================================================================

    H_work["datetime_aest"] = pd.to_datetime(
        H_work["obs_time_aest"],
        errors="coerce"
    )


    if H_work["datetime_aest"].isna().any():

        n_bad = (
            H_work["datetime_aest"]
            .isna()
            .sum()
        )

        raise RuntimeError(
            f"{n_bad} H rows have invalid obs_time_aest values."
        )


    # =========================================================================
    # NORMALISE H RECEPTOR
    # =========================================================================

    H_work["receptor_normalised"] = (
        H_work["receptor"]
        .apply(
            normalise_receptor
        )
    )


    # =========================================================================
    # FLOOR H TIME TO HOUR
    # =========================================================================

    H_work["datetime_aest"] = (
        H_work["datetime_aest"]
        .dt.floor("h")
    )


    # =========================================================================
    # PREPARE OBSERVATIONS
    # =========================================================================

    obs_work = obs_long.copy()


    obs_work["receptor_normalised"] = (
        obs_work["receptor"]
        .apply(
            normalise_receptor
        )
    )


    obs_work["datetime_aest"] = (
        pd.to_datetime(
            obs_work["datetime_aest"],
            errors="coerce"
        )
        .dt.floor("h")
    )


    # =========================================================================
    # CHECK FOR DUPLICATE OBSERVATION KEYS
    # =========================================================================

    duplicate_mask = (
        obs_work
        .duplicated(
            subset=[
                "receptor_normalised",
                "datetime_aest"
            ],
            keep=False
        )
    )


    n_duplicates = (
        duplicate_mask.sum()
    )


    print(
        f"\nDuplicate receptor/time observation rows: "
        f"{n_duplicates}"
    )


    if n_duplicates > 0:

        print(
            "\nWARNING:"
        )

        print(
            "Multiple cleaned Picarro values exist for the "
            "same receptor/time."
        )

        print(
            "Averaging duplicates before matching."
        )


        obs_work = (
            obs_work
            .groupby(
                [
                    "receptor_normalised",
                    "datetime_aest"
                ],
                as_index=False
            )["obs_ppm"]
            .mean()
        )


    # =========================================================================
    # EXACT RECEPTOR/TIME MATCH
    # =========================================================================

    merged = H_work.merge(
        obs_work[
            [
                "receptor_normalised",
                "datetime_aest",
                "obs_ppm",
            ]
        ],
        on=[
            "receptor_normalised",
            "datetime_aest",
        ],
        how="left",
        indicator=True,
    )


    n_total = len(
        merged
    )


    n_matched = (
        merged["obs_ppm"]
        .notna()
        .sum()
    )


    n_missing = (
        n_total
        -
        n_matched
    )


    print(
        f"\nTotal H rows:           {n_total}"
    )

    print(
        f"Matched observations:   {n_matched}"
    )

    print(
        f"Missing observations:   {n_missing}"
    )


    # =========================================================================
    # MATCHES BY RECEPTOR
    # =========================================================================

    print(
        "\nMatches by receptor:"
    )


    match_table = (
        merged
        .assign(
            matched=
            merged["obs_ppm"].notna()
        )
        .groupby(
            "receptor_normalised"
        )["matched"]
        .agg(
            [
                "count",
                "sum"
            ]
        )
    )


    match_table["missing"] = (
        match_table["count"]
        -
        match_table["sum"]
    )


    print(
        match_table.to_string()
    )


    # =========================================================================
    # PRINT UNMATCHED EXAMPLES
    # =========================================================================

    unmatched = merged[
        merged["obs_ppm"].isna()
    ]


    if len(unmatched) > 0:

        print(
            "\nFirst unmatched H rows:"
        )

        print(
            unmatched[
                [
                    "receptor",
                    "datetime_aest",
                    "release_utc",
                    "flexpart_file"
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )


    # =========================================================================
    # KEEP ONLY VALID OBSERVATIONS
    # =========================================================================

    matched = merged[
        merged["obs_ppm"].notna()
    ].copy()


    if len(matched) == 0:

        raise RuntimeError(
            "No H rows matched the cleaned Picarro observations."
        )


    return matched


# =============================================================================
# BUILD BACKGROUND
# =============================================================================

def build_background(
    matched
):

    """
    Current model:

        b = 1.85 ppm

    Later this can become:

        b =
            1.85
            + agriculture enhancement
            + power enhancement

    using the corresponding WRF-Chem output at each receptor/time.
    """

    print("\n" + "=" * 80)
    print("BUILDING BACKGROUND VECTOR")
    print("=" * 80)


    b = np.full(
        len(matched),
        BACKGROUND_PPM,
        dtype=np.float64
    )


    print(
        f"\nConstant background: "
        f"{BACKGROUND_PPM:.6f} ppm"
    )


    print(
        "\nAgriculture contribution: NOT YET INCLUDED"
    )

    print(
        "Power contribution:       NOT YET INCLUDED"
    )


    return b

# =============================================================================
# CONSTRAINED GAUSSIAN MAP INVERSION
# =============================================================================

def analytical_inversion(
    H,
    y,
    b,
    prior_scaling,
    prior_relative_error,
    observation_error,
):

    print("\n" + "=" * 80)
    print("CONSTRAINED GAUSSIAN MAP INVERSION")
    print("=" * 80)

    n_obs, n_param = H.shape

    print(f"\nObservations : {n_obs}")
    print(f"Parameters   : {n_param}")

    # =========================================================================
    # PRIOR STATE AND COVARIANCE
    # =========================================================================

    s_prior = np.full(
        n_param,
        prior_scaling,
        dtype=np.float64
    )

    sigma_prior = (
        prior_relative_error * np.abs(s_prior)
    )

    B_diag = sigma_prior ** 2

    # =========================================================================
    # OBSERVATION ERROR COVARIANCE
    # =========================================================================

    sigma_obs = np.full(
        n_obs,
        observation_error,
        dtype=np.float64
    )

    R_diag = sigma_obs ** 2

    # =========================================================================
    # REMOVE BACKGROUND
    # =========================================================================

    y_mine = y - b

    # =========================================================================
    # PRIOR FORWARD MODEL
    # =========================================================================

    y_prior_mine = H @ s_prior

    y_prior_total = b + y_prior_mine

    innovation = y_mine - y_prior_mine

    print("\nPrior state:")
    print(f"  Minimum = {s_prior.min():.6f}")
    print(f"  Maximum = {s_prior.max():.6f}")

    print("\nPrior uncertainty:")
    print(f"  Minimum sigma = {sigma_prior.min():.6f}")
    print(f"  Maximum sigma = {sigma_prior.max():.6f}")

    print("\nPrior prediction:")
    print(f"  Mean mine contribution = {y_prior_mine.mean():.8f} ppm")
    print(f"  Mean total CH4          = {y_prior_total.mean():.8f} ppm")

    print("\nObservation:")
    print(f"  Mean CH4 = {y.mean():.8f} ppm")

    print("\nInnovation:")
    print(f"  Mean = {innovation.mean():.8f} ppm")

    # =========================================================================
    # BUILD PRIOR AND OBSERVATION PRECISION
    # =========================================================================

    prior_precision = 1.0 / B_diag

    observation_precision = 1.0 / R_diag

    # =========================================================================
    # BUILD HESSIAN / POSTERIOR PRECISION MATRIX
    #
    # P = B^-1 + H.T R^-1 H
    #
    # This is the Hessian of the Gaussian negative log-posterior.
    # =========================================================================

    print("\nBuilding posterior precision matrix...")

    weighted_H = (
        H * observation_precision[:, None]
    )

    posterior_precision = (
        np.diag(prior_precision)
        +
        H.T @ weighted_H
    )

    # Ensure numerical symmetry.
    posterior_precision = (
        posterior_precision + posterior_precision.T
    ) / 2.0

    # =========================================================================
    # UNCONSTRAINED GAUSSIAN MAP SOLUTION
    #
    # Retained for comparison with the constrained solution.
    # =========================================================================

    print("\nCalculating unconstrained Gaussian MAP solution...")

    rhs = (
        prior_precision * s_prior
        +
        H.T @ (observation_precision * y_mine)
    )

    s_unconstrained = np.linalg.solve(
        posterior_precision,
        rhs
    )

    print("\nUnconstrained solution:")
    print(f"  Minimum = {s_unconstrained.min():.8f}")
    print(f"  Maximum = {s_unconstrained.max():.8f}")
    print(f"  Negative parameters = {np.sum(s_unconstrained < 0)}")

    # =========================================================================
    # OBJECTIVE FUNCTION
    #
    # J(s) =
    #
    # 0.5 * (y-b-Hs).T R^-1 (y-b-Hs)
    #
    # + 0.5 * (s-s_prior).T B^-1 (s-s_prior)
    #
    # =========================================================================

    def objective(s):

        residual = y_mine - H @ s

        prior_difference = s - s_prior

        observation_cost = 0.5 * np.sum(
            observation_precision * residual ** 2
        )

        prior_cost = 0.5 * np.sum(
            prior_precision * prior_difference ** 2
        )

        return observation_cost + prior_cost

    # =========================================================================
    # ANALYTICAL GRADIENT
    #
    # grad J =
    #
    # B^-1 (s-s_prior)
    # -
    # H.T R^-1 (y-b-Hs)
    #
    # =========================================================================

    def gradient(s):

        residual = y_mine - H @ s

        prior_difference = s - s_prior

        grad_prior = (
            prior_precision * prior_difference
        )

        grad_observation = (
            -H.T @ (observation_precision * residual)
        )

        return grad_prior + grad_observation

    # =========================================================================
    # CONSTRAINED OPTIMISATION
    #
    # Every mine scaling factor must satisfy:
    #
    # s_i >= 0
    #
    # There is no upper bound in this experiment.
    # =========================================================================

    print("\n" + "=" * 80)
    print("RUNNING NON-NEGATIVE MAP OPTIMISATION")
    print("=" * 80)

    bounds = [
        (0.0, None)
        for _ in range(n_param)
    ]

    # Start from the unconstrained solution after projecting
    # negative values onto the feasible non-negative region.
    x0 = np.maximum(
        s_unconstrained,
        0.0
    )

    result = minimize(
        fun=objective,
        x0=x0,
        jac=gradient,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 20000,
            "ftol": 1e-12,
            "gtol": 1e-9,
            "maxls": 50,
            "maxcor": 20,
        }
    )

    print("\nOptimisation result:")
    print(f"  Success       = {result.success}")
    print(f"  Status        = {result.status}")
    print(f"  Message       = {result.message}")
    print(f"  Iterations    = {result.nit}")
    print(f"  Function eval = {result.nfev}")
    print(f"  Objective     = {result.fun:.10f}")

    if not result.success:

        raise RuntimeError(
            "Constrained MAP optimisation did not converge:\n"
            + str(result.message)
        )

    # =========================================================================
    # CONSTRAINED MAP ESTIMATE
    # =========================================================================

    s_post = result.x.copy()

    # Remove tiny negative numerical round-off, if any.
    s_post = np.maximum(
        s_post,
        0.0
    )

    # =========================================================================
    # BOUNDARY DIAGNOSTICS
    # =========================================================================

    active_tolerance = 1e-7

    active_mask = (
        s_post <= active_tolerance
    )

    n_active = int(
        np.sum(active_mask)
    )

    n_free = n_param - n_active

    print("\nNon-negativity constraint diagnostics:")
    print(f"  Total parameters       = {n_param}")
    print(f"  Parameters at zero     = {n_active}")
    print(f"  Positive parameters    = {n_free}")
    print(f"  Minimum scaling factor = {s_post.min():.10f}")

    # =========================================================================
    # POSTERIOR CURVATURE / COVARIANCE
    #
    # IMPORTANT:
    #
    # The inverse Hessian is the covariance of the corresponding
    # unconstrained Gaussian posterior.
    #
    # With active non-negativity constraints, this is NOT the exact
    # covariance of the truncated constrained posterior.
    #
    # We retain it as a local Gaussian curvature diagnostic.
    # =========================================================================

    print("\nCalculating Gaussian curvature covariance...")

    S_post = np.linalg.solve(
        posterior_precision,
        np.eye(n_param)
    )

    S_post = (
        S_post + S_post.T
    ) / 2.0

    posterior_variance = np.diag(
        S_post
    )

    posterior_sigma = np.sqrt(
        np.maximum(
            posterior_variance,
            0.0
        )
    )

    # =========================================================================
    # POSTERIOR CORRELATION
    # =========================================================================

    denominator = (
        posterior_sigma[:, None]
        *
        posterior_sigma[None, :]
    )

    posterior_correlation = np.zeros_like(
        S_post
    )

    valid = denominator > 0

    posterior_correlation[valid] = (
        S_post[valid] / denominator[valid]
    )

    # =========================================================================
    # CONSTRAINED AND UNCONSTRAINED OBJECTIVE VALUES
    # =========================================================================

    objective_unconstrained = objective(
        s_unconstrained
    )

    objective_constrained = objective(
        s_post
    )

    # =========================================================================
    # KKT / GRADIENT DIAGNOSTICS
    #
    # At a valid constrained minimum:
    #
    # Free parameters:
    #     gradient approximately zero
    #
    # Parameters at lower bound:
    #     gradient >= 0
    #
    # =========================================================================

    final_gradient = gradient(
        s_post
    )

    free_gradient_norm = (
        np.max(np.abs(final_gradient[~active_mask]))
        if np.any(~active_mask)
        else 0.0
    )

    active_gradient_min = (
        np.min(final_gradient[active_mask])
        if np.any(active_mask)
        else np.nan
    )

    print("\nConstrained optimisation diagnostics:")
    print(
        f"  Unconstrained objective = "
        f"{objective_unconstrained:.10f}"
    )
    print(
        f"  Constrained objective   = "
        f"{objective_constrained:.10f}"
    )
    print(
        f"  Free-variable max |gradient| = "
        f"{free_gradient_norm:.6e}"
    )

    if np.any(active_mask):
        print(
            f"  Minimum active-bound gradient = "
            f"{active_gradient_min:.6e}"
        )

    # =========================================================================
    # POSTERIOR FORWARD MODEL
    # =========================================================================

    y_post_mine = H @ s_post

    y_post_total = b + y_post_mine

    # =========================================================================
    # RESIDUALS
    # =========================================================================

    residual_prior = (
        y - y_prior_total
    )

    residual_post = (
        y - y_post_total
    )

    prior_rmse = np.sqrt(
        np.mean(residual_prior ** 2)
    )

    posterior_rmse = np.sqrt(
        np.mean(residual_post ** 2)
    )

    # =========================================================================
    # UNCONSTRAINED VS CONSTRAINED COMPARISON
    # =========================================================================

    difference = (
        s_post - s_unconstrained
    )

    max_difference = np.max(
        np.abs(difference)
    )

    rms_difference = np.sqrt(
        np.mean(difference ** 2)
    )

    print("\nUnconstrained vs constrained MAP:")
    print(
        f"  Maximum absolute difference = "
        f"{max_difference:.8f}"
    )
    print(
        f"  RMS difference = "
        f"{rms_difference:.8f}"
    )

    # =========================================================================
    # FINAL POSTERIOR SUMMARY
    # =========================================================================

    print("\n" + "=" * 80)
    print("CONSTRAINED MAP POSTERIOR SUMMARY")
    print("=" * 80)

    print("\nConstrained posterior scaling factors:")
    print(f"  Minimum = {s_post.min():.8f}")
    print(f"  Maximum = {s_post.max():.8f}")
    print(f"  Mean    = {s_post.mean():.8f}")

    print("\nGaussian curvature sigma:")
    print(f"  Minimum = {posterior_sigma.min():.8f}")
    print(f"  Maximum = {posterior_sigma.max():.8f}")
    print(f"  Mean    = {posterior_sigma.mean():.8f}")

    print("\nForward-model RMSE:")
    print(f"  Prior      = {prior_rmse:.8f} ppm")
    print(f"  Constrained = {posterior_rmse:.8f} ppm")

    return {
        "s_prior": s_prior,

        "s_post": s_post,

        "s_unconstrained": s_unconstrained,

        "S_post": S_post,

        "posterior_sigma": posterior_sigma,

        "posterior_correlation": posterior_correlation,

        "y_prior_mine": y_prior_mine,

        "y_prior_total": y_prior_total,

        "y_post_mine": y_post_mine,

        "y_post_total": y_post_total,

        "residual_prior": residual_prior,

        "residual_post": residual_post,

        # Retain this key for compatibility with the existing
        # save_summary() function. It now represents the unconstrained
        # Gaussian MAP solution, not a Kalman comparison.
        "s_post_kalman": s_unconstrained,

        "max_difference": max_difference,

        "rms_difference": rms_difference,

        "prior_rmse": prior_rmse,

        "posterior_rmse": posterior_rmse,

        "n_active": n_active,

        "n_free": n_free,

        "objective_unconstrained": objective_unconstrained,

        "objective_constrained": objective_constrained,

        "optimisation_success": result.success,

        "optimisation_message": result.message,

        "free_gradient_norm": free_gradient_norm,

        "active_gradient_min": active_gradient_min,
    }

# =============================================================================
# SAVE POSTERIOR PARAMETERS
# =============================================================================

def save_posterior(
    parameter_names,
    results
):

    print("\n" + "=" * 80)
    print("SAVING POSTERIOR SCALING FACTORS")
    print("=" * 80)


    s_prior = results[
        "s_prior"
    ]


    s_post = results[
        "s_post"
    ]


    posterior_sigma = results[
        "posterior_sigma"
    ]


    output = pd.DataFrame(
        {
            "parameter":
                parameter_names,

            "prior_scaling":
                s_prior,

            "posterior_scaling":
                s_post,

            "posterior_sigma":
                posterior_sigma,
        }
    )


    # =========================================================================
    # Extract mine and date
    # =========================================================================

    output["mine"] = (
        output["parameter"]
        .str.extract(
            r"^(OC\d+)_"
        )[0]
    )


    output["date"] = (
        output["parameter"]
        .str.extract(
            r"_(\d{8})$"
        )[0]
    )


    output["date"] = pd.to_datetime(
        output["date"],
        format="%Y%m%d"
    )


    # =========================================================================
    # Save
    # =========================================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    output.to_csv(
        POSTERIOR_FILE,
        index=False
    )


    print(
        f"\nSaved:\n{POSTERIOR_FILE}"
    )


    return output


# =============================================================================
# SAVE OBSERVATION / FORWARD MODEL COMPARISON
# =============================================================================

def save_forward_comparison(
    matched,
    b,
    results
):

    output = pd.DataFrame(
        {
            "receptor":
                matched[
                    "receptor_normalised"
                ].to_numpy(),

            "obs_time_aest":
                matched[
                    "datetime_aest"
                ].to_numpy(),

            "observed_ppm":
                matched[
                    "obs_ppm"
                ].to_numpy(),

            "background_ppm":
                b,

            "prior_mine_ppm":
                results[
                    "y_prior_mine"
                ],

            "prior_total_ppm":
                results[
                    "y_prior_total"
                ],

            "posterior_mine_ppm":
                results[
                    "y_post_mine"
                ],

            "posterior_total_ppm":
                results[
                    "y_post_total"
                ],

            "prior_residual_ppm":
                results[
                    "residual_prior"
                ],

            "posterior_residual_ppm":
                results[
                    "residual_post"
                ],
        }
    )


    output.to_csv(
        OBS_PRED_FILE,
        index=False
    )


    print(
        f"\nSaved:\n{OBS_PRED_FILE}"
    )


# =============================================================================
# SAVE COVARIANCE MATRICES
# =============================================================================

def save_covariance(
    results
):

    S_post = results[
        "S_post"
    ]


    correlation = results[
        "posterior_correlation"
    ]


    np.save(
        COVARIANCE_FILE,
        S_post
    )


    np.save(
        CORRELATION_FILE,
        correlation
    )


    print(
        f"\nSaved posterior covariance:"
    )

    print(
        COVARIANCE_FILE
    )


    print(
        "\nSaved posterior correlation:"
    )

    print(
        CORRELATION_FILE
    )


# =============================================================================
# SAVE SUMMARY
# =============================================================================

def save_summary(
    H,
    y,
    b,
    results
):

    s_prior = results[
        "s_prior"
    ]

    s_post = results[
        "s_post"
    ]

    posterior_sigma = results[
        "posterior_sigma"
    ]

    residual_prior = results[
        "residual_prior"
    ]

    residual_post = results[
        "residual_post"
    ]


    lines = []


    lines.append(
        "UPPER HUNTER CH4 ANALYTICAL BAYESIAN INVERSION"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")


    lines.append(
        f"H dimensions: "
        f"{H.shape[0]} observations × "
        f"{H.shape[1]} parameters"
    )


    lines.append(
        f"Background: "
        f"{BACKGROUND_PPM:.6f} ppm"
    )


    lines.append(
        f"Prior scaling: "
        f"{PRIOR_SCALING:.6f}"
    )


    lines.append(
        f"Prior relative 1-sigma uncertainty: "
        f"{PRIOR_RELATIVE_ERROR:.6f}"
    )


    lines.append(
        f"Observation 1-sigma uncertainty: "
        f"{OBSERVATION_ERROR_PPM:.6f} ppm"
    )


    lines.append("")


    lines.append(
        "H already contains prior OC mine emissions."
    )


    lines.append(
        "Forward model:"
    )


    lines.append(
        "    y = b + Hs + epsilon"
    )


    lines.append("")


    lines.append(
        f"Number of observations: "
        f"{len(y)}"
    )


    lines.append(
        f"Number of parameters: "
        f"{H.shape[1]}"
    )


    lines.append("")


    lines.append(
        f"Prior scaling mean: "
        f"{np.mean(s_prior):.8f}"
    )


    lines.append(
        f"Posterior scaling mean: "
        f"{np.mean(s_post):.8f}"
    )


    lines.append(
        f"Posterior scaling minimum: "
        f"{np.min(s_post):.8f}"
    )


    lines.append(
        f"Posterior scaling maximum: "
        f"{np.max(s_post):.8f}"
    )


    lines.append("")


    lines.append(
        f"Posterior sigma minimum: "
        f"{np.min(posterior_sigma):.8f}"
    )


    lines.append(
        f"Posterior sigma maximum: "
        f"{np.max(posterior_sigma):.8f}"
    )


    lines.append(
        f"Posterior sigma mean: "
        f"{np.mean(posterior_sigma):.8f}"
    )


    lines.append("")


    lines.append(
        f"Prior RMSE: "
        f"{np.sqrt(np.mean(residual_prior ** 2)):.8f} ppm"
    )


    lines.append(
        f"Posterior RMSE: "
        f"{np.sqrt(np.mean(residual_post ** 2)):.8f} ppm"
    )


    lines.append("")


    lines.append(
        "Analytical/Kalman consistency:"
    )


    lines.append(
        f"Maximum absolute difference: "
        f"{results['max_difference']:.8e}"
    )


    lines.append(
        f"RMS difference: "
        f"{results['rms_difference']:.8e}"
    )


    lines.append("")


    lines.append(
        "NOTE:"
    )


    lines.append(
        "Agriculture and power WRF-Chem contributions are not "
        "included in b in this version."
    )


    lines.append(
        "Current background is the constant 1.85 ppm."
    )


    with open(
        SUMMARY_FILE,
        "w"
    ) as f:

        f.write(
            "\n".join(lines)
        )


    print(
        f"\nSaved summary:\n{SUMMARY_FILE}"
    )


# =============================================================================
# PRINT POSTERIOR BY MINE
# =============================================================================

def print_mine_summary(
    posterior_df
):

    print("\n" + "=" * 80)
    print("POSTERIOR SCALING FACTORS BY MINE")
    print("=" * 80)


    mine_summary = (
        posterior_df
        .groupby("mine")
        .agg(
            prior_scaling=(
                "prior_scaling",
                "mean"
            ),

            posterior_mean=(
                "posterior_scaling",
                "mean"
            ),

            posterior_std=(
                "posterior_scaling",
                "std"
            ),

            mean_posterior_sigma=(
                "posterior_sigma",
                "mean"
            ),
        )
    )


    print(
        mine_summary.to_string(
            float_format=
            lambda x: f"{x:.6f}"
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("\n")

    print("=" * 80)

    print(
        "UPPER HUNTER CH4 ANALYTICAL BAYESIAN INVERSION"
    )

    print("=" * 80)


    print(
        "\n13 open-cut mines × 28 days = "
        f"{N_PARAMETERS} parameters"
    )


    print(
        "FLEXPART-WRF H is already emission-weighted."
    )


    # =========================================================================
    # LOAD H
    # =========================================================================

    (
        H_df,
        H,
        mine_columns
    ) = load_H()


    # =========================================================================
    # LOAD CLEANED PICARRO OBSERVATIONS
    # =========================================================================

    obs_long = load_observations()


    # =========================================================================
    # MATCH H TO PICARRO
    # =========================================================================

    matched = match_observations(
        H_df,
        obs_long
    )


    # =========================================================================
    # OBSERVATION VECTOR
    # =========================================================================

    y = matched[
        "obs_ppm"
    ].to_numpy(
        dtype=np.float64
    )


    # =========================================================================
    # BACKGROUND
    # =========================================================================

    b = build_background(
        matched
    )


    # =========================================================================
    # H ROWS USED IN INVERSION
    # =========================================================================

    H_used = matched[
        mine_columns
    ].to_numpy(
        dtype=np.float64
    )


    print(
        "\nH used for inversion:"
    )


    print(
        f"  shape = "
        f"{H_used.shape}"
    )


    if not np.all(
        np.isfinite(H_used)
    ):

        raise RuntimeError(
            "H_used contains NaN or infinite values."
        )


    # =========================================================================
    # ANALYTICAL INVERSION
    # =========================================================================

    results = analytical_inversion(
        H=H_used,

        y=y,

        b=b,

        prior_scaling=
        PRIOR_SCALING,

        prior_relative_error=
        PRIOR_RELATIVE_ERROR,

        observation_error=
        OBSERVATION_ERROR_PPM,
    )


    # =========================================================================
    # SAVE POSTERIOR
    # =========================================================================

    posterior_df = save_posterior(
        parameter_names=mine_columns,

        results=results
    )


    # =========================================================================
    # SAVE FORWARD MODEL
    # =========================================================================

    save_forward_comparison(
        matched=matched,

        b=b,

        results=results
    )


    # =========================================================================
    # SAVE COVARIANCE
    # =========================================================================

    save_covariance(
        results
    )


    # =========================================================================
    # SAVE SUMMARY
    # =========================================================================

    save_summary(
        H=H_used,

        y=y,

        b=b,

        results=results
    )


    # =========================================================================
    # MINE SUMMARY
    # =========================================================================

    print_mine_summary(
        posterior_df
    )


    # =========================================================================
    # PRINT ALL 403 PARAMETERS
    # =========================================================================

    print("\n" + "=" * 80)
    print("ALL 403 POSTERIOR SCALING FACTORS")
    print("=" * 80)


    display_df = posterior_df[
        [
            "parameter",
            "prior_scaling",
            "posterior_scaling",
            "posterior_sigma",
        ]
    ]


    print(
        display_df.to_string(
            index=False,

            float_format=
            lambda x: f"{x:.8f}"
        )
    )


    # =========================================================================
    # FINAL
    # =========================================================================

    print("\n" + "=" * 80)
    print("INVERSION COMPLETE")
    print("=" * 80)


    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )


    print(
        "\nPosterior scaling factors:"
    )

    print(
        POSTERIOR_FILE
    )


    print(
        "\nForward-model comparison:"
    )

    print(
        OBS_PRED_FILE
    )


    print(
        "\nPosterior covariance:"
    )

    print(
        COVARIANCE_FILE
    )


    print(
        "\nPosterior correlation:"
    )

    print(
        CORRELATION_FILE
    )


    print(
        "\nSummary:"
    )

    print(
        SUMMARY_FILE
    )


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()

