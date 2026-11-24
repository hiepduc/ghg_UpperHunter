#!/usr/bin/env python3

"""
Compare constrained and unconstrained 4D-Var FLEXPART-WRF predictions
against Picarro CH4 observations.

Upper Hunter
------------
13 open-cut mines
27 daily scaling factors
351 parameters
January 2026

Compared predictions:

    1. Picarro observations
    2. Prior: all scaling factors = 1
    3. Unconstrained 4D-Var
    4. Constrained 4D-Var: 0 <= scaling <= 3
    5. Background = 1.85 ppm

The Picarro/H alignment is identical to the successful
run_4dvar_upperhunter.py workflow.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE = (
    "/home/duch/flexpart-wrf/"
    "upperhunter_flexpart"
)

H_FILE = (
    BASE +
    "/H_matrix_flexpart/"
    "H_flexpart_0_10m_13mine_Jan2026.csv"
)

# --------------------------------------------------------------------------
# CONSTRAINED 4D-Var
# --------------------------------------------------------------------------

CONSTRAINED_DIR = (
    BASE +
    "/H_matrix_flexpart/"
    "bayesian_4dvar_constrained"
)

CONSTRAINED_POSTERIOR = (
    CONSTRAINED_DIR +
    "/4dvar_posterior_scaling_factors.csv"
)

# --------------------------------------------------------------------------
# UNCONSTRAINED 4D-Var
# --------------------------------------------------------------------------

UNCONSTRAINED_DIR = (
    BASE +
    "/H_matrix_flexpart/"
    "bayesian_4dvar_unconstrained"
)

UNCONSTRAINED_POSTERIOR = (
    UNCONSTRAINED_DIR +
    "/4dvar_posterior_scaling_factors.csv"
)

# --------------------------------------------------------------------------
# PICARRO
# --------------------------------------------------------------------------

PICARRO_FILE = (
    "/home/duch/flexpart-wrf/"
    "emission_UH/RH_upperHunter/"
    "tapm_output/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

# --------------------------------------------------------------------------
# OUTPUT
# --------------------------------------------------------------------------

OUTPUT_DIR = (
    BASE +
    "/H_matrix_flexpart/"
    "bayesian_4dvar_constrained_vs_unconstrained"
)

PLOT_DIR = (
    OUTPUT_DIR +
    "/plots"
)

BACKGROUND_PPM = 1.85

START_DATE = pd.Timestamp(
    "2026-01-05 00:00:00"
)

END_DATE = pd.Timestamp(
    "2026-01-31 23:00:00"
)


# =============================================================================
# CREATE OUTPUT DIRECTORIES
# =============================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_receptor(series):
    """
    Normalize receptor names.
    """

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


def read_scaling_factors(
    filename,
    label,
):
    """
    Read a 4D-Var posterior scaling-factor CSV
    and return factors in exactly the H parameter order.
    """

    print()
    print("=" * 90)
    print(
        f"READING {label.upper()} "
        "4D-VAR SCALING FACTORS"
    )
    print("=" * 90)

    df = pd.read_csv(
        filename
    )

    print(
        "File:",
        filename
    )

    print(
        "Shape:",
        df.shape
    )

    print(
        "Columns:"
    )

    print(
        df.columns.tolist()
    )

    required = {
        "parameter",
        "prior_scaling",
        "posterior_scaling",
    }

    missing = (
        required
        -
        set(df.columns)
    )

    if missing:

        raise RuntimeError(
            f"{label} posterior file is missing "
            f"required columns: {sorted(missing)}"
        )

    factor_map = dict(
        zip(
            df["parameter"].astype(str),
            df["posterior_scaling"].astype(float),
        )
    )

    missing_parameters = [
        p
        for p in parameter_cols
        if p not in factor_map
    ]

    if missing_parameters:

        raise RuntimeError(
            f"{label}: missing "
            f"{len(missing_parameters)} H parameters. "
            f"First few: {missing_parameters[:10]}"
        )

    factors = np.array(
        [
            factor_map[p]
            for p in parameter_cols
        ],
        dtype=float,
    )

    print(
        f"Number of {label} factors:",
        len(factors)
    )

    print(
        f"{label} factor min:",
        factors.min()
    )

    print(
        f"{label} factor max:",
        factors.max()
    )

    print(
        f"{label} factor mean:",
        factors.mean()
    )

    print(
        f"{label} factor median:",
        np.median(factors)
    )

    return factors


def calculate_metrics(
    obs,
    pred,
):
    """
    Calculate prediction metrics.
    """

    obs = np.asarray(
        obs,
        dtype=float
    )

    pred = np.asarray(
        pred,
        dtype=float
    )

    mask = (
        np.isfinite(obs)
        &
        np.isfinite(pred)
    )

    obs = obs[mask]
    pred = pred[mask]

    if len(obs) == 0:

        return {
            "N": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan,
            "R": np.nan,
            "R2": np.nan,
        }

    residual = (
        pred - obs
    )

    rmse = np.sqrt(
        np.mean(
            residual ** 2
        )
    )

    mae = np.mean(
        np.abs(residual)
    )

    bias = np.mean(
        residual
    )

    if len(obs) > 1:

        r = np.corrcoef(
            obs,
            pred
        )[0, 1]

        r2 = r ** 2

    else:

        r = np.nan
        r2 = np.nan

    return {
        "N": len(obs),
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "R": r,
        "R2": r2,
    }


# =============================================================================
# HEADER
# =============================================================================

print()
print("=" * 90)
print(
    "CONSTRAINED vs UNCONSTRAINED 4D-VAR "
    "FLEXPART-WRF PREDICTIONS"
)
print("=" * 90)


# =============================================================================
# READ H MATRIX
# =============================================================================

print()
print("=" * 90)
print(
    "READING FLEXPART H MATRIX"
)
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

missing_metadata = [
    c
    for c in metadata_columns
    if c not in H_df.columns
]

if missing_metadata:

    raise RuntimeError(
        "Missing H metadata columns: "
        f"{missing_metadata}"
    )


parameter_cols = [
    c
    for c in H_df.columns
    if c not in metadata_columns
]

N_PARAM = len(
    parameter_cols
)

print(
    "Parameter count:",
    N_PARAM
)

if N_PARAM != 351:

    raise RuntimeError(
        f"Expected 351 parameters, "
        f"found {N_PARAM}"
    )


H_values_all = (
    H_df[
        parameter_cols
    ]
    .to_numpy(
        dtype=float
    )
)

print(
    "H array shape:",
    H_values_all.shape
)

print(
    "H min/max:",
    H_values_all.min(),
    H_values_all.max()
)


# =============================================================================
# READ UNCONSTRAINED FACTORS
# =============================================================================

s_unconstrained = read_scaling_factors(
    UNCONSTRAINED_POSTERIOR,
    "unconstrained",
)


# =============================================================================
# READ CONSTRAINED FACTORS
# =============================================================================

s_constrained = read_scaling_factors(
    CONSTRAINED_POSTERIOR,
    "constrained",
)


# =============================================================================
# VERIFY CONSTRAINT
# =============================================================================

print()
print(
    "Checking constrained solution..."
)

print(
    "Constrained minimum:",
    s_constrained.min()
)

print(
    "Constrained maximum:",
    s_constrained.max()
)

n_below = np.sum(
    s_constrained < -1e-10
)

n_above = np.sum(
    s_constrained > 3.0 + 1e-10
)

print(
    "Factors below 0:",
    n_below
)

print(
    "Factors above 3:",
    n_above
)

if n_below > 0 or n_above > 0:

    raise RuntimeError(
        "Constrained posterior contains "
        "factors outside [0, 3]."
    )


# =============================================================================
# CALCULATE PREDICTIONS
# =============================================================================

print()
print("=" * 90)
print(
    "CALCULATING PRIOR / UNCONSTRAINED / "
    "CONSTRAINED PREDICTIONS"
)
print("=" * 90)


s_prior = np.ones(
    N_PARAM,
    dtype=float
)


prior_enhancement = (
    H_values_all
    @
    s_prior
)

unconstrained_enhancement = (
    H_values_all
    @
    s_unconstrained
)

constrained_enhancement = (
    H_values_all
    @
    s_constrained
)


prior_prediction = (
    BACKGROUND_PPM
    +
    prior_enhancement
)

unconstrained_prediction = (
    BACKGROUND_PPM
    +
    unconstrained_enhancement
)

constrained_prediction = (
    BACKGROUND_PPM
    +
    constrained_enhancement
)


print()
print(
    "Prior prediction min/max:",
    prior_prediction.min(),
    prior_prediction.max()
)

print(
    "Unconstrained prediction min/max:",
    unconstrained_prediction.min(),
    unconstrained_prediction.max()
)

print(
    "Constrained prediction min/max:",
    constrained_prediction.min(),
    constrained_prediction.max()
)


# =============================================================================
# READ PICARRO
# =============================================================================

print()
print("=" * 90)
print(
    "READING PICARRO OBSERVATIONS"
)
print("=" * 90)


obs = pd.read_csv(
    PICARRO_FILE
)


# =============================================================================
# PARSE PICARRO DATETIME
# =============================================================================

# IMPORTANT:
# Picarro Datetime is DD/MM/YYYY HH:MM.

obs["time_aest"] = pd.to_datetime(
    obs["Datetime"],
    format="%d/%m/%Y %H:%M",
    errors="coerce",
)


if obs["time_aest"].isna().any():

    nbad = (
        obs["time_aest"]
        .isna()
        .sum()
    )

    raise ValueError(
        f"{nbad} Picarro rows have "
        "invalid Datetime values."
    )


print(
    "Picarro range:",
    obs["time_aest"].min(),
    "to",
    obs["time_aest"].max()
)


# =============================================================================
# CREATE LONG PICARRO TABLE
# =============================================================================

receptor_columns = {
    "singleton":
        "Singleton cleaned",

    "muswellbrook":
        "Muswellbrook cleaned",

    "merriwa":
        "Merriwa cleaned",
}


obs_long_parts = []


for receptor, column in receptor_columns.items():

    if column not in obs.columns:

        raise RuntimeError(
            f"Missing Picarro column: "
            f"{column}"
        )

    tmp = obs[
        [
            "time_aest",
            column,
        ]
    ].copy()

    tmp["receptor"] = receptor

    tmp["obs_ch4_ppm"] = pd.to_numeric(
        tmp[column],
        errors="coerce",
    )

    tmp = tmp[
        [
            "receptor",
            "time_aest",
            "obs_ch4_ppm",
        ]
    ]

    obs_long_parts.append(
        tmp
    )


obs_long = pd.concat(
    obs_long_parts,
    ignore_index=True,
)


# =============================================================================
# CLEAN OBSERVATIONS
# =============================================================================

obs_long = (
    obs_long
    .dropna(
        subset=[
            "time_aest",
            "obs_ch4_ppm",
        ]
    )
    .copy()
)


obs_long["receptor"] = normalize_receptor(
    obs_long["receptor"]
)


# =============================================================================
# JANUARY FILTER
# =============================================================================

obs_long = obs_long[
    (
        obs_long["time_aest"]
        >= START_DATE
    )
    &
    (
        obs_long["time_aest"]
        <= END_DATE
    )
].copy()


# =============================================================================
# HOURLY AVERAGING
# =============================================================================

obs_long["time_aest"] = (
    obs_long["time_aest"]
    .dt.floor("h")
)


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
    "Usable Picarro observations:",
    len(obs_long)
)

print(
    obs_long[
        "receptor"
    ].value_counts()
)


# =============================================================================
# ALIGN H WITH PICARRO
# =============================================================================

print()
print("=" * 90)
print(
    "ALIGNING PICARRO OBSERVATIONS "
    "WITH FLEXPART H"
)
print("=" * 90)


# =============================================================================
# PARSE H TIME
# =============================================================================

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
        "obs_time_aest."
    )


# =============================================================================
# NORMALIZE H RECEPTOR
# =============================================================================

H_df["receptor"] = normalize_receptor(
    H_df["receptor"]
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
# OBSERVATION KEY
# =============================================================================

obs_key = obs_long[
    [
        "receptor",
        "time_aest",
        "obs_ch4_ppm",
    ]
].copy()


# =============================================================================
# PRESERVE ORIGINAL H ROW NUMBER
# =============================================================================

H_key_indexed = H_key.copy()

H_key_indexed[
    "H_row"
] = np.arange(
    len(H_key_indexed)
)


# =============================================================================
# EXACT SAME MERGE AS SUCCESSFUL 4D-VAR
# =============================================================================

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
        "Example H keys:"
    )

    print(
        H_key.head(10)
    )

    print()
    print(
        "Example observation keys:"
    )

    print(
        obs_long.head(10)
    )

    raise RuntimeError(
        "No FLEXPART/Picarro observations "
        "could be aligned."
    )


# =============================================================================
# RECEPTOR COUNTS
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
# EXTRACT H ROWS
# =============================================================================

H_row_indices = (
    match[
        "H_row"
    ]
    .to_numpy(
        dtype=int
    )
)


y = (
    match[
        "obs_ch4_ppm"
    ]
    .to_numpy(
        dtype=float
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


prior_obs = (
    prior_prediction[
        H_row_indices
    ]
)


unconstrained_obs = (
    unconstrained_prediction[
        H_row_indices
    ]
)


constrained_obs = (
    constrained_prediction[
        H_row_indices
    ]
)


# =============================================================================
# CREATE PREDICTION TABLE
# =============================================================================

prediction_table = pd.DataFrame(
    {
        "receptor":
            receptors_obs,

        "time_aest":
            times_obs,

        "observed_ch4_ppm":
            y,

        "prior_prediction_ppm":
            prior_obs,

        "unconstrained_4dvar_prediction_ppm":
            unconstrained_obs,

        "constrained_4dvar_prediction_ppm":
            constrained_obs,

        "background_ppm":
            BACKGROUND_PPM,
    }
)


prediction_table[
    "prior_residual_ppm"
] = (
    prior_obs - y
)


prediction_table[
    "unconstrained_4dvar_residual_ppm"
] = (
    unconstrained_obs - y
)


prediction_table[
    "constrained_4dvar_residual_ppm"
] = (
    constrained_obs - y
)


prediction_table = (
    prediction_table
    .sort_values(
        [
            "receptor",
            "time_aest",
        ]
    )
    .reset_index(drop=True)
)


prediction_csv = (
    OUTPUT_DIR +
    "/4dvar_constrained_vs_"
    "unconstrained_predictions.csv"
)


prediction_table.to_csv(
    prediction_csv,
    index=False,
)


print()
print(
    "Saved prediction table:"
)

print(
    prediction_csv
)


# =============================================================================
# METRICS
# =============================================================================

print()
print("=" * 90)
print(
    "CALCULATING PREDICTION METRICS"
)
print("=" * 90)


metrics_rows = []


def add_metrics(
    receptor,
    method,
    obs_values,
    pred_values,
):

    metrics = calculate_metrics(
        obs_values,
        pred_values,
    )

    metrics_rows.append(
        {
            "receptor":
                receptor,

            "method":
                method,

            **metrics,
        }
    )

    return metrics


# =============================================================================
# ALL RECEPTORS
# =============================================================================

m_prior = add_metrics(
    "ALL",
    "Prior",
    y,
    prior_obs,
)

m_unconstrained = add_metrics(
    "ALL",
    "4DVar_unconstrained",
    y,
    unconstrained_obs,
)

m_constrained = add_metrics(
    "ALL",
    "4DVar_constrained",
    y,
    constrained_obs,
)


print()
print(
    "ALL RECEPTORS"
)

print(
    "Prior:",
    m_prior
)

print(
    "Unconstrained 4D-Var:",
    m_unconstrained
)

print(
    "Constrained 4D-Var:",
    m_constrained
)


# =============================================================================
# INDIVIDUAL RECEPTORS
# =============================================================================

for receptor in [
    "singleton",
    "muswellbrook",
    "merriwa",
]:

    mask = (
        receptors_obs
        ==
        receptor
    )

    if not np.any(mask):

        print()
        print(
            f"{receptor}: no observations"
        )

        continue


    obs_r = y[mask]

    prior_r = prior_obs[mask]

    unconstrained_r = (
        unconstrained_obs[mask]
    )

    constrained_r = (
        constrained_obs[mask]
    )


    m1 = add_metrics(
        receptor,
        "Prior",
        obs_r,
        prior_r,
    )

    m2 = add_metrics(
        receptor,
        "4DVar_unconstrained",
        obs_r,
        unconstrained_r,
    )

    m3 = add_metrics(
        receptor,
        "4DVar_constrained",
        obs_r,
        constrained_r,
    )


    print()
    print(
        receptor.upper()
    )

    print(
        "Prior:",
        m1
    )

    print(
        "Unconstrained 4D-Var:",
        m2
    )

    print(
        "Constrained 4D-Var:",
        m3
    )


metrics_df = pd.DataFrame(
    metrics_rows
)


metrics_csv = (
    OUTPUT_DIR +
    "/4dvar_constrained_vs_"
    "unconstrained_metrics.csv"
)


metrics_df.to_csv(
    metrics_csv,
    index=False,
)


print()
print(
    "Saved metrics:"
)

print(
    metrics_csv
)


# =============================================================================
# INDIVIDUAL RECEPTOR COMPARISON PLOTS
# =============================================================================

print()
print("=" * 90)
print(
    "CREATING RECEPTOR COMPARISON PLOTS"
)
print("=" * 90)


for receptor in [
    "singleton",
    "muswellbrook",
    "merriwa",
]:

    mask = (
        receptors_obs
        ==
        receptor
    )

    if not np.any(mask):

        print(
            f"No data for {receptor}; "
            "skipping."
        )

        continue


    t = pd.to_datetime(
        times_obs[mask]
    )

    obs_r = y[mask]

    prior_r = prior_obs[mask]

    unconstrained_r = (
        unconstrained_obs[mask]
    )

    constrained_r = (
        constrained_obs[mask]
    )


    # Sort chronologically
    order = np.argsort(t)

    t = t[order]

    obs_r = obs_r[order]

    prior_r = prior_r[order]

    unconstrained_r = (
        unconstrained_r[order]
    )

    constrained_r = (
        constrained_r[order]
    )


    plt.figure(
        figsize=(14, 6)
    )


    # ----------------------------------------------------------------------
    # Observations
    # ----------------------------------------------------------------------

    #plt.plot(
    #    t,
    #    obs_r,
    #    "o",
    #    markersize=3,
    #    label="Picarro observation",
    #)

    plt.plot(
        t,
        obs_r,
        "-",
        linewidth=1.5,
        label="Picarro observation",
    )


    # ----------------------------------------------------------------------
    # Prior
    # ----------------------------------------------------------------------

    plt.plot(
        t,
        prior_r,
        "-",
        linewidth=1.3,
        label="FLEXPART prior",
    )


    # ----------------------------------------------------------------------
    # Unconstrained
    # ----------------------------------------------------------------------

    plt.plot(
        t,
        unconstrained_r,
        "-",
        linewidth=1.8,
        label="4D-Var unconstrained",
    )


    # ----------------------------------------------------------------------
    # Constrained
    # ----------------------------------------------------------------------

    plt.plot(
        t,
        constrained_r,
        "-",
        linewidth=1.8,
        label="4D-Var constrained (0–3)",
    )


    # ----------------------------------------------------------------------
    # Background
    # ----------------------------------------------------------------------

    plt.axhline(
        BACKGROUND_PPM,
        linestyle="--",
        linewidth=1.2,
        label="Background = 1.85 ppm",
    )


    plt.xlabel(
        "AEST"
    )

    plt.ylabel(
        "CH$_4$ concentration (ppm)"
    )

    plt.title(
        f"{receptor.capitalize()} — "
        "FLEXPART-WRF CH$_4$\n"
        "Prior vs constrained and "
        "unconstrained 4D-Var"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()


    filename = (
        PLOT_DIR +
        f"/{receptor.capitalize()}_"
        "4DVar_constrained_vs_"
        "unconstrained_prediction.png"
    )


    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()


    print(
        "Saved:",
        filename
    )


# =============================================================================
# COMBINED PLOT
# =============================================================================

print()
print(
    "=" * 90
)
print(
    "CREATING COMBINED RECEPTOR PLOT"
)
print(
    "=" * 90
)


plt.figure(
    figsize=(16, 8)
)


# --------------------------------------------------------------------------
# Plot observations
# --------------------------------------------------------------------------

observation_styles = {
    "singleton": {
        "marker": "o",
        "label": "Singleton observation",
    },

    "muswellbrook": {
        "marker": "s",
        "label": "Muswellbrook observation",
    },

    "merriwa": {
        "marker": "^",
        "label": "Merriwa observation",
    },
}


for receptor in [
    "singleton",
    "muswellbrook",
    "merriwa",
]:

    mask = (
        receptors_obs
        ==
        receptor
    )

    if not np.any(mask):
        continue


    t = pd.to_datetime(
        times_obs[mask]
    )

    obs_r = y[mask]

    order = np.argsort(t)

    t = t[order]

    obs_r = obs_r[order]


    #plt.plot(
    #    t,
    #    obs_r,
    #    observation_styles[
    #        receptor
    #    ]["marker"],
    #    markersize=3,
    #    linestyle="none",
    #    label=observation_styles[
    #        receptor
    #    ]["label"],
    #)

    plt.plot(
        t,
        obs_r,
        "-",
        linewidth=1.2,
        label=observation_styles[
            receptor
        ]["label"],
    )


# =============================================================================
# COMBINED MODEL LINES
# =============================================================================

# To avoid an unreadable plot with many overlapping lines,
# use Singleton and Muswellbrook model lines, which are the
# receptors containing the matched observations.

for receptor in [
    "singleton",
    "muswellbrook",
]:

    mask = (
        receptors_obs
        ==
        receptor
    )

    if not np.any(mask):
        continue


    t = pd.to_datetime(
        times_obs[mask]
    )

    prior_r = prior_obs[mask]

    unconstrained_r = (
        unconstrained_obs[mask]
    )

    constrained_r = (
        constrained_obs[mask]
    )


    order = np.argsort(t)

    t = t[order]

    prior_r = prior_r[order]

    unconstrained_r = (
        unconstrained_r[order]
    )

    constrained_r = (
        constrained_r[order]
    )


    plt.plot(
        t,
        prior_r,
        "-",
        linewidth=1.0,
        alpha=0.65,
        label=f"{receptor.capitalize()} prior",
    )


    plt.plot(
        t,
        unconstrained_r,
        "-",
        linewidth=1.7,
        alpha=0.85,
        label=(
            f"{receptor.capitalize()} "
            "4D-Var unconstrained"
        ),
    )


    plt.plot(
        t,
        constrained_r,
        "-",
        linewidth=1.7,
        alpha=0.85,
        label=(
            f"{receptor.capitalize()} "
            "4D-Var constrained"
        ),
    )


# =============================================================================
# BACKGROUND
# =============================================================================

plt.axhline(
    BACKGROUND_PPM,
    linestyle="--",
    linewidth=1.2,
    label="Background = 1.85 ppm",
)


# =============================================================================
# LABELS
# =============================================================================

plt.xlabel(
    "AEST"
)

plt.ylabel(
    "CH$_4$ concentration (ppm)"
)

plt.title(
    "Upper Hunter — FLEXPART-WRF CH$_4$\n"
    "Prior vs constrained and "
    "unconstrained 4D-Var"
)

plt.legend(
    ncol=2
)

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()


combined_file = (
    PLOT_DIR +
    "/ALL_RECEPTORS_4DVar_"
    "constrained_vs_unconstrained_prediction.png"
)


plt.savefig(
    combined_file,
    dpi=200,
    bbox_inches="tight",
)

plt.show()
plt.close()


print(
    "Saved:",
    combined_file
)


# =============================================================================
# SCALING FACTOR COMPARISON
# =============================================================================

print()
print("=" * 90)
print(
    "CREATING SCALING-FACTOR COMPARISON"
)
print("=" * 90)


factor_comparison = pd.DataFrame(
    {
        "parameter":
            parameter_cols,

        "prior_scaling":
            s_prior,

        "unconstrained_4dvar":
            s_unconstrained,

        "constrained_4dvar":
            s_constrained,

        "constrained_minus_unconstrained":
            (
                s_constrained
                -
                s_unconstrained
            ),
    }
)


factor_csv = (
    OUTPUT_DIR +
    "/constrained_vs_unconstrained_"
    "scaling_factors.csv"
)


factor_comparison.to_csv(
    factor_csv,
    index=False,
)


print(
    "Saved:",
    factor_csv
)


# =============================================================================
# SCALING FACTOR SCATTER PLOT
# =============================================================================

plt.figure(
    figsize=(7, 7)
)


plt.scatter(
    s_unconstrained,
    s_constrained,
    s=15,
    alpha=0.7,
)


# Reference y = x
xmin = min(
    s_unconstrained.min(),
    s_constrained.min(),
)

xmax = max(
    s_unconstrained.max(),
    s_constrained.max(),
)

plt.plot(
    [xmin, xmax],
    [xmin, xmax],
    "--",
    linewidth=1.2,
)


# Constraint bounds
plt.axhline(
    0.0,
    linestyle=":",
    linewidth=1.0,
)

plt.axhline(
    3.0,
    linestyle=":",
    linewidth=1.0,
)


plt.xlabel(
    "Unconstrained 4D-Var scaling factor"
)

plt.ylabel(
    "Constrained 4D-Var scaling factor"
)

plt.title(
    "Daily mine scaling factors:\n"
    "constrained vs unconstrained 4D-Var"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()


scatter_file = (
    PLOT_DIR +
    "/constrained_vs_unconstrained_"
    "scaling_factors_scatter.png"
)


plt.savefig(
    scatter_file,
    dpi=200,
    bbox_inches="tight",
)

plt.show()
plt.close()


print(
    "Saved:",
    scatter_file
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 90)
print(
    "FINAL SUMMARY"
)
print("=" * 90)


print()
print(
    f"Matched observations : {len(y)}"
)

print(
    f"Parameters            : {N_PARAM}"
)


print()
print(
    "Scaling factors:"
)

print(
    f"  Unconstrained mean : "
    f"{s_unconstrained.mean():.8f}"
)

print(
    f"  Constrained mean   : "
    f"{s_constrained.mean():.8f}"
)

print(
    f"  Unconstrained median : "
    f"{np.median(s_unconstrained):.8f}"
)

print(
    f"  Constrained median   : "
    f"{np.median(s_constrained):.8f}"
)

print(
    f"  Unconstrained min : "
    f"{s_unconstrained.min():.8f}"
)

print(
    f"  Unconstrained max : "
    f"{s_unconstrained.max():.8f}"
)

print(
    f"  Constrained min   : "
    f"{s_constrained.min():.8f}"
)

print(
    f"  Constrained max   : "
    f"{s_constrained.max():.8f}"
)


print()
print(
    "ALL observations:"
)

print(
    f"  Prior RMSE                 : "
    f"{m_prior['RMSE']:.6f} ppm"
)

print(
    f"  Unconstrained 4D-Var RMSE  : "
    f"{m_unconstrained['RMSE']:.6f} ppm"
)

print(
    f"  Constrained 4D-Var RMSE    : "
    f"{m_constrained['RMSE']:.6f} ppm"
)


print()
print(
    f"  Prior MAE                  : "
    f"{m_prior['MAE']:.6f} ppm"
)

print(
    f"  Unconstrained 4D-Var MAE   : "
    f"{m_unconstrained['MAE']:.6f} ppm"
)

print(
    f"  Constrained 4D-Var MAE     : "
    f"{m_constrained['MAE']:.6f} ppm"
)


print()
print(
    f"  Prior Bias                 : "
    f"{m_prior['Bias']:.6f} ppm"
)

print(
    f"  Unconstrained 4D-Var Bias  : "
    f"{m_unconstrained['Bias']:.6f} ppm"
)

print(
    f"  Constrained 4D-Var Bias    : "
    f"{m_constrained['Bias']:.6f} ppm"
)


print()
print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Plot directory:"
)

print(
    PLOT_DIR
)

print()
print(
    "=" * 90
)

print(
    "DONE"
)

print(
    "=" * 90
)

