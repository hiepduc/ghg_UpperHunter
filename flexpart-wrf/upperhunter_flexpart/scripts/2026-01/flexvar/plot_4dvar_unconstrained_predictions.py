#!/usr/bin/env python3

"""
Plot Picarro CH4 observations against FLEXPART-WRF predictions
using the unconstrained 4D-Var posterior daily scaling factors.

Upper Hunter
------------
13 open-cut mines
27 daily scaling factors
351 parameters
January 2026

This script compares:

    1. Picarro observations
    2. Prior FLEXPART prediction
    3. Unconstrained 4D-Var posterior prediction
    4. Background concentration

The observation/H alignment is deliberately identical to the
successful run_4dvar_upperhunter.py workflow.
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

POSTERIOR_FILE = (
    BASE +
    "/H_matrix_flexpart/"
    "bayesian_4dvar_unconstrained/"
    "4dvar_posterior_scaling_factors.csv"
)

PICARRO_FILE = (
    "/home/duch/flexpart-wrf/"
    "emission_UH/RH_upperHunter/"
    "tapm_output/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

OUTPUT_DIR = (
    BASE +
    "/H_matrix_flexpart/"
    "bayesian_4dvar_unconstrained"
)

PLOT_DIR = OUTPUT_DIR + "/plots"

BACKGROUND_PPM = 1.85

START_DATE = pd.Timestamp("2026-01-05 00:00:00")
END_DATE = pd.Timestamp("2026-01-31 23:00:00")


# =============================================================================
# CREATE OUTPUT DIRECTORIES
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_receptor(series):
    """
    Normalize receptor names so H and Picarro use identical keys.
    """

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


def calculate_metrics(obs, pred):
    """
    Calculate standard prediction metrics.
    """

    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)

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

    residual = pred - obs

    rmse = np.sqrt(
        np.mean(residual ** 2)
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
    "FLEXPART-WRF / UNCONSTRAINED 4D-VAR "
    "PREDICTION PLOT"
)
print("=" * 90)
print()


# =============================================================================
# READ H MATRIX
# =============================================================================

print("=" * 90)
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

missing_metadata = [
    c for c in metadata_columns
    if c not in H_df.columns
]

if missing_metadata:

    raise RuntimeError(
        "Missing H metadata columns: "
        + str(missing_metadata)
    )


parameter_cols = [
    c for c in H_df.columns
    if c not in metadata_columns
]

N_PARAM = len(parameter_cols)

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
    .to_numpy(dtype=float)
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
# READ UNCONSTRAINED 4D-VAR SCALING FACTORS
# =============================================================================

print()
print("=" * 90)
print(
    "READING UNCONSTRAINED 4D-VAR "
    "SCALING FACTORS"
)
print("=" * 90)

post = pd.read_csv(
    POSTERIOR_FILE
)

print(
    "Posterior file shape:",
    post.shape
)

print(
    "Posterior columns:"
)

print(
    post.columns.tolist()
)

required = {
    "parameter",
    "prior_scaling",
    "posterior_scaling",
}

missing = (
    required
    -
    set(post.columns)
)

if missing:

    raise RuntimeError(
        "Missing required posterior columns: "
        + str(sorted(missing))
    )


# --------------------------------------------------------------------------
# Map posterior parameters to H parameter ordering
# --------------------------------------------------------------------------

factor_map = dict(
    zip(
        post["parameter"].astype(str),
        post["posterior_scaling"].astype(float),
    )
)


missing_parameters = [
    p
    for p in parameter_cols
    if p not in factor_map
]

if missing_parameters:

    raise RuntimeError(
        f"Missing {len(missing_parameters)} "
        "H parameters in posterior file. "
        f"First few: {missing_parameters[:10]}"
    )


s_post = np.array(
    [
        factor_map[p]
        for p in parameter_cols
    ],
    dtype=float,
)


print(
    "Using posterior factor column: "
    "posterior_scaling"
)

print(
    "Number of factors:",
    len(s_post)
)

print(
    "Posterior factor min:",
    s_post.min()
)

print(
    "Posterior factor max:",
    s_post.max()
)

print(
    "Posterior factor mean:",
    s_post.mean()
)

print(
    "Posterior factor median:",
    np.median(s_post)
)


# =============================================================================
# VERIFY POSTERIOR FACTOR ORDER
# =============================================================================

print()
print(
    "First 10 posterior factors:"
)

for i in range(
    min(10, N_PARAM)
):

    print(
        f"  {parameter_cols[i]:20s} "
        f"{s_post[i]: .8f}"
    )


# =============================================================================
# CALCULATE PRIOR AND POSTERIOR PREDICTIONS
# =============================================================================

print()
print("=" * 90)
print(
    "CALCULATING PRIOR AND POSTERIOR "
    "PREDICTIONS"
)
print("=" * 90)


# Prior scaling factor = 1 for every mine/day
s_prior = np.ones(
    N_PARAM,
    dtype=float
)


# --------------------------------------------------------------------------
# FLEXPART enhancement predictions
# --------------------------------------------------------------------------

prior_enhancement = (
    H_values_all @ s_prior
)

posterior_enhancement = (
    H_values_all @ s_post
)


# --------------------------------------------------------------------------
# Convert to total CH4 concentration
# --------------------------------------------------------------------------

prior_prediction = (
    BACKGROUND_PPM
    +
    prior_enhancement
)

post_prediction = (
    BACKGROUND_PPM
    +
    posterior_enhancement
)


print(
    "Prior prediction min/max:",
    prior_prediction.min(),
    prior_prediction.max()
)

print(
    "Posterior prediction min/max:",
    post_prediction.min(),
    post_prediction.max()
)


# =============================================================================
# READ PICARRO OBSERVATIONS
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


# --------------------------------------------------------------------------
# IMPORTANT:
# Picarro Datetime is DD/MM/YYYY HH:MM
# --------------------------------------------------------------------------

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
# CONSTRUCT LONG OBSERVATION TABLE
# =============================================================================

receptor_columns = {
    "singleton": "Singleton cleaned",
    "muswellbrook": "Muswellbrook cleaned",
    "merriwa": "Merriwa cleaned",
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


# --------------------------------------------------------------------------
# Remove invalid observations
# --------------------------------------------------------------------------

obs_long = obs_long.dropna(
    subset=[
        "time_aest",
        "obs_ch4_ppm",
    ]
).copy()


# --------------------------------------------------------------------------
# Normalize receptor names
# --------------------------------------------------------------------------

obs_long["receptor"] = normalize_receptor(
    obs_long["receptor"]
)


# --------------------------------------------------------------------------
# Restrict to January 5–31
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Floor to hourly
# --------------------------------------------------------------------------

obs_long["time_aest"] = (
    obs_long["time_aest"]
    .dt.floor("h")
)


# --------------------------------------------------------------------------
# Average duplicate receptor/hour observations
# --------------------------------------------------------------------------

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
# ALIGN PICARRO OBSERVATIONS WITH FLEXPART H
# =============================================================================

print()
print("=" * 90)
print(
    "ALIGNING PICARRO OBSERVATIONS "
    "WITH FLEXPART H"
)
print("=" * 90)


# --------------------------------------------------------------------------
# Parse H timestamps exactly as in successful 4D-Var script
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Normalize H receptor names
# --------------------------------------------------------------------------

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
# ALIGN OBSERVATIONS DIRECTLY WITH H
# =============================================================================

obs_key = obs_long[
    [
        "receptor",
        "time_aest",
        "obs_ch4_ppm",
    ]
].copy()


# --------------------------------------------------------------------------
# Preserve original H row number
# --------------------------------------------------------------------------

H_key_indexed = H_key.copy()

H_key_indexed[
    "H_row"
] = np.arange(
    len(H_key_indexed)
)


# --------------------------------------------------------------------------
# EXACT SAME MERGE AS SUCCESSFUL 4D-VAR / GIBBS / MH
# --------------------------------------------------------------------------

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
    print(
        "Example H times:"
    )

    print(
        H_key.head(10)
    )

    print()
    print(
        "Example observation times:"
    )

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
# EXTRACT MATCHED H ROWS
# =============================================================================

H_row_indices = (
    match[
        "H_row"
    ]
    .to_numpy(
        dtype=int
    )
)


H_obs = H_values_all[
    H_row_indices,
    :
]


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


posterior_obs = (
    post_prediction[
        H_row_indices
    ]
)


# =============================================================================
# FINAL ALIGNMENT DIAGNOSTICS
# =============================================================================

print()
print(
    "Final H shape:",
    H_obs.shape
)

print(
    "Observation vector shape:",
    y.shape
)

print(
    "Observed CH4 min/max:",
    f"{y.min():.6f},",
    f"{y.max():.6f} ppm"
)

print(
    "Observed CH4 mean:",
    f"{y.mean():.6f} ppm"
)


# =============================================================================
# CREATE OUTPUT DATAFRAME
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

        "posterior_4dvar_unconstrained_ppm":
            posterior_obs,

        "background_ppm":
            BACKGROUND_PPM,

        "prior_residual_ppm":
            prior_obs - y,

        "posterior_residual_ppm":
            posterior_obs - y,
    }
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
    "/4dvar_receptor_predictions_plot.csv"
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
    "PREDICTION METRICS"
)
print("=" * 90)


metrics_rows = []


# --------------------------------------------------------------------------
# Overall
# --------------------------------------------------------------------------

m_prior = calculate_metrics(
    y,
    prior_obs,
)

m_post = calculate_metrics(
    y,
    posterior_obs,
)


metrics_rows.append(
    {
        "receptor": "ALL",
        "method": "Prior",
        **m_prior,
    }
)


metrics_rows.append(
    {
        "receptor": "ALL",
        "method":
            "4DVar_unconstrained",
        **m_post,
    }
)


print()
print(
    "ALL — Prior:"
)

print(m_prior)


print()
print(
    "ALL — 4D-Var unconstrained:"
)

print(m_post)


# --------------------------------------------------------------------------
# Individual receptors
# --------------------------------------------------------------------------

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

    obs_r = y[mask]

    prior_r = prior_obs[mask]

    post_r = posterior_obs[mask]

    mp = calculate_metrics(
        obs_r,
        prior_r,
    )

    mo = calculate_metrics(
        obs_r,
        post_r,
    )

    metrics_rows.append(
        {
            "receptor": receptor,
            "method": "Prior",
            **mp,
        }
    )

    metrics_rows.append(
        {
            "receptor": receptor,
            "method":
                "4DVar_unconstrained",
            **mo,
        }
    )

    print()
    print(
        f"{receptor} — Prior:"
    )

    print(mp)

    print(
        f"{receptor} — "
        "4D-Var unconstrained:"
    )

    print(mo)


metrics_df = pd.DataFrame(
    metrics_rows
)


metrics_csv = (
    OUTPUT_DIR +
    "/4dvar_unconstrained_prediction_metrics.csv"
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
# PLOT INDIVIDUAL RECEPTOR
# =============================================================================

print()
print("=" * 90)
print(
    "CREATING RECEPTOR PLOTS"
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
            "skipping plot."
        )

        continue

    t = (
        pd.to_datetime(
            times_obs[mask]
        )
    )

    obs_r = y[mask]

    prior_r = prior_obs[mask]

    post_r = posterior_obs[mask]


    # Sort by time
    order = np.argsort(t)

    t = t[order]

    obs_r = obs_r[order]

    prior_r = prior_r[order]

    post_r = post_r[order]


    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        t,
        obs_r,
        "o",
        markersize=3,
        label="Picarro observation",
    )

    plt.plot(
        t,
        prior_r,
        "-",
        linewidth=1.5,
        label="FLEXPART prior",
    )

    plt.plot(
        t,
        post_r,
        "-",
        linewidth=1.8,
        label="4D-Var unconstrained",
    )

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
        "FLEXPART-WRF CH$_4$ prediction\n"
        "Prior vs unconstrained 4D-Var"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    filename = (
        PLOT_DIR +
        f"/{receptor.capitalize()}"
        "_4DVar_unconstrained_prediction.png"
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
# COMBINED RECEPTOR PLOT
# =============================================================================

print()
print(
    "CREATING COMBINED RECEPTOR PLOT"
)


plt.figure(
    figsize=(16, 8)
)


receptor_styles = {
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

    plt.plot(
        t,
        obs_r,
        receptor_styles[
            receptor
        ]["marker"],
        markersize=3,
        linestyle="none",
        label=receptor_styles[
            receptor
        ]["label"],
    )


# --------------------------------------------------------------------------
# Prior / posterior are plotted separately for each receptor
# --------------------------------------------------------------------------

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

    post_r = posterior_obs[mask]

    order = np.argsort(t)

    t = t[order]

    prior_r = prior_r[order]

    post_r = post_r[order]

    plt.plot(
        t,
        prior_r,
        "-",
        linewidth=1.2,
        alpha=0.8,
        label=f"{receptor.capitalize()} prior",
    )

    plt.plot(
        t,
        post_r,
        "-",
        linewidth=1.8,
        alpha=0.9,
        label=f"{receptor.capitalize()} 4D-Var",
    )


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
    "Upper Hunter — "
    "FLEXPART-WRF CH$_4$ predictions\n"
    "Prior vs unconstrained 4D-Var"
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
    "unconstrained_prediction.png"
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
# PRINT FINAL SUMMARY
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

print(
    f"Posterior mean factor : "
    f"{s_post.mean():.8f}"
)

print(
    f"Posterior median      : "
    f"{np.median(s_post):.8f}"
)

print(
    f"Posterior min         : "
    f"{s_post.min():.8f}"
)

print(
    f"Posterior max         : "
    f"{s_post.max():.8f}"
)

print()

print(
    "ALL observations:"
)

print(
    f"  Prior RMSE          : "
    f"{m_prior['RMSE']:.6f} ppm"
)

print(
    f"  4D-Var RMSE         : "
    f"{m_post['RMSE']:.6f} ppm"
)

print(
    f"  Prior MAE           : "
    f"{m_prior['MAE']:.6f} ppm"
)

print(
    f"  4D-Var MAE          : "
    f"{m_post['MAE']:.6f} ppm"
)

print(
    f"  Prior Bias          : "
    f"{m_prior['Bias']:.6f} ppm"
)

print(
    f"  4D-Var Bias         : "
    f"{m_post['Bias']:.6f} ppm"
)

print(
    f"  Prior R             : "
    f"{m_prior['R']:.6f}"
)

print(
    f"  4D-Var R            : "
    f"{m_post['R']:.6f}"
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
print("=" * 90)
print(
    "DONE"
)
print("=" * 90)

