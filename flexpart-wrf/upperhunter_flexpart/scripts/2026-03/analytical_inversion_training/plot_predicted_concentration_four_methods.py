#!/usr/bin/env python3

"""
FLEXPART-WRF Upper Hunter CH4
Observed vs Prior vs Unconstrained Analytical vs
Truncated Analytical vs MCMC

March 2026
13 mines x 31 days = 403 scaling factors

Methods:

    Prior:
        scaling = 1 for all parameters

    Linear unconstrained analytical:
        posterior_scaling

    Truncated analytical:
        truncated_posterior_mean

    MCMC:
        FLEXPART_MCMC_daily_factor_matrix_Mar2026.csv

Forward model:

    CH4 = BACKGROUND_PPM + H @ scaling_factors

IMPORTANT:
    H is already emission-weighted using the prior mine emissions.
    Therefore prior emissions must NOT be multiplied into H again.

Outputs:

    inversion_results/predicted_concentration_four_methods/

        predicted_concentrations_four_methods.csv

        predicted_Singleton_four_methods.png
        predicted_Muswellbrook_four_methods.png
        predicted_Merriwa_four_methods.png

        predicted_all_receptors_four_methods.png

        prediction_statistics_four_methods.csv
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

BASE = "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart"

# ------------------------------------------------------------
# H matrix
# ------------------------------------------------------------

H_FILE = os.path.join(
    BASE,
    "H_matrix_flexpart/2026-03/",
    "H_flexpart_0_10m_13mine_Mar2026.csv",
)

# ------------------------------------------------------------
# Picarro observations
# ------------------------------------------------------------

OBS_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/"
    "piccaro_obs/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

# ------------------------------------------------------------
# Analytical posterior
#
# IMPORTANT:
# This one file contains both:
#
#   posterior_scaling
#       = unconstrained Gaussian analytical solution
#
#   truncated_posterior_mean
#       = positive-orthant truncated analytical solution
# ------------------------------------------------------------

ANALYTICAL_FILE = os.path.join(
    BASE,
    "inversion_results/2026-03",
    "truncated_gaussian_analytical_posterior.csv",
)

# ------------------------------------------------------------
# MCMC daily posterior factors
# ------------------------------------------------------------

MCMC_FILE = os.path.join(
    BASE,
    "H_matrix_flexpart/2026-03",
    "MCMC_13mine_daily_Mar2026",
    "FLEXPART_MCMC_daily_factor_matrix_Mar2026.csv",
)

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    BASE,
    "inversion_results",
    "2026-03",
    "predicted_concentration_four_methods",
)

# ------------------------------------------------------------
# Model settings
# ------------------------------------------------------------

BACKGROUND_PPM = 1.85

START_DATE = "2026-03-01"
END_DATE = "2026-03-31"

OBS_COLUMNS = {
    "singleton": "Singleton cleaned",
    "muswellbrook": "Muswellbrook cleaned",
    "merriwa": "Merriwa cleaned",
}


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("FLEXPART-WRF UPPER HUNTER CH4")
print("OBSERVATION vs PRIOR vs ANALYTICAL vs TRUNCATED vs MCMC")
print("=" * 90)


# ============================================================
# READ H MATRIX
# ============================================================

print()
print("=" * 90)
print("1. READING H MATRIX")
print("=" * 90)

H_df = pd.read_csv(H_FILE)

print("H file:")
print(H_FILE)

print()
print("H shape:", H_df.shape)


H_metadata_columns = [
    "receptor",
    "release_utc",
    "obs_time_aest",
    "flexpart_file",
]


missing_metadata = [
    c
    for c in H_metadata_columns
    if c not in H_df.columns
]

if missing_metadata:

    raise ValueError(
        "Missing H metadata columns: "
        + str(missing_metadata)
    )


# ------------------------------------------------------------
# H parameter columns
# ------------------------------------------------------------

parameter_columns = [
    c
    for c in H_df.columns
    if c not in H_metadata_columns
]

print(
    "Number of H parameters:",
    len(parameter_columns),
)


if len(parameter_columns) != 403:

    raise ValueError(
        f"Expected 403 H parameters, "
        f"found {len(parameter_columns)}"
    )


H = H_df[
    parameter_columns
].to_numpy(
    dtype=float
)

print(
    "Numerical H shape:",
    H.shape
)


# ============================================================
# CONVERT H TIMES
# ============================================================

H_df["obs_time_aest"] = pd.to_datetime(
    H_df["obs_time_aest"],
    errors="coerce",
)

if H_df["obs_time_aest"].isna().any():

    nbad = H_df["obs_time_aest"].isna().sum()

    raise ValueError(
        f"{nbad} invalid H obs_time_aest values."
    )


# ============================================================
# READ ANALYTICAL RESULTS
# ============================================================

print()
print("=" * 90)
print("2. READING ANALYTICAL POSTERIOR")
print("=" * 90)

analytical = pd.read_csv(
    ANALYTICAL_FILE
)

print("Analytical file:")
print(ANALYTICAL_FILE)

print()
print(
    "Analytical shape:",
    analytical.shape
)

print()
print(
    "Analytical columns:"
)

print(
    list(analytical.columns)
)


required_analytical_columns = [
    "parameter",
    "mine",
    "date",
    "posterior_scaling",
    "posterior_sigma",
    "truncated_posterior_mean",
    "truncated_posterior_sd",
]


missing = [
    c
    for c in required_analytical_columns
    if c not in analytical.columns
]

if missing:

    raise ValueError(
        "Missing analytical columns: "
        + str(missing)
    )


# ------------------------------------------------------------
# Make parameter index
# ------------------------------------------------------------

analytical_index = analytical.set_index(
    "parameter"
)


# ------------------------------------------------------------
# Verify exact parameter set
# ------------------------------------------------------------

analytical_parameters = set(
    analytical["parameter"]
)

H_parameters = set(
    parameter_columns
)


if analytical_parameters != H_parameters:

    missing_from_analytical = sorted(
        H_parameters - analytical_parameters
    )

    extra_in_analytical = sorted(
        analytical_parameters - H_parameters
    )

    print()
    print(
        "ERROR: analytical and H parameter sets differ."
    )

    print(
        "Missing from analytical:",
        missing_from_analytical[:20]
    )

    print(
        "Extra in analytical:",
        extra_in_analytical[:20]
    )

    raise ValueError(
        "Analytical parameter set does not "
        "match H parameter set."
    )


print()
print(
    "Analytical parameter set exactly matches H."
)


# ============================================================
# BUILD ANALYTICAL VECTORS
# ============================================================

print()
print(
    "Building analytical scaling vectors..."
)


# ------------------------------------------------------------
# Unconstrained linear Gaussian analytical solution
# ------------------------------------------------------------

unconstrained_scaling = np.array(
    [
        float(
            analytical_index.loc[
                p,
                "posterior_scaling"
            ]
        )
        for p in parameter_columns
    ]
)


# ------------------------------------------------------------
# Positive-truncated analytical solution
# ------------------------------------------------------------

truncated_scaling = np.array(
    [
        float(
            analytical_index.loc[
                p,
                "truncated_posterior_mean"
            ]
        )
        for p in parameter_columns
    ]
)


print()
print("Unconstrained analytical scaling:")
print(
    "  min =",
    unconstrained_scaling.min()
)

print(
    "  max =",
    unconstrained_scaling.max()
)

print(
    "  mean =",
    unconstrained_scaling.mean()
)


print()
print("Truncated analytical scaling:")
print(
    "  min =",
    truncated_scaling.min()
)

print(
    "  max =",
    truncated_scaling.max()
)

print(
    "  mean =",
    truncated_scaling.mean()
)


# ============================================================
# READ MCMC RESULTS
# ============================================================

print()
print("=" * 90)
print("3. READING MCMC POSTERIOR DAILY FACTORS")
print("=" * 90)

mcmc = pd.read_csv(
    MCMC_FILE
)

print("MCMC file:")
print(MCMC_FILE)

print()
print(
    "MCMC shape:",
    mcmc.shape
)

if "date" not in mcmc.columns:

    raise ValueError(
        "MCMC factor matrix has no date column."
    )


mcmc["date"] = pd.to_datetime(
    mcmc["date"],
    errors="coerce"
)


if mcmc["date"].isna().any():

    raise ValueError(
        "Invalid dates found in MCMC file."
    )


# ------------------------------------------------------------
# Restrict to March 1-31
# ------------------------------------------------------------

mcmc = mcmc[
    (mcmc["date"] >= pd.Timestamp(START_DATE))
    &
    (mcmc["date"] <= pd.Timestamp(END_DATE))
].copy()


# ============================================================
# BUILD MCMC 403-PARAMETER VECTOR
# ============================================================

print()
print(
    "Building MCMC scaling vector..."
)


mcmc_scaling = np.ones(
    len(parameter_columns),
    dtype=float
)


missing_mcmc = []


for j, parameter in enumerate(
    parameter_columns
):

    # Example:
    #
    # OC1_20260105
    #
    parts = parameter.split("_")

    if len(parts) != 2:

        raise ValueError(
            f"Unexpected parameter name: {parameter}"
        )


    mine = parts[0]

    date = pd.to_datetime(
        parts[1],
        format="%Y%m%d"
    )


    rows = mcmc[
        mcmc["date"] == date
    ]


    if len(rows) != 1:

        missing_mcmc.append(
            parameter
        )

        continue


    if mine not in rows.columns:

        missing_mcmc.append(
            parameter
        )

        continue


    mcmc_scaling[j] = float(
        rows.iloc[0][mine]
    )


if missing_mcmc:

    print()
    print(
        "Missing MCMC factors:",
        len(missing_mcmc)
    )

    print(
        missing_mcmc[:20]
    )

    raise ValueError(
        "MCMC factor matrix does not contain "
        "all 403 mine/day factors."
    )


print()
print(
    "MCMC scaling vector:",
    mcmc_scaling.shape
)

print(
    "MCMC range:",
    mcmc_scaling.min(),
    "to",
    mcmc_scaling.max()
)


# ============================================================
# PRIOR VECTOR
# ============================================================

prior_scaling = np.ones(
    len(parameter_columns),
    dtype=float
)


# ============================================================
# FORWARD MODEL
# ============================================================

print()
print("=" * 90)
print("4. FORWARD MODEL")
print("=" * 90)

print()
print(
    "CH4 = 1.85 + H @ scaling"
)


# ------------------------------------------------------------
# Prior
# ------------------------------------------------------------

prior_prediction = (
    BACKGROUND_PPM
    +
    H @ prior_scaling
)


# ------------------------------------------------------------
# Unconstrained analytical
# ------------------------------------------------------------

unconstrained_prediction = (
    BACKGROUND_PPM
    +
    H @ unconstrained_scaling
)


# ------------------------------------------------------------
# Truncated analytical
# ------------------------------------------------------------

truncated_prediction = (
    BACKGROUND_PPM
    +
    H @ truncated_scaling
)


# ------------------------------------------------------------
# MCMC
# ------------------------------------------------------------

mcmc_prediction = (
    BACKGROUND_PPM
    +
    H @ mcmc_scaling
)


# ------------------------------------------------------------
# Print ranges
# ------------------------------------------------------------

methods = {
    "Prior":
        prior_prediction,

    "Unconstrained analytical":
        unconstrained_prediction,

    "Truncated analytical":
        truncated_prediction,

    "MCMC":
        mcmc_prediction,
}


for name, prediction in methods.items():

    print()
    print(name)

    print(
        "  min:",
        np.nanmin(prediction),
        "ppm"
    )

    print(
        "  max:",
        np.nanmax(prediction),
        "ppm"
    )

    print(
        "  mean:",
        np.nanmean(prediction),
        "ppm"
    )


# ============================================================
# READ PICARRO
# ============================================================

print()
print("=" * 90)
print("5. READING PICARRO OBSERVATIONS")
print("=" * 90)

obs = pd.read_csv(
    OBS_FILE
)

print(
    "Observation shape:",
    obs.shape
)


# ------------------------------------------------------------
# Determine datetime field
# ------------------------------------------------------------

datetime_candidates = [
    "Datetime",
    "datetime",
    "DateTime",
    "datetime_aest",
]


datetime_column = None

for col in datetime_candidates:

    if col in obs.columns:

        datetime_column = col

        break


if datetime_column is not None:

    print(
        "Using datetime column:",
        datetime_column
    )

    obs["datetime_aest"] = pd.to_datetime(
        obs[datetime_column],
        format="%d/%m/%Y %H:%M",
        errors="coerce"
    )

else:

    if (
        "Date" not in obs.columns
        or
        "Time" not in obs.columns
    ):

        raise ValueError(
            "Could not find observation datetime."
        )


    print(
        "Using Date + Time columns."
    )

    obs["datetime_aest"] = pd.to_datetime(
        obs["Date"].astype(str)
        + " "
        + obs["Time"].astype(str),
        format="%d/%m/%Y %H:%M",
        errors="coerce"
    )


nbad = obs[
    "datetime_aest"
].isna().sum()


print(
    "Invalid observation datetimes:",
    nbad
)


if nbad > 0:

    raise ValueError(
        "Observation datetime parsing failed."
    )


# ============================================================
# MATCH H ROWS WITH OBSERVATIONS
# ============================================================

print()
print("=" * 90)
print("6. MATCHING OBSERVATIONS")
print("=" * 90)


records = []


receptors = [
    "singleton",
    "muswellbrook",
    "merriwa",
]


for receptor in receptors:

    print()
    print("-" * 70)
    print(
        "RECEPTOR:",
        receptor.upper()
    )
    print("-" * 70)


    # --------------------------------------------------------
    # H receptor selection
    # --------------------------------------------------------

    mask = (
        H_df["receptor"]
        .astype(str)
        .str.lower()
        .str.contains(
            receptor
        )
    )


    indices = np.where(
        mask.to_numpy()
    )[0]


    print(
        "H rows:",
        len(indices)
    )


    # --------------------------------------------------------
    # Observation column
    # --------------------------------------------------------

    obs_column = OBS_COLUMNS[
        receptor
    ]


    if obs_column not in obs.columns:

        print(
            "WARNING: observation column missing:",
            obs_column
        )

        continue


    obs_lookup = (
        obs[
            [
                "datetime_aest",
                obs_column,
            ]
        ]
        .dropna(
            subset=[obs_column]
        )
        .drop_duplicates(
            subset=["datetime_aest"]
        )
        .set_index(
            "datetime_aest"
        )[obs_column]
    )


    matched = 0
    missing = 0


    # --------------------------------------------------------
    # Build records
    # --------------------------------------------------------

    for idx in indices:

        time = H_df.loc[
            idx,
            "obs_time_aest"
        ]


        if time in obs_lookup.index:

            observation = float(
                obs_lookup.loc[time]
            )

            matched += 1

        else:

            observation = np.nan

            missing += 1


        records.append(
            {
                "receptor":
                    receptor,

                "datetime_aest":
                    time,

                "observation_ppm":
                    observation,

                "prior_prediction_ppm":
                    prior_prediction[idx],

                "unconstrained_analytical_prediction_ppm":
                    unconstrained_prediction[idx],

                "truncated_analytical_prediction_ppm":
                    truncated_prediction[idx],

                "mcmc_prediction_ppm":
                    mcmc_prediction[idx],
            }
        )


    print(
        "Matched:",
        matched
    )

    print(
        "Missing:",
        missing
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(
    records
)


results = results.sort_values(
    [
        "receptor",
        "datetime_aest",
    ]
).reset_index(
    drop=True
)


# ------------------------------------------------------------
# Restrict period
# ------------------------------------------------------------

results = results[
    (
        results["datetime_aest"]
        >=
        pd.Timestamp(START_DATE)
    )
    &
    (
        results["datetime_aest"]
        <
        pd.Timestamp(END_DATE)
        +
        pd.Timedelta(days=1)
    )
].copy()


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_csv = os.path.join(
    OUTPUT_DIR,
    "predicted_concentrations_four_methods.csv"
)


results.to_csv(
    prediction_csv,
    index=False
)


print()
print("=" * 90)
print("7. SAVED PREDICTION DATA")
print("=" * 90)

print(
    prediction_csv
)


# ============================================================
# CALCULATE STATISTICS
# ============================================================

print()
print("=" * 90)
print("8. PREDICTION STATISTICS")
print("=" * 90)


prediction_columns = {

    "prior":
        "prior_prediction_ppm",

    "unconstrained_analytical":
        "unconstrained_analytical_prediction_ppm",

    "truncated_analytical":
        "truncated_analytical_prediction_ppm",

    "mcmc":
        "mcmc_prediction_ppm",
}


statistics = []


for receptor in receptors:

    tmp = results[
        results["receptor"]
        ==
        receptor
    ].copy()


    valid = tmp.dropna(
        subset=["observation_ppm"]
    )


    print()
    print(
        receptor.upper()
    )

    print(
        "H rows:",
        len(tmp)
    )

    print(
        "Observed:",
        len(valid)
    )


    if len(valid) == 0:

        continue


    y = valid[
        "observation_ppm"
    ].to_numpy()


    for method, column in (
        prediction_columns.items()
    ):

        prediction = valid[
            column
        ].to_numpy()


        residual = (
            y - prediction
        )


        bias = np.mean(
            residual
        )


        mae = np.mean(
            np.abs(residual)
        )


        rmse = np.sqrt(
            np.mean(
                residual ** 2
            )
        )


        # Correlation
        if (
            np.std(y) > 0
            and
            np.std(prediction) > 0
        ):

            correlation = np.corrcoef(
                y,
                prediction
            )[0, 1]

        else:

            correlation = np.nan


        statistics.append(
            {
                "receptor":
                    receptor,

                "method":
                    method,

                "n_observations":
                    len(valid),

                "bias_obs_minus_pred_ppm":
                    bias,

                "MAE_ppm":
                    mae,

                "RMSE_ppm":
                    rmse,

                "correlation":
                    correlation,
            }
        )


        print()
        print(
            f"{method}:"
        )

        print(
            "  Bias:",
            f"{bias:.6f}",
            "ppm"
        )

        print(
            "  MAE:",
            f"{mae:.6f}",
            "ppm"
        )

        print(
            "  RMSE:",
            f"{rmse:.6f}",
            "ppm"
        )

        print(
            "  R:",
            f"{correlation:.6f}"
        )


statistics_df = pd.DataFrame(
    statistics
)


statistics_csv = os.path.join(
    OUTPUT_DIR,
    "prediction_statistics_four_methods.csv"
)


statistics_df.to_csv(
    statistics_csv,
    index=False
)


print()
print(
    "Saved statistics:"
)

print(
    statistics_csv
)


# ============================================================
# INDIVIDUAL RECEPTOR PLOTS
# ============================================================

print()
print("=" * 90)
print("9. CREATING INDIVIDUAL RECEPTOR PLOTS")
print("=" * 90)


for receptor in receptors:

    tmp = results[
        results["receptor"]
        ==
        receptor
    ].copy()


    if len(tmp) == 0:

        continue


    fig, ax = plt.subplots(
        figsize=(15, 6)
    )


    # --------------------------------------------------------
    # Observation
    # --------------------------------------------------------

    observed = tmp.dropna(
        subset=[
            "observation_ppm"
        ]
    )


    if len(observed) > 0:

        ax.plot(
            observed[
                "datetime_aest"
            ],
            observed[
                "observation_ppm"
            ],
            linestyle="none",
            marker="o",
            markersize=3.5,
            label="Picarro observation",
        )


    # --------------------------------------------------------
    # Prior
    # --------------------------------------------------------

    ax.plot(
        tmp["datetime_aest"],
        tmp[
            "prior_prediction_ppm"
        ],
        linewidth=1.5,
        linestyle="--",
        label="Prior",
    )


    # --------------------------------------------------------
    # Unconstrained analytical
    # --------------------------------------------------------

    ax.plot(
        tmp["datetime_aest"],
        tmp[
            "unconstrained_analytical_prediction_ppm"
        ],
        linewidth=1.7,
        label="Linear analytical",
    )


    # --------------------------------------------------------
    # Truncated analytical
    # --------------------------------------------------------

    ax.plot(
        tmp["datetime_aest"],
        tmp[
            "truncated_analytical_prediction_ppm"
        ],
        linewidth=1.8,
        label="Truncated analytical",
    )


    # --------------------------------------------------------
    # MCMC
    # --------------------------------------------------------

    ax.plot(
        tmp["datetime_aest"],
        tmp[
            "mcmc_prediction_ppm"
        ],
        linewidth=2.0,
        linestyle=":",
        label="MCMC",
    )


    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    ax.axhline(
        BACKGROUND_PPM,
        linestyle="-.",
        linewidth=1.1,
        label="Background = 1.85 ppm",
    )


    ax.set_title(
        receptor.capitalize()
        +
        " — Upper Hunter CH₄"
    )


    ax.set_xlabel(
        "Time (AEST)"
    )


    ax.set_ylabel(
        "CH₄ concentration (ppm)"
    )


    ax.grid(
        True,
        alpha=0.25
    )


    ax.legend(
        loc="best"
    )


    fig.autofmt_xdate()


    fig.tight_layout()


    output_png = os.path.join(
        OUTPUT_DIR,
        f"predicted_{receptor}_four_methods.png"
    )


    fig.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()
    plt.close(fig)


    print(
        "Saved:",
        output_png
    )


# ============================================================
# COMBINED THREE-RECEPTOR PLOT
# ============================================================

print()
print("=" * 90)
print("10. CREATING COMBINED THREE-RECEPTOR PLOT")
print("=" * 90)


fig, axes = plt.subplots(
    3,
    1,
    figsize=(16, 12),
    sharex=True
)


for ax, receptor in zip(
    axes,
    receptors
):

    tmp = results[
        results["receptor"]
        ==
        receptor
    ].copy()


    if len(tmp) == 0:

        ax.text(
            0.5,
            0.5,
            "No H-matrix data",
            transform=ax.transAxes,
            ha="center",
            va="center"
        )

        continue


    # --------------------------------------------------------
    # Observation
    # --------------------------------------------------------

    observed = tmp.dropna(
        subset=[
            "observation_ppm"
        ]
    )


    if len(observed) > 0:

        ax.plot(
            observed[
                "datetime_aest"
            ],
            observed[
                "observation_ppm"
            ],
            linestyle="none",
            marker="o",
            markersize=3,
            label="Picarro observation"
        )


    # --------------------------------------------------------
    # Prior
    # --------------------------------------------------------

    ax.plot(
        tmp["datetime_aest"],
        tmp[
            "prior_prediction_ppm"
        ],
        linestyle="--",
        linewidth=1.5,
        label="Prior"
    )


    # --------------------------------------------------------
    # Unconstrained analytical
    # --------------------------------------------------------

    ax.plot(
        tmp["datetime_aest"],
        tmp[
            "unconstrained_analytical_prediction_ppm"
        ],
        linewidth=1.6,
        label="Linear analytical"
    )


    # --------------------------------------------------------
    # Truncated analytical
    # --------------------------------------------------------

    ax.plot(
        tmp["datetime_aest"],
        tmp[
            "truncated_analytical_prediction_ppm"
        ],
        linewidth=1.8,
        label="Truncated analytical"
    )


    # --------------------------------------------------------
    # MCMC
    # --------------------------------------------------------

    ax.plot(
        tmp["datetime_aest"],
        tmp[
            "mcmc_prediction_ppm"
        ],
        linestyle=":",
        linewidth=2.0,
        label="MCMC"
    )


    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    ax.axhline(
        BACKGROUND_PPM,
        linestyle="-.",
        linewidth=1.0,
        label="Background"
    )


    ax.set_title(
        receptor.capitalize()
    )


    ax.set_ylabel(
        "CH₄ (ppm)"
    )


    ax.grid(
        True,
        alpha=0.25
    )


    ax.legend(
        loc="best",
        fontsize=9
    )


axes[-1].set_xlabel(
    "Time (AEST)"
)


fig.suptitle(
    "Upper Hunter CH₄ — Observation vs prior, "
    "linear analytical, truncated analytical and MCMC",
    fontsize=15,
    y=0.995
)


fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.98
    ]
)


combined_png = os.path.join(
    OUTPUT_DIR,
    "predicted_all_receptors_four_methods.png"
)


fig.savefig(
    combined_png,
    dpi=300,
    bbox_inches="tight"
)


plt.show()
plt.close(fig)


print()
print(
    "Saved:"
)

print(
    combined_png
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 90)
print("DONE")
print("=" * 90)

print()
print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Prediction CSV:"
)

print(
    "  predicted_concentrations_four_methods.csv"
)

print()
print(
    "Statistics CSV:"
)

print(
    "  prediction_statistics_four_methods.csv"
)

print()
print(
    "Figures:"
)

print(
    "  predicted_Singleton_four_methods.png"
)

print(
    "  predicted_Muswellbrook_four_methods.png"
)

print(
    "  predicted_Merriwa_four_methods.png"
)

print(
    "  predicted_all_receptors_four_methods.png"
)

print()
print("=" * 90)

