#!/usr/bin/env python3

"""
Upper Hunter CH4:
Observation vs Prior vs Analytical Linear Unconstrained
vs Analytical Linear Constrained

Three receptor plots:
    Singleton
    Muswellbrook
    Merriwa

Observation is plotted as a continuous line.

Methods:
    1. Prior
    2. Analytical linear unconstrained
    3. Analytical linear constrained
       (positive-orthant truncated Gaussian; essentially same as MCMC)

No MCMC curve is plotted because the constrained analytical solution
has already been independently validated against MCMC.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# PATHS
# ============================================================================

BASE = "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart"

H_FILE = os.path.join(
    BASE,
    "H_matrix_flexpart",
    "2026-02",
    "H_flexpart_0_10m_13mine_Feb2026.csv",
)

PICARRO_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/"
    "piccaro_obs/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

ANALYTICAL_FILE = os.path.join(
    BASE,
    "inversion_results",
    "2026-02",
    "truncated_gaussian_analytical_posterior.csv",
)

OUTPUT_DIR = os.path.join(
    BASE,
    "inversion_results",
    "2026-02",
    "predicted_concentration_three_methods",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# SETTINGS
# ============================================================================

BACKGROUND_PPM = 1.85

START_DATE = pd.Timestamp("2026-02-01")
END_DATE = pd.Timestamp("2026-02-28")

RECEPTORS = {
    "Singleton": "Singleton cleaned",
    "Muswellbrook": "Muswellbrook cleaned",
    "Merriwa": "Merriwa cleaned",
}


# ============================================================================
# READ H MATRIX
# ============================================================================

print("=" * 90)
print("READING FLEXPART H MATRIX")
print("=" * 90)

H_df = pd.read_csv(H_FILE)

print("H shape:", H_df.shape)

# --------------------------------------------------------------------------
# Parameter columns are all columns beginning with OC
# --------------------------------------------------------------------------

parameter_columns = [
    c for c in H_df.columns
    if c.startswith("OC")
]

metadata_columns = [
    c for c in H_df.columns
    if c not in parameter_columns
]

print("Metadata columns:")
print(metadata_columns)

print("\nNumber of parameters:", len(parameter_columns))

if len(parameter_columns) != 364:
    raise ValueError(
        f"Expected 364 parameters, found {len(parameter_columns)}"
    )

# --------------------------------------------------------------------------
# H matrix
# --------------------------------------------------------------------------

H = H_df[parameter_columns].to_numpy(dtype=float)

print("H matrix shape:", H.shape)


# ============================================================================
# READ PICARRO
# ============================================================================

print("\n" + "=" * 90)
print("READING PICARRO OBSERVATIONS")
print("=" * 90)

obs = pd.read_csv(PICARRO_FILE)

print("Observation shape:", obs.shape)

# --------------------------------------------------------------------------
# Parse the actual datetime column
# --------------------------------------------------------------------------

if "Datetime" in obs.columns:

    obs["datetime_aest"] = pd.to_datetime(
        obs["Datetime"],
        errors="coerce",
    )

else:

    # Fallback if Datetime is not available
    time_column = "Date"

    if "Time" in obs.columns:
        obs["datetime_aest"] = pd.to_datetime(
            obs["Date"].astype(str)
            + " "
            + obs["Time"].astype(str),
            format="%d/%m/%Y %H:%M",
            errors="coerce",
        )

    else:
        obs["datetime_aest"] = pd.to_datetime(
            obs[time_column],
            format="%d/%m/%Y %H:%M",
            errors="coerce",
        )

print(
    "Invalid observation datetimes:",
    obs["datetime_aest"].isna().sum(),
)


# ============================================================================
# FILTER OBSERVATIONS TO JAN 5-31
# ============================================================================

obs = obs[
    (obs["datetime_aest"] >= START_DATE)
    & (
        obs["datetime_aest"]
        < END_DATE + pd.Timedelta(days=1)
    )
].copy()

print(
    "February observation rows:",
    len(obs),
)


# ============================================================================
# MATCH H ROWS TO OBSERVATION TIMES
# ============================================================================

print("\n" + "=" * 90)
print("MATCHING H MATRIX TIMES TO PICARRO")
print("=" * 90)

H_df["obs_time_aest"] = pd.to_datetime(
    H_df["obs_time_aest"],
    errors="coerce",
)

# Keep only Feb 1-28
H_df = H_df[
    (H_df["obs_time_aest"] >= START_DATE)
    & (
        H_df["obs_time_aest"]
        < END_DATE + pd.Timedelta(days=1)
    )
].copy()

print("H rows in February:", len(H_df))

# Rebuild H after filtering
H = H_df[parameter_columns].to_numpy(dtype=float)


# ============================================================================
# READ ANALYTICAL POSTERIOR
# ============================================================================

print("\n" + "=" * 90)
print("READING ANALYTICAL POSTERIOR")
print("=" * 90)

analytical = pd.read_csv(ANALYTICAL_FILE)

print("Analytical shape:", analytical.shape)
print("Analytical columns:")
print(list(analytical.columns))


# ============================================================================
# CHECK ANALYTICAL PARAMETER ORDER
# ============================================================================

analytical = analytical.set_index("parameter")

missing = [
    p for p in parameter_columns
    if p not in analytical.index
]

if missing:
    raise ValueError(
        "Analytical posterior is missing parameters:\n"
        + "\n".join(missing)
    )

analytical = analytical.loc[parameter_columns]


# ============================================================================
# THREE SCALING FACTOR SETS
# ============================================================================

# --------------------------------------------------------------------------
# 1. Prior
# --------------------------------------------------------------------------

prior_scaling = np.ones(len(parameter_columns))


# --------------------------------------------------------------------------
# 2. Linear analytical unconstrained
#
# This is the ordinary Gaussian analytical posterior:
#
#     posterior_scaling
#
# It allows negative scaling factors.
# --------------------------------------------------------------------------

unconstrained_scaling = (
    analytical["posterior_scaling"]
    .to_numpy(dtype=float)
)


# --------------------------------------------------------------------------
# 3. Linear analytical constrained
#
# This is the positive-orthant truncated Gaussian posterior:
#
#     truncated_posterior_mean
#
# It is the analytical equivalent that matches the MCMC constraint:
#
#     scaling factor >= 0
# --------------------------------------------------------------------------

constrained_scaling = (
    analytical["truncated_posterior_mean"]
    .to_numpy(dtype=float)
)


# ============================================================================
# PREDICT CONCENTRATION
# ============================================================================

print("\n" + "=" * 90)
print("CALCULATING PREDICTED CH4")
print("=" * 90)

prior_prediction = (
    BACKGROUND_PPM
    + H @ prior_scaling
)

unconstrained_prediction = (
    BACKGROUND_PPM
    + H @ unconstrained_scaling
)

constrained_prediction = (
    BACKGROUND_PPM
    + H @ constrained_scaling
)

print(
    "Prior prediction range:",
    prior_prediction.min(),
    prior_prediction.max(),
)

print(
    "Unconstrained prediction range:",
    unconstrained_prediction.min(),
    unconstrained_prediction.max(),
)

print(
    "Constrained prediction range:",
    constrained_prediction.min(),
    constrained_prediction.max(),
)


# ============================================================================
# BUILD LONG OUTPUT TABLE
# ============================================================================

prediction_rows = []

for i, row in H_df.reset_index(drop=True).iterrows():

    receptor = row["receptor"]
    datetime_aest = row["obs_time_aest"]

    prediction_rows.append(
        {
            "receptor": receptor,
            "datetime_aest": datetime_aest,
            "prior_ppm": prior_prediction[i],
            "analytical_unconstrained_ppm":
                unconstrained_prediction[i],
            "analytical_constrained_ppm":
                constrained_prediction[i],
        }
    )

predictions = pd.DataFrame(prediction_rows)


# ============================================================================
# ADD OBSERVATIONS
# ============================================================================

print("\n" + "=" * 90)
print("MATCHING PICARRO OBSERVATIONS TO FLEXPART RECEPTORS")
print("=" * 90)

# --------------------------------------------------------------------------
# Standardise receptor names
# --------------------------------------------------------------------------

H_df["receptor_std"] = (
    H_df["receptor"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# --------------------------------------------------------------------------
# Build observation table
# --------------------------------------------------------------------------

observation_records = []

for receptor_name, obs_column in RECEPTORS.items():

    tmp = obs[
        [
            "datetime_aest",
            obs_column,
        ]
    ].copy()

    tmp = tmp.rename(
        columns={
            obs_column: "observation_ppm"
        }
    )

    tmp["receptor_std"] = (
        receptor_name
        .strip()
        .lower()
    )

    observation_records.append(tmp)


observations = pd.concat(
    observation_records,
    ignore_index=True,
)

# Make absolutely sure the datetime type is identical
observations["datetime_aest"] = pd.to_datetime(
    observations["datetime_aest"],
    errors="coerce",
)

H_df["obs_time_aest"] = pd.to_datetime(
    H_df["obs_time_aest"],
    errors="coerce",
)

# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

print("\nH receptor names:")
print(
    H_df["receptor_std"]
    .value_counts()
)

print("\nObservation receptor names:")
print(
    observations["receptor_std"]
    .value_counts()
)

print("\nExample H times:")
print(
    H_df[
        ["receptor_std", "obs_time_aest"]
    ].head(10).to_string(index=False)
)

print("\nExample observation times:")
print(
    observations[
        ["receptor_std", "datetime_aest", "observation_ppm"]
    ].head(10).to_string(index=False)
)

# --------------------------------------------------------------------------
# Merge using standardised receptor + datetime
# --------------------------------------------------------------------------

predictions["receptor_std"] = (
    predictions["receptor"]
    .astype(str)
    .str.strip()
    .str.lower()
)

results = predictions.merge(
    observations[
        [
            "receptor_std",
            "datetime_aest",
            "observation_ppm",
        ]
    ],
    left_on=[
        "receptor_std",
        "datetime_aest",
    ],
    right_on=[
        "receptor_std",
        "datetime_aest",
    ],
    how="left",
)

# --------------------------------------------------------------------------
# Restore clean receptor name
# --------------------------------------------------------------------------

results["receptor"] = (
    results["receptor_std"]
    .map(
        {
            "singleton": "Singleton",
            "muswellbrook": "Muswellbrook",
            "merriwa": "Merriwa",
        }
    )
    .fillna(results["receptor_std"].str.title())
)

results = results.sort_values(
    [
        "receptor",
        "datetime_aest",
    ]
).reset_index(drop=True)

# --------------------------------------------------------------------------
# Matching diagnostics
# --------------------------------------------------------------------------

print("\n" + "=" * 90)
print("OBSERVATION MATCHING RESULTS")
print("=" * 90)

for receptor in RECEPTORS:

    tmp = results[
        results["receptor"] == receptor
    ]

    n_h = len(tmp)

    n_obs = tmp["observation_ppm"].notna().sum()

    print(
        f"{receptor:15s}: "
        f"H rows = {n_h:4d}, "
        f"matched observations = {n_obs:4d}"
    )


# ============================================================================
# SORT
# ============================================================================

results = results.sort_values(
    [
        "receptor",
        "datetime_aest",
    ]
).reset_index(drop=True)


# ============================================================================
# SAVE DATA
# ============================================================================

prediction_csv = os.path.join(
    OUTPUT_DIR,
    "predicted_concentrations_three_methods.csv",
)

results.to_csv(
    prediction_csv,
    index=False,
)

print("\nSaved:")
print(prediction_csv)


# ============================================================================
# STATISTICS
# ============================================================================

print("\n" + "=" * 90)
print("PREDICTION STATISTICS")
print("=" * 90)

statistics = []

method_columns = {
    "Prior": "prior_ppm",
    "Analytical linear unconstrained":
        "analytical_unconstrained_ppm",
    "Analytical linear constrained":
        "analytical_constrained_ppm",
}

for receptor in RECEPTORS:

    tmp = results[
        results["receptor"] == receptor
    ].copy()

    # Only calculate statistics where observations exist
    tmp_obs = tmp.dropna(
        subset=["observation_ppm"]
    )

    print(
        f"\n{receptor}: "
        f"{len(tmp_obs)} matched observations"
    )

    for method, column in method_columns.items():

        if len(tmp_obs) == 0:

            bias = np.nan
            mae = np.nan
            rmse = np.nan
            correlation = np.nan

        else:

            obs_values = (
                tmp_obs["observation_ppm"]
                .to_numpy()
            )

            pred_values = (
                tmp_obs[column]
                .to_numpy()
            )

            residual = (
                pred_values
                - obs_values
            )

            bias = np.mean(residual)

            mae = np.mean(
                np.abs(residual)
            )

            rmse = np.sqrt(
                np.mean(residual ** 2)
            )

            if (
                len(obs_values) > 1
                and np.std(obs_values) > 0
                and np.std(pred_values) > 0
            ):

                correlation = np.corrcoef(
                    obs_values,
                    pred_values,
                )[0, 1]

            else:

                correlation = np.nan

        statistics.append(
            {
                "receptor": receptor,
                "method": method,
                "bias_ppm": bias,
                "MAE_ppm": mae,
                "RMSE_ppm": rmse,
                "correlation": correlation,
                "n_observations": len(tmp_obs),
            }
        )

        print(
            f"  {method:35s}"
            f" Bias={bias:8.4f}"
            f" MAE={mae:8.4f}"
            f" RMSE={rmse:8.4f}"
            f" R={correlation:8.4f}"
        )


statistics_df = pd.DataFrame(statistics)

statistics_csv = os.path.join(
    OUTPUT_DIR,
    "prediction_statistics_three_methods.csv",
)

statistics_df.to_csv(
    statistics_csv,
    index=False,
)

print("\nSaved:")
print(statistics_csv)


# ============================================================================
# PLOT SETTINGS
# ============================================================================

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    }
)


# ============================================================================
# INDIVIDUAL RECEPTOR PLOTS
# ============================================================================

print("\n" + "=" * 90)
print("CREATING RECEPTOR PLOTS")
print("=" * 90)


for receptor in RECEPTORS:

    tmp = results[
        results["receptor"] == receptor
    ].copy()

    tmp = tmp.sort_values(
        "datetime_aest"
    )

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    # ------------------------------------------------------------
    # Observation
    # Continuous line rather than points
    # ------------------------------------------------------------

    ax.plot(
        tmp["datetime_aest"],
        tmp["observation_ppm"],
        linewidth=1.2,
        label="Observation",
    )

    # ------------------------------------------------------------
    # Prior
    # ------------------------------------------------------------

    ax.plot(
        tmp["datetime_aest"],
        tmp["prior_ppm"],
        linestyle="--",
        linewidth=1.8,
        label="Prior",
    )

    # ------------------------------------------------------------
    # Analytical unconstrained
    # ------------------------------------------------------------

    ax.plot(
        tmp["datetime_aest"],
        tmp["analytical_unconstrained_ppm"],
        linewidth=1.8,
        label="Analytical linear — unconstrained",
    )

    # ------------------------------------------------------------
    # Analytical constrained
    # ------------------------------------------------------------

    ax.plot(
        tmp["datetime_aest"],
        tmp["analytical_constrained_ppm"],
        linewidth=1.8,
        label="Analytical linear — constrained",
    )

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------

    ax.set_title(
        f"{receptor} — Upper Hunter CH₄"
    )

    ax.set_xlabel(
        "Date / time (AEST)"
    )

    ax.set_ylabel(
        "CH₄ concentration (ppm)"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        loc="best",
        frameon=True,
    )

    fig.autofmt_xdate()

    plt.tight_layout()

    filename = os.path.join(
        OUTPUT_DIR,
        f"predicted_{receptor}_three_methods.png",
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(filename)


# ============================================================================
# COMBINED THREE-RECEPTOR FIGURE
# ============================================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(15, 13),
    sharex=True,
)

for ax, receptor in zip(
    axes,
    RECEPTORS,
):

    tmp = results[
        results["receptor"] == receptor
    ].copy()

    tmp = tmp.sort_values(
        "datetime_aest"
    )

    # Observation
    ax.plot(
        tmp["datetime_aest"],
        tmp["observation_ppm"],
        linewidth=1.1,
        label="Observation",
    )

    # Prior
    ax.plot(
        tmp["datetime_aest"],
        tmp["prior_ppm"],
        linestyle="--",
        linewidth=1.6,
        label="Prior",
    )

    # Unconstrained analytical
    ax.plot(
        tmp["datetime_aest"],
        tmp["analytical_unconstrained_ppm"],
        linewidth=1.6,
        label="Analytical linear — unconstrained",
    )

    # Constrained analytical
    ax.plot(
        tmp["datetime_aest"],
        tmp["analytical_constrained_ppm"],
        linewidth=1.6,
        label="Analytical linear — constrained",
    )

    ax.set_title(
        receptor,
        loc="left",
        fontweight="bold",
    )

    ax.set_ylabel(
        "CH₄ (ppm)"
    )

    ax.grid(
        True,
        alpha=0.25,
    )


axes[-1].set_xlabel(
    "Date / time (AEST)"
)

axes[0].legend(
    loc="upper right",
    ncol=2,
)

fig.suptitle(
    "Upper Hunter CH₄ — Observation and Analytical Inversion",
    fontsize=16,
    fontweight="bold",
)

plt.tight_layout(
    rect=[0, 0, 1, 0.97]
)

combined_filename = os.path.join(
    OUTPUT_DIR,
    "predicted_all_receptors_three_methods.png",
)

fig.savefig(
    combined_filename,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

print("\nSaved:")
print(combined_filename)


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 90)
print("COMPLETE")
print("=" * 90)

print(f"Output directory:")
print(OUTPUT_DIR)

print("\nFiles:")
print(
    "  predicted_concentrations_three_methods.csv"
)
print(
    "  prediction_statistics_three_methods.csv"
)
print(
    "  predicted_Singleton_three_methods.png"
)
print(
    "  predicted_Muswellbrook_three_methods.png"
)
print(
    "  predicted_Merriwa_three_methods.png"
)
print(
    "  predicted_all_receptors_three_methods.png"
)

print("\nMethods plotted:")
print("  Observation")
print("  Prior")
print("  Analytical linear — unconstrained")
print("  Analytical linear — constrained")

