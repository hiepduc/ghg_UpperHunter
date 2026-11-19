#!/usr/bin/env python3

"""
FLEXPART-WRF Upper Hunter
Observed vs Prior-Predicted vs Posterior-Predicted CH4

Uses:
    H matrix:
        H_flexpart_0_10m_13mine_Jun2026.csv

    Observations:
        Picarro_CH4_Upper_Hunter_Jan-July_2026.csv

    Posterior daily scaling factors:
        FLEXPART_MCMC_daily_factor_matrix_Jun2026.csv

Forward model:

    CH4 = BACKGROUND_PPM + H @ scaling_factors

Important:
    H is already emission-weighted using the prior mine emissions.
    Therefore DO NOT multiply H by the prior emissions again.

Outputs:
    inversion_results/predicted_concentration/
        predicted_concentrations_prior_posterior.csv

        predicted_Singleton.png
        predicted_Muswellbrook.png
        predicted_Merriwa.png
        predicted_all_receptors.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

BASE = "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart"

H_FILE = os.path.join(
    BASE,
    "H_matrix_flexpart/2026-06",
    "H_flexpart_0_10m_13mine_Jun2026.csv",
)

OBS_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/"
    "piccaro_obs/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

POSTERIOR_FILE = os.path.join(
    BASE,
    "H_matrix_flexpart",
    "2026-06",
    "MCMC_13mine_daily_Jun2026",
    "FLEXPART_MCMC_daily_factor_matrix_Jun2026.csv",
)

OUTPUT_DIR = os.path.join(
    BASE,
    "inversion_results",
    "2026-06",
    "predicted_concentration",
)

BACKGROUND_PPM = 1.85

START_DATE = "2026-06-01"
END_DATE   = "2026-06-30"

OBS_SIGMA = 0.05


# ============================================================
# RECEPTOR / OBSERVATION COLUMN SETTINGS
# ============================================================

OBS_COLUMNS = {
    "singleton": "Singleton cleaned",
    "muswellbrook": "Muswellbrook cleaned",
    "merriwa": "Merriwa cleaned",
}


# ============================================================
# HELPER
# ============================================================

def find_datetime_column(df):

    candidates = [
        "Datetime",
        "datetime",
        "DateTime",
        "datetime_aest",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    # If Date and Time are separate
    if "Date" in df.columns and "Time" in df.columns:
        return None

    raise ValueError(
        "Could not find a datetime column. "
        f"Available columns:\n{list(df.columns)}"
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 80)
print("FLEXPART-WRF PRIOR vs POSTERIOR CONCENTRATION with GIBBS truncated sampler Jun 2026")
print("=" * 80)

print()
print("H matrix:")
print(H_FILE)

print()
print("Observations:")
print(OBS_FILE)

print()
print("Posterior daily factors:")
print(POSTERIOR_FILE)


# ============================================================
# READ H MATRIX
# ============================================================

print()
print("=" * 80)
print("READING H MATRIX")
print("=" * 80)

H_df = pd.read_csv(H_FILE)

print("H shape:", H_df.shape)
print("H columns:", len(H_df.columns))

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
    raise ValueError(
        f"Missing expected H metadata columns: {missing_metadata}"
    )


# Parameter columns are everything except metadata
parameter_columns = [
    c for c in H_df.columns
    if c not in metadata_columns
]

print("Number of scaling-factor parameters:", len(parameter_columns))

if len(parameter_columns) != 390:
    raise ValueError(
        f"Expected 390 parameters, found {len(parameter_columns)}"
    )


H = H_df[parameter_columns].to_numpy(dtype=float)

print("H numerical shape:", H.shape)


# ============================================================
# READ POSTERIOR DAILY FACTORS
# ============================================================

print()
print("=" * 80)
print("READING POSTERIOR DAILY FACTORS")
print("=" * 80)

post_df = pd.read_csv(POSTERIOR_FILE)

print("Posterior factor matrix shape:", post_df.shape)
print("Columns:")
print(list(post_df.columns))


if "date" not in post_df.columns:
    raise ValueError("Posterior factor file has no 'date' column.")


# ------------------------------------------------------------
# The MCMC factor matrix contains one row per day and
# one column per mine.
# ------------------------------------------------------------

post_df["date"] = pd.to_datetime(
    post_df["date"],
    errors="coerce",
)

if post_df["date"].isna().any():
    raise ValueError("Invalid dates found in posterior factor file.")


post_df = post_df[
    (post_df["date"] >= pd.Timestamp(START_DATE))
    & (post_df["date"] <= pd.Timestamp(END_DATE))
].copy()


# ------------------------------------------------------------
# Build the exact 390 parameter vector:
#
# OC1_20260301
# OC2_20260301
# ...
#
# matching the H matrix columns exactly.
# ------------------------------------------------------------

posterior_scaling = np.ones(len(parameter_columns))


missing_factor_columns = []

for j, parameter in enumerate(parameter_columns):

    # Example:
    # OC1_20260105
    mine = parameter.split("_")[0]
    date_string = parameter.split("_")[1]

    date = pd.to_datetime(
        date_string,
        format="%Y%m%d",
    )

    rows = post_df[
        post_df["date"] == date
    ]

    if len(rows) != 1:
        missing_factor_columns.append(parameter)
        continue

    if mine not in rows.columns:
        missing_factor_columns.append(parameter)
        continue

    posterior_scaling[j] = float(
        rows.iloc[0][mine]
    )


if missing_factor_columns:

    print()
    print("WARNING:")
    print(
        "Missing posterior factors for "
        f"{len(missing_factor_columns)} parameters."
    )

    print(
        "First missing parameters:",
        missing_factor_columns[:20],
    )

    raise ValueError(
        "Posterior factor matrix does not contain all "
        "required mine/day combinations."
    )


print()
print(
    "Posterior scaling vector:",
    posterior_scaling.shape
)

print(
    "Posterior scaling range:",
    posterior_scaling.min(),
    "to",
    posterior_scaling.max(),
)


# ============================================================
# PRIOR SCALING
# ============================================================

prior_scaling = np.ones(len(parameter_columns))

print()
print("Prior scaling factors:")
print("All parameters = 1.0")


# ============================================================
# FORWARD MODEL
# ============================================================

print()
print("=" * 80)
print("CALCULATING CONCENTRATIONS")
print("=" * 80)

print()
print("Prior:")
print("    CH4_prior = background + H @ 1")

print()
print("Posterior:")
print("    CH4_post = background + H @ posterior_scaling")


# H is already emission weighted
prior_enhancement = H @ prior_scaling
posterior_enhancement = H @ posterior_scaling

prior_prediction = (
    BACKGROUND_PPM
    + prior_enhancement
)

posterior_prediction = (
    BACKGROUND_PPM
    + posterior_enhancement
)


print()
print("Prior prediction range:")
print(
    np.nanmin(prior_prediction),
    "to",
    np.nanmax(prior_prediction),
    "ppm",
)

print()
print("Posterior prediction range:")
print(
    np.nanmin(posterior_prediction),
    "to",
    np.nanmax(posterior_prediction),
    "ppm",
)


# ============================================================
# CONVERT H TIMES TO DATETIME
# ============================================================

print()
print("=" * 80)
print("PROCESSING H TIMES")
print("=" * 80)


H_df["obs_time_aest"] = pd.to_datetime(
    H_df["obs_time_aest"],
    errors="coerce",
)

if H_df["obs_time_aest"].isna().any():

    bad = H_df["obs_time_aest"].isna().sum()

    raise ValueError(
        f"{bad} H rows have invalid obs_time_aest values."
    )


# ============================================================
# READ PICARRO OBSERVATIONS
# ============================================================

print()
print("=" * 80)
print("READING PICARRO OBSERVATIONS")
print("=" * 80)

obs = pd.read_csv(OBS_FILE)

print("Observation shape:", obs.shape)

datetime_column = find_datetime_column(obs)


if datetime_column is not None:

    print(
        "Using datetime column:",
        datetime_column
    )

    obs["datetime_aest"] = pd.to_datetime(
        obs[datetime_column],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )

else:

    print("Using Date + Time columns")

    obs["datetime_aest"] = pd.to_datetime(
        obs["Date"].astype(str)
        + " "
        + obs["Time"].astype(str),
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )


invalid_datetime = obs["datetime_aest"].isna().sum()

print(
    "Invalid observation datetimes:",
    invalid_datetime,
)


if invalid_datetime > 0:

    raise ValueError(
        "Observation datetime parsing failed."
    )


# ============================================================
# MATCH OBSERVATIONS TO H ROWS
# ============================================================

print()
print("=" * 80)
print("MATCHING OBSERVATIONS TO H ROWS")
print("=" * 80)


records = []


for receptor in [
    "singleton",
    "muswellbrook",
    "merriwa",
]:

    print()
    print("-" * 60)
    print("Receptor:", receptor)
    print("-" * 60)

    receptor_mask = (
        H_df["receptor"]
        .astype(str)
        .str.lower()
        .str.contains(receptor)
    )

    indices = np.where(
        receptor_mask.to_numpy()
    )[0]

    print(
        "H rows:",
        len(indices)
    )

    obs_column = OBS_COLUMNS[receptor]

    if obs_column not in obs.columns:

        print(
            "WARNING:",
            obs_column,
            "not found."
        )

        continue


    obs_lookup = (
        obs[
            [
                "datetime_aest",
                obs_column,
            ]
        ]
        .dropna(subset=[obs_column])
        .drop_duplicates(
            subset=["datetime_aest"]
        )
        .set_index("datetime_aest")[obs_column]
    )


    matched = 0
    missing = 0


    for idx in indices:

        time = H_df.loc[
            idx,
            "obs_time_aest"
        ]

        prior_value = prior_prediction[idx]
        posterior_value = posterior_prediction[idx]


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
                "receptor": receptor,
                "datetime_aest": time,
                "observation_ppm": observation,
                "prior_prediction_ppm": prior_value,
                "posterior_prediction_ppm": posterior_value,
                "prior_enhancement_ppm":
                    prior_value - BACKGROUND_PPM,
                "posterior_enhancement_ppm":
                    posterior_value - BACKGROUND_PPM,
            }
        )


    print(
        "Matched observations:",
        matched
    )

    print(
        "Missing observations:",
        missing
    )


# ============================================================
# BUILD OUTPUT DATAFRAME
# ============================================================

results = pd.DataFrame(records)

results = results.sort_values(
    [
        "receptor",
        "datetime_aest",
    ]
).reset_index(drop=True)


# Restrict to requested period
results = results[
    (results["datetime_aest"] >= pd.Timestamp(START_DATE))
    &
    (results["datetime_aest"] <
     pd.Timestamp(END_DATE) + pd.Timedelta(days=1))
].copy()


# ============================================================
# SAVE CSV
# ============================================================

output_csv = os.path.join(
    OUTPUT_DIR,
    "predicted_concentrations_prior_posterior.csv",
)

results.to_csv(
    output_csv,
    index=False,
)

print()
print("=" * 80)
print("SAVED PREDICTION DATA")
print("=" * 80)

print(output_csv)

print()
print(results.head())


# ============================================================
# STATISTICS
# ============================================================

print()
print("=" * 80)
print("PREDICTION STATISTICS")
print("=" * 80)


for receptor in [
    "singleton",
    "muswellbrook",
    "merriwa",
]:

    tmp = results[
        results["receptor"] == receptor
    ].copy()

    valid = tmp.dropna(
        subset=["observation_ppm"]
    )

    print()
    print(receptor.upper())

    print(
        "H rows:",
        len(tmp)
    )

    print(
        "Observed:",
        len(valid)
    )

    if len(valid) == 0:

        print("No observations available.")

        continue


    for name in [
        "prior_prediction_ppm",
        "posterior_prediction_ppm",
    ]:

        residual = (
            valid["observation_ppm"].to_numpy()
            -
            valid[name].to_numpy()
        )

        rmse = np.sqrt(
            np.mean(residual ** 2)
        )

        bias = np.mean(residual)

        mae = np.mean(
            np.abs(residual)
        )

        print()
        print(name)

        print(
            "  Bias (obs - pred):",
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


# ============================================================
# PLOT INDIVIDUAL RECEPTORS
# ============================================================

print()
print("=" * 80)
print("CREATING PLOTS")
print("=" * 80)


for receptor in [
    "singleton",
    "muswellbrook",
    "merriwa",
]:

    tmp = results[
        results["receptor"] == receptor
    ].copy()

    if len(tmp) == 0:
        continue


    fig, ax = plt.subplots(
        figsize=(14, 5.5)
    )


    # --------------------------------------------------------
    # Observation
    # --------------------------------------------------------

    observed = tmp.dropna(
        subset=["observation_ppm"]
    )

    if len(observed) > 0:

        ax.plot(
            observed["datetime_aest"],
            observed["observation_ppm"],
            linestyle="none",
            marker="o",
            markersize=3.5,
            label="Picarro observation",
        )


    # --------------------------------------------------------
    # Prior prediction
    # --------------------------------------------------------

    ax.plot(
        tmp["datetime_aest"],
        tmp["prior_prediction_ppm"],
        linewidth=1.5,
        linestyle="--",
        label="Prior emissions",
    )


    # --------------------------------------------------------
    # Posterior prediction
    # --------------------------------------------------------

    ax.plot(
        tmp["datetime_aest"],
        tmp["posterior_prediction_ppm"],
        linewidth=2.0,
        label="Posterior emissions",
    )


    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    ax.axhline(
        BACKGROUND_PPM,
        linestyle=":",
        linewidth=1.2,
        label="Background (1.85 ppm)",
    )


    ax.set_title(
        receptor.capitalize()
        + " — Observed vs prior/posterior predicted CH₄ with Gibbs truncated Gaussian sampler Jun 2026"
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
        f"predicted_{receptor}.png",
    )

    fig.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


    print(
        "Saved:",
        output_png
    )


# ============================================================
# COMBINED THREE-RECEPTOR FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(15, 12),
    sharex=True,
)


for ax, receptor in zip(
    axes,
    [
        "singleton",
        "muswellbrook",
        "merriwa",
    ],
):

    tmp = results[
        results["receptor"] == receptor
    ].copy()


    if len(tmp) == 0:

        ax.text(
            0.5,
            0.5,
            "No H-matrix data",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

        continue


    observed = tmp.dropna(
        subset=["observation_ppm"]
    )


    if len(observed) > 0:

        ax.plot(
            observed["datetime_aest"],
            observed["observation_ppm"],
            linestyle="none",
            marker="o",
            markersize=3,
            label="Picarro observation",
        )


    ax.plot(
        tmp["datetime_aest"],
        tmp["prior_prediction_ppm"],
        linestyle="--",
        linewidth=1.5,
        label="Prior emissions",
    )


    ax.plot(
        tmp["datetime_aest"],
        tmp["posterior_prediction_ppm"],
        linewidth=2.0,
        label="Posterior emissions (Gibbs truncated sampler)",
    )


    ax.axhline(
        BACKGROUND_PPM,
        linestyle=":",
        linewidth=1.0,
        label="Background",
    )


    ax.set_ylabel(
        "CH₄ (ppm)"
    )

    ax.set_title(
        receptor.capitalize()
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        loc="best"
    )


axes[-1].set_xlabel(
    "Time (AEST)"
)

fig.suptitle(
    "Upper Hunter CH₄ — Prior and posterior emission predictions",
    fontsize=15,
    y=0.995,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.98]
)


combined_png = os.path.join(
    OUTPUT_DIR,
    "predicted_all_receptors.png",
)

fig.savefig(
    combined_png,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)


print()
print("Saved:")
print(combined_png)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("DONE")
print("=" * 80)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Files:")
print(
    "  predicted_concentrations_prior_posterior.csv"
)

print(
    "  predicted_Singleton.png"
)

print(
    "  predicted_Muswellbrook.png"
)

print(
    "  predicted_Merriwa.png"
)

print(
    "  predicted_all_receptors.png"
)

print()
print("Forward model:")
print(
    "  CH4 = 1.85 + H @ scaling_factors"
)

print()
print("Prior:")
print(
    "  scaling_factors = 1.0 for all 390 parameters"
)

print()
print("Posterior:")
print(
    "  MCMC posterior mean daily scaling factors"
)

print("=" * 80)

