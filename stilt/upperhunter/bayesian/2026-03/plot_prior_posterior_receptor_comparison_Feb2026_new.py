#!/usr/bin/env python3
"""
===============================================================================
Upper Hunter CH4 Bayesian inversion
Prior vs posterior predictions vs Picarro observations
February 2026

Plots:
    1. Three-receptor time series:
         - Picarro observation
         - Prior prediction
         - Posterior prediction

    2. Individual receptor time series for:
         - Singleton
         - Muswellbrook
         - Merriwa

    3. Prior vs observation scatter
    4. Posterior vs observation scatter

    5. Metrics CSV
    6. Complete prediction/observation CSV

IMPORTANT:
    - Missing observations are NOT replaced by zero.
    - Missing observations are NOT interpolated.
    - Missing observations are plotted as gaps.
    - WRF/STILT prior and posterior predictions remain continuous.
    - Picarro timestamps are AEST (UTC+10) for February 2026.
    - H matrix contains datetime_aest, so no timezone conversion is
      required when matching H matrix to observations.

Prior:
    CH4_prior = BACKGROUND_PPM + sum(H_m)

Posterior:
    CH4_post = BACKGROUND_PPM
                + sum(H_m * scaling_factor[m, day])

where:
    H_m = STILT source-receptor response for mine m
    scaling_factor[m, day] = posterior daily emission scaling factor

===============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

BASE_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/bayesian/2026-02/"
)

H_FILE = os.path.join(
    BASE_DIR,
    "H_matrix/STILT_H_matrix_mines_Feb2026.csv"
)

SCALING_FILE = os.path.join(
    BASE_DIR,
    "Bayesian_daily_inversion_Feb2026",
    "daily_scaling_factors_Feb2026.csv"
)

OBS_FILE = (
    "/home/duch/flexpart-wrf/emission_UH/"
    "RH_upperHunter/tapm_output/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Bayesian_daily_inversion_Feb2026",
    "prior_posterior_receptor_comparison"
)

BACKGROUND_PPM = 1.85

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

SITES = [
    "Singleton",
    "Muswellbrook",
    "Merriwa",
]

OBS_COLUMNS = {
    "Singleton": "Singleton cleaned",
    "Muswellbrook": "Muswellbrook cleaned",
    "Merriwa": "Merriwa cleaned",
}


# =============================================================================
# CREATE OUTPUT DIRECTORY
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# FUNCTIONS
# =============================================================================

def clean_datetime_column(df, column):
    """
    Convert datetime column to pandas datetime.
    Invalid values become NaT.
    """
    df[column] = pd.to_datetime(
        df[column],
        errors="coerce"
    )

    return df


def calculate_metrics(observation, prediction):
    """
    Calculate standard comparison metrics using only valid
    observation/prediction pairs.
    """

    mask = (
        np.isfinite(observation)
        & np.isfinite(prediction)
    )

    obs = np.asarray(observation)[mask]
    pred = np.asarray(prediction)[mask]

    n = len(obs)

    if n == 0:
        return {
            "N": 0,
            "R": np.nan,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan,
            "FAC2": np.nan,
        }

    residual = pred - obs

    rmse = np.sqrt(
        np.mean(residual ** 2)
    )

    mae = np.mean(
        np.abs(residual)
    )

    bias = np.mean(residual)

    if n > 1:
        r = np.corrcoef(obs, pred)[0, 1]
        r2 = r ** 2
    else:
        r = np.nan
        r2 = np.nan

    # FAC2:
    # 0.5 <= prediction / observation <= 2
    positive_mask = (
        np.isfinite(obs)
        & np.isfinite(pred)
        & (obs > 0)
    )

    if np.any(positive_mask):
        ratio = (
            pred[positive_mask]
            / obs[positive_mask]
        )

        fac2 = np.mean(
            (ratio >= 0.5)
            & (ratio <= 2.0)
        )
    else:
        fac2 = np.nan

    return {
        "N": n,
        "R": r,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "FAC2": fac2,
    }


# =============================================================================
# READ H MATRIX
# =============================================================================

print()
print("=" * 80)
print("READING H MATRIX")
print("=" * 80)

h = pd.read_csv(H_FILE)

print("H matrix:")
print("  File :", H_FILE)
print("  Rows :", len(h))
print("  Cols :", len(h.columns))

print()
print("Columns:")
print(list(h.columns))


# =============================================================================
# DETERMINE DATETIME COLUMN
# =============================================================================

if "datetime_aest" in h.columns:
    H_TIME_COLUMN = "datetime_aest"
elif "datetime_utc" in h.columns:
    H_TIME_COLUMN = "datetime_utc"
else:
    raise RuntimeError(
        "H matrix must contain either "
        "'datetime_aest' or 'datetime_utc'."
    )


h[H_TIME_COLUMN] = pd.to_datetime(
    h[H_TIME_COLUMN],
    errors="coerce"
)

if h[H_TIME_COLUMN].isna().any():
    bad = h[H_TIME_COLUMN].isna().sum()

    raise RuntimeError(
        f"H matrix contains {bad} invalid datetime values."
    )


# =============================================================================
# CONVERT TO AEST TIMESTAMP
# =============================================================================

if H_TIME_COLUMN == "datetime_aest":

    h["datetime_aest"] = h["datetime_aest"]

else:

    # H matrix contains UTC.
    # February 2026 = AEST = UTC + 10 hours.
    h["datetime_aest"] = (
        h["datetime_utc"]
        + pd.Timedelta(hours=10)
    )


# =============================================================================
# CHECK SITE COLUMN
# =============================================================================

if "site" not in h.columns:
    raise RuntimeError(
        "H matrix does not contain a 'site' column."
    )


print()
print("H matrix sites:")

for site in SITES:
    count = np.sum(
        h["site"].astype(str).str.lower()
        == site.lower()
    )

    print(
        f"  {site:<15s}: {count:5d} rows"
    )


# =============================================================================
# CONVERT H COLUMNS TO NUMERIC
# =============================================================================

for mine in MINES:

    if mine not in h.columns:
        raise RuntimeError(
            f"Missing H_matrix column: {mine}"
        )

    h[mine] = pd.to_numeric(
        h[mine],
        errors="coerce"
    )


# =============================================================================
# READ DAILY POSTERIOR SCALING FACTORS
# =============================================================================

print()
print("=" * 80)
print("READING POSTERIOR DAILY SCALING FACTORS")
print("=" * 80)

scaling = pd.read_csv(SCALING_FILE)

print("Scaling file:")
print("  File :", SCALING_FILE)
print("  Rows :", len(scaling))
print("  Cols :", len(scaling.columns))


scaling["date"] = pd.to_datetime(
    scaling["date"],
    errors="coerce"
)

if scaling["date"].isna().any():
    raise RuntimeError(
        "Scaling file contains invalid dates."
    )


# Make date-only column for matching
scaling["date_only"] = scaling["date"].dt.normalize()


# =============================================================================
# CHECK SCALING COLUMNS
# =============================================================================

for mine in MINES:

    if mine not in scaling.columns:
        raise RuntimeError(
            f"Missing posterior scaling column: {mine}"
        )

    scaling[mine] = pd.to_numeric(
        scaling[mine],
        errors="coerce"
    )


print()
print("Posterior scaling factor ranges:")

for mine in MINES:

    values = scaling[mine].dropna()

    print(
        f"  {mine:>4s}: "
        f"min={values.min():.4f}, "
        f"max={values.max():.4f}, "
        f"mean={values.mean():.4f}"
    )


# =============================================================================
# READ PICARRO OBSERVATIONS
# =============================================================================

print()
print("=" * 80)
print("READING PICARRO OBSERVATIONS")
print("=" * 80)

obs = pd.read_csv(OBS_FILE)

print("Observation file:")
print("  File :", OBS_FILE)
print("  Rows :", len(obs))
print("  Cols :", len(obs.columns))


# =============================================================================
# CREATE AEST DATETIME
# =============================================================================

if "Datetime" not in obs.columns:
    raise RuntimeError(
        "Observation file does not contain 'Datetime'."
    )

obs["datetime_aest"] = pd.to_datetime(
    obs["Datetime"],
    errors="coerce"
)

if obs["datetime_aest"].isna().any():
    print(
        "WARNING:",
        obs["datetime_aest"].isna().sum(),
        "invalid observation timestamps."
    )


# =============================================================================
# SELECT FEBRUARY 2026
# =============================================================================

obs = obs[
    (obs["datetime_aest"] >= "2026-02-01 00:00:00")
    &
    (obs["datetime_aest"] < "2026-03-01 00:00:00")
].copy()

print()
print(
    "February observation rows:",
    len(obs)
)


# =============================================================================
# CHECK OBSERVATION COUNTS
# =============================================================================

print()
print("Valid February 2026 observations:")

for site in SITES:

    column = OBS_COLUMNS[site]

    if column not in obs.columns:
        raise RuntimeError(
            f"Missing observation column: {column}"
        )

    obs[column] = pd.to_numeric(
        obs[column],
        errors="coerce"
    )

    valid = np.isfinite(
        obs[column].to_numpy()
    )

    print(
        f"  {site:<15s}: "
        f"{np.sum(valid):5d}"
    )


# =============================================================================
# SHOW MERRIWA DAILY OBSERVATION COUNTS
# =============================================================================

print()
print("=" * 80)
print("MERRIWA JANUARY 2026 OBSERVATION CHECK")
print("=" * 80)

merriwa_column = OBS_COLUMNS["Merriwa"]

merriwa_valid = obs[
    np.isfinite(
        obs[merriwa_column].to_numpy()
    )
].copy()

print(
    "Total valid Merriwa observations:",
    len(merriwa_valid)
)

if len(merriwa_valid) > 0:

    merriwa_valid["date"] = (
        merriwa_valid["datetime_aest"]
        .dt.normalize()
    )

    daily_counts = (
        merriwa_valid
        .groupby("date")
        .size()
    )

    print()
    print("Merriwa observations by day:")
    print(daily_counts.to_string())

    print()
    print(
        "Merriwa first observation:",
        merriwa_valid["datetime_aest"].min()
    )

    print(
        "Merriwa last observation:",
        merriwa_valid["datetime_aest"].max()
    )


# =============================================================================
# BUILD SITE DATA
# =============================================================================

all_predictions = []
all_metrics = []


# =============================================================================
# PROCESS EACH RECEPTOR
# =============================================================================

for site in SITES:

    print()
    print("=" * 80)
    print("PROCESSING:", site)
    print("=" * 80)

    # -------------------------------------------------------------------------
    # H matrix for this site
    # -------------------------------------------------------------------------

    h_site = h[
        h["site"].astype(str).str.lower()
        == site.lower()
    ].copy()

    if len(h_site) == 0:

        print(
            "WARNING: no H_matrix rows for",
            site
        )

        continue

    # -------------------------------------------------------------------------
    # Remove duplicate datetime rows if present
    # -------------------------------------------------------------------------

    h_site = (
        h_site
        .sort_values("datetime_aest")
        .drop_duplicates(
            subset=["datetime_aest"],
            keep="first"
        )
        .copy()
    )

    # -------------------------------------------------------------------------
    # Observation for this site
    # -------------------------------------------------------------------------

    obs_column = OBS_COLUMNS[site]

    site_obs = obs[
        [
            "datetime_aest",
            obs_column,
        ]
    ].copy()

    site_obs = site_obs.rename(
        columns={
            obs_column: "observation"
        }
    )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # H matrix is the master time axis.
    #
    # This keeps model predictions continuous while observation
    # gaps remain NaN.
    # -------------------------------------------------------------------------

    merged = pd.merge(
        h_site[
            ["datetime_aest"] + MINES
        ],
        site_obs,
        on="datetime_aest",
        how="left"
    )

    merged = (
        merged
        .sort_values("datetime_aest")
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Make sure observations are numeric
    # -------------------------------------------------------------------------

    merged["observation"] = pd.to_numeric(
        merged["observation"],
        errors="coerce"
    )

    # -------------------------------------------------------------------------
    # PRIOR PREDICTION
    #
    # Current inventory = scaling factor 1 for every mine.
    # -------------------------------------------------------------------------

    H_values = (
        merged[MINES]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    merged["prior_prediction"] = (
        BACKGROUND_PPM
        + np.sum(
            H_values,
            axis=1
        )
    )

    # -------------------------------------------------------------------------
    # DATE USED TO SELECT DAILY POSTERIOR FACTORS
    # -------------------------------------------------------------------------

    merged["date_only"] = (
        merged["datetime_aest"]
        .dt.normalize()
    )

    # -------------------------------------------------------------------------
    # MERGE DAILY POSTERIOR SCALING FACTORS
    # -------------------------------------------------------------------------

    merged = pd.merge(
        merged,
        scaling[
            ["date_only"] + MINES
        ].rename(
            columns={
                mine: f"{mine}_scale"
                for mine in MINES
            }
        ),
        on="date_only",
        how="left"
    )

    # -------------------------------------------------------------------------
    # CHECK DAILY SCALING FACTORS
    # -------------------------------------------------------------------------

    missing_scaling_days = (
        merged[
            [f"{mine}_scale" for mine in MINES]
        ]
        .isna()
        .any(axis=1)
    )

    if missing_scaling_days.any():

        missing_dates = sorted(
            merged.loc[
                missing_scaling_days,
                "date_only"
            ].dropna().unique()
        )

        print()
        print(
            "WARNING:",
            len(missing_dates),
            "dates have missing posterior scaling factors."
        )

        for date in missing_dates:
            print("  ", date)

    # -------------------------------------------------------------------------
    # POSTERIOR PREDICTION
    # -------------------------------------------------------------------------

    posterior_enhancement = np.zeros(
        len(merged),
        dtype=float
    )

    for mine in MINES:

        h_value = (
            pd.to_numeric(
                merged[mine],
                errors="coerce"
            )
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        scale_value = (
            pd.to_numeric(
                merged[f"{mine}_scale"],
                errors="coerce"
            )
            .fillna(1.0)
            .to_numpy(dtype=float)
        )

        posterior_enhancement += (
            h_value * scale_value
        )

    merged["posterior_prediction"] = (
        BACKGROUND_PPM
        + posterior_enhancement
    )

    # -------------------------------------------------------------------------
    # SELECT OUTPUT COLUMNS
    # -------------------------------------------------------------------------

    output_site = merged[
        [
            "datetime_aest",
            "observation",
            "prior_prediction",
            "posterior_prediction",
        ]
    ].copy()

    output_site["site"] = site

    all_predictions.append(
        output_site
    )

    # -------------------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------------------

    prior_metrics = calculate_metrics(
        merged["observation"].to_numpy(),
        merged["prior_prediction"].to_numpy()
    )

    posterior_metrics = calculate_metrics(
        merged["observation"].to_numpy(),
        merged["posterior_prediction"].to_numpy()
    )

    print()
    print("Prior metrics:")
    for key, value in prior_metrics.items():
        print(
            f"  {key:<8s}: {value:.6f}"
            if isinstance(value, float)
            else
            f"  {key:<8s}: {value}"
        )

    print()
    print("Posterior metrics:")
    for key, value in posterior_metrics.items():
        print(
            f"  {key:<8s}: {value:.6f}"
            if isinstance(value, float)
            else
            f"  {key:<8s}: {value}"
        )

    prior_metrics["site"] = site
    prior_metrics["model"] = "Prior"

    posterior_metrics["site"] = site
    posterior_metrics["model"] = "Posterior"

    all_metrics.append(
        prior_metrics
    )

    all_metrics.append(
        posterior_metrics
    )

    # =============================================================================
    # INDIVIDUAL SITE TIME SERIES
    # =============================================================================

    fig, ax = plt.subplots(
        figsize=(15, 6)
    )

    # -------------------------------------------------------------------------
    # Observation
    #
    # NaN values are deliberately retained.
    #
    # Matplotlib therefore leaves a gap during periods with no observation.
    # -------------------------------------------------------------------------

    ax.plot(
        merged["datetime_aest"],
        merged["observation"],
        linestyle="-",
        linewidth=1.8,
        marker=None,
        label="Picarro observation"
    )

    # -------------------------------------------------------------------------
    # Prior
    # -------------------------------------------------------------------------

    ax.plot(
        merged["datetime_aest"],
        merged["prior_prediction"],
        linestyle="--",
        linewidth=1.5,
        label="Prior"
    )

    # -------------------------------------------------------------------------
    # Posterior
    # -------------------------------------------------------------------------

    ax.plot(
        merged["datetime_aest"],
        merged["posterior_prediction"],
        linestyle="-",
        linewidth=1.8,
        label="Posterior"
    )

    ax.set_title(
        f"{site}: CH4 prior and posterior vs Picarro "
        "— February 2026"
    )

    ax.set_xlabel(
        "AEST"
    )

    ax.set_ylabel(
        "CH4 (ppm)"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    fig.autofmt_xdate()

    fig.tight_layout()

    site_png = os.path.join(
        OUTPUT_DIR,
        f"{site.lower()}_prior_posterior_Feb2026.png"
    )

    site_pdf = os.path.join(
        OUTPUT_DIR,
        f"{site.lower()}_prior_posterior_Feb2026.pdf"
    )

    fig.savefig(
        site_png,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        site_pdf,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print()
    print("Saved:")
    print(" ", site_png)
    print(" ", site_pdf)


# =============================================================================
# COMBINE ALL PREDICTIONS
# =============================================================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions_file = os.path.join(
    OUTPUT_DIR,
    "prior_posterior_predictions_Feb2026.csv"
)

predictions_df.to_csv(
    predictions_file,
    index=False
)

print()
print("Saved:")
print(" ", predictions_file)


# =============================================================================
# COMBINE METRICS
# =============================================================================

metrics_df = pd.DataFrame(
    all_metrics
)

metrics_columns = [
    "site",
    "model",
    "N",
    "R",
    "R2",
    "RMSE",
    "MAE",
    "Bias",
    "FAC2",
]

metrics_df = metrics_df[
    metrics_columns
]

metrics_file = os.path.join(
    OUTPUT_DIR,
    "prior_posterior_metrics_Feb2026.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)

print()
print("Saved:")
print(" ", metrics_file)


# =============================================================================
# PRINT METRICS TABLE
# =============================================================================

print()
print("=" * 80)
print("METRICS")
print("=" * 80)

print(
    metrics_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)


# =============================================================================
# THREE-RECEPTOR TIME SERIES
# =============================================================================

print()
print("=" * 80)
print("CREATING THREE-RECEPTOR TIME SERIES")
print("=" * 80)

fig, axes = plt.subplots(
    len(SITES),
    1,
    figsize=(16, 12),
    sharex=True
)

for ax, site in zip(
    axes,
    SITES
):

    data = predictions_df[
        predictions_df["site"] == site
    ].copy()

    data = data.sort_values(
        "datetime_aest"
    )

    # -------------------------------------------------------------------------
    # OBSERVATION
    #
    # Missing observations remain NaN.
    # Therefore the line is NOT drawn across missing periods.
    # -------------------------------------------------------------------------

    ax.plot(
        data["datetime_aest"],
        data["observation"],
        linestyle="-",
        linewidth=1.8,
        marker=None,
        label="Picarro observation"
    )

    # -------------------------------------------------------------------------
    # PRIOR
    # -------------------------------------------------------------------------

    ax.plot(
        data["datetime_aest"],
        data["prior_prediction"],
        linestyle="--",
        linewidth=1.5,
        label="Prior"
    )

    # -------------------------------------------------------------------------
    # POSTERIOR
    # -------------------------------------------------------------------------

    ax.plot(
        data["datetime_aest"],
        data["posterior_prediction"],
        linestyle="-",
        linewidth=1.8,
        label="Posterior"
    )

    ax.set_ylabel(
        f"{site}\nCH4 (ppm)"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend(
        loc="upper right"
    )


axes[-1].set_xlabel(
    "AEST"
)

fig.suptitle(
    "Upper Hunter CH4: Prior vs Posterior vs Picarro "
    "— February 2026",
    fontsize=15
)

fig.tight_layout(
    rect=[0, 0, 1, 0.97]
)

three_png = os.path.join(
    OUTPUT_DIR,
    "prior_posterior_receptor_timeseries_Feb2026.png"
)

three_pdf = os.path.join(
    OUTPUT_DIR,
    "prior_posterior_receptor_timeseries_Feb2026.pdf"
)

fig.savefig(
    three_png,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    three_pdf,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print()
print("Saved:")
print(" ", three_png)
print(" ", three_pdf)


# =============================================================================
# SCATTER PLOTS
# =============================================================================

print()
print("=" * 80)
print("CREATING SCATTER PLOTS")
print("=" * 80)


# =============================================================================
# PRIOR SCATTER
# =============================================================================

fig, axes = plt.subplots(
    1,
    len(SITES),
    figsize=(16, 5)
)

for ax, site in zip(
    axes,
    SITES
):

    data = predictions_df[
        predictions_df["site"] == site
    ].copy()

    mask = (
        np.isfinite(
            data["observation"].to_numpy()
        )
        &
        np.isfinite(
            data["prior_prediction"].to_numpy()
        )
    )

    x = data.loc[
        mask,
        "observation"
    ]

    y = data.loc[
        mask,
        "prior_prediction"
    ]

    ax.scatter(
        x,
        y,
        s=18,
        alpha=0.65
    )

    # 1:1 line
    finite_values = np.concatenate(
        [
            x.to_numpy(),
            y.to_numpy()
        ]
    )

    if len(finite_values) > 0:

        vmin = np.nanmin(
            finite_values
        )

        vmax = np.nanmax(
            finite_values
        )

        ax.plot(
            [vmin, vmax],
            [vmin, vmax],
            linestyle="--",
            linewidth=1.2
        )

    ax.set_title(
        site
    )

    ax.set_xlabel(
        "Picarro CH4 (ppm)"
    )

    ax.set_ylabel(
        "Prior CH4 (ppm)"
    )

    ax.grid(
        True,
        alpha=0.3
    )


fig.suptitle(
    "Prior prediction vs Picarro observations "
    "— February 2026",
    fontsize=14
)

fig.tight_layout(
    rect=[0, 0, 1, 0.95]
)

prior_scatter_png = os.path.join(
    OUTPUT_DIR,
    "prior_vs_observation_scatter_Feb2026.png"
)

prior_scatter_pdf = os.path.join(
    OUTPUT_DIR,
    "prior_vs_observation_scatter_Feb2026.pdf"
)

fig.savefig(
    prior_scatter_png,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    prior_scatter_pdf,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print()
print("Saved:")
print(" ", prior_scatter_png)
print(" ", prior_scatter_pdf)


# =============================================================================
# POSTERIOR SCATTER
# =============================================================================

fig, axes = plt.subplots(
    1,
    len(SITES),
    figsize=(16, 5)
)

for ax, site in zip(
    axes,
    SITES
):

    data = predictions_df[
        predictions_df["site"] == site
    ].copy()

    mask = (
        np.isfinite(
            data["observation"].to_numpy()
        )
        &
        np.isfinite(
            data["posterior_prediction"].to_numpy()
        )
    )

    x = data.loc[
        mask,
        "observation"
    ]

    y = data.loc[
        mask,
        "posterior_prediction"
    ]

    ax.scatter(
        x,
        y,
        s=18,
        alpha=0.65
    )

    # 1:1 line
    finite_values = np.concatenate(
        [
            x.to_numpy(),
            y.to_numpy()
        ]
    )

    if len(finite_values) > 0:

        vmin = np.nanmin(
            finite_values
        )

        vmax = np.nanmax(
            finite_values
        )

        ax.plot(
            [vmin, vmax],
            [vmin, vmax],
            linestyle="--",
            linewidth=1.2
        )

    ax.set_title(
        site
    )

    ax.set_xlabel(
        "Picarro CH4 (ppm)"
    )

    ax.set_ylabel(
        "Posterior CH4 (ppm)"
    )

    ax.grid(
        True,
        alpha=0.3
    )


fig.suptitle(
    "Posterior prediction vs Picarro observations "
    "— February 2026",
    fontsize=14
)

fig.tight_layout(
    rect=[0, 0, 1, 0.95]
)

posterior_scatter_png = os.path.join(
    OUTPUT_DIR,
    "posterior_vs_observation_scatter_Feb2026.png"
)

posterior_scatter_pdf = os.path.join(
    OUTPUT_DIR,
    "posterior_vs_observation_scatter_Feb2026.pdf"
)

fig.savefig(
    posterior_scatter_png,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    posterior_scatter_pdf,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print()
print("Saved:")
print(" ", posterior_scatter_png)
print(" ", posterior_scatter_pdf)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Files created:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(
        " ",
        filename
    )

print()
print("=" * 80)

