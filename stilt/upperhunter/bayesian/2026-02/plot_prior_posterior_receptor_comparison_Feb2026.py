#!/usr/bin/env python3

"""
===============================================================================
Upper Hunter CH4 Bayesian Inversion
Prior vs Posterior Prediction at 3 Picarro Receptors
February 2026

This script reconstructs receptor CH4 concentrations from the STILT H matrix.

PRIOR:
    CH4_prior = BACKGROUND + sum(H_i,m)

POSTERIOR:
    CH4_post = BACKGROUND + sum(H_i,m * scaling_factor_m,day)

where:
    H_i,m = STILT source-receptor response for mine m
    scaling_factor_m,day = posterior daily emission scaling factor

Inputs
------
1. Clean STILT H matrix:

   STILT_H_matrix_mines_Feb2026.csv

2. Bayesian posterior daily scaling factors:

   Bayesian_daily_inversion_Feb2026/daily_scaling_factors_Feb2026.csv

3. Picarro observations:

   Picarro_CH4_Upper_Hunter_Jan-July_2026.csv

Outputs
-------
1. prior_posterior_receptor_timeseries_Feb2026.png
2. prior_posterior_receptor_timeseries_Feb2026.pdf

3. prior_posterior_timeseries_Singleton_Feb2026.png
4. prior_posterior_timeseries_Muswellbrook_Feb2026.png
5. prior_posterior_timeseries_Merriwa_Feb2026.png

6. prior_posterior_scatter_Feb2026.png
7. prior_posterior_metrics_Feb2026.csv
8. prior_posterior_predictions_Feb2026.csv

Time:
-----
Picarro and H matrix are matched using AEST = UTC + 10 hours.

Background:
-----------
BACKGROUND_PPM = 1.85 ppm

===============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/bayesian/2026-02"
)

H_FILE = os.path.join(
    BASE_DIR,
    "H_matrix/STILT_H_matrix_mines_Feb2026.csv"
)

POSTERIOR_FILE = os.path.join(
    BASE_DIR,
    "Bayesian_daily_inversion_Feb2026",
    "daily_scaling_factors_Feb2026.csv"
)

OBS_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/piccaro_obs/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Bayesian_daily_inversion_Feb2026"
)

BACKGROUND_PPM = 1.85

MINES = [f"OC{i}" for i in range(1, 14)]

SITES = [
    "Singleton",
    "Muswellbrook",
    "Merriwa"
]

OBS_COLUMNS = {
    "Singleton": "Singleton cleaned",
    "Muswellbrook": "Muswellbrook cleaned",
    "Merriwa": "Merriwa cleaned"
}


# =============================================================================
# OUTPUT FILES
# =============================================================================

TIMESERIES_ALL_PNG = os.path.join(
    OUTPUT_DIR,
    "prior_posterior_receptor_timeseries_Feb2026.png"
)

TIMESERIES_ALL_PDF = os.path.join(
    OUTPUT_DIR,
    "prior_posterior_receptor_timeseries_Feb2026.pdf"
)

SCATTER_PNG = os.path.join(
    OUTPUT_DIR,
    "prior_posterior_scatter_Feb2026.png"
)

METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "prior_posterior_metrics_Feb2026.csv"
)

PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "prior_posterior_predictions_Feb2026.csv"
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_metrics(obs, pred):
    """
    Calculate standard model-observation statistics.
    """

    mask = (
        np.isfinite(obs)
        & np.isfinite(pred)
    )

    obs = np.asarray(obs)[mask]
    pred = np.asarray(pred)[mask]

    n = len(obs)

    if n == 0:
        return {
            "N": 0,
            "R": np.nan,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "FAC2": np.nan
        }

    residual = pred - obs

    rmse = np.sqrt(
        np.mean(residual ** 2)
    )

    mae = np.mean(
        np.abs(residual)
    )

    if n > 1:
        r = np.corrcoef(obs, pred)[0, 1]
        r2 = r ** 2
    else:
        r = np.nan
        r2 = np.nan

    # FAC2:
    # fraction of predictions within factor of 2 of observations
    valid_fac2 = obs > 0

    if np.any(valid_fac2):
        ratio = pred[valid_fac2] / obs[valid_fac2]

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
        "FAC2": fac2
    }


# =============================================================================
# READ H MATRIX
# =============================================================================

print()
print("=" * 80)
print("READING STILT H MATRIX")
print("=" * 80)

print(H_FILE)

H = pd.read_csv(H_FILE)

print(f"Rows: {len(H)}")
print(f"Columns: {len(H.columns)}")

print()
print("H matrix columns:")
print(H.columns.tolist())


# =============================================================================
# IDENTIFY H COLUMNS
# =============================================================================

missing_h = [
    mine for mine in MINES
    if mine not in H.columns
]

if missing_h:
    raise RuntimeError(
        "Missing H matrix columns: "
        + ", ".join(missing_h)
    )


# =============================================================================
# DATETIME
# =============================================================================

if "datetime_aest" in H.columns:

    H["datetime_aest"] = pd.to_datetime(
        H["datetime_aest"]
    )

elif "datetime_utc" in H.columns:

    print()
    print(
        "WARNING: datetime_aest not found. "
        "Converting datetime_utc to AEST (+10 h)."
    )

    H["datetime_utc"] = pd.to_datetime(
        H["datetime_utc"]
    )

    H["datetime_aest"] = (
        H["datetime_utc"]
        + pd.Timedelta(hours=10)
    )

else:

    raise RuntimeError(
        "Neither datetime_aest nor datetime_utc "
        "was found in H matrix."
    )


# =============================================================================
# SITE COLUMN
# =============================================================================

if "site" not in H.columns:
    raise RuntimeError(
        "The H matrix does not contain a 'site' column."
    )


# =============================================================================
# REMOVE POSSIBLE DUPLICATES AS A SAFETY CHECK
# =============================================================================

before = len(H)

H = H.drop_duplicates(
    subset=["site", "datetime_aest"],
    keep="first"
).copy()

after = len(H)

if after != before:

    print()
    print(
        f"WARNING: removed {before-after} duplicate "
        f"site/time rows from H matrix."
    )

print()
print(f"Final H rows: {len(H)}")


# =============================================================================
# READ POSTERIOR DAILY SCALING FACTORS
# =============================================================================

print()
print("=" * 80)
print("READING POSTERIOR DAILY SCALING FACTORS")
print("=" * 80)

print(POSTERIOR_FILE)

post = pd.read_csv(
    POSTERIOR_FILE
)

post["date"] = pd.to_datetime(
    post["date"]
).dt.date

print(
    f"Posterior days: {len(post)}"
)

print(
    f"Date range: "
    f"{post['date'].min()} to {post['date'].max()}"
)


# =============================================================================
# CHECK POSTERIOR COLUMNS
# =============================================================================

missing_post = [
    mine for mine in MINES
    if mine not in post.columns
]

if missing_post:
    raise RuntimeError(
        "Missing posterior scaling-factor columns: "
        + ", ".join(missing_post)
    )


# =============================================================================
# READ PICARRO OBSERVATIONS
# =============================================================================

print()
print("=" * 80)
print("READING PICARRO OBSERVATIONS")
print("=" * 80)

print(OBS_FILE)

obs = pd.read_csv(
    OBS_FILE
)

print(
    f"Observation rows: {len(obs)}"
)


# =============================================================================
# CONSTRUCT AEST DATETIME
# =============================================================================

if "Datetime" in obs.columns:

    obs["datetime_aest"] = pd.to_datetime(
        obs["Datetime"],
        errors="coerce"
    )

else:

    obs["datetime_aest"] = pd.to_datetime(
        obs["Date"].astype(str)
        + " "
        + obs["Time"].astype(str),
        errors="coerce"
    )


# =============================================================================
# RESTRICT TO FEBRUARY 2026
# =============================================================================

obs = obs[
    (obs["datetime_aest"] >= "2026-02-01 00:00:00")
    &
    (obs["datetime_aest"] < "2026-03-01 00:00:00")
].copy()

print(
    f"February observation rows: {len(obs)}"
)


# =============================================================================
# BUILD PREDICTIONS
# =============================================================================

print()
print("=" * 80)
print("CALCULATING PRIOR AND POSTERIOR PREDICTIONS")
print("=" * 80)


all_results = []


for site in SITES:

    print()
    print("-" * 80)
    print(f"Site: {site}")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # H matrix for this site
    # -------------------------------------------------------------------------

    h_site = H[
        H["site"].astype(str).str.lower()
        == site.lower()
    ].copy()

    h_site = h_site.sort_values(
        "datetime_aest"
    )

    print(
        f"H rows for {site}: {len(h_site)}"
    )

    # -------------------------------------------------------------------------
    # Observation column
    # -------------------------------------------------------------------------

    obs_column = OBS_COLUMNS[site]

    site_obs = obs[
        [
            "datetime_aest",
            obs_column
        ]
    ].copy()

    site_obs = site_obs.rename(
        columns={
            obs_column: "observation"
        }
    )

    # -------------------------------------------------------------------------
    # Merge observations and H matrix
    # -------------------------------------------------------------------------

    merged = pd.merge(
        h_site[
            ["datetime_aest"] + MINES
        ],
        site_obs,
        on="datetime_aest",
        how="outer"
    )

    merged = merged.sort_values(
        "datetime_aest"
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # PRIOR PREDICTION
    #
    # s = 1 for every mine
    # -------------------------------------------------------------------------

    H_values = (
        merged[MINES]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy()
    )

    merged["prior_prediction"] = (
        BACKGROUND_PPM
        + np.sum(H_values, axis=1)
    )

    # -------------------------------------------------------------------------
    # POSTERIOR PREDICTION
    # -------------------------------------------------------------------------

    merged["date"] = (
        merged["datetime_aest"]
        .dt.date
    )

    # Start with background
    posterior_mine_contribution = np.zeros(
        len(merged)
    )

    # Loop over days
    for day in merged["date"].dropna().unique():

        day_mask = (
            merged["date"] == day
        )

        post_day = post[
            post["date"] == day
        ]

        if len(post_day) == 0:

            print(
                f"WARNING: no posterior factors for {day}"
            )

            # Fall back to prior scaling = 1
            scaling = np.ones(13)

        else:

            scaling = np.array([
                float(
                    post_day.iloc[0][mine]
                )
                for mine in MINES
            ])

        posterior_mine_contribution[day_mask] = np.sum(
            H_values[day_mask, :]
            * scaling[np.newaxis, :],
            axis=1
        )

    merged["posterior_prediction"] = (
        BACKGROUND_PPM
        + posterior_mine_contribution
    )

    # -------------------------------------------------------------------------
    # Site identifier
    # -------------------------------------------------------------------------

    merged["site"] = site

    # -------------------------------------------------------------------------
    # Keep useful columns
    # -------------------------------------------------------------------------

    merged = merged[
        [
            "site",
            "datetime_aest",
            "observation",
            "prior_prediction",
            "posterior_prediction"
        ]
    ]

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    valid = (
        np.isfinite(
            merged["observation"]
        )
    )

    metrics_prior = calculate_metrics(
        merged.loc[valid, "observation"],
        merged.loc[valid, "prior_prediction"]
    )

    metrics_post = calculate_metrics(
        merged.loc[valid, "observation"],
        merged.loc[valid, "posterior_prediction"]
    )

    print()
    print("Prior:")
    print(
        f"  N    = {metrics_prior['N']}"
    )
    print(
        f"  R    = {metrics_prior['R']:.5f}"
    )
    print(
        f"  R2   = {metrics_prior['R2']:.5f}"
    )
    print(
        f"  RMSE = {metrics_prior['RMSE']:.5f} ppm"
    )
    print(
        f"  MAE  = {metrics_prior['MAE']:.5f} ppm"
    )
    print(
        f"  FAC2 = {metrics_prior['FAC2']:.5f}"
    )

    print()
    print("Posterior:")
    print(
        f"  N    = {metrics_post['N']}"
    )
    print(
        f"  R    = {metrics_post['R']:.5f}"
    )
    print(
        f"  R2   = {metrics_post['R2']:.5f}"
    )
    print(
        f"  RMSE = {metrics_post['RMSE']:.5f} ppm"
    )
    print(
        f"  MAE  = {metrics_post['MAE']:.5f} ppm"
    )
    print(
        f"  FAC2 = {metrics_post['FAC2']:.5f}"
    )

    # -------------------------------------------------------------------------
    # Improvement
    # -------------------------------------------------------------------------

    rmse_improvement = (
        100.0
        * (
            metrics_prior["RMSE"]
            - metrics_post["RMSE"]
        )
        / metrics_prior["RMSE"]
    )

    mae_improvement = (
        100.0
        * (
            metrics_prior["MAE"]
            - metrics_post["MAE"]
        )
        / metrics_prior["MAE"]
    )

    print()
    print(
        f"RMSE improvement = {rmse_improvement:.2f}%"
    )

    print(
        f"MAE improvement  = {mae_improvement:.2f}%"
    )

    # -------------------------------------------------------------------------
    # Store metrics
    # -------------------------------------------------------------------------

    all_results.append({
        "site": site,
        "model": "Prior",
        **metrics_prior
    })

    all_results.append({
        "site": site,
        "model": "Posterior",
        **metrics_post
    })

    # -------------------------------------------------------------------------
    # Save individual site plot
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(15, 5.5)
    )

    ax.plot(
        merged["datetime_aest"],
        merged["prior_prediction"],
        linewidth=1.5,
        label="Prior STILT"
    )

    ax.plot(
        merged["datetime_aest"],
        merged["posterior_prediction"],
        linewidth=1.8,
        label="Posterior STILT"
    )

    ax.plot(
        merged["datetime_aest"],
        merged["observation"],
        linestyle="None",
        marker="o",
        markersize=3,
        label="Picarro observation"
    )

    ax.set_title(
        f"{site} — CH$_4$ prior vs posterior prediction",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylabel(
        "CH$_4$ (ppm)",
        fontsize=11
    )

    ax.set_xlabel(
        "Date (AEST)",
        fontsize=11
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend(
        loc="best"
    )

    ax.xaxis.set_major_locator(
        mdates.DayLocator(interval=2)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b")
    )

    fig.autofmt_xdate()

    plt.tight_layout()

    individual_file = os.path.join(
        OUTPUT_DIR,
        f"prior_posterior_timeseries_{site}_Feb2026.png"
    )

    fig.savefig(
        individual_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print(
        f"Saved: {individual_file}"
    )

    all_results[-2]["RMSE_improvement_percent"] = (
        rmse_improvement
    )

    all_results[-1]["RMSE_improvement_percent"] = (
        rmse_improvement
    )


    # Store the predictions
    if "prediction_results" not in locals():
        prediction_results = []

    prediction_results.append(
        merged
    )


# =============================================================================
# COMBINE ALL PREDICTIONS
# =============================================================================

prediction_df = pd.concat(
    prediction_results,
    ignore_index=True
)

prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)

print()
print(
    f"Saved prediction CSV: {PREDICTIONS_FILE}"
)


# =============================================================================
# SAVE METRICS
# =============================================================================

metrics_df = pd.DataFrame(
    all_results
)

metrics_df.to_csv(
    METRICS_FILE,
    index=False
)

print(
    f"Saved metrics CSV: {METRICS_FILE}"
)


# =============================================================================
# THREE-RECEPTOR TIME-SERIES FIGURE
# =============================================================================

print()
print("=" * 80)
print("CREATING THREE-RECEPTOR TIME SERIES")
print("=" * 80)

fig, axes = plt.subplots(
    3,
    1,
    figsize=(16, 12),
    sharex=True
)

for ax, site in zip(axes, SITES):

    data = prediction_df[
        prediction_df["site"] == site
    ].copy()

    ax.plot(
        data["datetime_aest"],
        data["prior_prediction"],
        linewidth=1.5,
        label="Prior STILT"
    )

    ax.plot(
        data["datetime_aest"],
        data["posterior_prediction"],
        linewidth=1.8,
        label="Posterior STILT"
    )

    ax.plot(
        data["datetime_aest"],
        data["observation"],
        linestyle="None",
        marker="o",
        markersize=3,
        label="Picarro observation"
    )

    ax.set_title(
        site,
        fontsize=13,
        fontweight="bold"
    )

    ax.set_ylabel(
        "CH$_4$ (ppm)"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend(
        loc="best"
    )

    # Calculate metrics for title annotation
    valid = np.isfinite(
        data["observation"]
    )

    if valid.sum() > 0:

        prior_metrics = calculate_metrics(
            data.loc[valid, "observation"],
            data.loc[valid, "prior_prediction"]
        )

        post_metrics = calculate_metrics(
            data.loc[valid, "observation"],
            data.loc[valid, "posterior_prediction"]
        )

        text = (
            f"Prior RMSE = {prior_metrics['RMSE']:.3f} ppm    "
            f"Posterior RMSE = {post_metrics['RMSE']:.3f} ppm"
        )

        ax.text(
            0.01,
            0.94,
            text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top"
        )


axes[-1].set_xlabel(
    "Date (AEST)"
)

axes[-1].xaxis.set_major_locator(
    mdates.DayLocator(interval=2)
)

axes[-1].xaxis.set_major_formatter(
    mdates.DateFormatter("%d %b")
)

fig.suptitle(
    "Upper Hunter CH$_4$ — Prior vs Posterior Bayesian Inversion\n"
    "February 2026",
    fontsize=17,
    fontweight="bold"
)

fig.autofmt_xdate()

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

fig.savefig(
    TIMESERIES_ALL_PNG,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    TIMESERIES_ALL_PDF,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print(
    f"Saved: {TIMESERIES_ALL_PNG}"
)

print(
    f"Saved: {TIMESERIES_ALL_PDF}"
)


# =============================================================================
# SCATTER PLOTS
# =============================================================================

print()
print("=" * 80)
print("CREATING OBSERVED VS PREDICTED SCATTER PLOTS")
print("=" * 80)

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 10)
)

for j, site in enumerate(SITES):

    data = prediction_df[
        prediction_df["site"] == site
    ].copy()

    valid = (
        np.isfinite(data["observation"])
        &
        np.isfinite(data["prior_prediction"])
        &
        np.isfinite(data["posterior_prediction"])
    )

    data = data[valid]

    observed = data["observation"].to_numpy()

    prior = data["prior_prediction"].to_numpy()

    posterior = data["posterior_prediction"].to_numpy()

    # -------------------------------------------------------------------------
    # Determine common axis
    # -------------------------------------------------------------------------

    combined = np.concatenate([
        observed,
        prior,
        posterior
    ])

    lo = np.nanmin(combined)
    hi = np.nanmax(combined)

    margin = 0.05 * (hi - lo)

    lo -= margin
    hi += margin

    # -------------------------------------------------------------------------
    # Prior
    # -------------------------------------------------------------------------

    ax = axes[0, j]

    ax.scatter(
        observed,
        prior,
        s=18,
        alpha=0.6
    )

    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=1.5
    )

    metrics = calculate_metrics(
        observed,
        prior
    )

    ax.set_title(
        f"{site} — Prior",
        fontweight="bold"
    )

    ax.set_xlabel(
        "Observed CH$_4$ (ppm)"
    )

    ax.set_ylabel(
        "Predicted CH$_4$ (ppm)"
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.grid(
        True,
        alpha=0.3
    )

    ax.text(
        0.05,
        0.95,
        f"R = {metrics['R']:.3f}\n"
        f"R² = {metrics['R2']:.3f}\n"
        f"RMSE = {metrics['RMSE']:.3f} ppm\n"
        f"MAE = {metrics['MAE']:.3f} ppm\n"
        f"FAC2 = {metrics['FAC2']:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9
    )

    # -------------------------------------------------------------------------
    # Posterior
    # -------------------------------------------------------------------------

    ax = axes[1, j]

    ax.scatter(
        observed,
        posterior,
        s=18,
        alpha=0.6
    )

    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=1.5
    )

    metrics = calculate_metrics(
        observed,
        posterior
    )

    ax.set_title(
        f"{site} — Posterior",
        fontweight="bold"
    )

    ax.set_xlabel(
        "Observed CH$_4$ (ppm)"
    )

    ax.set_ylabel(
        "Predicted CH$_4$ (ppm)"
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.grid(
        True,
        alpha=0.3
    )

    ax.text(
        0.05,
        0.95,
        f"R = {metrics['R']:.3f}\n"
        f"R² = {metrics['R2']:.3f}\n"
        f"RMSE = {metrics['RMSE']:.3f} ppm\n"
        f"MAE = {metrics['MAE']:.3f} ppm\n"
        f"FAC2 = {metrics['FAC2']:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9
    )


fig.suptitle(
    "Upper Hunter CH$_4$ — Observed vs Predicted\n"
    "Prior and Posterior Bayesian Inversion, February 2026",
    fontsize=17,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

fig.savefig(
    SCATTER_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print(
    f"Saved: {SCATTER_PNG}"
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 80)
print("FINAL PRIOR vs POSTERIOR SUMMARY")
print("=" * 80)

summary = metrics_df[
    [
        "site",
        "model",
        "N",
        "R",
        "R2",
        "RMSE",
        "MAE",
        "FAC2"
    ]
]

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

print()
print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(TIMESERIES_ALL_PNG)
print(TIMESERIES_ALL_PDF)
print(SCATTER_PNG)
print(METRICS_FILE)
print(PREDICTIONS_FILE)

print()
print("DONE")
print("=" * 80)

