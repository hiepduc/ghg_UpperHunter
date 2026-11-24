
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot FLEXPART-WRF CH4 prior/posterior predictions against Picarro observations
for Singleton, Muswellbrook and Merriwa, Jul 2026.

Uses existing:
  1. FLEXPART H matrix
  2. MCMC posterior samples
  3. Picarro observations

No MCMC rerun is required.

Model:
    CH4_total = BACKGROUND_PPM + H @ scaling_factors

Prior:
    scaling_factor = 1 for all 403 source/day parameters

Posterior:
    Uses the saved MCMC samples.

Output:
    - FLEXPART_MCMC_prior_posterior_CH4_Jul2026.png
    - FLEXPART_MCMC_prior_posterior_CH4_Jul2026.pdf
    - FLEXPART_MCMC_prior_posterior_predictions_Jul2026.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

BASE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/"
    "flexpart-wrf/upperhunter_flexpart"
)

H_FILE = os.path.join(
    BASE,
    "H_matrix_flexpart/2026-07",
    "H_flexpart_0_10m_13mine_Jul2026.csv"
)

MCMC_DIR = os.path.join(
    BASE,
    "H_matrix_flexpart/2026-07",
    "MCMC_13mine_daily_Jul2026"
)

POSTERIOR_FILE = os.path.join(
    MCMC_DIR,
    "FLEXPART_MCMC_posterior_samples_Jul2026.npz"
)

OBS_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/"
    "piccaro_obs/Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

OUTPUT_DIR = MCMC_DIR

PNG_FILE = os.path.join(
    OUTPUT_DIR,
    "FLEXPART_MCMC_prior_posterior_CH4_Jul2026.png"
)

PDF_FILE = os.path.join(
    OUTPUT_DIR,
    "FLEXPART_MCMC_prior_posterior_CH4_Jul2026.pdf"
)

CSV_FILE = os.path.join(
    OUTPUT_DIR,
    "FLEXPART_MCMC_prior_posterior_predictions_Jul2026.csv"
)

# ============================================================
# PARAMETERS
# ============================================================

BACKGROUND_PPM = 1.85

START_DATE = pd.Timestamp("2026-07-01 00:00:00")
END_DATE = pd.Timestamp("2026-07-31 23:00:00")

# Picarro data and FLEXPART H obs_time_aest are both AEST.
OBS_TIME_FORMAT = "%d/%m/%Y %H:%M"

# ============================================================
# PLOT FONT SETTINGS
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_metrics(obs, pred):
    """
    Calculate standard model-observation statistics.
    """

    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(pred)

    obs = obs[mask]
    pred = pred[mask]

    if len(obs) == 0:
        return {
            "N": 0,
            "R": np.nan,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan,
            "FAC2": np.nan,
        }

    if (
        len(obs) > 1
        and np.std(obs) > 0
        and np.std(pred) > 0
    ):
        R = np.corrcoef(obs, pred)[0, 1]
    else:
        R = np.nan

    R2 = R ** 2 if np.isfinite(R) else np.nan

    error = pred - obs

    RMSE = np.sqrt(np.mean(error ** 2))
    MAE = np.mean(np.abs(error))
    Bias = np.mean(error)

    positive = (obs > 0) & (pred > 0)

    if np.any(positive):
        ratio = pred[positive] / obs[positive]
        FAC2 = np.mean(
            (ratio >= 0.5) & (ratio <= 2.0)
        )
    else:
        FAC2 = np.nan

    return {
        "N": len(obs),
        "R": R,
        "R2": R2,
        "RMSE": RMSE,
        "MAE": MAE,
        "Bias": Bias,
        "FAC2": FAC2,
    }


def print_metrics(receptor, model_name, metrics):
    """
    Print model performance statistics.
    """

    print(f"{model_name}:")
    print(
        f"  N={metrics['N']}, "
        f"R={metrics['R']:.5f}, "
        f"R2={metrics['R2']:.5f}, "
        f"RMSE={metrics['RMSE']:.5f}, "
        f"MAE={metrics['MAE']:.5f}, "
        f"Bias={metrics['Bias']:.5f}, "
        f"FAC2={metrics['FAC2']:.5f}"
    )


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 90)
print("FLEXPART-WRF CH4 PRIOR / POSTERIOR PREDICTION PLOT")
print("=" * 90)

for f in [H_FILE, POSTERIOR_FILE, OBS_FILE]:

    if not os.path.exists(f):
        raise FileNotFoundError(
            f"File not found:\n{f}"
        )

print("\nInput files:")
print("H matrix:")
print(H_FILE)

print("\nPosterior:")
print(POSTERIOR_FILE)

print("\nObservations:")
print(OBS_FILE)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# READ H MATRIX
# ============================================================

print("\n" + "=" * 90)
print("READING H MATRIX")
print("=" * 90)

H_df = pd.read_csv(H_FILE)

print(f"H dataframe shape: {H_df.shape}")

required_metadata = [
    "receptor",
    "release_utc",
    "obs_time_aest",
    "flexpart_file",
]

for col in required_metadata:

    if col not in H_df.columns:
        raise ValueError(
            f"Missing H metadata column: {col}"
        )

source_columns = [
    c for c in H_df.columns
    if c not in required_metadata
]

print(
    f"Number of source/day parameters: "
    f"{len(source_columns)}"
)

if len(source_columns) != 403:
    raise ValueError(
        f"Expected 403 source/day columns, "
        f"found {len(source_columns)}"
    )

H = H_df[source_columns].to_numpy(dtype=float)

print(f"Numerical H shape: {H.shape}")
print(f"H minimum: {np.nanmin(H):.8g}")
print(f"H maximum: {np.nanmax(H):.8g}")
print(f"H nonzero values: {np.count_nonzero(H)}")

# ============================================================
# READ POSTERIOR
# ============================================================

print("\n" + "=" * 90)
print("READING MCMC POSTERIOR")
print("=" * 90)

npz = np.load(POSTERIOR_FILE)

print("Arrays stored in posterior file:")
print(npz.files)

if "samples" not in npz.files:
    raise ValueError(
        "Expected array 'samples' in posterior NPZ file."
    )

samples = np.asarray(
    npz["samples"],
    dtype=float
)

print(f"Posterior sample shape: {samples.shape}")

if samples.ndim != 2:
    raise ValueError(
        f"Expected posterior samples with shape "
        f"(n_samples, n_parameters), got {samples.shape}"
    )

if samples.shape[1] != len(source_columns):
    raise ValueError(
        f"Posterior has {samples.shape[1]} parameters "
        f"but H has {len(source_columns)} source/day columns."
    )

print(f"Number of posterior samples: {samples.shape[0]}")
print(f"Number of parameters: {samples.shape[1]}")

# ============================================================
# CONVERT H TIMES
# ============================================================

H_df["obs_time_aest"] = pd.to_datetime(
    H_df["obs_time_aest"],
    errors="coerce"
)

H_df = H_df[
    (H_df["obs_time_aest"] >= START_DATE) &
    (H_df["obs_time_aest"] <= END_DATE)
].copy()

print("\nH rows in July 2026:")
print(len(H_df))

if H_df.empty:
    raise ValueError(
        "No H matrix rows found for July 2026. "
        "Check the obs_time_aest column and date range."
    )

# ============================================================
# READ PICARRO OBSERVATIONS
# ============================================================

print("\n" + "=" * 90)
print("READING PICARRO OBSERVATIONS")
print("=" * 90)

obs = pd.read_csv(OBS_FILE)

print(f"Observation dataframe shape: {obs.shape}")

print("\nObservation columns:")
for c in obs.columns:
    print("  ", c)

# ------------------------------------------------------------
# Parse Datetime
# ------------------------------------------------------------

if "Datetime" not in obs.columns:
    raise ValueError(
        "Picarro file does not contain 'Datetime' column."
    )

obs["Datetime"] = pd.to_datetime(
    obs["Datetime"],
    format=OBS_TIME_FORMAT,
    errors="coerce"
)

print(
    "\nUnparsed observation timestamps:",
    obs["Datetime"].isna().sum()
)

obs = obs[
    (obs["Datetime"] >= START_DATE) &
    (obs["Datetime"] <= END_DATE)
].copy()

# ------------------------------------------------------------
# Clean receptor columns
# ------------------------------------------------------------

required_obs_columns = [
    "Singleton cleaned",
    "Muswellbrook cleaned",
    "Merriwa cleaned",
]

for c in required_obs_columns:

    if c not in obs.columns:
        raise ValueError(
            f"Missing observation column: {c}"
        )

obs["Singleton_clean"] = pd.to_numeric(
    obs["Singleton cleaned"],
    errors="coerce"
)

obs["Muswellbrook_clean"] = pd.to_numeric(
    obs["Muswellbrook cleaned"],
    errors="coerce"
)

obs["Merriwa_clean"] = pd.to_numeric(
    obs["Merriwa cleaned"],
    errors="coerce"
)

print("\nMay observations:")

print(
    "Singleton:",
    obs["Singleton_clean"].notna().sum()
)

print(
    "Muswellbrook:",
    obs["Muswellbrook_clean"].notna().sum()
)

print(
    "Merriwa:",
    obs["Merriwa_clean"].notna().sum()
)

# ============================================================
# BUILD PRIOR PREDICTION
# ============================================================

print("\n" + "=" * 90)
print("CALCULATING PRIOR PREDICTION")
print("=" * 90)

# Prior scaling factor = 1 for every mine/day.
prior_scaling = np.ones(len(source_columns))

prior_enhancement = H @ prior_scaling

prior_prediction = (
    BACKGROUND_PPM + prior_enhancement
)

print(
    f"Prior prediction range: "
    f"{np.nanmin(prior_prediction):.6f} - "
    f"{np.nanmax(prior_prediction):.6f} ppm"
)

# ============================================================
# CALCULATE POSTERIOR PREDICTION
# ============================================================

print("\n" + "=" * 90)
print("CALCULATING POSTERIOR PREDICTION")
print("=" * 90)

print(
    "Calculating posterior predictions from "
    f"{samples.shape[0]:,} samples..."
)

# H shape:
#     observations x parameters
#
# samples:
#     posterior samples x parameters
#
# H @ samples.T gives:
#     observations x posterior_samples

posterior_predictions = (
    BACKGROUND_PPM
    + H @ samples.T
)

print(
    "Posterior prediction array shape:",
    posterior_predictions.shape
)

# ------------------------------------------------------------
# Posterior summary
# ------------------------------------------------------------

posterior_mean = np.mean(
    posterior_predictions,
    axis=1
)

posterior_median = np.median(
    posterior_predictions,
    axis=1
)

posterior_q025 = np.percentile(
    posterior_predictions,
    2.5,
    axis=1
)

posterior_q975 = np.percentile(
    posterior_predictions,
    97.5,
    axis=1
)

# ============================================================
# BUILD PREDICTION DATAFRAME
# ============================================================

pred_df = H_df[
    [
        "receptor",
        "release_utc",
        "obs_time_aest",
        "flexpart_file",
    ]
].copy()

pred_df["prior_prediction_ppm"] = prior_prediction

pred_df["posterior_mean_ppm"] = posterior_mean
pred_df["posterior_median_ppm"] = posterior_median
pred_df["posterior_q025_ppm"] = posterior_q025
pred_df["posterior_q975_ppm"] = posterior_q975

# ============================================================
# MATCH OBSERVATIONS
# ============================================================

print("\n" + "=" * 90)
print("MATCHING OBSERVATIONS")
print("=" * 90)

obs_lookup_singleton = (
    obs[
        ["Datetime", "Singleton_clean"]
    ]
    .dropna(subset=["Singleton_clean"])
    .drop_duplicates(subset=["Datetime"])
    .set_index("Datetime")["Singleton_clean"]
)

obs_lookup_muswellbrook = (
    obs[
        ["Datetime", "Muswellbrook_clean"]
    ]
    .dropna(subset=["Muswellbrook_clean"])
    .drop_duplicates(subset=["Datetime"])
    .set_index("Datetime")["Muswellbrook_clean"]
)

obs_lookup_merriwa = (
    obs[
        ["Datetime", "Merriwa_clean"]
    ]
    .dropna(subset=["Merriwa_clean"])
    .drop_duplicates(subset=["Datetime"])
    .set_index("Datetime")["Merriwa_clean"]
)


def get_observation(row):
    """
    Match the H matrix observation time to the correct
    Picarro station observation.
    """

    t = row["obs_time_aest"]

    receptor = str(
        row["receptor"]
    ).strip().lower()

    if receptor == "singleton":
        return obs_lookup_singleton.get(t, np.nan)

    elif receptor == "muswellbrook":
        return obs_lookup_muswellbrook.get(t, np.nan)

    elif receptor == "merriwa":
        return obs_lookup_merriwa.get(t, np.nan)

    else:
        return np.nan


pred_df["observation_ppm"] = pred_df.apply(
    get_observation,
    axis=1
)

# ============================================================
# PRINT MATCHING SUMMARY
# ============================================================

print("\nMatched observation counts by receptor:")

for receptor in [
    "Singleton",
    "Muswellbrook",
    "Merriwa",
]:

    sub = pred_df[
        pred_df["receptor"].astype(str).str.strip().str.lower()
        == receptor.lower()
    ].copy()

    n_h = len(sub)

    n_obs = sub[
        "observation_ppm"
    ].notna().sum()

    print(
        f"{receptor:15s}: "
        f"H rows = {n_h:4d}, "
        f"matched observations = {n_obs:4d}"
    )

# ============================================================
# CHECK RECEPTOR NAMES
# ============================================================

print("\nReceptor names found in H matrix:")

print(
    pred_df["receptor"]
    .value_counts(dropna=False)
    .to_string()
)

# ============================================================
# METRICS
# ============================================================

print("\n" + "=" * 90)
print("MODEL METRICS")
print("=" * 90)

metrics_rows = []

for receptor in [
    "Singleton",
    "Muswellbrook",
    "Merriwa",
]:

    sub = pred_df[
        pred_df["receptor"].astype(str).str.strip().str.lower()
        == receptor.lower()
    ].copy()

    mask = sub["observation_ppm"].notna()

    y = sub.loc[
        mask,
        "observation_ppm"
    ].to_numpy()

    prior = sub.loc[
        mask,
        "prior_prediction_ppm"
    ].to_numpy()

    post = sub.loc[
        mask,
        "posterior_mean_ppm"
    ].to_numpy()

    prior_metrics = calculate_metrics(y, prior)
    post_metrics = calculate_metrics(y, post)

    print(f"\n{receptor}")
    print("-" * 70)

    print_metrics(
        receptor,
        "Prior",
        prior_metrics
    )

    print_metrics(
        receptor,
        "Posterior",
        post_metrics
    )

    metrics_rows.append({
        "receptor": receptor,
        "model": "prior",
        **prior_metrics,
    })

    metrics_rows.append({
        "receptor": receptor,
        "model": "posterior",
        **post_metrics,
    })

metrics_df = pd.DataFrame(metrics_rows)

# ============================================================
# SAVE PREDICTION CSV
# ============================================================

pred_df.to_csv(
    CSV_FILE,
    index=False
)

print("\nPrediction data saved:")
print(CSV_FILE)

# ============================================================
# PLOT
# ============================================================

print("\n" + "=" * 90)
print("CREATING FIGURE")
print("=" * 90)

# Three panels are required for the three receptors.
fig, axes = plt.subplots(
    3,
    1,
    figsize=(15, 13),
    sharex=True
)

receptors = [
    "Singleton",
    "Muswellbrook",
    "Merriwa",
]

for ax, receptor in zip(axes, receptors):

    print(f"\nPlotting {receptor}...")

    sub = pred_df[
        pred_df["receptor"].astype(str).str.strip().str.lower()
        == receptor.lower()
    ].copy()

    sub = sub.sort_values("obs_time_aest")

    if sub.empty:
        print(
            f"WARNING: No FLEXPART H matrix rows "
            f"found for {receptor}."
        )
        ax.text(
            0.5,
            0.5,
            f"No FLEXPART data for {receptor}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=13
        )
        ax.set_title(
            receptor,
            fontsize=15,
            fontweight="bold"
        )
        continue

    t = sub["obs_time_aest"]

    # --------------------------------------------------------
    # Posterior 95% interval
    # --------------------------------------------------------

    ax.fill_between(
        t,
        sub["posterior_q025_ppm"].to_numpy(),
        sub["posterior_q975_ppm"].to_numpy(),
        alpha=0.20,
        label="Posterior 95% interval"
    )

    # --------------------------------------------------------
    # Prior prediction
    # --------------------------------------------------------

    ax.plot(
        t,
        sub["prior_prediction_ppm"],
        linewidth=1.6,
        linestyle="--",
        label="Prior prediction"
    )

    # --------------------------------------------------------
    # Posterior mean
    # --------------------------------------------------------

    ax.plot(
        t,
        sub["posterior_mean_ppm"],
        linewidth=2.0,
        label="Posterior mean"
    )

    # --------------------------------------------------------
    # Observations
    # --------------------------------------------------------

    obs_mask = sub["observation_ppm"].notna()

    obs_t = t[obs_mask]

    obs_y = sub.loc[
        obs_mask,
        "observation_ppm"
    ]

    if len(obs_y) > 0:

        ax.plot(
            obs_t,
            obs_y,
            linewidth=1.4,
            marker="o",
            markersize=3.5,
            alpha=0.85,
            label="Picarro observation",
            zorder=6
        )

    else:

        print(
            f"WARNING: No matched Picarro observations "
            f"for {receptor}."
        )

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    ax.axhline(
        BACKGROUND_PPM,
        linestyle=":",
        linewidth=1.3,
        label="Background = 1.85 ppm"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    y = sub.loc[
        obs_mask,
        "observation_ppm"
    ].to_numpy()

    prior = sub.loc[
        obs_mask,
        "prior_prediction_ppm"
    ].to_numpy()

    post = sub.loc[
        obs_mask,
        "posterior_mean_ppm"
    ].to_numpy()

    prior_m = calculate_metrics(y, prior)
    post_m = calculate_metrics(y, post)

    if len(y) > 0:

        text = (
            f"N = {len(y)}\n"
            f"Prior:     R = {prior_m['R']:.3f}, "
            f"RMSE = {prior_m['RMSE']:.3f} ppm\n"
            f"Posterior: R = {post_m['R']:.3f}, "
            f"RMSE = {post_m['RMSE']:.3f} ppm"
        )

    else:

        text = "N = 0 matched observations"

    ax.text(
        0.015,
        0.97,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85
        )
    )

    # --------------------------------------------------------
    # Panel formatting
    # --------------------------------------------------------

    ax.set_title(
        receptor,
        fontsize=15,
        fontweight="bold"
    )

    ax.set_ylabel(
        "CH₄ (ppmv)",
        fontsize=14
    )

    ax.grid(
        True,
        color="0.85",
        linestyle="-",
        alpha=0.5,
        linewidth=0.6
    )

    ax.set_axisbelow(True)

    ax.legend(
        loc="upper right",
        fontsize=10,
        ncol=2,
        frameon=True
    )

# ============================================================
# X-AXIS
# ============================================================

axes[-1].set_xlabel(
    "July 2026 (AEST)",
    fontsize=14
)

# ============================================================
# FIGURE TITLE
# ============================================================

fig.suptitle(
    "FLEXPART-WRF CH₄: Prior and Posterior Predictions vs Picarro Observations",
    fontsize=16,
    fontweight="bold"
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)

# ============================================================
# SAVE
# ============================================================

fig.savefig(
    PNG_FILE,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    PDF_FILE,
    bbox_inches="tight"
)

print("\nFigures saved:")
print(PNG_FILE)
print(PDF_FILE)

plt.show()
plt.close(fig)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("COMPLETE")
print("=" * 90)

print("\nPNG:")
print(PNG_FILE)

print("\nPDF:")
print(PDF_FILE)

print("\nCSV:")
print(CSV_FILE)

print("\nMetrics:")
print(metrics_df.to_string(index=False))

print("\n" + "=" * 90)

