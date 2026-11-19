#!/usr/bin/env python3
"""
===============================================================================
POSTERIOR PREDICTIVE CH4 COMPARISON
===============================================================================

Upper Hunter FLEXPART-WRF Bayesian inversion

Calculate posterior predictive CH4 concentrations using ALL retained
MCMC samples and compare them with Picarro observations.

Model:

    CH4_i = BACKGROUND + sum_j H_ij * E0_j * s_j

where:

    H_ij       = FLEXPART-WRF source-receptor sensitivity
                 [ppm / (kg CH4/s)]

    E0_j       = prior mine emission [kg CH4/s]

    s_j        = MCMC daily scaling factor

    BACKGROUND = 1.85 ppm

MCMC:
    351 parameters
    6000 retained samples

Outputs:
    - posterior predictive median
    - posterior predictive lower 95%
    - posterior predictive upper 95%
    - posterior predictive mean
    - posterior predictive standard deviation
    - prior prediction
    - observations

Plots:
    - three-receptor time series with 95% posterior interval
    - individual receptor time series
    - observation vs posterior median
    - observation vs prior
    - posterior residuals
    - posterior predictive interval coverage

===============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE = (
    Path.home()
    / "flexpart-wrf"
    / "upperhunter_flexpart"
)

HMATRIX_DIR = (
    BASE
    / "hmatrix"
    / "daily_20260105_20260131"
)

MCMC_DIR = (
    HMATRIX_DIR
    / "bayesian_analytical"
    / "mcmc_blocked"
)

OUTPUT_DIR = (
    HMATRIX_DIR
    / "posterior_predictive"
)

H_FILE = (
    HMATRIX_DIR
    / "H_daily_13mine_1124x351_ppm.npy"
)

OBS_FILE = (
    HMATRIX_DIR
    / "observation_vector.csv"
)

SAMPLES_FILE = (
    MCMC_DIR
    / "mcmc_samples.npy"
)

PRIOR_EMISSION_FILE = (
    Path.home()
    / "flexpart-wrf"
    / "emission_UH"
    / "RH_upperHunter"
    / "UpperHuntercoalmines.csv"
)

BACKGROUND_PPM = 1.85

G_TO_KG = 1.0e-3

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

RECEPTORS = [
    "Singleton",
    "Muswellbrook",
    "Merriwa",
]

N_MINES = 13
N_DAYS = 27
N_PARAMS = 351


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_metrics(obs, pred):
    """Calculate standard model-observation statistics."""

    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)

    mask = (
        np.isfinite(obs)
        & np.isfinite(pred)
    )

    obs = obs[mask]
    pred = pred[mask]

    n = len(obs)

    if n == 0:
        return {
            "N": 0,
            "R": np.nan,
            "R2": np.nan,
            "Bias": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "FAC2": np.nan,
            "Mean_obs": np.nan,
            "Mean_pred": np.nan,
        }

    diff = pred - obs

    if n > 1:
        r = np.corrcoef(
            obs,
            pred
        )[0, 1]
    else:
        r = np.nan

    if np.any(obs > 0):

        ratio = (
            pred[obs > 0]
            / obs[obs > 0]
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
        "R2": r ** 2 if np.isfinite(r) else np.nan,
        "Bias": np.mean(diff),
        "RMSE": np.sqrt(
            np.mean(diff ** 2)
        ),
        "MAE": np.mean(
            np.abs(diff)
        ),
        "FAC2": fac2,
        "Mean_obs": np.mean(obs),
        "Mean_pred": np.mean(pred),
    }


# =============================================================================
# CREATE OUTPUT DIRECTORY
# =============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("\n")
print("=" * 80)
print("UPPER HUNTER CH4 POSTERIOR PREDICTIVE ANALYSIS")
print("=" * 80)


# =============================================================================
# READ H MATRIX
# =============================================================================

print("\n" + "=" * 80)
print("READING H MATRIX")
print("=" * 80)

H = np.load(
    H_FILE
)

print(
    f"\nH shape: {H.shape}"
)

print(
    f"H min : {H.min():.10e}"
)

print(
    f"H max : {H.max():.10e}"
)

if H.shape != (1124, 351):

    raise RuntimeError(
        f"Unexpected H shape {H.shape}. "
        "Expected (1124, 351)."
    )


# =============================================================================
# READ OBSERVATIONS
# =============================================================================

print("\n" + "=" * 80)
print("READING OBSERVATIONS")
print("=" * 80)

obs_df = pd.read_csv(
    OBS_FILE
)

print(
    f"\nObservation rows: {len(obs_df)}"
)

required_obs_columns = [
    "row",
    "receptor",
    "datetime_aest",
    "datetime_utc",
    "observation_ppm",
]

missing = [
    c
    for c in required_obs_columns
    if c not in obs_df.columns
]

if missing:

    raise RuntimeError(
        f"Missing observation columns: {missing}"
    )

obs_df["datetime_aest"] = pd.to_datetime(
    obs_df["datetime_aest"],
    errors="coerce"
)

obs_df["datetime_utc"] = pd.to_datetime(
    obs_df["datetime_utc"],
    errors="coerce"
)

obs_df["observation_ppm"] = pd.to_numeric(
    obs_df["observation_ppm"],
    errors="coerce"
)

obs = (
    obs_df["observation_ppm"]
    .values
)


# =============================================================================
# READ PRIOR EMISSIONS
# =============================================================================

print("\n" + "=" * 80)
print("READING PRIOR MINE EMISSIONS")
print("=" * 80)

prior_df = pd.read_csv(
    PRIOR_EMISSION_FILE
)

prior_df["Facility_code_name"] = (
    prior_df["Facility_code_name"]
    .astype(str)
    .str.strip()
)

mine_df = prior_df[
    prior_df["Facility_code_name"]
    .isin(MINES)
].copy()

if len(mine_df) != N_MINES:

    found = set(
        mine_df["Facility_code_name"]
    )

    missing = [
        m
        for m in MINES
        if m not in found
    ]

    raise RuntimeError(
        f"Missing prior emissions: {missing}"
    )

mine_df["order"] = (
    mine_df["Facility_code_name"]
    .map(
        {
            m: i
            for i, m in enumerate(MINES)
        }
    )
)

mine_df = mine_df.sort_values(
    "order"
)

prior_emission_kg_s = (
    mine_df["gCH4/s"]
    .astype(float)
    .values
    * G_TO_KG
)

print("\nPrior emissions:")

for mine, emission in zip(
    MINES,
    prior_emission_kg_s
):

    print(
        f"{mine:>5s} : "
        f"{emission:.6f} kg/s"
    )

print(
    f"\nTotal : "
    f"{prior_emission_kg_s.sum():.6f} kg/s"
)


# =============================================================================
# READ MCMC SAMPLES
# =============================================================================

print("\n" + "=" * 80)
print("READING MCMC SAMPLES")
print("=" * 80)

samples = np.load(
    SAMPLES_FILE
)

print(
    f"\nMCMC samples shape: "
    f"{samples.shape}"
)

print(
    f"dtype: {samples.dtype}"
)

if samples.ndim != 2:

    raise RuntimeError(
        "mcmc_samples.npy must be a "
        "2-dimensional array."
    )

n_samples, n_parameters = samples.shape

if n_parameters != N_PARAMS:

    raise RuntimeError(
        f"Expected {N_PARAMS} parameters, "
        f"found {n_parameters}."
    )

print(
    f"\nNumber of retained samples: "
    f"{n_samples}"
)

print(
    f"Number of parameters: "
    f"{n_parameters}"
)

print(
    f"Minimum scaling factor: "
    f"{samples.min():.6f}"
)

print(
    f"Maximum scaling factor: "
    f"{samples.max():.6f}"
)

if np.any(samples <= 0):

    raise RuntimeError(
        "MCMC samples contain non-positive "
        "scaling factors."
    )


# =============================================================================
# PRIOR 351-ELEMENT EMISSION VECTOR
# =============================================================================

E_prior_351 = np.tile(
    prior_emission_kg_s,
    N_DAYS
)

print(
    f"\nPrior emission vector: "
    f"{E_prior_351.shape}"
)


# =============================================================================
# PRIOR PREDICTION
# =============================================================================

print("\n" + "=" * 80)
print("CALCULATING PRIOR PREDICTION")
print("=" * 80)

prior_enhancement = (
    H @ E_prior_351
)

prior_prediction = (
    BACKGROUND_PPM
    + prior_enhancement
)

print(
    f"\nPrior CH4:"
)

print(
    f"  min  = {prior_prediction.min():.6f}"
)

print(
    f"  max  = {prior_prediction.max():.6f}"
)

print(
    f"  mean = {prior_prediction.mean():.6f}"
)


# =============================================================================
# POSTERIOR PREDICTIVE CALCULATION
# =============================================================================

print("\n" + "=" * 80)
print("CALCULATING POSTERIOR PREDICTIVE DISTRIBUTION")
print("=" * 80)

print(
    "\nCalculating predictions for "
    f"{n_samples} MCMC samples..."
)

# Each MCMC sample contains 351 scaling factors.
#
# Emission for each sample:
#
#   E_sample = E_prior_351 * scaling
#
# We calculate:
#
#   H @ E_sample
#
# for every sample.

# To avoid excessive memory use, process in chunks.

CHUNK_SIZE = 250

posterior_predictions = np.empty(
    (
        n_samples,
        H.shape[0]
    ),
    dtype=np.float64
)

for start in range(
    0,
    n_samples,
    CHUNK_SIZE
):

    end = min(
        start + CHUNK_SIZE,
        n_samples
    )

    scaling_chunk = (
        samples[start:end]
    )

    emissions_chunk = (
        scaling_chunk
        * E_prior_351[None, :]
    )

    # Matrix multiplication:
    #
    # (chunk, 351) @ (351, 1124)
    #
    # = (chunk, 1124)

    enhancement_chunk = (
        emissions_chunk
        @ H.T
    )

    posterior_predictions[
        start:end,
        :
    ] = (
        BACKGROUND_PPM
        + enhancement_chunk
    )

    if (
        start == 0
        or end == n_samples
        or end % 1000 == 0
    ):

        print(
            f"  processed "
            f"{end:5d} / "
            f"{n_samples} samples"
        )


# =============================================================================
# POSTERIOR SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("CALCULATING POSTERIOR PREDICTIVE SUMMARY")
print("=" * 80)

posterior_mean = np.mean(
    posterior_predictions,
    axis=0
)

posterior_median = np.median(
    posterior_predictions,
    axis=0
)

posterior_sd = np.std(
    posterior_predictions,
    axis=0,
    ddof=1
)

posterior_lower95 = np.percentile(
    posterior_predictions,
    2.5,
    axis=0
)

posterior_upper95 = np.percentile(
    posterior_predictions,
    97.5,
    axis=0
)

print(
    f"\nPosterior predictive mean:"
)

print(
    f"  min  = {posterior_mean.min():.6f}"
)

print(
    f"  max  = {posterior_mean.max():.6f}"
)

print(
    f"  mean = {posterior_mean.mean():.6f}"
)

print(
    f"\nPosterior predictive median:"
)

print(
    f"  min  = {posterior_median.min():.6f}"
)

print(
    f"  max  = {posterior_median.max():.6f}"
)

print(
    f"  mean = {posterior_median.mean():.6f}"
)


# =============================================================================
# ADD RESULTS TO DATAFRAME
# =============================================================================

obs_df["CH4_prior_ppm"] = (
    prior_prediction
)

obs_df["CH4_posterior_mean_ppm"] = (
    posterior_mean
)

obs_df["CH4_posterior_median_ppm"] = (
    posterior_median
)

obs_df["CH4_posterior_sd_ppm"] = (
    posterior_sd
)

obs_df["CH4_posterior_lower95_ppm"] = (
    posterior_lower95
)

obs_df["CH4_posterior_upper95_ppm"] = (
    posterior_upper95
)

obs_df["prior_residual_ppm"] = (
    prior_prediction - obs
)

obs_df["posterior_mean_residual_ppm"] = (
    posterior_mean - obs
)

obs_df["posterior_median_residual_ppm"] = (
    posterior_median - obs
)

obs_df["posterior_interval_contains_obs"] = (
    (obs >= posterior_lower95)
    & (obs <= posterior_upper95)
)


# =============================================================================
# SAVE PREDICTIONS
# =============================================================================

prediction_file = (
    OUTPUT_DIR
    / "posterior_predictive_predictions.csv"
)

obs_df.to_csv(
    prediction_file,
    index=False
)

print(
    f"\nSaved:"
    f"\n{prediction_file}"
)


# =============================================================================
# SAVE POSTERIOR PREDICTIVE ARRAY
# =============================================================================

summary_array_file = (
    OUTPUT_DIR
    / "posterior_predictive_summary.npz"
)

np.savez_compressed(
    summary_array_file,
    mean=posterior_mean,
    median=posterior_median,
    sd=posterior_sd,
    lower95=posterior_lower95,
    upper95=posterior_upper95,
)

print(
    f"Saved:"
    f"\n{summary_array_file}"
)


# =============================================================================
# OVERALL METRICS
# =============================================================================

print("\n" + "=" * 80)
print("OVERALL MODEL PERFORMANCE")
print("=" * 80)

prior_metrics = calculate_metrics(
    obs,
    prior_prediction
)

posterior_mean_metrics = calculate_metrics(
    obs,
    posterior_mean
)

posterior_median_metrics = calculate_metrics(
    obs,
    posterior_median
)

overall_metrics = pd.DataFrame(
    [
        {
            "Model": "Prior",
            **prior_metrics
        },
        {
            "Model": "Posterior mean",
            **posterior_mean_metrics
        },
        {
            "Model": "Posterior median",
            **posterior_median_metrics
        },
    ]
)

print(
    "\n"
    + overall_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

overall_metrics.to_csv(
    OUTPUT_DIR / "overall_metrics.csv",
    index=False
)


# =============================================================================
# POSTERIOR 95% INTERVAL COVERAGE
# =============================================================================

coverage = np.mean(
    obs_df[
        "posterior_interval_contains_obs"
    ]
)

print("\n" + "=" * 80)
print("POSTERIOR PREDICTIVE INTERVAL COVERAGE")
print("=" * 80)

print(
    f"\nObservations inside posterior 95% interval:"
    f" {coverage * 100:.2f}%"
)

print(
    f"Observations outside posterior 95% interval:"
    f" {(1 - coverage) * 100:.2f}%"
)


# =============================================================================
# RECEPTOR METRICS
# =============================================================================

print("\n" + "=" * 80)
print("RECEPTOR-BY-RECEPTOR RESULTS")
print("=" * 80)

receptor_results = []

for receptor in RECEPTORS:

    subset = obs_df[
        obs_df["receptor"]
        .str.lower()
        == receptor.lower()
    ].copy()

    if len(subset) == 0:

        print(
            f"\nNo observations for {receptor}"
        )

        continue

    obs_r = (
        subset["observation_ppm"]
        .values
    )

    prior_r = (
        subset["CH4_prior_ppm"]
        .values
    )

    mean_r = (
        subset[
            "CH4_posterior_mean_ppm"
        ].values
    )

    median_r = (
        subset[
            "CH4_posterior_median_ppm"
        ].values
    )

    lower_r = (
        subset[
            "CH4_posterior_lower95_ppm"
        ].values
    )

    upper_r = (
        subset[
            "CH4_posterior_upper95_ppm"
        ].values
    )

    mp = calculate_metrics(
        obs_r,
        prior_r
    )

    mm = calculate_metrics(
        obs_r,
        mean_r
    )

    mmed = calculate_metrics(
        obs_r,
        median_r
    )

    coverage_r = np.mean(
        (obs_r >= lower_r)
        & (obs_r <= upper_r)
    )

    print(
        f"\n{receptor}"
    )

    print(
        f"  N = {len(subset)}"
    )

    print(
        f"\n  Prior:"
    )

    print(
        f"    R    = {mp['R']:.6f}"
    )

    print(
        f"    RMSE = {mp['RMSE']:.6f} ppm"
    )

    print(
        f"    MAE  = {mp['MAE']:.6f} ppm"
    )

    print(
        f"    Bias = {mp['Bias']:.6f} ppm"
    )

    print(
        f"\n  Posterior mean:"
    )

    print(
        f"    R    = {mm['R']:.6f}"
    )

    print(
        f"    RMSE = {mm['RMSE']:.6f} ppm"
    )

    print(
        f"    MAE  = {mm['MAE']:.6f} ppm"
    )

    print(
        f"    Bias = {mm['Bias']:.6f} ppm"
    )

    print(
        f"\n  Posterior median:"
    )

    print(
        f"    R    = {mmed['R']:.6f}"
    )

    print(
        f"    RMSE = {mmed['RMSE']:.6f} ppm"
    )

    print(
        f"    MAE  = {mmed['MAE']:.6f} ppm"
    )

    print(
        f"    Bias = {mmed['Bias']:.6f} ppm"
    )

    print(
        f"\n  95% interval coverage:"
        f" {coverage_r * 100:.2f}%"
    )

    receptor_results.append(
        {
            "Receptor": receptor,
            "Model": "Prior",
            **mp
        }
    )

    receptor_results.append(
        {
            "Receptor": receptor,
            "Model": "Posterior mean",
            **mm
        }
    )

    receptor_results.append(
        {
            "Receptor": receptor,
            "Model": "Posterior median",
            **mmed
        }
    )

receptor_metrics = pd.DataFrame(
    receptor_results
)

receptor_metrics.to_csv(
    OUTPUT_DIR / "receptor_metrics.csv",
    index=False
)


# =============================================================================
# TIME SERIES — INDIVIDUAL RECEPTORS
# =============================================================================

print("\n" + "=" * 80)
print("CREATING TIME-SERIES PLOTS")
print("=" * 80)

for receptor in RECEPTORS:

    subset = obs_df[
        obs_df["receptor"]
        .str.lower()
        == receptor.lower()
    ].copy()

    if len(subset) == 0:
        continue

    subset = subset.sort_values(
        "datetime_aest"
    )

    fig, ax = plt.subplots(
        figsize=(15, 6)
    )

    x = subset[
        "datetime_aest"
    ]

    obs_r = subset[
        "observation_ppm"
    ].values

    prior_r = subset[
        "CH4_prior_ppm"
    ].values

    median_r = subset[
        "CH4_posterior_median_ppm"
    ].values

    lower_r = subset[
        "CH4_posterior_lower95_ppm"
    ].values

    upper_r = subset[
        "CH4_posterior_upper95_ppm"
    ].values

    # Posterior 95% interval
    ax.fill_between(
        x,
        lower_r,
        upper_r,
        alpha=0.25,
        label="Posterior 95% credible interval"
    )

    # Observation
    ax.plot(
        x,
        obs_r,
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="Picarro observation"
    )

    # Prior
    ax.plot(
        x,
        prior_r,
        linewidth=1.2,
        label="Prior mine emissions"
    )

    # Posterior median
    ax.plot(
        x,
        median_r,
        linewidth=1.8,
        label="Posterior median"
    )

    ax.axhline(
        BACKGROUND_PPM,
        linestyle="--",
        linewidth=1.0,
        label=f"Background = {BACKGROUND_PPM:.2f} ppm"
    )

    ax.set_title(
        f"{receptor} — Posterior Predictive CH₄"
    )

    ax.set_xlabel(
        "Date (AEST)"
    )

    ax.set_ylabel(
        "CH₄ concentration (ppm)"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend(
        loc="best"
    )

    fig.autofmt_xdate()

    plt.tight_layout()

    outfile = (
        OUTPUT_DIR
        / f"{receptor.lower()}_posterior_predictive.png"
    )

    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(
        f"Saved: {outfile}"
    )


# =============================================================================
# ALL THREE RECEPTORS
# =============================================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(15, 13)
)

for ax, receptor in zip(
    axes,
    RECEPTORS
):

    subset = obs_df[
        obs_df["receptor"]
        .str.lower()
        == receptor.lower()
    ].copy()

    if len(subset) == 0:

        ax.set_title(
            f"{receptor} — no observations"
        )

        continue

    subset = subset.sort_values(
        "datetime_aest"
    )

    x = subset[
        "datetime_aest"
    ]

    ax.fill_between(
        x,
        subset[
            "CH4_posterior_lower95_ppm"
        ],
        subset[
            "CH4_posterior_upper95_ppm"
        ],
        alpha=0.25,
        label="Posterior 95%"
    )

    ax.plot(
        x,
        subset[
            "observation_ppm"
        ],
        marker="o",
        markersize=2.5,
        linewidth=1.0,
        label="Observation"
    )

    ax.plot(
        x,
        subset[
            "CH4_prior_ppm"
        ],
        linewidth=1.1,
        label="Prior"
    )

    ax.plot(
        x,
        subset[
            "CH4_posterior_median_ppm"
        ],
        linewidth=1.5,
        label="Posterior median"
    )

    ax.axhline(
        BACKGROUND_PPM,
        linestyle="--",
        linewidth=0.9,
        label="Background"
    )

    ax.set_title(
        receptor
    )

    ax.set_ylabel(
        "CH₄ (ppm)"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend(
        loc="best"
    )

axes[-1].set_xlabel(
    "Date (AEST)"
)

fig.suptitle(
    "Upper Hunter CH₄ — Prior and Posterior Predictive Comparison",
    fontsize=16
)

plt.tight_layout(
    rect=[0, 0, 1, 0.97]
)

outfile = (
    OUTPUT_DIR
    / "all_three_receptors_posterior_predictive.png"
)

plt.savefig(
    outfile,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print(
    f"Saved: {outfile}"
)


# =============================================================================
# OBSERVATION VS POSTERIOR MEDIAN
# =============================================================================

for receptor in RECEPTORS:

    subset = obs_df[
        obs_df["receptor"]
        .str.lower()
        == receptor.lower()
    ].copy()

    if len(subset) == 0:
        continue

    obs_r = subset[
        "observation_ppm"
    ].values

    pred_r = subset[
        "CH4_posterior_median_ppm"
    ].values

    mask = (
        np.isfinite(obs_r)
        & np.isfinite(pred_r)
    )

    obs_r = obs_r[mask]
    pred_r = pred_r[mask]

    if len(obs_r) == 0:
        continue

    minval = min(
        obs_r.min(),
        pred_r.min()
    )

    maxval = max(
        obs_r.max(),
        pred_r.max()
    )

    margin = (
        0.05
        * (maxval - minval)
    )

    minval -= margin
    maxval += margin

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.scatter(
        obs_r,
        pred_r,
        s=22,
        alpha=0.65
    )

    ax.plot(
        [minval, maxval],
        [minval, maxval],
        linestyle="--",
        linewidth=1.2,
        label="1:1"
    )

    ax.set_xlim(
        minval,
        maxval
    )

    ax.set_ylim(
        minval,
        maxval
    )

    ax.set_xlabel(
        "Picarro observed CH₄ (ppm)"
    )

    ax.set_ylabel(
        "Posterior median predicted CH₄ (ppm)"
    )

    ax.set_title(
        f"{receptor} — Posterior Predictive"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    outfile = (
        OUTPUT_DIR
        / f"{receptor.lower()}_posterior_scatter.png"
    )

    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(
        f"Saved: {outfile}"
    )


# =============================================================================
# RESIDUAL TIME SERIES
# =============================================================================

for receptor in RECEPTORS:

    subset = obs_df[
        obs_df["receptor"]
        .str.lower()
        == receptor.lower()
    ].copy()

    if len(subset) == 0:
        continue

    subset = subset.sort_values(
        "datetime_aest"
    )

    fig, ax = plt.subplots(
        figsize=(15, 5)
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.plot(
        subset["datetime_aest"],
        subset["posterior_median_residual_ppm"],
        linewidth=1.4,
        label="Posterior median residual"
    )

    ax.set_title(
        f"{receptor} — Posterior Residual"
    )

    ax.set_xlabel(
        "Date (AEST)"
    )

    ax.set_ylabel(
        "Predicted − observed CH₄ (ppm)"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    fig.autofmt_xdate()

    plt.tight_layout()

    outfile = (
        OUTPUT_DIR
        / f"{receptor.lower()}_posterior_residual.png"
    )

    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(
        f"Saved: {outfile}"
    )


# =============================================================================
# SAVE COVERAGE SUMMARY
# =============================================================================

coverage_df = pd.DataFrame(
    [
        {
            "Scope": "All receptors",
            "N": len(obs_df),
            "coverage_95_percent": coverage * 100.0,
        }
    ]
)

for receptor in RECEPTORS:

    subset = obs_df[
        obs_df["receptor"]
        .str.lower()
        == receptor.lower()
    ]

    if len(subset) == 0:
        continue

    cov = np.mean(
        subset[
            "posterior_interval_contains_obs"
        ]
    )

    coverage_df = pd.concat(
        [
            coverage_df,
            pd.DataFrame(
                [
                    {
                        "Scope": receptor,
                        "N": len(subset),
                        "coverage_95_percent": cov * 100.0,
                    }
                ]
            ),
        ],
        ignore_index=True
    )

coverage_df.to_csv(
    OUTPUT_DIR
    / "posterior_95_interval_coverage.csv",
    index=False
)


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print("\n")
print("=" * 80)
print("POSTERIOR PREDICTIVE ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"\nOutput directory:"
    f"\n{OUTPUT_DIR}"
)

print("\nOverall metrics:")

print(
    overall_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print(
    "\n95% posterior predictive coverage:"
    f" {coverage * 100:.2f}%"
)

print("\nDone.")

