#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Direct Gibbs sampler for the non-negative truncated Bayesian posterior
of 13 Upper Hunter coal-mine daily scaling factors.

This implementation calculates the Gibbs conditional distributions
directly from the FLEXPART H matrix, observations, background and prior.

It does NOT load the analytical posterior mean or covariance.

Model
-----
    y = background + H @ scaling_factors + observation_error

    observation_error ~ N(0, sigma_obs**2 I)

    scaling_factors_j ~ N(prior_mean, prior_sigma**2), scaling_factors_j >= 0

The H matrix already contains the prior mine emissions. Do NOT multiply
H by the prior emissions again.

Parameter ordering
------------------
Date-major:
    OC1_20260501, OC2_20260501, ..., OC13_20260501,
    OC1_20260502, OC2_20260502, ..., OC13_20260502,
    ...

For comparison with the independent analytical-posterior Gibbs sampler,
the two implementations must use exactly the same H matrix, observations,
background, prior and parameter ordering.
"""

from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri


# ================================================================
# 1. USER SETTINGS
# ================================================================

# Try the current scratch paths first, then the earlier home-directory
# paths used in the previous version of this workflow.

H_MATRIX_CANDIDATES = [
    Path(
        "/mnt/scratch_lustre/nzem_meim_scratch/"
        "flexpart-wrf/upperhunter_flexpart/"
        "H_matrix_flexpart/2026-05/"
        "H_flexpart_0_10m_13mine_May2026.csv"
    ),
    Path(
        "/home/duch/flexpart-wrf/upperhunter_flexpart/"
        "H_matrix_flexpart/H_flexpart_0_10m_13mine_May2026.csv"
    ),
]

OBSERVATION_CANDIDATES = [
    Path(
        "/mnt/scratch_lustre/nzem_meim_scratch/data/piccaro_obs/"
        "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
    ),
    Path(
        "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter/"
        "tapm_output/Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
    ),
]

# Model settings
BACKGROUND_PPM = 1.85

# Observation uncertainty
SIGMA_OBS = 0.05

# Prior: scaling factor ~ Normal(1, 1^2), truncated at zero
PRIOR_MEAN = 1.0
PRIOR_SIGMA = 1.0

# MCMC settings
N_CHAINS = 4
BURN_IN = 5000
N_SAMPLES = 5000
THIN = 5

# Reproducibility
RANDOM_SEED = 20261041

# Output folder
OUTPUT_FOLDER_NAME = "MCMC_direct_Gibbs_13mine_daily_May2026"
OUTPUT_DIR = Path("/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart/inversion_results/2026-05") / OUTPUT_FOLDER_NAME


# ================================================================
# 2. FIND INPUT FILES
# ================================================================

def find_existing_file(candidates, description):

    for path in candidates:
        if path.is_file():
            print(f"{description}:")
            print(f"  {path}")
            return path

    print(f"\nERROR: Cannot find {description}.")
    print("Checked the following paths:")

    for path in candidates:
        print(f"  {path}")

    sys.exit(1)


H_FILE = find_existing_file(
    H_MATRIX_CANDIDATES,
    "FLEXPART H matrix"
)

OBS_FILE = find_existing_file(
    OBSERVATION_CANDIDATES,
    "Picarro observation file"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# 3. ROBUST NORMAL DRAW TRUNCATED AT ZERO
# ================================================================

def sample_normal_truncated_zero(rng, mu, sd):
    """
    Draw exactly from:

        X ~ Normal(mu, sd**2), conditional on X >= 0.

    Uses inverse survival-function sampling for ordinary cases and
    Robert's exponential rejection method for the far upper tail.

    This avoids the numerical problems that can occur when calculating
    1 - ndtr(alpha) for large positive alpha.
    """

    if not np.isfinite(mu):
        raise RuntimeError(
            f"Non-finite conditional mean: {mu}"
        )

    if not np.isfinite(sd) or sd <= 0.0:
        raise RuntimeError(
            f"Invalid conditional standard deviation: {sd}"
        )

    alpha = -mu / sd

    # ------------------------------------------------------------
    # Ordinary case: inverse survival-function sampling
    # ------------------------------------------------------------
    if alpha < 8.0:

        survival_alpha = ndtr(-alpha)

        u = rng.random()

        q = u * survival_alpha

        # Protect against a zero probability caused by floating point
        # rounding. q is a survival probability.
        q = max(q, np.finfo(float).tiny)

        z = -ndtri(q)

        x = sd * (z - alpha)

        # Protect against tiny negative round-off values.
        return max(0.0, float(x))

    # ------------------------------------------------------------
    # Extreme positive alpha: Robert exponential rejection method
    # ------------------------------------------------------------

    lam = 0.5 * (
        alpha + np.hypot(alpha, 2.0)
    )

    while True:

        t = rng.exponential(scale=1.0 / lam)

        accept_probability = np.exp(
            -0.5 * (t - 1.0 / lam)**2
        )

        if rng.random() <= accept_probability:

            # Stable expression: avoids cancellation in mu + sd*z.
            return float(sd * t)


# ================================================================
# 4. READ AND PREPARE THE H MATRIX
# ================================================================

print("\n" + "=" * 78)
print("READING FLEXPART H MATRIX")
print("=" * 78)

df_h = pd.read_csv(H_FILE, low_memory=False)

df_h.columns = df_h.columns.str.strip()

print("H matrix shape:", df_h.shape)

required_h_columns = [
    "receptor",
    "obs_time_aest",
]

for col in required_h_columns:
    if col not in df_h.columns:
        raise ValueError(
            f"Required H-matrix column not found: {col}"
        )

# Identify mine/date parameter columns.
parameter_pattern = re.compile(
    r"^OC(\d+)_(\d{8})$"
)

parameter_columns = []

for col in df_h.columns:

    if parameter_pattern.match(str(col)):
        parameter_columns.append(col)

if len(parameter_columns) != 403:

    raise ValueError(
        "Expected 403 mine/date parameters, but found "
        f"{len(parameter_columns)}."
    )


# Explicitly enforce date-major ordering:
# date first, then mine number.
def parameter_sort_key(col):

    match = parameter_pattern.match(str(col))

    mine_number = int(match.group(1))
    date_string = match.group(2)

    return date_string, mine_number


parameter_columns = sorted(
    parameter_columns,
    key=parameter_sort_key
)

# Check the expected 13 mines and 27 dates.
parameter_info = []

for col in parameter_columns:

    match = parameter_pattern.match(str(col))

    mine_number = int(match.group(1))
    date_string = match.group(2)

    parameter_info.append(
        {
            "parameter": col,
            "mine": f"OC{mine_number}",
            "date": pd.to_datetime(
                date_string,
                format="%Y%m%d"
            ),
        }
    )

parameter_info = pd.DataFrame(parameter_info)

mine_names = sorted(
    parameter_info["mine"].unique(),
    key=lambda x: int(x.replace("OC", ""))
)

dates = sorted(
    parameter_info["date"].unique()
)

print("Number of mine/date parameters:", len(parameter_columns))
print("Number of mines:", len(mine_names))
print("Number of dates:", len(dates))
print("Mines:", mine_names)
print(
    "Date range:",
    pd.Timestamp(dates[0]).strftime("%Y-%m-%d"),
    "to",
    pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")
)

print("\nFirst 15 parameters:")
for col in parameter_columns[:15]:
    print(" ", col)

print("\nLast 5 parameters:")
for col in parameter_columns[-5:]:
    print(" ", col)


# ================================================================
# 5. READ PICARRO OBSERVATIONS
# ================================================================

print("\n" + "=" * 78)
print("READING PICARRO OBSERVATIONS")
print("=" * 78)

df_obs = pd.read_csv(
    OBS_FILE,
    low_memory=False
)

df_obs.columns = df_obs.columns.str.strip()

required_obs_columns = [
    "Datetime",
    "Singleton cleaned",
    "Muswellbrook cleaned",
    "Merriwa cleaned",
]

for col in required_obs_columns:

    if col not in df_obs.columns:
        raise ValueError(
            f"Required Picarro column not found: {col}"
        )

# Picarro datetime format is day/month/year hour:minute.
df_obs["datetime_hour"] = pd.to_datetime(
    df_obs["Datetime"],
    format="%d/%m/%Y %H:%M",
    errors="coerce"
)

# Fallback for any rows that did not match the explicit format.
missing_datetime = df_obs["datetime_hour"].isna()

if missing_datetime.any():

    df_obs.loc[
        missing_datetime,
        "datetime_hour"
    ] = pd.to_datetime(
        df_obs.loc[missing_datetime, "Datetime"],
        dayfirst=True,
        errors="coerce"
    )

df_obs["datetime_hour"] = (
    df_obs["datetime_hour"].dt.floor("h")
)

# Convert station observations to numeric.
station_columns = {
    "singleton": "Singleton cleaned",
    "muswellbrook": "Muswellbrook cleaned",
    "merriwa": "Merriwa cleaned",
}

for col in station_columns.values():

    df_obs[col] = pd.to_numeric(
        df_obs[col],
        errors="coerce"
    )


# Convert the three station columns into a long-format table.
obs_long_list = []

for receptor, obs_column in station_columns.items():

    tmp = df_obs[
        [
            "datetime_hour",
            obs_column
        ]
    ].copy()

    tmp = tmp.rename(
        columns={
            obs_column: "observed_ch4"
        }
    )

    tmp["receptor_key"] = receptor

    obs_long_list.append(tmp)


df_obs_long = pd.concat(
    obs_long_list,
    ignore_index=True
)

df_obs_long = df_obs_long.dropna(
    subset=[
        "datetime_hour",
        "observed_ch4"
    ]
)

# Average duplicate station/hour records if any.
df_obs_long = (
    df_obs_long
    .groupby(
        ["receptor_key", "datetime_hour"],
        as_index=False
    )["observed_ch4"]
    .mean()
)

print("Picarro observation records:", len(df_obs_long))


# ================================================================
# 6. MATCH H MATRIX ROWS TO OBSERVATIONS
# ================================================================

print("\n" + "=" * 78)
print("MATCHING H MATRIX TO OBSERVATIONS")
print("=" * 78)

df_h["datetime_hour"] = pd.to_datetime(
    df_h["obs_time_aest"],
    errors="coerce"
).dt.floor("h")

df_h["receptor_key"] = (
    df_h["receptor"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df_matched = df_h.merge(
    df_obs_long,
    on=[
        "receptor_key",
        "datetime_hour"
    ],
    how="inner",
    validate="many_to_one"
)

print("Original H matrix rows:", len(df_h))
print("Matched observation rows:", len(df_matched))

if len(df_matched) == 0:
    raise RuntimeError(
        "No observations matched. Check receptor names and timestamps."
    )

print("\nMatched observations by receptor:")

print(
    df_matched["receptor_key"]
    .value_counts()
    .to_string()
)


# ================================================================
# 7. BUILD H, OBSERVATION AND BACKGROUND ARRAYS
# ================================================================

print("\n" + "=" * 78)
print("BUILDING INVERSION ARRAYS")
print("=" * 78)

H = df_matched[
    parameter_columns
].to_numpy(dtype=float)

observed = df_matched[
    "observed_ch4"
].to_numpy(dtype=float)

# Remove any rows with invalid observations or H values.
valid_rows = (
    np.isfinite(observed)
    & np.all(np.isfinite(H), axis=1)
)

H = H[valid_rows]
observed = observed[valid_rows]

df_matched = df_matched.loc[
    valid_rows
].reset_index(drop=True)

# Background-subtracted observations.
y = observed - BACKGROUND_PPM

n_obs, n_parameters = H.shape

print("Final number of observations:", n_obs)
print("Number of parameters:", n_parameters)
print("H shape:", H.shape)

print("Background:", BACKGROUND_PPM, "ppm")
print("Observation sigma:", SIGMA_OBS, "ppm")
print("Prior mean:", PRIOR_MEAN)
print("Prior sigma:", PRIOR_SIGMA)

if n_parameters != 403:
    raise RuntimeError(
        f"Expected 403 parameters, got {n_parameters}"
    )

if not np.all(np.isfinite(H)):
    raise RuntimeError("H still contains non-finite values.")

if not np.all(np.isfinite(y)):
    raise RuntimeError("Observations still contain non-finite values.")


# Save the actual matched observation table for reproducibility.
df_matched.to_csv(
    OUTPUT_DIR / "matched_observations_used.csv",
    index=False
)


# ================================================================
# 8. CONSTRUCT THE DIRECT GIBBS CONDITIONALS
# ================================================================

print("\n" + "=" * 78)
print("CONSTRUCTING GIBBS CONDITIONAL DISTRIBUTIONS")
print("=" * 78)

"""
Posterior log density, ignoring constants:

  -0.5 * ||(y - Hs) / sigma_obs||^2
  -0.5 * ||(s - prior_mean) / prior_sigma||^2

Define:

  Q = H.T @ H / sigma_obs^2 + I / prior_sigma^2

  rhs = H.T @ y / sigma_obs^2
        + prior_mean / prior_sigma^2

Then the conditional distribution for parameter j is:

  s_j | s_-j, y ~ Normal(cond_mean_j, cond_sd_j^2)
                  truncated to s_j >= 0

where:

  cond_sd_j = 1 / sqrt(Q[j,j])

  cond_mean_j =
      (rhs[j] - sum_{k != j} Q[j,k] * s[k]) / Q[j,j]

The implementation maintains Q @ s and updates it after each
coordinate draw. This avoids repeatedly calculating H @ s.
"""

prior_precision = 1.0 / PRIOR_SIGMA**2

# Whiten H and y by the observation uncertainty.
H_weighted = H / SIGMA_OBS
y_weighted = y / SIGMA_OBS

# Direct precision matrix from H and the prior.
Q = H_weighted.T @ H_weighted

Q[np.diag_indices_from(Q)] += prior_precision

# Symmetrise against floating-point round-off.
Q = 0.5 * (Q + Q.T)

rhs = (
    H_weighted.T @ y_weighted
    + prior_precision * PRIOR_MEAN
)

q_diagonal = np.diag(Q)

if np.any(q_diagonal <= 0.0):
    raise RuntimeError(
        "Non-positive diagonal found in Gibbs precision matrix."
    )

conditional_sd = 1.0 / np.sqrt(q_diagonal)

print("Precision matrix shape:", Q.shape)
print("Minimum conditional SD:", conditional_sd.min())
print("Maximum conditional SD:", conditional_sd.max())


# ================================================================
# 9. ONE COMPLETE GIBBS SWEEP
# ================================================================

def gibbs_sweep(
    rng,
    scaling,
    q_scaling
):
    """
    One systematic-scan Gibbs sweep over all 403 parameters.

    scaling:
        Current scaling-factor vector.

    q_scaling:
        Q @ scaling, maintained incrementally.

    Returns:
        Updated scaling and Q @ scaling.
    """

    for j in range(n_parameters):

        old_value = scaling[j]

        # Remove the current j contribution from Q @ scaling.
        q_without_j = (
            q_scaling[j]
            - q_diagonal[j] * old_value
        )

        # Exact univariate Gaussian conditional mean.
        conditional_mean = (
            rhs[j] - q_without_j
        ) / q_diagonal[j]

        # Draw conditional value subject to scaling >= 0.
        new_value = sample_normal_truncated_zero(
            rng,
            conditional_mean,
            conditional_sd[j]
        )

        delta = new_value - old_value

        scaling[j] = new_value

        # Incrementally update Q @ scaling.
        if delta != 0.0:
            q_scaling += Q[:, j] * delta

    return scaling, q_scaling


# ================================================================
# 10. INITIALISE CHAINS
# ================================================================

def make_initial_state(rng):

    """
    Start each chain from a different positive prior-like state.

    Initial values are only starting points. The burn-in allows each
    chain to move towards the posterior distribution.
    """

    x = rng.normal(
        loc=PRIOR_MEAN,
        scale=PRIOR_SIGMA,
        size=n_parameters
    )

    # Ensure a strictly positive starting point.
    x = np.maximum(x, 0.01)

    return x


# ================================================================
# 11. SPLIT R-HAT DIAGNOSTIC
# ================================================================

def calculate_split_rhat(chains):
    """
    Basic split-Rhat diagnostic.

    chains shape:
        (n_chains, n_samples, n_parameters)

    Values close to 1 indicate better agreement between chains.
    This is a diagnostic, not a substitute for checking trace plots
    and effective sample sizes.
    """

    n_chains, n_draws, n_params = chains.shape

    half = n_draws // 2

    if half < 2:
        return np.full(n_params, np.nan)

    first_half = chains[:, :half, :]
    second_half = chains[:, -half:, :]

    split_chains = np.concatenate(
        [
            first_half,
            second_half
        ],
        axis=0
    )

    m, n, _ = split_chains.shape

    chain_means = np.mean(
        split_chains,
        axis=1
    )

    chain_variances = np.var(
        split_chains,
        axis=1,
        ddof=1
    )

    W = np.mean(
        chain_variances,
        axis=0
    )

    B = n * np.var(
        chain_means,
        axis=0,
        ddof=1
    )

    var_hat = (
        ((n - 1.0) / n) * W
        + B / n
    )

    rhat = np.sqrt(
        var_hat / W
    )

    rhat[W == 0.0] = np.nan

    return rhat


# ================================================================
# 12. RUN FOUR GIBBS CHAINS
# ================================================================

print("\n" + "=" * 78)
print("RUNNING DIRECT TRUNCATED-POSTERIOR GIBBS SAMPLER")
print("=" * 78)

print("Number of chains:", N_CHAINS)
print("Burn-in sweeps:", BURN_IN)
print("Retained samples per chain:", N_SAMPLES)
print("Thinning interval:", THIN)

total_sweeps_per_chain = (
    BURN_IN + N_SAMPLES * THIN
)

print(
    "Total sweeps per chain:",
    total_sweeps_per_chain
)

print(
    "Total sweeps across all chains:",
    N_CHAINS * total_sweeps_per_chain
)

rng_master = np.random.default_rng(
    RANDOM_SEED
)

chains = np.zeros(
    (
        N_CHAINS,
        N_SAMPLES,
        n_parameters
    ),
    dtype=np.float64
)

start_time = time.time()

for chain_id in range(N_CHAINS):

    print("\n" + "-" * 70)
    print(f"STARTING CHAIN {chain_id + 1}/{N_CHAINS}")
    print("-" * 70)

    # Independent random-number generator for each chain.
    chain_seed = int(
        rng_master.integers(
            0,
            2**32 - 1
        )
    )

    rng = np.random.default_rng(
        chain_seed
    )

    scaling = make_initial_state(rng)

    # Initialise Q @ scaling.
    q_scaling = Q @ scaling

    chain_start = time.time()

    sample_index = 0

    for sweep in range(1, total_sweeps_per_chain + 1):

        scaling, q_scaling = gibbs_sweep(
            rng,
            scaling,
            q_scaling
        )

        # Save after burn-in, then apply thinning.
        if (
            sweep > BURN_IN
            and (sweep - BURN_IN) % THIN == 0
        ):

            chains[
                chain_id,
                sample_index,
                :
            ] = scaling

            sample_index += 1

        # Progress information.
        if sweep % 1000 == 0:

            elapsed = time.time() - chain_start

            print(
                f"Chain {chain_id + 1}: "
                f"sweep {sweep:6d}/"
                f"{total_sweeps_per_chain}, "
                f"saved {sample_index:5d}/"
                f"{N_SAMPLES}, "
                f"elapsed {elapsed / 60.0:.1f} min"
            )

    if sample_index != N_SAMPLES:
        raise RuntimeError(
            f"Chain {chain_id + 1}: expected "
            f"{N_SAMPLES} samples, got {sample_index}"
        )

    print(
        f"Chain {chain_id + 1} completed in "
        f"{(time.time() - chain_start) / 60.0:.2f} minutes."
    )


total_elapsed = time.time() - start_time

print("\nAll chains completed.")
print(
    f"Total Gibbs sampling time: "
    f"{total_elapsed / 60.0:.2f} minutes"
)


# ================================================================
# 13. CHECK NON-NEGATIVITY
# ================================================================

negative_count = np.count_nonzero(
    chains < 0.0
)

print("\n" + "=" * 78)
print("NON-NEGATIVITY CHECK")
print("=" * 78)

print("Number of negative retained samples:", negative_count)

if negative_count > 0:

    raise RuntimeError(
        "Negative scaling factors found in retained Gibbs samples."
    )

print("PASS: all retained scaling factors are non-negative.")


# ================================================================
# 14. POSTERIOR SUMMARIES
# ================================================================

print("\n" + "=" * 78)
print("CALCULATING TRUNCATED POSTERIOR SUMMARIES")
print("=" * 78)

# Combine all four chains.
all_samples = chains.reshape(
    N_CHAINS * N_SAMPLES,
    n_parameters
)

# ================================================================
# POSTERIOR COVARIANCE MATRIX
# ================================================================

posterior_covariance = np.cov(
    all_samples,
    rowvar=False,
    ddof=1
)

covariance_file = (
    OUTPUT_DIR
    / "direct_Gibbs_truncated_posterior_covariance.npy"
)

np.save(
    covariance_file,
    posterior_covariance
)

print("\nSaved truncated posterior covariance matrix:")
print(covariance_file)

print("Covariance matrix shape:", posterior_covariance.shape)

posterior_mean = np.mean(
    all_samples,
    axis=0
)

posterior_median = np.median(
    all_samples,
    axis=0
)

posterior_sd = np.std(
    all_samples,
    axis=0,
    ddof=1
)

q05 = np.quantile(
    all_samples,
    0.05,
    axis=0
)

q25 = np.quantile(
    all_samples,
    0.25,
    axis=0
)

q75 = np.quantile(
    all_samples,
    0.75,
    axis=0
)

q95 = np.quantile(
    all_samples,
    0.95,
    axis=0
)

probability_above_one = np.mean(
    all_samples > 1.0,
    axis=0
)

probability_above_1_5 = np.mean(
    all_samples > 1.5,
    axis=0
)

probability_near_zero = np.mean(
    all_samples < 0.05,
    axis=0
)

rhat = calculate_split_rhat(
    chains
)


# ================================================================
# 15. CREATE POSTERIOR SUMMARY CSV
# ================================================================

summary = parameter_info.copy()

# Prior and posterior scaling-factor outputs
summary["prior_scaling"] = PRIOR_MEAN

summary["posterior_scaling"] = posterior_mean
summary["posterior_sigma"] = posterior_sd

# Retain the original descriptive column names as well
summary["posterior_mean"] = posterior_mean
summary["posterior_median"] = posterior_median
summary["posterior_sd"] = posterior_sd

summary["truncated_posterior_mean"] = posterior_mean
summary["truncated_posterior_median"] = posterior_median
summary["truncated_posterior_sd"] = posterior_sd

summary["q05"] = q05
summary["q25"] = q25
summary["q75"] = q75
summary["q95"] = q95

summary["prob_scaling_gt_1"] = probability_above_one
summary["prob_scaling_gt_1_5"] = probability_above_1_5
summary["prob_scaling_lt_0_05"] = probability_near_zero

summary["split_rhat"] = rhat

summary["n_chains"] = N_CHAINS
summary["n_samples_per_chain"] = N_SAMPLES
summary["total_retained_samples"] = N_CHAINS * N_SAMPLES

summary_file = (
    OUTPUT_DIR
    / "direct_Gibbs_truncated_posterior_scaling_factors.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print("\nSaved posterior summary:")
print(summary_file)

# ================================================================
# 15A. PLOT PRIOR AND POSTERIOR SCALING FACTORS FOR 13 MINES
# ================================================================

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print("\n" + "=" * 78)
print("PLOTTING PRIOR AND POSTERIOR SCALING FACTORS")
print("=" * 78)

# Ensure dates are datetime objects
summary["date"] = pd.to_datetime(summary["date"])

# Sort by date
summary = summary.sort_values(
    ["mine", "date"]
)

# Create a 4 x 4 panel figure for the 13 mines
fig, axes = plt.subplots(
    nrows=4,
    ncols=4,
    figsize=(18, 14),
    sharex=True,
    sharey=True
)

axes = axes.flatten()

for i, mine in enumerate(mine_names):

    ax = axes[i]

    mine_data = summary[
        summary["mine"] == mine
    ].sort_values("date")

    dates_plot = mine_data["date"]

    prior = mine_data["prior_scaling"].to_numpy()

    posterior = mine_data["posterior_scaling"].to_numpy()

    lower = mine_data["q05"].to_numpy()
    upper = mine_data["q95"].to_numpy()

    # ------------------------------------------------------------
    # Prior scaling factor
    # ------------------------------------------------------------

    ax.plot(
        dates_plot,
        prior,
        color="orange",
        linestyle="--",
        linewidth=1.8,
        label="Prior"
    )

    # ------------------------------------------------------------
    # Truncated posterior mean
    # ------------------------------------------------------------

    ax.plot(
        dates_plot,
        posterior,
        color="green",
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label="Posterior mean"
    )

    # ------------------------------------------------------------
    # 90% posterior credible interval
    # ------------------------------------------------------------

    ax.fill_between(
        dates_plot,
        lower,
        upper,
        color="green",
        alpha=0.18,
        label="90% credible interval"
    )

    ax.set_title(
        mine,
        fontsize=12,
        fontweight="bold"
    )

    ax.grid(
        True,
        linestyle=":",
        alpha=0.5
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d-%b")
    )

    ax.tick_params(
        axis="x",
        rotation=45,
        labelsize=8
    )

    ax.tick_params(
        axis="y",
        labelsize=9
    )

    # Show legend only in the first panel
    if i == 0:
        ax.legend(
            fontsize=8,
            loc="best"
        )

# Remove the three unused panels
for j in range(len(mine_names), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle(
    "Upper Hunter: Prior and Direct Gibbs Truncated Posterior "
    "Mine Scaling Factors\nMay 2026",
    fontsize=16,
    fontweight="bold",
    y=0.99
)

fig.supxlabel(
    "Date (May 2026)",
    fontsize=12
)

fig.supylabel(
    "Daily emission scaling factor",
    fontsize=12
)

fig.tight_layout(
    rect=[0.03, 0.03, 1, 0.95]
)

plot_file = (
    OUTPUT_DIR
    / "direct_Gibbs_prior_posterior_scaling_factors_13mines.png"
)

fig.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close(fig)

print("\nSaved 13-mine scaling-factor plot:")
print(plot_file)

# ================================================================
# 16. SAVE THE FULL MCMC SAMPLES
# ================================================================

samples_file = (
    OUTPUT_DIR
    / "direct_Gibbs_truncated_posterior_samples.npz"
)

np.savez_compressed(
    samples_file,
    chains=chains,
    all_samples=all_samples,
    parameter_names=np.asarray(parameter_columns),
    mines=np.asarray(mine_names),
    dates=np.asarray(
        [
            pd.Timestamp(d).strftime("%Y%m%d")
            for d in dates
        ]
    ),
    background_ppm=BACKGROUND_PPM,
    sigma_obs=SIGMA_OBS,
    prior_mean=PRIOR_MEAN,
    prior_sigma=PRIOR_SIGMA,
    burn_in=BURN_IN,
    n_samples=N_SAMPLES,
    thin=THIN,
    random_seed=RANDOM_SEED
)

print("\nSaved full MCMC samples:")
print(samples_file)


# ================================================================
# 17. SAVE CHAIN DIAGNOSTICS
# ================================================================

diagnostics = pd.DataFrame(
    {
        "parameter": parameter_columns,
        "split_rhat": rhat,
    }
)

diagnostics_file = (
    OUTPUT_DIR
    / "direct_Gibbs_chain_diagnostics.csv"
)

diagnostics.to_csv(
    diagnostics_file,
    index=False
)

print("\nSaved chain diagnostics:")
print(diagnostics_file)


# ================================================================
# 18. PRINT SUMMARY
# ================================================================

print("\n" + "=" * 78)
print("DIRECT GIBBS TRUNCATED POSTERIOR SUMMARY")
print("=" * 78)

print("Total retained samples:", len(all_samples))
print("Number of parameters:", n_parameters)

print(
    "Minimum posterior mean:",
    f"{posterior_mean.min():.6f}"
)

print(
    "Maximum posterior mean:",
    f"{posterior_mean.max():.6f}"
)

print(
    "Minimum posterior median:",
    f"{posterior_median.min():.6f}"
)

print(
    "Maximum posterior median:",
    f"{posterior_median.max():.6f}"
)

print(
    "Maximum split R-hat:",
    f"{np.nanmax(rhat):.6f}"
)

print(
    "Parameters with split R-hat > 1.05:",
    int(np.sum(rhat > 1.05))
)

print("\nFirst 20 posterior scaling factors:")

print(
    summary[
        [
            "parameter",
            "posterior_mean",
            "posterior_median",
            "posterior_sd",
            "q05",
            "q95",
            "split_rhat",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\n" + "=" * 78)
print("DIRECT GIBBS SAMPLER FINISHED")
print("=" * 78)

print("Output directory:")
print(OUTPUT_DIR)

