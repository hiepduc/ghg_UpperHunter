#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE = "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart"

MCMC_DIR = os.path.join(
    BASE,
    "H_matrix_flexpart/2026-01",
    "MCMC_13mine_daily_Jan2026"
)

POST_FILE = os.path.join(
    MCMC_DIR,
    "FLEXPART_MCMC_posterior_samples_Jan2026.npz"
)

PRIOR_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/emission_location/"
    "UpperHuntercoalmines.csv"
)

OUT_CSV = os.path.join(
    MCMC_DIR,
    "FLEXPART_MCMC_daily_scaling_factors_Jan2026.csv"
)

OUT_PNG = os.path.join(
    MCMC_DIR,
    "FLEXPART_MCMC_daily_scaling_factors_Jan2026.png"
)


# ============================================================
# SETTINGS
# ============================================================

DATES = pd.date_range(
    "2026-01-05",
    "2026-01-31",
    freq="D"
)

MINES = [f"OC{i}" for i in range(1, 14)]

BACKGROUND_PPM = 1.85


# ============================================================
# PRIOR MINE EMISSIONS
# ============================================================

prior_df = pd.read_csv(PRIOR_FILE)

prior_df["mine"] = (
    prior_df["Facility_code_name"]
    .astype(str)
    .str.extract(r"(OC\d+)", expand=False)
)

prior_df["gCH4_s"] = pd.to_numeric(
    prior_df["gCH4/s"],
    errors="coerce"
)

prior_lookup = (
    prior_df
    .dropna(subset=["mine", "gCH4_s"])
    .groupby("mine")["gCH4_s"]
    .sum()
    .to_dict()
)

prior_emissions = np.array(
    [prior_lookup[m] for m in MINES],
    dtype=float
)

total_prior = prior_emissions.sum()

print("=" * 80)
print("FLEXPART-WRF MCMC DAILY SCALING FACTORS")
print("=" * 80)

print("\nPrior mine emissions:")
for mine, emission in zip(MINES, prior_emissions):
    print(f"{mine:5s} {emission:12.3f} g CH4/s")

print(f"\nTotal prior emission = {total_prior:.3f} g CH4/s")


# ============================================================
# LOAD GIBBS POSTERIOR SAMPLES
# ============================================================

print("\nReading posterior samples:")
print(POST_FILE)

data = np.load(POST_FILE)

print("\nNPZ contents:")
print(data.files)

# Find posterior sample array
if "samples" in data.files:
    samples = data["samples"]
elif "posterior_samples" in data.files:
    samples = data["posterior_samples"]
else:
    # Fall back to first array
    samples = data[data.files[0]]

print("\nRaw sample shape:", samples.shape)


# ============================================================
# HANDLE MCMC ARRAY SHAPE
# ============================================================

# Expected:
#   (chains, samples, parameters)
# or already:
#   (samples, parameters)

if samples.ndim == 3:

    n_chains, n_iter, n_param = samples.shape

    print(
        f"Chains       : {n_chains}"
    )
    print(
        f"Iterations   : {n_iter}"
    )
    print(
        f"Parameters   : {n_param}"
    )

    # Combine chains
    samples = samples.reshape(
        n_chains * n_iter,
        n_param
    )

elif samples.ndim == 2:

    n_samples, n_param = samples.shape

    print(
        f"Samples      : {n_samples}"
    )
    print(
        f"Parameters   : {n_param}"
    )

else:
    raise ValueError(
        f"Unexpected posterior sample shape: {samples.shape}"
    )


expected_parameters = len(DATES) * len(MINES)

if n_param != expected_parameters:
    raise ValueError(
        f"Expected {expected_parameters} parameters "
        f"(27 days x 13 mines), but found {n_param}"
    )


print("\nCombined posterior shape:", samples.shape)


# ============================================================
# RESHAPE:
#
# samples =
#   [sample, day, mine]
#
# parameter ordering is:
#   OC1_20260105
#   OC2_20260105
#   ...
#   OC13_20260105
#   OC1_20260106
#   ...
# ============================================================

samples_3d = samples.reshape(
    samples.shape[0],
    len(DATES),
    len(MINES)
)

print("Reshaped posterior:", samples_3d.shape)


# ============================================================
# CONVERT SCALING FACTORS TO ACTUAL EMISSIONS
# ============================================================

# prior emissions shape:
#
#       day       mine
#       27         13
#
prior_2d = np.tile(
    prior_emissions,
    (len(DATES), 1)
)

# Posterior daily total emission:
#
# E(day) = sum_mine(
#             scaling(day,mine)
#             * prior_emission(mine)
#         )

posterior_daily_gps = (
    samples_3d * prior_2d[None, :, :]
).sum(axis=2)


# ============================================================
# DAILY SCALING FACTOR
# ============================================================

daily_scaling = (
    posterior_daily_gps
    / total_prior
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary = []

for i, date in enumerate(DATES):

    x = daily_scaling[:, i]

    summary.append({
        "date": date.strftime("%Y-%m-%d"),

        "prior_factor": 1.0,

        "posterior_mean_factor": np.mean(x),
        "posterior_median_factor": np.median(x),
        "posterior_sd_factor": np.std(x, ddof=1),

        "q025_factor": np.percentile(x, 2.5),
        "q975_factor": np.percentile(x, 97.5),

        "prior_total_gps": total_prior,

        "posterior_mean_gps": np.mean(
            posterior_daily_gps[:, i]
        ),

        "posterior_median_gps": np.median(
            posterior_daily_gps[:, i]
        ),

        "posterior_q025_gps": np.percentile(
            posterior_daily_gps[:, i],
            2.5
        ),

        "posterior_q975_gps": np.percentile(
            posterior_daily_gps[:, i],
            97.5
        ),

        "posterior_mean_tonnes_day": (
            np.mean(posterior_daily_gps[:, i])
            * 86400.0 / 1e6
        ),

        "posterior_median_tonnes_day": (
            np.median(posterior_daily_gps[:, i])
            * 86400.0 / 1e6
        ),

        "posterior_q025_tonnes_day": (
            np.percentile(
                posterior_daily_gps[:, i],
                2.5
            )
            * 86400.0 / 1e6
        ),

        "posterior_q975_tonnes_day": (
            np.percentile(
                posterior_daily_gps[:, i],
                97.5
            )
            * 86400.0 / 1e6
        ),
    })


summary_df = pd.DataFrame(summary)


# ============================================================
# SAVE CSV
# ============================================================

summary_df.to_csv(
    OUT_CSV,
    index=False
)

print("\nSaved:")
print(OUT_CSV)


# ============================================================
# PRINT DAILY FACTORS
# ============================================================

print("\n")
print("=" * 95)
print("DAILY POSTERIOR SCALING FACTORS")
print("=" * 95)

print(
    f"{'Date':12s}"
    f"{'Mean':>10s}"
    f"{'Median':>10s}"
    f"{'2.5%':>10s}"
    f"{'97.5%':>10s}"
    f"{'Mean g/s':>14s}"
)

print("-" * 95)

for _, r in summary_df.iterrows():

    print(
        f"{r['date']:12s}"
        f"{r['posterior_mean_factor']:10.3f}"
        f"{r['posterior_median_factor']:10.3f}"
        f"{r['q025_factor']:10.3f}"
        f"{r['q975_factor']:10.3f}"
        f"{r['posterior_mean_gps']:14.1f}"
    )


# ============================================================
# PLOT
# ============================================================

dates = pd.to_datetime(summary_df["date"])

mean_factor = summary_df["posterior_mean_factor"].values
median_factor = summary_df["posterior_median_factor"].values

lower = summary_df["q025_factor"].values
upper = summary_df["q975_factor"].values

prior = summary_df["prior_factor"].values


plt.figure(figsize=(14, 7))

# 95% credible interval
plt.fill_between(
    dates,
    lower,
    upper,
    alpha=0.25,
    label="95% credible interval"
)

# Posterior median
plt.plot(
    dates,
    median_factor,
    marker="o",
    linewidth=2,
    markersize=5,
    label="Posterior median"
)

# Posterior mean
plt.plot(
    dates,
    mean_factor,
    marker="s",
    linestyle="--",
    linewidth=1.5,
    markersize=4,
    label="Posterior mean"
)

# Prior factor
plt.axhline(
    1.0,
    linestyle=":",
    linewidth=2,
    label="Prior factor = 1"
)

plt.xlabel("Date")
plt.ylabel("Daily emission scaling factor")

plt.title(
    "FLEXPART-WRF Bayesian MCMC — Daily Mine Emission Scaling Factors\n"
    "13 mines, Jan 5–31 2026"
)

plt.grid(
    True,
    alpha=0.3
)

plt.xticks(
    dates,
    [d.strftime("%d %b") for d in dates],
    rotation=45,
    ha="right"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved plot:")
print(OUT_PNG)

print("\nDone.")

