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
    "H_matrix_flexpart/2026-03",
    "MCMC_13mine_daily_Mar2026"
)

POST_FILE = os.path.join(
    MCMC_DIR,
    "FLEXPART_MCMC_posterior_samples_Mar2026.npz"
)

H_FILE = os.path.join(
    BASE,
    "H_matrix_flexpart/2026-03",
    "H_flexpart_0_10m_13mine_Mar2026.csv"
)

PRIOR_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/"
    "emission_location/UpperHuntercoalmines.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

OUT_STATS = os.path.join(
    MCMC_DIR,
    "FLEXPART_MCMC_mine_daily_scaling_statistics_Mar2026.csv"
)

OUT_DAILY = os.path.join(
    MCMC_DIR,
    "FLEXPART_MCMC_mine_daily_scaling_factors_Mar2026.csv"
)

OUT_TS = os.path.join(
    MCMC_DIR,
    "FLEXPART_MCMC_mine_daily_scaling_timeseries_Mar2026.png"
)

OUT_BOX = os.path.join(
    MCMC_DIR,
    "FLEXPART_MCMC_mine_scaling_boxplots_Mar2026.png"
)


# ============================================================
# DATES / MINES
# ============================================================

DATES = pd.date_range(
    "2026-03-01",
    "2026-03-31",
    freq="D"
)

MINES = [f"OC{i}" for i in range(1, 14)]

N_DAYS = len(DATES)
N_MINES = len(MINES)

print("=" * 90)
print("FLEXPART-WRF GIBBS/MCMC — 13 MINE POSTERIOR ANALYSIS")
print("=" * 90)

print(f"Dates       : {DATES[0].date()} to {DATES[-1].date()}")
print(f"Number days : {N_DAYS}")
print(f"Number mines: {N_MINES}")
print(f"Parameters  : {N_DAYS * N_MINES}")


# ============================================================
# READ PRIOR EMISSIONS
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

print("\nPrior emissions:")
for mine, e in zip(MINES, prior_emissions):
    print(f"{mine:5s}: {e:12.3f} g/s")

print(f"\nTotal prior: {prior_emissions.sum():.3f} g CH4/s")


# ============================================================
# READ H MATRIX TO DETERMINE INFORMATIVE DAYS
# ============================================================

print("\nReading H matrix:")
print(H_FILE)

Hdf = pd.read_csv(H_FILE)

h_columns = [
    f"{mine}_{date.strftime('%Y%m%d')}"
    for date in DATES
    for mine in MINES
]

missing = [
    c for c in h_columns
    if c not in Hdf.columns
]

if missing:
    raise ValueError(
        f"Missing {len(missing)} H columns. "
        f"First few: {missing[:10]}"
    )

H = Hdf[h_columns].to_numpy(dtype=float)

# A parameter is informative if it has at least one
# non-zero sensitivity in the observation set.
informative = np.any(H != 0, axis=0)

informative_2d = informative.reshape(
    N_DAYS,
    N_MINES
)

# Number of informative observations for each parameter
n_nonzero_obs = np.count_nonzero(
    H,
    axis=0
).reshape(N_DAYS, N_MINES)


# ============================================================
# LOAD POSTERIOR SAMPLES
# ============================================================

print("\nReading Gibbs posterior:")
print(POST_FILE)

data = np.load(POST_FILE)

print("Available arrays:", data.files)

if "samples" in data.files:
    samples = data["samples"]

elif "posterior_samples" in data.files:
    samples = data["posterior_samples"]

else:
    samples = data[data.files[0]]

print("Raw shape:", samples.shape)


# ============================================================
# COMBINE CHAINS
# ============================================================

if samples.ndim == 3:

    n_chains, n_iter, n_parameters = samples.shape

    print(f"Chains     : {n_chains}")
    print(f"Iterations : {n_iter}")
    print(f"Parameters : {n_parameters}")

    samples = samples.reshape(
        n_chains * n_iter,
        n_parameters
    )

elif samples.ndim == 2:

    n_samples, n_parameters = samples.shape

    print(f"Samples    : {n_samples}")
    print(f"Parameters : {n_parameters}")

else:
    raise ValueError(
        f"Unexpected sample shape: {samples.shape}"
    )


expected = N_DAYS * N_MINES

if n_parameters != expected:
    raise ValueError(
        f"Expected {expected} parameters, "
        f"got {n_parameters}"
    )

print("Combined posterior:", samples.shape)


# ============================================================
# RESHAPE TO:
#
# sample × day × mine
# ============================================================

samples_3d = samples.reshape(
    samples.shape[0],
    N_DAYS,
    N_MINES
)

print(
    "3-D posterior:",
    samples_3d.shape
)


# ============================================================
# DAILY MINE POSTERIOR STATISTICS
# ============================================================

rows = []

for d, date in enumerate(DATES):

    for m, mine in enumerate(MINES):

        x = samples_3d[:, d, m]

        rows.append({

            "date":
                date.strftime("%Y-%m-%d"),

            "mine":
                mine,

            "prior_factor":
                1.0,

            "prior_emission_gps":
                prior_emissions[m],

            "posterior_mean_factor":
                np.mean(x),

            "posterior_median_factor":
                np.median(x),

            "posterior_sd_factor":
                np.std(x, ddof=1),

            "q025_factor":
                np.percentile(x, 2.5),

            "q975_factor":
                np.percentile(x, 97.5),

            "min_factor":
                np.min(x),

            "max_factor":
                np.max(x),

            "n_nonzero_H_observations":
                n_nonzero_obs[d, m],

            "informative":
                bool(informative_2d[d, m])
        })


daily_df = pd.DataFrame(rows)

daily_df.to_csv(
    OUT_DAILY,
    index=False
)

print("\nSaved daily mine results:")
print(OUT_DAILY)


# ============================================================
# OVERALL MINE STATISTICS ACROSS THE 27 DAYS
# ============================================================

stats = []

for m, mine in enumerate(MINES):

    # 20,000 samples × 27 days
    x = samples_3d[:, :, m].ravel()

    # Number of informative days
    info_days = informative_2d[:, m].sum()

    # Number of uninformative days
    zero_days = N_DAYS - info_days

    stats.append({

        "mine":
            mine,

        "prior_emission_gps":
            prior_emissions[m],

        "prior_emission_tonnes_day":
            prior_emissions[m] * 86400 / 1e6,

        "prior_factor":
            1.0,

        "posterior_mean_factor":
            np.mean(x),

        "posterior_median_factor":
            np.median(x),

        "posterior_sd_factor":
            np.std(x, ddof=1),

        "posterior_q025_factor":
            np.percentile(x, 2.5),

        "posterior_q975_factor":
            np.percentile(x, 97.5),

        "posterior_min_factor":
            np.min(x),

        "posterior_max_factor":
            np.max(x),

        "n_informative_days":
            int(info_days),

        "n_uninformative_days":
            int(zero_days),

        "percent_informative_days":
            100.0 * info_days / N_DAYS,

        "mean_daily_emission_gps":
            np.mean(
                samples_3d[:, :, m]
                * prior_emissions[m]
            ),

        "median_daily_emission_gps":
            np.median(
                samples_3d[:, :, m]
                * prior_emissions[m]
            ),

        "q025_daily_emission_gps":
            np.percentile(
                samples_3d[:, :, m]
                * prior_emissions[m],
                2.5
            ),

        "q975_daily_emission_gps":
            np.percentile(
                samples_3d[:, :, m]
                * prior_emissions[m],
                97.5
            )
    })


stats_df = pd.DataFrame(stats)

stats_df.to_csv(
    OUT_STATS,
    index=False
)

print("\nSaved mine statistics:")
print(OUT_STATS)


# ============================================================
# PRINT STATISTICS
# ============================================================

print("\n")
print("=" * 125)
print("POSTERIOR MINE SCALING FACTOR STATISTICS")
print("=" * 125)

header = (
    f"{'Mine':5s}"
    f"{'Prior g/s':>12s}"
    f"{'Mean':>10s}"
    f"{'Median':>10s}"
    f"{'SD':>10s}"
    f"{'2.5%':>10s}"
    f"{'97.5%':>10s}"
    f"{'Min':>10s}"
    f"{'Max':>10s}"
    f"{'Info days':>11s}"
)

print(header)
print("-" * len(header))

for _, r in stats_df.iterrows():

    print(
        f"{r['mine']:5s}"
        f"{r['prior_emission_gps']:12.3f}"
        f"{r['posterior_mean_factor']:10.3f}"
        f"{r['posterior_median_factor']:10.3f}"
        f"{r['posterior_sd_factor']:10.3f}"
        f"{r['posterior_q025_factor']:10.3f}"
        f"{r['posterior_q975_factor']:10.3f}"
        f"{r['posterior_min_factor']:10.3f}"
        f"{r['posterior_max_factor']:10.3f}"
        f"{int(r['n_informative_days']):11d}"
    )


# ============================================================
# FIGURE 1:
# 13 MINE TIME SERIES
# ============================================================

fig, axes = plt.subplots(
    13,
    1,
    figsize=(15, 32),
    sharex=True
)

for m, mine in enumerate(MINES):

    ax = axes[m]

    median = []
    lower = []
    upper = []
    mean = []

    for d in range(N_DAYS):

        x = samples_3d[:, d, m]

        mean.append(np.mean(x))
        median.append(np.median(x))
        lower.append(np.percentile(x, 2.5))
        upper.append(np.percentile(x, 97.5))

    median = np.array(median)
    lower = np.array(lower)
    upper = np.array(upper)
    mean = np.array(mean)

    # 95% credible interval
    ax.fill_between(
        DATES,
        lower,
        upper,
        alpha=0.25,
        label="95% CI"
    )

    # posterior median
    ax.plot(
        DATES,
        median,
        marker="o",
        linewidth=1.8,
        markersize=4,
        label="Median"
    )

    # posterior mean
    ax.plot(
        DATES,
        mean,
        linestyle="--",
        linewidth=1.2,
        label="Mean"
    )

    # prior scaling factor
    ax.axhline(
        1.0,
        linestyle=":",
        linewidth=1.5
    )

    ax.set_ylabel(
        f"{mine}\nScaling",
        fontsize=9
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.set_title(
        f"{mine}  |  Prior emission = "
        f"{prior_emissions[m]:.2f} g CH4/s",
        loc="left",
        fontsize=10
    )

    # Mark days with zero H sensitivity
    for d in range(N_DAYS):

        if not informative_2d[d, m]:

            ax.axvline(
                DATES[d],
                alpha=0.08,
                linewidth=3
            )


axes[0].legend(
    loc="upper right",
    ncol=3
)

axes[-1].set_xticks(DATES)
axes[-1].set_xticklabels(
    [d.strftime("%d %b") for d in DATES],
    rotation=45,
    ha="right"
)

axes[-1].set_xlabel(
    "Date"
)

fig.suptitle(
    "FLEXPART-WRF Gibbs/MCMC Posterior Daily Scaling Factors\n"
    "13 open-cut coal mines — 1-31 March 2026",
    fontsize=16,
    y=0.995
)

plt.tight_layout(
    rect=[0, 0, 1, 0.99]
)

plt.savefig(
    OUT_TS,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nSaved time-series plot:")
print(OUT_TS)


# ============================================================
# FIGURE 2:
# BOX PLOTS FOR EACH MINE
# ============================================================

box_data = []

for m in range(N_MINES):

    # 27 daily posterior distributions
    # represented by all posterior samples
    x = samples_3d[:, :, m].ravel()

    box_data.append(x)


plt.figure(
    figsize=(15, 8)
)

plt.boxplot(
    box_data,
    labels=MINES,
    showfliers=False
)

plt.axhline(
    1.0,
    linestyle=":",
    linewidth=2,
    label="Prior factor = 1"
)

plt.ylabel(
    "Posterior scaling factor"
)

plt.xlabel(
    "Mine"
)

plt.title(
    "FLEXPART-WRF Gibbs/MCMC Posterior Scaling Factors\n"
    "Distribution across 31 days — March 2026"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT_BOX,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nSaved box plot:")
print(OUT_BOX)


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)

print("\nOutput files:")
print(f"1. {OUT_DAILY}")
print(f"2. {OUT_STATS}")
print(f"3. {OUT_TS}")
print(f"4. {OUT_BOX}")

