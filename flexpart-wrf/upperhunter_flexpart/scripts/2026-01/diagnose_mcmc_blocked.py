#!/usr/bin/env python3
"""
===============================================================================
UPPER HUNTER CH4
BLOCKED METROPOLIS-HASTINGS MCMC DIAGNOSTICS
===============================================================================

Reads the completed blocked MH MCMC chain:

    mcmc_samples.npy
    mcmc_logposterior.npy
    mcmc_daily_scaling.csv
    mcmc_daily_scaling_lower95.csv
    mcmc_daily_scaling_upper95.csv
    mcmc_block_acceptance.csv

Produces:

    mcmc_diagnostics_v2/
    ├── mcmc_diagnostic_summary.csv
    ├── parameter_ess.csv
    ├── logposterior_trace.png
    ├── daily_scaling_all_mines.png
    ├── daily_scaling_individual_mines.png
    ├── daily_scaling_individual_mines_95CI.png
    ├── trace_representative_parameters.png
    ├── autocorrelation_representative_parameters.png
    ├── posterior_histograms_representative_parameters.png
    ├── mine_mean_scaling.png
    ├── mine_mean_scaling_95CI.png
    └── high_autocorrelation_parameters.csv

===============================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


# =============================================================================
# PATHS
# =============================================================================

BASE = Path.home() / "flexpart-wrf/upperhunter_flexpart"

MCMC_DIR = (
    BASE
    / "hmatrix/daily_20260105_20260131"
    / "bayesian_analytical/mcmc_blocked"
)

OUTDIR = MCMC_DIR / "mcmc_diagnostics_v2"
OUTDIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SETTINGS
# =============================================================================

BURNIN_ALREADY_REMOVED = True

# Number of representative parameters to plot
N_REPRESENTATIVE = 12

# Maximum lag for autocorrelation
MAX_LAG = 200

# Number of histogram bins
HIST_BINS = 50


# =============================================================================
# MINE / DATE DEFINITIONS
# =============================================================================

MINES = [f"OC{i}" for i in range(1, 14)]

DATES = pd.date_range(
    "2026-01-05",
    "2026-01-31",
    freq="D"
)

PARAMETER_NAMES = [
    f"{mine}_{date.strftime('%Y%m%d')}"
    for date in DATES
    for mine in MINES
]

N_MINE = 13
N_DAY = 27
N_PARAM = N_MINE * N_DAY


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def autocorrelation_1d(x, max_lag=200):
    """
    Calculate normalized autocorrelation for lags 0...max_lag.
    """
    x = np.asarray(x, dtype=float)

    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.full(max_lag + 1, np.nan)

    x = x - np.mean(x)

    variance = np.dot(x, x)

    if variance <= 0:
        return np.full(max_lag + 1, np.nan)

    max_lag = min(max_lag, len(x) - 1)

    acf = np.empty(max_lag + 1)
    acf[0] = 1.0

    for lag in range(1, max_lag + 1):
        acf[lag] = np.dot(
            x[:-lag],
            x[lag:]
        ) / variance

    return acf


def integrated_autocorrelation_time(x, max_lag=200):
    """
    Estimate integrated autocorrelation time using the initial-positive
    sequence.

        tau = 1 + 2 * sum rho_k

    Stop when autocorrelation becomes non-positive.
    """
    acf = autocorrelation_1d(x, max_lag)

    if len(acf) <= 1:
        return np.nan

    tau = 1.0

    for k in range(1, len(acf)):
        if not np.isfinite(acf[k]):
            break

        if acf[k] <= 0:
            break

        tau += 2.0 * acf[k]

    return max(tau, 1.0)


def effective_sample_size(x):
    """
    ESS = N / tau.
    """
    n = len(x)

    tau = integrated_autocorrelation_time(
        x,
        max_lag=MAX_LAG
    )

    if not np.isfinite(tau) or tau <= 0:
        return np.nan

    return n / tau


def lag1_autocorrelation(x):
    acf = autocorrelation_1d(x, max_lag=1)

    if len(acf) < 2:
        return np.nan

    return acf[1]


# =============================================================================
# LOAD MCMC CHAIN
# =============================================================================

print("=" * 90)
print("UPPER HUNTER CH4")
print("BLOCKED METROPOLIS-HASTINGS MCMC DIAGNOSTICS")
print("=" * 90)

print()
print("MCMC directory:")
print(MCMC_DIR)

samples_file = MCMC_DIR / "mcmc_samples.npy"
logp_file = MCMC_DIR / "mcmc_logposterior.npy"

if not samples_file.exists():
    raise FileNotFoundError(samples_file)

if not logp_file.exists():
    raise FileNotFoundError(logp_file)


print()
print("Reading posterior samples:")
print(samples_file)

samples = np.load(samples_file)

print("Sample shape:", samples.shape)

if samples.ndim != 2:
    raise ValueError(
        f"Expected 2-D MCMC array, got {samples.ndim}-D"
    )

n_samples, n_parameters = samples.shape

if n_parameters != N_PARAM:
    raise ValueError(
        f"Expected {N_PARAM} parameters, got {n_parameters}"
    )


print()
print("Reading log posterior:")
print(logp_file)

logp = np.load(logp_file)

print("Log posterior shape:", logp.shape)

if len(logp) != n_samples:
    print(
        "WARNING: log posterior length does not equal sample count."
    )


# =============================================================================
# BASIC CHAIN DIAGNOSTICS
# =============================================================================

print()
print("=" * 90)
print("BASIC CHAIN DIAGNOSTICS")
print("=" * 90)

print(f"Samples              : {n_samples}")
print(f"Parameters           : {n_parameters}")
print(f"Samples / parameter  : {n_samples / n_parameters:.2f}")

print()
print("Posterior scaling range:")
print(f"Minimum: {np.min(samples):.8f}")
print(f"Maximum: {np.max(samples):.8f}")

print()
print("All positive:", bool(np.all(samples > 0)))


# =============================================================================
# PARAMETER STATISTICS + ESS
# =============================================================================

print()
print("=" * 90)
print("CALCULATING ESS")
print("=" * 90)

rows = []

start_time = time.time()

for j, name in enumerate(PARAMETER_NAMES):

    x = samples[:, j]

    mean = np.mean(x)
    median = np.median(x)
    sd = np.std(x, ddof=1)

    q025, q975 = np.percentile(
        x,
        [2.5, 97.5]
    )

    minimum = np.min(x)
    maximum = np.max(x)

    acf1 = lag1_autocorrelation(x)

    tau = integrated_autocorrelation_time(
        x,
        max_lag=MAX_LAG
    )

    ess = effective_sample_size(x)

    rows.append({
        "parameter": name,
        "mean": mean,
        "median": median,
        "sd": sd,
        "lower95": q025,
        "upper95": q975,
        "minimum": minimum,
        "maximum": maximum,
        "lag1_autocorrelation": acf1,
        "integrated_autocorrelation_time": tau,
        "effective_sample_size": ess,
        "ess_fraction": ess / n_samples
    })

    if (j + 1) % 25 == 0 or j == 0:
        elapsed = time.time() - start_time
        print(
            f"Parameter {j+1:3d}/{n_parameters} "
            f"| elapsed {elapsed/60:.2f} min"
        )


ess_df = pd.DataFrame(rows)

ess_df.to_csv(
    OUTDIR / "parameter_ess.csv",
    index=False
)


# =============================================================================
# ESS SUMMARY
# =============================================================================

print()
print("=" * 90)
print("ESS SUMMARY")
print("=" * 90)

print(
    f"Minimum ESS : "
    f"{ess_df.effective_sample_size.min():.2f}"
)

print(
    f"Median ESS  : "
    f"{ess_df.effective_sample_size.median():.2f}"
)

print(
    f"Mean ESS    : "
    f"{ess_df.effective_sample_size.mean():.2f}"
)

print(
    f"Maximum ESS : "
    f"{ess_df.effective_sample_size.max():.2f}"
)

print()
print("Lowest ESS parameters:")

print(
    ess_df
    .sort_values("effective_sample_size")
    .head(20)
    .to_string(index=False)
)


# =============================================================================
# HIGH AUTOCORRELATION PARAMETERS
# =============================================================================

high_acf = (
    ess_df
    .sort_values(
        "lag1_autocorrelation",
        ascending=False
    )
)

high_acf.to_csv(
    OUTDIR / "high_autocorrelation_parameters.csv",
    index=False
)


# =============================================================================
# GLOBAL MCMC DIAGNOSTIC SUMMARY
# =============================================================================

summary = {
    "n_samples": n_samples,
    "n_parameters": n_parameters,
    "minimum_scaling": np.min(samples),
    "maximum_scaling": np.max(samples),
    "median_ess": ess_df.effective_sample_size.median(),
    "mean_ess": ess_df.effective_sample_size.mean(),
    "minimum_ess": ess_df.effective_sample_size.min(),
    "maximum_ess": ess_df.effective_sample_size.max(),
    "median_lag1_acf": ess_df.lag1_autocorrelation.median(),
    "maximum_lag1_acf": ess_df.lag1_autocorrelation.max(),
    "all_samples_positive": bool(np.all(samples > 0))
}

pd.DataFrame(
    [summary]
).to_csv(
    OUTDIR / "mcmc_diagnostic_summary.csv",
    index=False
)


# =============================================================================
# LOG POSTERIOR TRACE
# =============================================================================

print()
print("Plotting log posterior trace...")

plt.figure(figsize=(12, 5))

plt.plot(
    np.arange(len(logp)),
    logp,
    linewidth=0.8
)

plt.xlabel("Retained sample")
plt.ylabel("Log posterior")
plt.title(
    "Upper Hunter blocked MH MCMC: log-posterior trace"
)

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTDIR / "logposterior_trace.png",
    dpi=200
)

plt.show()
plt.close()


# =============================================================================
# DAILY SCALING DATA
# =============================================================================

print()
print("Building daily posterior scaling table...")

mean_file = MCMC_DIR / "mcmc_daily_scaling.csv"
lower_file = MCMC_DIR / "mcmc_daily_scaling_lower95.csv"
upper_file = MCMC_DIR / "mcmc_daily_scaling_upper95.csv"

mean_df = pd.read_csv(mean_file, index_col=0)
lower_df = pd.read_csv(lower_file, index_col=0)
upper_df = pd.read_csv(upper_file, index_col=0)

# Convert index to datetime
mean_df.index = pd.to_datetime(mean_df.index)
lower_df.index = pd.to_datetime(lower_df.index)
upper_df.index = pd.to_datetime(upper_df.index)


# =============================================================================
# ALL-MINE DAILY TIME SERIES
# =============================================================================

print("Plotting all 13 mines...")

plt.figure(figsize=(15, 8))

for mine in MINES:

    plt.plot(
        mean_df.index,
        mean_df[mine],
        marker="o",
        markersize=3,
        linewidth=1.2,
        label=mine
    )

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1.5,
    label="Prior scaling = 1"
)

plt.xlabel("Date")
plt.ylabel("Daily scaling factor")
plt.title(
    "Upper Hunter CH4 MCMC posterior daily scaling factors"
)

plt.xticks(
    rotation=45
)

plt.grid(True, alpha=0.3)
plt.legend(
    ncol=4,
    fontsize=9
)

plt.tight_layout()

plt.savefig(
    OUTDIR / "daily_scaling_all_mines.png",
    dpi=200
)

plt.show()
plt.close()


# =============================================================================
# INDIVIDUAL MINE TIME SERIES
# =============================================================================

print("Plotting individual mine time series...")

fig, axes = plt.subplots(
    N_MINE,
    1,
    figsize=(15, 32),
    sharex=True
)

for i, mine in enumerate(MINES):

    ax = axes[i]

    ax.plot(
        mean_df.index,
        mean_df[mine],
        marker="o",
        markersize=3,
        linewidth=1.2
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0
    )

    ax.set_ylabel(
        mine,
        rotation=0,
        labelpad=30
    )

    ax.grid(
        True,
        alpha=0.3
    )

axes[-1].set_xlabel("Date")

fig.suptitle(
    "Upper Hunter CH4 MCMC daily scaling factors by mine",
    fontsize=16
)

plt.tight_layout(
    rect=[0, 0, 1, 0.98]
)

plt.savefig(
    OUTDIR / "daily_scaling_individual_mines.png",
    dpi=200
)

plt.show()
plt.close()


# =============================================================================
# INDIVIDUAL MINE TIME SERIES + 95% CI
# =============================================================================

print("Plotting individual mine time series with 95% CI...")

fig, axes = plt.subplots(
    N_MINE,
    1,
    figsize=(15, 32),
    sharex=True
)

for i, mine in enumerate(MINES):

    ax = axes[i]

    ax.fill_between(
        mean_df.index,
        lower_df[mine].values,
        upper_df[mine].values,
        alpha=0.25
    )

    ax.plot(
        mean_df.index,
        mean_df[mine],
        marker="o",
        markersize=3,
        linewidth=1.2
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0
    )

    ax.set_ylabel(
        mine,
        rotation=0,
        labelpad=30
    )

    ax.grid(
        True,
        alpha=0.3
    )

axes[-1].set_xlabel("Date")

fig.suptitle(
    "Upper Hunter CH4 MCMC posterior daily scaling factors\n"
    "Mean and 95% credible interval",
    fontsize=16
)

plt.tight_layout(
    rect=[0, 0, 1, 0.98]
)

plt.savefig(
    OUTDIR / "daily_scaling_individual_mines_95CI.png",
    dpi=200
)

plt.show()
plt.close()


# =============================================================================
# MINE-LEVEL MEAN SCALING
# =============================================================================

mine_summary = []

for mine in MINES:

    x = samples[
        :,
        [j for j, p in enumerate(PARAMETER_NAMES)
         if p.startswith(mine + "_")]
    ]

    # Flatten all days for this mine
    xflat = x.reshape(-1)

    mine_summary.append({
        "mine": mine,
        "mean_scaling": np.mean(xflat),
        "median_scaling": np.median(xflat),
        "sd": np.std(xflat, ddof=1),
        "lower95": np.percentile(xflat, 2.5),
        "upper95": np.percentile(xflat, 97.5)
    })

mine_summary_df = pd.DataFrame(
    mine_summary
)

mine_summary_df.to_csv(
    OUTDIR / "mine_scaling_summary.csv",
    index=False
)


# =============================================================================
# MINE MEAN SCALING PLOT
# =============================================================================

plt.figure(figsize=(12, 6))

plt.bar(
    mine_summary_df["mine"],
    mine_summary_df["mean_scaling"]
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1.5
)

plt.xlabel("Mine")
plt.ylabel("Mean daily scaling factor")
plt.title(
    "Mean January 2026 posterior scaling by mine"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTDIR / "mine_mean_scaling.png",
    dpi=200
)

plt.show()
plt.close()


# =============================================================================
# MINE MEAN SCALING + 95% CI
# =============================================================================

plt.figure(figsize=(12, 6))

means = mine_summary_df["mean_scaling"].values

lower = (
    means -
    mine_summary_df["lower95"].values
)

upper = (
    mine_summary_df["upper95"].values -
    means
)

plt.errorbar(
    np.arange(N_MINE),
    means,
    yerr=[lower, upper],
    fmt="o",
    capsize=4
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1.5
)

plt.xticks(
    np.arange(N_MINE),
    MINES
)

plt.xlabel("Mine")
plt.ylabel("Scaling factor")
plt.title(
    "January 2026 posterior scaling by mine\n"
    "95% credible intervals"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTDIR / "mine_mean_scaling_95CI.png",
    dpi=200
)

plt.show()
plt.close()


# =============================================================================
# REPRESENTATIVE PARAMETERS
# =============================================================================

print()
print("=" * 90)
print("SELECTING REPRESENTATIVE PARAMETERS")
print("=" * 90)

# Select:
# - strongest sensitivity / identifiable-looking examples indirectly through
#   posterior uncertainty
# - low ESS
# - middle ESS
# - high ESS

sorted_ess = ess_df.sort_values(
    "effective_sample_size"
)

candidate_names = []

# Lowest ESS
candidate_names.extend(
    sorted_ess.head(4)["parameter"].tolist()
)

# Highest ESS
candidate_names.extend(
    sorted_ess.tail(4)["parameter"].tolist()
)

# Add several well-spread parameters
candidate_names.extend([
    "OC1_20260105",
    "OC3_20260114",
    "OC5_20260115",
    "OC9_20260118",
    "OC9_20260128",
    "OC13_20260131"
])

# Remove duplicates while preserving order
representative = []

for name in candidate_names:
    if name not in representative:
        representative.append(name)

representative = representative[:N_REPRESENTATIVE]

print("Representative parameters:")

for name in representative:
    print(" ", name)


# =============================================================================
# TRACE PLOTS
# =============================================================================

print()
print("Plotting representative parameter traces...")

nrep = len(representative)

fig, axes = plt.subplots(
    nrep,
    1,
    figsize=(14, 2.3 * nrep),
    sharex=True
)

if nrep == 1:
    axes = [axes]

for ax, name in zip(axes, representative):

    j = PARAMETER_NAMES.index(name)

    ax.plot(
        samples[:, j],
        linewidth=0.5
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0
    )

    row = ess_df[
        ess_df.parameter == name
    ].iloc[0]

    ax.set_ylabel(name, fontsize=8)

    ax.set_title(
        f"ESS={row.effective_sample_size:.1f}, "
        f"lag-1 ACF={row.lag1_autocorrelation:.3f}",
        fontsize=8,
        loc="right"
    )

    ax.grid(
        True,
        alpha=0.3
    )

axes[-1].set_xlabel(
    "Retained MCMC sample"
)

fig.suptitle(
    "Representative MCMC parameter traces",
    fontsize=15
)

plt.tight_layout(
    rect=[0, 0, 1, 0.98]
)

plt.savefig(
    OUTDIR / "trace_representative_parameters.png",
    dpi=200
)

plt.show()
plt.close()


# =============================================================================
# AUTOCORRELATION PLOTS
# =============================================================================

print("Plotting autocorrelation functions...")

fig, axes = plt.subplots(
    nrep,
    1,
    figsize=(12, 2.2 * nrep)
)

if nrep == 1:
    axes = [axes]

for ax, name in zip(axes, representative):

    j = PARAMETER_NAMES.index(name)

    acf = autocorrelation_1d(
        samples[:, j],
        MAX_LAG
    )

    ax.plot(
        np.arange(len(acf)),
        acf,
        linewidth=1.0
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=0.8
    )

    ax.set_ylabel(
        name,
        fontsize=8
    )

    ax.grid(
        True,
        alpha=0.3
    )

axes[-1].set_xlabel(
    "Lag"
)

fig.suptitle(
    "Representative posterior autocorrelation",
    fontsize=15
)

plt.tight_layout(
    rect=[0, 0, 1, 0.98]
)

plt.savefig(
    OUTDIR / "autocorrelation_representative_parameters.png",
    dpi=200
)

plt.show()
plt.close()


# =============================================================================
# POSTERIOR HISTOGRAMS
# =============================================================================

print("Plotting representative posterior distributions...")

fig, axes = plt.subplots(
    4,
    3,
    figsize=(14, 12)
)

axes = axes.flatten()

for i, name in enumerate(representative[:12]):

    ax = axes[i]

    j = PARAMETER_NAMES.index(name)

    ax.hist(
        samples[:, j],
        bins=HIST_BINS,
        density=True
    )

    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=1.2
    )

    ax.set_title(
        name,
        fontsize=9
    )

    ax.set_xlabel("Scaling factor")
    ax.set_ylabel("Density")

    ax.grid(
        True,
        alpha=0.25
    )

# Hide unused axes
for i in range(len(representative), len(axes)):
    axes[i].axis("off")

fig.suptitle(
    "Representative posterior distributions",
    fontsize=15
)

plt.tight_layout(
    rect=[0, 0, 1, 0.97]
)

plt.savefig(
    OUTDIR / "posterior_histograms_representative_parameters.png",
    dpi=200
)

plt.show()
plt.close()


# =============================================================================
# FINAL REPORT
# =============================================================================

print()
print("=" * 90)
print("MCMC DIAGNOSTICS COMPLETE")
print("=" * 90)

print()
print("Output directory:")
print(OUTDIR)

print()
print("Files:")

for f in sorted(OUTDIR.iterdir()):
    print(" ", f.name)

print()
print("=" * 90)
print("IMPORTANT DIAGNOSTIC VALUES")
print("=" * 90)

print(
    f"Samples                  : {n_samples}"
)

print(
    f"Parameters               : {n_parameters}"
)

print(
    f"Minimum ESS              : "
    f"{ess_df.effective_sample_size.min():.2f}"
)

print(
    f"Median ESS               : "
    f"{ess_df.effective_sample_size.median():.2f}"
)

print(
    f"Mean ESS                 : "
    f"{ess_df.effective_sample_size.mean():.2f}"
)

print(
    f"Maximum ESS              : "
    f"{ess_df.effective_sample_size.max():.2f}"
)

print(
    f"Median lag-1 ACF         : "
    f"{ess_df.lag1_autocorrelation.median():.4f}"
)

print(
    f"Maximum lag-1 ACF        : "
    f"{ess_df.lag1_autocorrelation.max():.4f}"
)

print(
    f"All samples positive     : "
    f"{np.all(samples > 0)}"
)

print()
print("Lowest ESS parameters:")
print(
    ess_df
    .sort_values("effective_sample_size")
    .head(10)[
        [
            "parameter",
            "effective_sample_size",
            "ess_fraction",
            "lag1_autocorrelation",
            "integrated_autocorrelation_time"
        ]
    ]
    .to_string(index=False)
)

print()
print("===============================================================================")
print("NEXT STEP")
print("===============================================================================")
print()
print(
    "Use ESS and trace plots to determine whether another longer MCMC run "
    "is necessary."
)
print()

