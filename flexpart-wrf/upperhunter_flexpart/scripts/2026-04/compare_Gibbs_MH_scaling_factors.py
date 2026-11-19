#!/usr/bin/env python3

"""
Compare Gibbs and Metropolis-Hastings posterior scaling factors
for the FLEXPART-WRF Upper Hunter 13-mine × 28-day inversion.

Gibbs:
  .../MCMC_13mine_daily_Apr2026/

MH:
  .../MCMC_13mine_daily_Apr2026_MH/

Both samplers should use exactly the same:
    y = BACKGROUND + H @ s + epsilon

with:
    background = 1.85 ppm
    prior       = truncated N(1, 1^2)
    390 parameters = 13 mines × 30 days

This script compares the posterior samples directly.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path("/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart")

GIBBS_DIR = (
    BASE_DIR
    / "H_matrix_flexpart/2026-04"
    / "MCMC_13mine_daily_Apr2026"
)

MH_DIR = (
    BASE_DIR
    / "H_matrix_flexpart/2026-04"
    / "MCMC_13mine_daily_Apr2026_MH"
)

GIBBS_SAMPLES_FILE = (
    GIBBS_DIR / "FLEXPART_MCMC_posterior_samples_Apr2026.npz"
)

MH_SAMPLES_FILE = (
    MH_DIR / "FLEXPART_MH_posterior_samples_Apr2026.npz"
)

OUTPUT_DIR = (
    BASE_DIR
    / "H_matrix_flexpart/2026-04"
    / "MCMC_13mine_daily_Apr2026_MH"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SETTINGS
# =============================================================================

MINES = [f"OC{i}" for i in range(1, 14)]

DATES = pd.date_range(
    "2026-04-01",
    "2026-04-30",
    freq="D"
)

PARAMETERS = [
    f"{mine}_{date.strftime('%Y%m%d')}"
    for date in DATES
    for mine in MINES
]

N_PARAMS = len(PARAMETERS)

EXPECTED_CHAINS = 4
EXPECTED_SAMPLES_PER_CHAIN = 5000


# =============================================================================
# HEADER
# =============================================================================

print()
print("=" * 100)
print("FLEXPART-WRF GIBBS vs METROPOLIS-HASTINGS POSTERIOR COMPARISON")
print("=" * 100)
print("Upper Hunter — 13 mines × 28 daily scaling factors")
print()
print(f"Parameters : {N_PARAMS}")
print(f"Dates      : {DATES[0].date()} to {DATES[-1].date()}")
print()
print("Gibbs samples:")
print(GIBBS_SAMPLES_FILE)
print()
print("MH samples:")
print(MH_SAMPLES_FILE)
print()


# =============================================================================
# CHECK FILES
# =============================================================================

if not GIBBS_SAMPLES_FILE.exists():
    raise FileNotFoundError(
        f"Gibbs posterior samples not found:\n{GIBBS_SAMPLES_FILE}"
    )

if not MH_SAMPLES_FILE.exists():
    raise FileNotFoundError(
        f"MH posterior samples not found:\n{MH_SAMPLES_FILE}"
    )


# =============================================================================
# READ NPZ
# =============================================================================

print("=" * 100)
print("READING POSTERIOR SAMPLES")
print("=" * 100)

gibbs_npz = np.load(GIBBS_SAMPLES_FILE)
mh_npz = np.load(MH_SAMPLES_FILE)

print("\nGibbs NPZ keys:")
print(gibbs_npz.files)

print("\nMH NPZ keys:")
print(mh_npz.files)


# =============================================================================
# IDENTIFY SAMPLE ARRAY
# =============================================================================

def get_sample_array(npz, name):

    preferred = [
        "samples",
        "posterior_samples",
        "combined_samples",
        "chain_samples",
    ]

    for key in preferred:
        if key in npz.files:
            arr = np.asarray(npz[key])
            print(f"{name}: using key '{key}'")
            print(f"{name} array shape: {arr.shape}")
            return arr

    # Fall back to first multidimensional array
    for key in npz.files:
        arr = np.asarray(npz[key])

        if arr.ndim >= 2:
            print(f"{name}: using multidimensional key '{key}'")
            print(f"{name} array shape: {arr.shape}")
            return arr

    raise RuntimeError(
        f"Could not identify posterior sample array in {name} NPZ."
    )


gibbs_samples = get_sample_array(gibbs_npz, "Gibbs")
mh_samples = get_sample_array(mh_npz, "MH")


# =============================================================================
# NORMALIZE SAMPLE SHAPE
# =============================================================================

def normalize_samples(arr, name):

    """
    Return shape:

        (n_posterior_samples, n_parameters)

    Accepted input shapes:

        (chains, samples, parameters)
        (samples, parameters)
    """

    if arr.ndim == 3:

        chains, samples, parameters = arr.shape

        print(
            f"{name}: chains={chains}, "
            f"samples/chain={samples}, "
            f"parameters={parameters}"
        )

        arr = arr.reshape(
            chains * samples,
            parameters
        )

    elif arr.ndim == 2:

        samples, parameters = arr.shape

        print(
            f"{name}: samples={samples}, "
            f"parameters={parameters}"
        )

    else:

        raise ValueError(
            f"{name} posterior array has unsupported shape {arr.shape}"
        )

    return np.asarray(arr, dtype=float)


gibbs_samples = normalize_samples(
    gibbs_samples,
    "Gibbs"
)

mh_samples = normalize_samples(
    mh_samples,
    "MH"
)


# =============================================================================
# CHECK DIMENSIONS
# =============================================================================

print()
print("=" * 100)
print("CHECKING SAMPLE DIMENSIONS")
print("=" * 100)

print(
    f"Gibbs posterior samples : {gibbs_samples.shape}"
)

print(
    f"MH posterior samples    : {mh_samples.shape}"
)

if gibbs_samples.shape[1] != N_PARAMS:
    raise RuntimeError(
        f"Gibbs has {gibbs_samples.shape[1]} parameters; "
        f"expected {N_PARAMS}."
    )

if mh_samples.shape[1] != N_PARAMS:
    raise RuntimeError(
        f"MH has {mh_samples.shape[1]} parameters; "
        f"expected {N_PARAMS}."
    )


# =============================================================================
# SUMMARY FUNCTION
# =============================================================================

def summarize(samples):

    return {
        "mean": np.mean(samples, axis=0),
        "median": np.median(samples, axis=0),
        "sd": np.std(samples, axis=0, ddof=1),

        "q025": np.percentile(samples, 2.5, axis=0),
        "q05": np.percentile(samples, 5.0, axis=0),

        "q95": np.percentile(samples, 95.0, axis=0),
        "q975": np.percentile(samples, 97.5, axis=0),

        "prob_lt_1": np.mean(samples < 1.0, axis=0),
        "prob_gt_1": np.mean(samples > 1.0, axis=0),
    }


print()
print("=" * 100)
print("CALCULATING POSTERIOR SUMMARIES")
print("=" * 100)

g = summarize(gibbs_samples)
m = summarize(mh_samples)


# =============================================================================
# BUILD COMPARISON TABLE
# =============================================================================

comparison = pd.DataFrame({
    "parameter": PARAMETERS,

    "Gibbs_mean": g["mean"],
    "MH_mean": m["mean"],

    "Gibbs_median": g["median"],
    "MH_median": m["median"],

    "Gibbs_sd": g["sd"],
    "MH_sd": m["sd"],

    "Gibbs_q025": g["q025"],
    "MH_q025": m["q025"],

    "Gibbs_q975": g["q975"],
    "MH_q975": m["q975"],

    "Gibbs_prob_lt_1": g["prob_lt_1"],
    "MH_prob_lt_1": m["prob_lt_1"],

    "Gibbs_prob_gt_1": g["prob_gt_1"],
    "MH_prob_gt_1": m["prob_gt_1"],
})


# =============================================================================
# MINE AND DATE
# =============================================================================

comparison["mine"] = (
    comparison["parameter"]
    .str.extract(r"^(OC\d+)_")[0]
)

comparison["date"] = pd.to_datetime(
    comparison["parameter"].str.extract(r"_(\d{8})$")[0],
    format="%Y%m%d"
)


# =============================================================================
# DIFFERENCES
# =============================================================================

comparison["mean_difference_MH_minus_Gibbs"] = (
    comparison["MH_mean"]
    - comparison["Gibbs_mean"]
)

comparison["median_difference_MH_minus_Gibbs"] = (
    comparison["MH_median"]
    - comparison["Gibbs_median"]
)

comparison["sd_difference_MH_minus_Gibbs"] = (
    comparison["MH_sd"]
    - comparison["Gibbs_sd"]
)

comparison["mean_abs_difference"] = np.abs(
    comparison["mean_difference_MH_minus_Gibbs"]
)

comparison["mean_relative_difference_percent"] = np.where(
    np.abs(comparison["Gibbs_mean"]) > 1e-12,
    100.0
    * comparison["mean_difference_MH_minus_Gibbs"]
    / comparison["Gibbs_mean"],
    np.nan
)


# =============================================================================
# 95% INTERVAL OVERLAP
# =============================================================================

comparison["CI_overlap"] = (
    np.maximum(
        comparison["Gibbs_q025"],
        comparison["MH_q025"]
    )
    <=
    np.minimum(
        comparison["Gibbs_q975"],
        comparison["MH_q975"]
    )
)


# =============================================================================
# SAVE FULL COMPARISON
# =============================================================================

comparison_file = (
    OUTPUT_DIR
    / "FLEXPART_Gibbs_vs_MH_scaling_factor_comparison_Apr2026.csv"
)

comparison.to_csv(
    comparison_file,
    index=False,
    float_format="%.8f"
)

print()
print("Full parameter comparison:")
print(comparison_file)


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

mean_corr = np.corrcoef(
    comparison["Gibbs_mean"],
    comparison["MH_mean"]
)[0, 1]

median_corr = np.corrcoef(
    comparison["Gibbs_median"],
    comparison["MH_median"]
)[0, 1]

mean_abs_diff = comparison["mean_abs_difference"].mean()

max_abs_diff = comparison["mean_abs_difference"].max()

max_diff_row = comparison.loc[
    comparison["mean_abs_difference"].idxmax()
]

mean_relative_abs_diff = (
    np.abs(
        comparison["mean_relative_difference_percent"]
    ).mean()
)

ci_overlap_fraction = (
    comparison["CI_overlap"].mean()
)


# =============================================================================
# PRINT GLOBAL COMPARISON
# =============================================================================

print()
print("=" * 100)
print("GIBBS vs MH GLOBAL COMPARISON")
print("=" * 100)

print()
print(f"Correlation of posterior means : {mean_corr:.6f}")
print(f"Correlation of posterior medians: {median_corr:.6f}")

print()
print(
    f"Mean absolute difference "
    f"in posterior mean factors     : {mean_abs_diff:.6f}"
)

print(
    f"Maximum absolute difference "
    f"in posterior mean factors     : {max_abs_diff:.6f}"
)

print(
    f"Mean absolute relative "
    f"difference                     : {mean_relative_abs_diff:.3f}%"
)

print(
    f"95% CI overlap                  : "
    f"{100.0 * ci_overlap_fraction:.2f}%"
)

print()
print("Largest Gibbs/MH mean difference:")
print(
    f"  Parameter : {max_diff_row['parameter']}"
)
print(
    f"  Gibbs     : {max_diff_row['Gibbs_mean']:.6f}"
)
print(
    f"  MH        : {max_diff_row['MH_mean']:.6f}"
)
print(
    f"  Difference: "
    f"{max_diff_row['mean_difference_MH_minus_Gibbs']:.6f}"
)


# =============================================================================
# MINE-LEVEL COMPARISON
# =============================================================================

mine_summary = (
    comparison
    .groupby("mine")
    .agg(
        Gibbs_mean_factor=(
            "Gibbs_mean",
            "mean"
        ),

        MH_mean_factor=(
            "MH_mean",
            "mean"
        ),

        Gibbs_mean_median=(
            "Gibbs_median",
            "mean"
        ),

        MH_mean_median=(
            "MH_median",
            "mean"
        ),

        Gibbs_mean_sd=(
            "Gibbs_sd",
            "mean"
        ),

        MH_mean_sd=(
            "MH_sd",
            "mean"
        ),

        mean_abs_difference=(
            "mean_abs_difference",
            "mean"
        ),

        max_abs_difference=(
            "mean_abs_difference",
            "max"
        ),

        CI_overlap_fraction=(
            "CI_overlap",
            "mean"
        ),
    )
    .reset_index()
)

mine_summary[
    "difference_MH_minus_Gibbs"
] = (
    mine_summary["MH_mean_factor"]
    - mine_summary["Gibbs_mean_factor"]
)

mine_summary[
    "relative_difference_percent"
] = np.where(
    np.abs(mine_summary["Gibbs_mean_factor"]) > 1e-12,
    100.0
    * mine_summary["difference_MH_minus_Gibbs"]
    / mine_summary["Gibbs_mean_factor"],
    np.nan
)

mine_summary_file = (
    OUTPUT_DIR
    / "FLEXPART_Gibbs_vs_MH_mine_summary_Apr2026.csv"
)

mine_summary.to_csv(
    mine_summary_file,
    index=False,
    float_format="%.8f"
)


# =============================================================================
# PRINT MINE SUMMARY
# =============================================================================

print()
print("=" * 100)
print("MINE-LEVEL POSTERIOR SCALING FACTORS")
print("=" * 100)

print()

display_columns = [
    "mine",
    "Gibbs_mean_factor",
    "MH_mean_factor",
    "difference_MH_minus_Gibbs",
    "relative_difference_percent",
    "mean_abs_difference",
    "CI_overlap_fraction",
]

print(
    mine_summary[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)


# =============================================================================
# IDENTIFY LARGEST DIFFERENCES
# =============================================================================

print()
print("=" * 100)
print("10 LARGEST GIBBS vs MH DIFFERENCES")
print("=" * 100)

largest = (
    comparison
    .sort_values(
        "mean_abs_difference",
        ascending=False
    )
    .head(10)
)

print()

print(
    largest[
        [
            "parameter",
            "Gibbs_mean",
            "MH_mean",
            "mean_difference_MH_minus_Gibbs",
            "mean_relative_difference_percent",
            "Gibbs_q025",
            "Gibbs_q975",
            "MH_q025",
            "MH_q975",
            "CI_overlap",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)


# =============================================================================
# INFORMED PARAMETERS
# =============================================================================

print()
print("=" * 100)
print("PARAMETERS WITH LARGEST ABSOLUTE POSTERIOR DIFFERENCES")
print("=" * 100)

thresholds = [0.01, 0.02, 0.05, 0.10]

for threshold in thresholds:

    n = np.sum(
        comparison["mean_abs_difference"] > threshold
    )

    print(
        f"|MH mean - Gibbs mean| > {threshold:.2f}: "
        f"{n}/{N_PARAMS}"
    )


# =============================================================================
# POSTERIOR MEAN AGREEMENT
# =============================================================================

print()
print("=" * 100)
print("POSTERIOR MEAN AGREEMENT")
print("=" * 100)

gibbs_mean = comparison["Gibbs_mean"].values
mh_mean = comparison["MH_mean"].values

print()
print(
    f"Gibbs mean factor range: "
    f"{np.min(gibbs_mean):.5f} – {np.max(gibbs_mean):.5f}"
)

print(
    f"MH mean factor range   : "
    f"{np.min(mh_mean):.5f} – {np.max(mh_mean):.5f}"
)

print()
print(
    "Correlation between Gibbs and MH posterior means:"
)
print(
    f"  r = {mean_corr:.6f}"
)


# =============================================================================
# DAILY COMPARISON
# =============================================================================

daily_summary = (
    comparison
    .groupby("date")
    .agg(
        Gibbs_mean_factor=(
            "Gibbs_mean",
            "mean"
        ),

        MH_mean_factor=(
            "MH_mean",
            "mean"
        ),

        mean_abs_difference=(
            "mean_abs_difference",
            "mean"
        ),

        max_abs_difference=(
            "mean_abs_difference",
            "max"
        ),
    )
    .reset_index()
)

daily_summary["difference_MH_minus_Gibbs"] = (
    daily_summary["MH_mean_factor"]
    - daily_summary["Gibbs_mean_factor"]
)

daily_file = (
    OUTPUT_DIR
    / "FLEXPART_Gibbs_vs_MH_daily_summary_Apr2026.csv"
)

daily_summary.to_csv(
    daily_file,
    index=False,
    float_format="%.8f"
)


# =============================================================================
# FINAL INTERPRETATION
# =============================================================================

print()
print("=" * 100)
print("INTERPRETATION")
print("=" * 100)

print()

if mean_corr >= 0.99:
    print(
        "EXCELLENT: Gibbs and MH posterior means are extremely "
        "strongly correlated."
    )
elif mean_corr >= 0.95:
    print(
        "GOOD: Gibbs and MH posterior means are strongly correlated."
    )
else:
    print(
        "WARNING: Gibbs and MH posterior means show substantial "
        "differences. Investigate sampler/proposal settings."
    )

if mean_abs_diff < 0.01:
    print(
        "The average absolute difference in scaling factors is "
        "less than 0.01."
    )
elif mean_abs_diff < 0.05:
    print(
        "The average absolute difference in scaling factors is "
        "less than 0.05."
    )
else:
    print(
        "The average absolute difference exceeds 0.05; "
        "the two samplers should be investigated further."
    )

if ci_overlap_fraction > 0.95:
    print(
        "More than 95% of parameter credible intervals overlap."
    )
elif ci_overlap_fraction > 0.80:
    print(
        "Most parameter credible intervals overlap."
    )
else:
    print(
        "A substantial fraction of credible intervals do not overlap."
    )

print()
print("Output files:")
print()
print("1.", comparison_file)
print("2.", mine_summary_file)
print("3.", daily_file)

print()
print("=" * 100)
print("COMPARISON COMPLETE")
print("=" * 100)

