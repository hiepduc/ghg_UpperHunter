#!/usr/bin/env python3
"""
===============================================================================
Compare STILT combined mining scaling factor with WRF-GHG Mining factor
Upper Hunter - February 2026

STILT:
    13 individual open-cut mine posterior daily scaling factors
    OC1 ... OC13

WRF-GHG:
    Combined Mining posterior daily scaling factor

STILT combined factor is emission-weighted using the prior emission
of each mine:

    S_STILT = sum(E_prior_m * S_m) / sum(E_prior_m)

where:
    E_prior_m = prior emission of mine m [g CH4/s]
    S_m       = STILT posterior scaling factor for mine m

The script also calculates:
    - approximate STILT combined uncertainty
    - WRF-GHG posterior uncertainty
    - total scaled mine emissions
    - STILT - WRF-GHG factor difference
    - daily comparison CSV
    - summary statistics
    - PNG and PDF plots

Period:
    1 February 2026 - 31 February 2026
===============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

# ---------------------------------------------------------------------------
# STILT posterior daily scaling factors
# ---------------------------------------------------------------------------
STILT_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/bayesian/2026-02/"
    "Bayesian_daily_inversion_Feb2026/daily_scaling_factors_Feb2026.csv"
)

# ---------------------------------------------------------------------------
# WRF-GHG posterior daily scaling factors
# ---------------------------------------------------------------------------
WRF_FILE = (
    "/mnt/scratch_lustre/duch/runnetzero/"
    "bayesian_daily_MH_5_31Jan2026/bayesian_daily_posterior.csv"
)

# ---------------------------------------------------------------------------
# Prior emissions for individual open-cut mines
# ---------------------------------------------------------------------------
PRIOR_FILE = (
    "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter/"
    "UpperHuntercoalmines.csv"
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTDIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/bayesian/2026-02/"
    "Bayesian_daily_inversion_Feb2026/"
    "STILT_vs_WRFGHG_combined_mining"
)

# ---------------------------------------------------------------------------
# Date range for comparison
# ---------------------------------------------------------------------------
START_DATE = "2026-02-01"
END_DATE   = "2026-02-28"

# ---------------------------------------------------------------------------
# Number of mines
# ---------------------------------------------------------------------------
MINES = [f"OC{i}" for i in range(1, 14)]

# =============================================================================
# CREATE OUTPUT DIRECTORY
# =============================================================================

os.makedirs(OUTDIR, exist_ok=True)

print("=" * 90)
print("STILT vs WRF-GHG COMBINED MINING SCALING FACTOR")
print("Upper Hunter - February 2026")
print("=" * 90)


# =============================================================================
# 1. READ PRIOR MINE EMISSIONS
# =============================================================================

print("\nReading prior mine emissions:")
print(PRIOR_FILE)

prior = pd.read_csv(PRIOR_FILE)

# Clean column names
prior.columns = prior.columns.str.strip()

# Clean facility names
prior["Facility_code_name"] = (
    prior["Facility_code_name"]
    .astype(str)
    .str.strip()
)

# Clean emission column
prior["gCH4/s"] = (
    prior["gCH4/s"]
    .astype(str)
    .str.strip()
    .replace("", np.nan)
)

prior["gCH4/s"] = pd.to_numeric(
    prior["gCH4/s"],
    errors="coerce"
)

# Extract OC1-OC13
mine_prior = prior[
    prior["Facility_code_name"].isin(MINES)
].copy()

# Make sure all 13 mines exist
missing_mines = [
    m for m in MINES
    if m not in mine_prior["Facility_code_name"].values
]

if missing_mines:
    raise ValueError(
        "Missing prior emissions for: "
        + ", ".join(missing_mines)
    )

# Reorder exactly OC1 ... OC13
mine_prior["mine_order"] = mine_prior["Facility_code_name"].apply(
    lambda x: int(x.replace("OC", ""))
)

mine_prior = (
    mine_prior
    .sort_values("mine_order")
    .reset_index(drop=True)
)

# Dictionary
prior_emissions = dict(
    zip(
        mine_prior["Facility_code_name"],
        mine_prior["gCH4/s"]
    )
)

# Check
print("\nPrior emissions:")
print("-" * 60)

for mine in MINES:
    print(
        f"{mine:5s} "
        f"{prior_emissions[mine]:12.4f} g CH4/s"
    )

total_prior = sum(
    prior_emissions[mine]
    for mine in MINES
)

print("-" * 60)
print(
    f"{'TOTAL':5s} "
    f"{total_prior:12.4f} g CH4/s"
)


# =============================================================================
# 2. CALCULATE EMISSION WEIGHTS
# =============================================================================

weights = {
    mine: prior_emissions[mine] / total_prior
    for mine in MINES
}

print("\nEmission weights:")
print("-" * 60)

for mine in MINES:
    print(
        f"{mine:5s} "
        f"{prior_emissions[mine]:12.4f} g/s   "
        f"{100.0 * weights[mine]:8.3f} %"
    )

print("-" * 60)
print(
    f"{'TOTAL':5s} "
    f"{total_prior:12.4f} g/s   "
    f"{100.0 * sum(weights.values()):8.3f} %"
)


# =============================================================================
# 3. READ STILT POSTERIOR FACTORS
# =============================================================================

print("\nReading STILT posterior factors:")
print(STILT_FILE)

stilt = pd.read_csv(STILT_FILE)

stilt.columns = stilt.columns.str.strip()

stilt["date"] = pd.to_datetime(
    stilt["date"],
    errors="coerce"
)

# Check required columns
required_stilt_columns = ["date"]

for mine in MINES:
    required_stilt_columns.extend(
        [mine, f"{mine}_std"]
    )

missing = [
    c for c in required_stilt_columns
    if c not in stilt.columns
]

if missing:
    raise ValueError(
        "Missing STILT columns:\n"
        + "\n".join(missing)
    )

# Restrict dates
stilt = stilt[
    (stilt["date"] >= START_DATE)
    & (stilt["date"] <= END_DATE)
].copy()

stilt = stilt.sort_values("date")

print(
    f"STILT days available: {len(stilt)}"
)


# =============================================================================
# 4. CALCULATE COMBINED STILT FACTOR
# =============================================================================

print("\nCalculating emission-weighted STILT combined factor...")

# Initialize
stilt_combined = np.zeros(len(stilt))

# For approximate variance
stilt_variance = np.zeros(len(stilt))

for mine in MINES:

    w = weights[mine]

    factor = pd.to_numeric(
        stilt[mine],
        errors="coerce"
    ).values

    std = pd.to_numeric(
        stilt[f"{mine}_std"],
        errors="coerce"
    ).values

    # Weighted mean
    stilt_combined += w * factor

    # Independent-error approximation:
    #
    # Var(S_combined) ~= sum(w_m^2 * sigma_m^2)
    #
    # NOTE:
    # This does NOT include posterior covariance between mines.
    stilt_variance += (w ** 2) * (std ** 2)

stilt_combined_std = np.sqrt(stilt_variance)

stilt["STILT_combined_mining_factor"] = stilt_combined
stilt["STILT_combined_mining_std_approx"] = (
    stilt_combined_std
)

stilt["STILT_combined_lower95_approx"] = (
    stilt_combined
    - 1.96 * stilt_combined_std
)

stilt["STILT_combined_upper95_approx"] = (
    stilt_combined
    + 1.96 * stilt_combined_std
)


# =============================================================================
# 5. CALCULATE TOTAL STILT MINE EMISSIONS
# =============================================================================

stilt_total_emission = np.zeros(len(stilt))

for mine in MINES:

    E_prior = prior_emissions[mine]

    factor = pd.to_numeric(
        stilt[mine],
        errors="coerce"
    ).values

    stilt_total_emission += E_prior * factor

stilt["STILT_total_mine_emission_gCH4_s"] = (
    stilt_total_emission
)

stilt["STILT_total_mine_emission_tCH4_day"] = (
    stilt_total_emission
    * 86400.0
    / 1.0e6
)


# =============================================================================
# 6. READ WRF-GHG POSTERIOR
# =============================================================================

print("\nReading WRF-GHG posterior:")
print(WRF_FILE)

wrf = pd.read_csv(WRF_FILE)

wrf.columns = wrf.columns.str.strip()

wrf["date"] = pd.to_datetime(
    wrf["date"],
    errors="coerce"
)

# Keep Mining only
wrf_mining = wrf[
    wrf["sector"].astype(str).str.strip().str.lower()
    == "mining"
].copy()

# Restrict dates
wrf_mining = wrf_mining[
    (wrf_mining["date"] >= START_DATE)
    & (wrf_mining["date"] <= END_DATE)
].copy()

wrf_mining = wrf_mining.sort_values("date")

print(
    f"WRF-GHG Mining days available: {len(wrf_mining)}"
)

# Rename columns
wrf_mining = wrf_mining[
    [
        "date",
        "posterior_mean",
        "posterior_std",
        "posterior_median",
        "lower_2.5pct",
        "upper_97.5pct"
    ]
].copy()

wrf_mining = wrf_mining.rename(
    columns={
        "posterior_mean":
            "WRF_GHG_mining_factor",

        "posterior_std":
            "WRF_GHG_mining_std",

        "posterior_median":
            "WRF_GHG_mining_median",

        "lower_2.5pct":
            "WRF_GHG_mining_lower95",

        "upper_97.5pct":
            "WRF_GHG_mining_upper95"
    }
)


# =============================================================================
# 7. MERGE STILT AND WRF-GHG
# =============================================================================

comparison = pd.merge(
    stilt,
    wrf_mining,
    on="date",
    how="inner"
)

comparison = comparison.sort_values("date").reset_index(
    drop=True
)

print(
    f"\nCommon days: {len(comparison)}"
)

if len(comparison) == 0:
    raise ValueError(
        "No common dates between STILT and WRF-GHG."
    )


# =============================================================================
# 8. CALCULATE DIFFERENCE AND TOTAL WRF EMISSION
# =============================================================================

comparison["factor_difference_STILT_minus_WRF"] = (
    comparison["STILT_combined_mining_factor"]
    - comparison["WRF_GHG_mining_factor"]
)

comparison["absolute_factor_difference"] = (
    comparison["factor_difference_STILT_minus_WRF"]
    .abs()
)

# WRF-GHG prior mining emission is assumed to correspond
# to the sum of the same 13 OC mine emissions.
comparison["WRF_GHG_total_mine_emission_gCH4_s"] = (
    total_prior
    * comparison["WRF_GHG_mining_factor"]
)

comparison["WRF_GHG_total_mine_emission_tCH4_day"] = (
    comparison["WRF_GHG_total_mine_emission_gCH4_s"]
    * 86400.0
    / 1.0e6
)

comparison["mine_emission_difference_gCH4_s"] = (
    comparison["STILT_total_mine_emission_gCH4_s"]
    - comparison["WRF_GHG_total_mine_emission_gCH4_s"]
)

comparison["mine_emission_difference_tCH4_day"] = (
    comparison["mine_emission_difference_gCH4_s"]
    * 86400.0
    / 1.0e6
)


# =============================================================================
# 9. SAVE DAILY COMPARISON
# =============================================================================

comparison_file = os.path.join(
    OUTDIR,
    "STILT_vs_WRFGHG_combined_mining_daily_Jan2026.csv"
)

comparison.to_csv(
    comparison_file,
    index=False,
    float_format="%.8f"
)

print("\nDaily comparison saved:")
print(comparison_file)


# =============================================================================
# 10. SUMMARY STATISTICS
# =============================================================================

x = comparison[
    "STILT_combined_mining_factor"
].values

y = comparison[
    "WRF_GHG_mining_factor"
].values

difference = x - y

# Correlation
if len(x) > 1:
    correlation = np.corrcoef(x, y)[0, 1]
else:
    correlation = np.nan

# R2
if len(x) > 1:
    r2 = correlation ** 2
else:
    r2 = np.nan

# RMSE
rmse = np.sqrt(
    np.mean(
        (x - y) ** 2
    )
)

# MAE
mae = np.mean(
    np.abs(x - y)
)

# Mean bias
bias = np.mean(
    x - y
)

# Mean factors
mean_stilt = np.mean(x)
mean_wrf = np.mean(y)

# Median factors
median_stilt = np.median(x)
median_wrf = np.median(y)


summary = pd.DataFrame({
    "Metric": [
        "N",
        "STILT mean factor",
        "WRF-GHG mean factor",
        "STILT median factor",
        "WRF-GHG median factor",
        "Correlation R",
        "R2",
        "RMSE factor",
        "MAE factor",
        "Mean bias (STILT-WRF)"
    ],

    "Value": [
        len(comparison),
        mean_stilt,
        mean_wrf,
        median_stilt,
        median_wrf,
        correlation,
        r2,
        rmse,
        mae,
        bias
    ]
})

summary_file = os.path.join(
    OUTDIR,
    "STILT_vs_WRFGHG_combined_mining_summary_Jan2026.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print("\n")
print("=" * 90)
print("SUMMARY")
print("=" * 90)

for _, row in summary.iterrows():

    metric = row["Metric"]
    value = row["Value"]

    if metric == "N":
        print(f"{metric:35s}: {int(value)}")
    else:
        print(f"{metric:35s}: {value:.5f}")

print("=" * 90)


# =============================================================================
# 11. PRINT DAILY COMPARISON
# =============================================================================

print("\nDAILY COMPARISON")
print("=" * 90)

print(
    f"{'Date':12s}"
    f"{'STILT':>12s}"
    f"{'WRF-GHG':>12s}"
    f"{'STILT-WRF':>14s}"
    f"{'STILT g/s':>15s}"
    f"{'WRF g/s':>15s}"
)

print("-" * 90)

for _, row in comparison.iterrows():

    print(
        f"{row['date'].strftime('%Y-%m-%d'):12s}"
        f"{row['STILT_combined_mining_factor']:12.4f}"
        f"{row['WRF_GHG_mining_factor']:12.4f}"
        f"{row['factor_difference_STILT_minus_WRF']:14.4f}"
        f"{row['STILT_total_mine_emission_gCH4_s']:15.2f}"
        f"{row['WRF_GHG_total_mine_emission_gCH4_s']:15.2f}"
    )


# =============================================================================
# 12. PLOT 1 - DAILY COMBINED FACTORS
# =============================================================================

fig, ax = plt.subplots(
    figsize=(13, 6)
)

ax.plot(
    comparison["date"],
    comparison["STILT_combined_mining_factor"],
    marker="o",
    linewidth=2.0,
    label="STILT combined mining"
)

ax.plot(
    comparison["date"],
    comparison["WRF_GHG_mining_factor"],
    marker="s",
    linewidth=2.0,
    label="WRF-GHG Mining"
)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.2,
    label="Prior factor = 1"
)

# STILT approximate uncertainty
ax.fill_between(
    comparison["date"],
    comparison["STILT_combined_lower95_approx"],
    comparison["STILT_combined_upper95_approx"],
    alpha=0.15,
    label="STILT approximate 95% CI"
)

# WRF uncertainty
ax.fill_between(
    comparison["date"],
    comparison["WRF_GHG_mining_lower95"],
    comparison["WRF_GHG_mining_upper95"],
    alpha=0.15,
    label="WRF-GHG 95% CI"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Daily mining scaling factor"
)

ax.set_title(
    "Upper Hunter CH₄ Mining Scaling Factors\n"
    "STILT vs WRF-GHG — 1-28 February 2026"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()

fig.autofmt_xdate()

plt.tight_layout()

png1 = os.path.join(
    OUTDIR,
    "STILT_vs_WRFGHG_combined_mining_factor_Jan2026.png"
)

pdf1 = os.path.join(
    OUTDIR,
    "STILT_vs_WRFGHG_combined_mining_factor_Jan2026.pdf"
)

plt.savefig(
    png1,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    pdf1,
    bbox_inches="tight"
)

plt.show()
plt.close()


# =============================================================================
# 13. PLOT 2 - DIFFERENCE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(13, 5)
)

ax.plot(
    comparison["date"],
    comparison["factor_difference_STILT_minus_WRF"],
    marker="o",
    linewidth=2.0
)

ax.axhline(
    0.0,
    linestyle="--",
    linewidth=1.2
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "STILT − WRF-GHG scaling factor"
)

ax.set_title(
    "Difference in Inferred Mining Scaling Factor\n"
    "STILT − WRF-GHG"
)

ax.grid(
    True,
    alpha=0.3
)

fig.autofmt_xdate()

plt.tight_layout()

png2 = os.path.join(
    OUTDIR,
    "STILT_minus_WRFGHG_mining_factor_difference_Jan2026.png"
)

pdf2 = os.path.join(
    OUTDIR,
    "STILT_minus_WRFGHG_mining_factor_difference_Jan2026.pdf"
)

plt.savefig(
    png2,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    pdf2,
    bbox_inches="tight"
)

plt.show()
plt.close()


# =============================================================================
# 14. PLOT 3 - TOTAL MINE EMISSIONS
# =============================================================================

fig, ax = plt.subplots(
    figsize=(13, 6)
)

ax.plot(
    comparison["date"],
    comparison["STILT_total_mine_emission_gCH4_s"],
    marker="o",
    linewidth=2.0,
    label="STILT total mines"
)

ax.plot(
    comparison["date"],
    comparison["WRF_GHG_total_mine_emission_gCH4_s"],
    marker="s",
    linewidth=2.0,
    label="WRF-GHG total mines"
)

ax.axhline(
    total_prior,
    linestyle="--",
    linewidth=1.2,
    label=f"Prior total = {total_prior:.2f} g CH₄/s"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Total mine emission (g CH₄/s)"
)

ax.set_title(
    "Upper Hunter Total Open-Cut Mine Emissions\n"
    "STILT vs WRF-GHG — 1-28 February 2026"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()

fig.autofmt_xdate()

plt.tight_layout()

png3 = os.path.join(
    OUTDIR,
    "STILT_vs_WRFGHG_total_mine_emissions_Jan2026.png"
)

pdf3 = os.path.join(
    OUTDIR,
    "STILT_vs_WRFGHG_total_mine_emissions_Jan2026.pdf"
)

plt.savefig(
    png3,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    pdf3,
    bbox_inches="tight"
)

plt.show()
plt.close()


# =============================================================================
# 15. PLOT 4 - SCATTER COMPARISON
# =============================================================================

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.scatter(
    comparison["WRF_GHG_mining_factor"],
    comparison["STILT_combined_mining_factor"],
    s=60,
    alpha=0.8
)

# Determine common range
all_values = np.concatenate([
    comparison["WRF_GHG_mining_factor"].values,
    comparison["STILT_combined_mining_factor"].values
])

vmin = np.nanmin(all_values)
vmax = np.nanmax(all_values)

padding = 0.05 * (
    vmax - vmin
)

if padding == 0:
    padding = 0.1

plot_min = max(
    0,
    vmin - padding
)

plot_max = vmax + padding

# 1:1 line
ax.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    linestyle="--",
    linewidth=1.5,
    label="1:1"
)

ax.set_xlim(
    plot_min,
    plot_max
)

ax.set_ylim(
    plot_min,
    plot_max
)

ax.set_xlabel(
    "WRF-GHG Mining scaling factor"
)

ax.set_ylabel(
    "STILT combined Mining scaling factor"
)

ax.set_title(
    "STILT vs WRF-GHG Daily Mining Factors\n"
    "1–28 February 2026"
)

ax.text(
    0.05,
    0.95,
    f"R = {correlation:.3f}\n"
    f"R² = {r2:.3f}\n"
    f"RMSE = {rmse:.3f}\n"
    f"MAE = {mae:.3f}",
    transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        alpha=0.8
    )
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()

plt.tight_layout()

png4 = os.path.join(
    OUTDIR,
    "STILT_vs_WRFGHG_mining_factor_scatter_Jan2026.png"
)

pdf4 = os.path.join(
    OUTDIR,
    "STILT_vs_WRFGHG_mining_factor_scatter_Jan2026.pdf"
)

plt.savefig(
    png4,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    pdf4,
    bbox_inches="tight"
)

plt.show()
plt.close()


# =============================================================================
# 16. SAVE MINE WEIGHTS
# =============================================================================

weights_df = pd.DataFrame({
    "mine": MINES,
    "prior_emission_gCH4_s": [
        prior_emissions[m] for m in MINES
    ],
    "emission_weight": [
        weights[m] for m in MINES
    ],
    "emission_weight_percent": [
        100.0 * weights[m] for m in MINES
    ]
})

weights_file = os.path.join(
    OUTDIR,
    "STILT_mine_emission_weights.csv"
)

weights_df.to_csv(
    weights_file,
    index=False,
    float_format="%.8f"
)


# =============================================================================
# 17. FINAL MESSAGE
# =============================================================================

print("\n")
print("=" * 90)
print("OUTPUT FILES")
print("=" * 90)

print(f"Daily comparison:")
print(f"  {comparison_file}")

print(f"\nSummary:")
print(f"  {summary_file}")

print(f"\nMine emission weights:")
print(f"  {weights_file}")

print("\nPlots:")
print(f"  {png1}")
print(f"  {pdf1}")
print(f"  {png2}")
print(f"  {pdf2}")
print(f"  {png3}")
print(f"  {pdf3}")
print(f"  {png4}")
print(f"  {pdf4}")

print("\n")
print("=" * 90)
print("DONE")
print("=" * 90)

