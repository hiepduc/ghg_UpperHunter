#!/usr/bin/env python3
"""
===============================================================================
Upper Hunter CH4
STILT vs WRF-GHG Mining Inversion Diagnostics
February 2026

Purpose
-------
Compare:
    1. STILT emission-weighted combined mining factor
    2. WRF-GHG combined Mining factor
    3. Individual STILT OC1-OC13 factors
    4. Emission-weighted contribution of each mine
    5. STILT - WRF-GHG difference

The script also identifies which mines drive the aggregate STILT solution.

STILT combined mining factor:

    S_STILT(d) =
        sum_m [ E_m * S_m(d) ] / sum_m(E_m)

where:
    E_m     = prior emission of mine m [g CH4/s]
    S_m(d)  = posterior scaling factor of mine m on day d

Emission-weighted contribution:

    C_m(d) = w_m * S_m(d)

where:

    w_m = E_m / sum(E_m)

Therefore:

    S_STILT(d) = sum_m C_m(d)

Period:
    1 February 2026 - 28 February 2026

===============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

STILT_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/bayesian/2026-02/"
    "Bayesian_daily_inversion_Feb2026/"
    "daily_scaling_factors_Feb2026.csv"
)

WRF_FILE = (
    "/mnt/scratch_lustre/duch/runnetzero/"
    "bayesian_daily_MH_5_31Feb2026/"
    "bayesian_daily_posterior.csv"
)

PRIOR_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/emission_location/"
    "UpperHuntercoalmines.csv"
)

OUTDIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/bayesian/2026-01/"
    "Bayesian_daily_inversion_Feb2026/"
    "STILT_vs_WRFGHG_mining_diagnostics"
)

START_DATE = "2026-02-01"
END_DATE   = "2026-02-28"

MINES = [f"OC{i}" for i in range(1, 14)]


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

os.makedirs(OUTDIR, exist_ok=True)


print("=" * 100)
print("STILT vs WRF-GHG MINING INVERSION DIAGNOSTICS")
print("Upper Hunter - 1 to 28 February 2026")
print("=" * 100)


# =============================================================================
# 1. READ PRIOR MINE EMISSIONS
# =============================================================================

print("\nReading prior mine emissions:")
print(PRIOR_FILE)

prior = pd.read_csv(PRIOR_FILE)

prior.columns = prior.columns.str.strip()

prior["Facility_code_name"] = (
    prior["Facility_code_name"]
    .astype(str)
    .str.strip()
)

prior["gCH4/s"] = pd.to_numeric(
    prior["gCH4/s"].astype(str).str.strip(),
    errors="coerce"
)

mine_prior = prior[
    prior["Facility_code_name"].isin(MINES)
].copy()

missing = [
    mine for mine in MINES
    if mine not in mine_prior["Facility_code_name"].values
]

if missing:
    raise ValueError(
        "Missing prior emissions for: "
        + ", ".join(missing)
    )

mine_prior["mine_number"] = mine_prior[
    "Facility_code_name"
].str.replace("OC", "", regex=False).astype(int)

mine_prior = (
    mine_prior
    .sort_values("mine_number")
    .reset_index(drop=True)
)

prior_emissions = dict(
    zip(
        mine_prior["Facility_code_name"],
        mine_prior["gCH4/s"]
    )
)

total_prior = sum(
    prior_emissions[mine]
    for mine in MINES
)

weights = {
    mine: prior_emissions[mine] / total_prior
    for mine in MINES
}


print("\nPrior emissions and weights:")
print("-" * 75)
print(
    f"{'Mine':>6s}"
    f"{'Prior g/s':>15s}"
    f"{'Weight %':>15s}"
)
print("-" * 75)

for mine in MINES:
    print(
        f"{mine:>6s}"
        f"{prior_emissions[mine]:15.4f}"
        f"{100.0 * weights[mine]:15.3f}"
    )

print("-" * 75)
print(
    f"{'TOTAL':>6s}"
    f"{total_prior:15.4f}"
    f"{100.0:15.3f}"
)


# =============================================================================
# 2. READ STILT POSTERIOR
# =============================================================================

print("\nReading STILT posterior factors:")
print(STILT_FILE)

stilt = pd.read_csv(STILT_FILE)

stilt.columns = stilt.columns.str.strip()

stilt["date"] = pd.to_datetime(
    stilt["date"],
    errors="coerce"
)

required_columns = ["date"]

for mine in MINES:
    required_columns.extend(
        [mine, f"{mine}_std"]
    )

missing = [
    c for c in required_columns
    if c not in stilt.columns
]

if missing:
    raise ValueError(
        "Missing STILT columns:\n"
        + "\n".join(missing)
    )

stilt = stilt[
    (stilt["date"] >= START_DATE)
    & (stilt["date"] <= END_DATE)
].copy()

stilt = stilt.sort_values("date").reset_index(drop=True)

print(
    f"STILT days: {len(stilt)}"
)


# =============================================================================
# 3. CALCULATE STILT COMBINED FACTOR
# =============================================================================

stilt_combined = np.zeros(len(stilt))

stilt_variance_approx = np.zeros(len(stilt))

for mine in MINES:

    factor = pd.to_numeric(
        stilt[mine],
        errors="coerce"
    ).values

    std = pd.to_numeric(
        stilt[f"{mine}_std"],
        errors="coerce"
    ).values

    w = weights[mine]

    stilt_combined += w * factor

    # Approximate uncertainty assuming independent mine factors.
    #
    # Exact uncertainty requires the posterior covariance matrix.
    stilt_variance_approx += (
        w ** 2
        * std ** 2
    )

stilt["STILT_combined_factor"] = stilt_combined

stilt["STILT_combined_std_approx"] = np.sqrt(
    stilt_variance_approx
)

stilt["STILT_combined_lower95_approx"] = (
    stilt["STILT_combined_factor"]
    - 1.96 * stilt["STILT_combined_std_approx"]
)

stilt["STILT_combined_upper95_approx"] = (
    stilt["STILT_combined_factor"]
    + 1.96 * stilt["STILT_combined_std_approx"]
)


# =============================================================================
# 4. CALCULATE INDIVIDUAL MINE CONTRIBUTIONS
# =============================================================================

print("\nCalculating emission-weighted mine contributions...")

for mine in MINES:

    w = weights[mine]

    stilt[f"{mine}_weighted_contribution"] = (
        w * stilt[mine]
    )


# Check that weighted contributions sum to combined factor
weighted_columns = [
    f"{mine}_weighted_contribution"
    for mine in MINES
]

check = stilt[weighted_columns].sum(axis=1)

max_difference = np.nanmax(
    np.abs(
        check
        - stilt["STILT_combined_factor"]
    )
)

print(
    f"Maximum weighted-sum check difference: "
    f"{max_difference:.12e}"
)


# =============================================================================
# 5. TOTAL STILT MINE EMISSIONS
# =============================================================================

stilt_total_emission = np.zeros(len(stilt))

for mine in MINES:

    stilt_total_emission += (
        prior_emissions[mine]
        * stilt[mine].values
    )

stilt["STILT_total_mine_gCH4_s"] = (
    stilt_total_emission
)

stilt["STILT_total_mine_tCH4_day"] = (
    stilt_total_emission
    * 86400.0
    / 1.0e6
)


# =============================================================================
# 6. READ WRF-GHG MINING POSTERIOR
# =============================================================================

print("\nReading WRF-GHG posterior:")
print(WRF_FILE)

wrf = pd.read_csv(WRF_FILE)

wrf.columns = wrf.columns.str.strip()

wrf["date"] = pd.to_datetime(
    wrf["date"],
    errors="coerce"
)

wrf_mining = wrf[
    wrf["sector"]
    .astype(str)
    .str.strip()
    .str.lower()
    == "mining"
].copy()

wrf_mining = wrf_mining[
    (wrf_mining["date"] >= START_DATE)
    & (wrf_mining["date"] <= END_DATE)
].copy()

wrf_mining = wrf_mining.sort_values(
    "date"
).reset_index(drop=True)

wrf_mining = wrf_mining[
    [
        "date",
        "posterior_mean",
        "posterior_std",
        "posterior_median",
        "lower_2.5pct",
        "upper_97.5pct"
    ]
].rename(
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

comparison = comparison.sort_values(
    "date"
).reset_index(drop=True)

comparison["STILT_minus_WRF"] = (
    comparison["STILT_combined_factor"]
    - comparison["WRF_GHG_mining_factor"]
)

comparison["WRF_minus_STILT"] = (
    -comparison["STILT_minus_WRF"]
)

comparison["WRF_GHG_total_mine_gCH4_s"] = (
    total_prior
    * comparison["WRF_GHG_mining_factor"]
)

comparison["WRF_GHG_total_mine_tCH4_day"] = (
    comparison["WRF_GHG_total_mine_gCH4_s"]
    * 86400.0
    / 1.0e6
)

comparison["mine_emission_difference_gCH4_s"] = (
    comparison["STILT_total_mine_gCH4_s"]
    - comparison["WRF_GHG_total_mine_gCH4_s"]
)

comparison["mine_emission_difference_tCH4_day"] = (
    comparison["mine_emission_difference_gCH4_s"]
    * 86400.0
    / 1.0e6
)


print(
    f"\nCommon days: {len(comparison)}"
)


# =============================================================================
# 8. SAVE FULL DAILY DIAGNOSTIC FILE
# =============================================================================

full_csv = os.path.join(
    OUTDIR,
    "STILT_WRFGHG_mining_full_daily_diagnostics_Feb2026.csv"
)

comparison.to_csv(
    full_csv,
    index=False,
    float_format="%.8f"
)

print("\nFull daily diagnostic CSV:")
print(full_csv)


# =============================================================================
# 9. MINE CONTRIBUTION ANALYSIS
# =============================================================================

print("\nCalculating aggregate mine contributions...")

# Use only common STILT/WRF days
n_days = len(comparison)

mine_rows = []

for mine in MINES:

    factor_mean = comparison[mine].mean()

    weighted_mean = (
        comparison[
            f"{mine}_weighted_contribution"
        ].mean()
    )

    total_emission_mean = (
        comparison[mine]
        * prior_emissions[mine]
    ).mean()

    weighted_contribution_percent = (
        100.0
        * weighted_mean
        / comparison[
            "STILT_combined_factor"
        ].mean()
    )

    mine_rows.append({
        "mine": mine,

        "prior_emission_gCH4_s":
            prior_emissions[mine],

        "prior_weight_percent":
            100.0 * weights[mine],

        "mean_STILT_factor":
            factor_mean,

        "mean_weighted_contribution":
            weighted_mean,

        "mean_contribution_percent_of_STILT":
            weighted_contribution_percent,

        "mean_scaled_emission_gCH4_s":
            total_emission_mean,

        "mean_emission_adjustment_gCH4_s":
            total_emission_mean
            - prior_emissions[mine]
    })

mine_contribution = pd.DataFrame(
    mine_rows
)

# Sort by contribution to aggregate STILT solution
mine_contribution = mine_contribution.sort_values(
    "mean_contribution_percent_of_STILT",
    ascending=False
).reset_index(drop=True)

mine_csv = os.path.join(
    OUTDIR,
    "STILT_mine_contribution_analysis_Feb2026.csv"
)

mine_contribution.to_csv(
    mine_csv,
    index=False,
    float_format="%.8f"
)


# =============================================================================
# 10. PRINT MINE CONTRIBUTION TABLE
# =============================================================================

print("\n")
print("=" * 110)
print("STILT MINE CONTRIBUTION TO AGGREGATE SOLUTION")
print("=" * 110)

print(
    f"{'Mine':>6s}"
    f"{'Prior g/s':>13s}"
    f"{'Prior %':>11s}"
    f"{'Mean factor':>14s}"
    f"{'Weighted':>14s}"
    f"{'Contribution %':>18s}"
    f"{'Mean scaled g/s':>18s}"
)

print("-" * 110)

for _, row in mine_contribution.iterrows():

    print(
        f"{row['mine']:>6s}"
        f"{row['prior_emission_gCH4_s']:13.2f}"
        f"{row['prior_weight_percent']:11.2f}"
        f"{row['mean_STILT_factor']:14.4f}"
        f"{row['mean_weighted_contribution']:14.4f}"
        f"{row['mean_contribution_percent_of_STILT']:18.2f}"
        f"{row['mean_scaled_emission_gCH4_s']:18.2f}"
    )

print("-" * 110)

print(
    f"\nSaved to:\n{mine_csv}"
)


# =============================================================================
# 11. CALCULATE PERIOD-WIDE MINE CONTRIBUTION TO STILT ADJUSTMENT
# =============================================================================
#
# This is a different diagnostic:
#
#     mean scaled emission - prior emission
#
# Positive = mine is increased relative to prior
# Negative = mine is decreased relative to prior
#
# This helps identify which mines are responsible for the total adjustment.
# =============================================================================

adjustment_rows = []

for mine in MINES:

    mean_factor = comparison[mine].mean()

    adjustment_gs = (
        prior_emissions[mine]
        * (mean_factor - 1.0)
    )

    adjustment_rows.append({
        "mine": mine,
        "prior_emission_gCH4_s":
            prior_emissions[mine],
        "mean_STILT_factor":
            mean_factor,
        "mean_adjustment_gCH4_s":
            adjustment_gs
    })

adjustment_df = pd.DataFrame(
    adjustment_rows
)

adjustment_df["absolute_adjustment_gCH4_s"] = (
    adjustment_df["mean_adjustment_gCH4_s"]
    .abs()
)

adjustment_df = adjustment_df.sort_values(
    "absolute_adjustment_gCH4_s",
    ascending=False
).reset_index(drop=True)

adjustment_csv = os.path.join(
    OUTDIR,
    "STILT_mine_emission_adjustment_analysis_Feb2026.csv"
)

adjustment_df.to_csv(
    adjustment_csv,
    index=False,
    float_format="%.8f"
)


# =============================================================================
# 12. PRINT EMISSION ADJUSTMENT
# =============================================================================

print("\n")
print("=" * 100)
print("STILT MEAN EMISSION ADJUSTMENT RELATIVE TO PRIOR")
print("=" * 100)

print(
    f"{'Mine':>6s}"
    f"{'Prior g/s':>15s}"
    f"{'Mean factor':>15s}"
    f"{'Adjustment g/s':>18s}"
)

print("-" * 100)

for _, row in adjustment_df.iterrows():

    print(
        f"{row['mine']:>6s}"
        f"{row['prior_emission_gCH4_s']:15.2f}"
        f"{row['mean_STILT_factor']:15.4f}"
        f"{row['mean_adjustment_gCH4_s']:18.2f}"
    )

print("-" * 100)

total_adjustment = (
    adjustment_df["mean_adjustment_gCH4_s"]
    .sum()
)

print(
    f"{'TOTAL':>6s}"
    f"{'':15s}"
    f"{'':15s}"
    f"{total_adjustment:18.2f}"
)

print(
    f"\nSaved to:\n{adjustment_csv}"
)


# =============================================================================
# 13. FIGURE 1
#     Combined STILT vs WRF-GHG
# =============================================================================

fig, ax = plt.subplots(
    figsize=(14, 6)
)

ax.plot(
    comparison["date"],
    comparison["STILT_combined_factor"],
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
    label="Prior = 1"
)

# STILT approximate uncertainty
ax.fill_between(
    comparison["date"],
    comparison["STILT_combined_lower95_approx"],
    comparison["STILT_combined_upper95_approx"],
    alpha=0.15,
    label="STILT approximate 95% CI"
)

# WRF-GHG uncertainty
ax.fill_between(
    comparison["date"],
    comparison["WRF_GHG_mining_lower95"],
    comparison["WRF_GHG_mining_upper95"],
    alpha=0.15,
    label="WRF-GHG 95% CI"
)

ax.set_xlabel("Date")
ax.set_ylabel("Mining scaling factor")

ax.set_title(
    "Upper Hunter CH₄ Mining Scaling Factor\n"
    "STILT vs WRF-GHG — 1–28 February 2026"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()

fig.autofmt_xdate()

plt.tight_layout()

fig1_png = os.path.join(
    OUTDIR,
    "01_STILT_vs_WRFGHG_combined_factor_Feb2026.png"
)

fig1_pdf = os.path.join(
    OUTDIR,
    "01_STILT_vs_WRFGHG_combined_factor_Feb2026.pdf"
)

plt.savefig(
    fig1_png,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    fig1_pdf,
    bbox_inches="tight"
)

plt.show()
plt.close()


# =============================================================================
# 14. FIGURE 2
#     Individual STILT Mine Factors
# =============================================================================

fig, ax = plt.subplots(
    figsize=(15, 8)
)

for mine in MINES:

    ax.plot(
        comparison["date"],
        comparison[mine],
        linewidth=1.5,
        label=mine
    )

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.2,
    label="Prior = 1"
)

ax.set_xlabel("Date")
ax.set_ylabel("STILT posterior scaling factor")

ax.set_title(
    "Individual STILT Mine Scaling Factors\n"
    "OC1–OC13 — 5–31 February 2026"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend(
    ncol=4,
    fontsize=9
)

fig.autofmt_xdate()

plt.tight_layout()

fig2_png = os.path.join(
    OUTDIR,
    "02_STILT_individual_mine_factors_Feb2026.png"
)

fig2_pdf = os.path.join(
    OUTDIR,
    "02_STILT_individual_mine_factors_Feb2026.pdf"
)

plt.savefig(
    fig2_png,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    fig2_pdf,
    bbox_inches="tight"
)

plt.show()
plt.close()


# =============================================================================
# 15. FIGURE 3
#     Emission-weighted mine contributions
# =============================================================================

fig, ax = plt.subplots(
    figsize=(15, 8)
)

for mine in MINES:

    ax.plot(
        comparison["date"],
        comparison[
            f"{mine}_weighted_contribution"
        ],
        linewidth=1.5,
        label=mine
    )

ax.set_xlabel("Date")

ax.set_ylabel(
    "Emission-weighted contribution\n"
    r"$w_m \times s_m$"
)

ax.set_title(
    "Emission-Weighted Contribution of Individual Mines\n"
    "to the Combined STILT Mining Factor"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend(
    ncol=4,
    fontsize=9
)

fig.autofmt_xdate()

plt.tight_layout()

fig3_png = os.path.join(
    OUTDIR,
    "03_STILT_emission_weighted_mine_contributions_Feb2026.png"
)

fig3_pdf = os.path.join(
    OUTDIR,
    "03_STILT_emission_weighted_mine_contributions_Feb2026.pdf"
)

plt.savefig(
    fig3_png,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    fig3_pdf,
    bbox_inches="tight"
)

plt.show()
plt.close()


# =============================================================================
# 16. FIGURE 4
#     STILT - WRF-GHG difference
# =============================================================================

fig, ax = plt.subplots(
    figsize=(14, 5)
)

ax.plot(
    comparison["date"],
    comparison["STILT_minus_WRF"],
    marker="o",
    linewidth=2.0
)

ax.axhline(
    0.0,
    linestyle="--",
    linewidth=1.2
)

ax.set_xlabel("Date")

ax.set_ylabel(
    "STILT − WRF-GHG"
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

fig4_png = os.path.join(
    OUTDIR,
    "04_STILT_minus_WRFGHG_factor_difference_Feb2026.png"
)

fig4_pdf = os.path.join(
    OUTDIR,
    "04_STILT_minus_WRFGHG_factor_difference_Feb2026.pdf"
)

plt.savefig(
    fig4_png,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    fig4_pdf,
    bbox_inches="tight"
)

plt.show()
plt.close()


# =============================================================================
# 17. FIGURE 5
#     Aggregate total mine emissions
# =============================================================================

fig, ax = plt.subplots(
    figsize=(14, 6)
)

ax.plot(
    comparison["date"],
    comparison["STILT_total_mine_gCH4_s"],
    marker="o",
    linewidth=2.0,
    label="STILT total mines"
)

ax.plot(
    comparison["date"],
    comparison["WRF_GHG_total_mine_gCH4_s"],
    marker="s",
    linewidth=2.0,
    label="WRF-GHG total mines"
)

ax.axhline(
    total_prior,
    linestyle="--",
    linewidth=1.2,
    label=f"Prior = {total_prior:.2f} g CH₄/s"
)

ax.set_xlabel("Date")

ax.set_ylabel(
    "Total mine emission (g CH₄/s)"
)

ax.set_title(
    "Total Open-Cut Mine Emission\n"
    "STILT vs WRF-GHG"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()

fig.autofmt_xdate()

plt.tight_layout()

fig5_png = os.path.join(
    OUTDIR,
    "05_STILT_vs_WRFGHG_total_mine_emission_Feb2026.png"
)

fig5_pdf = os.path.join(
    OUTDIR,
    "05_STILT_vs_WRFGHG_total_mine_emission_Feb2026.pdf"
)

plt.savefig(
    fig5_png,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    fig5_pdf,
    bbox_inches="tight"
)

plt.show()
plt.close()


# =============================================================================
# 18. FIGURE 6
#     Mean mine contribution bar chart
# =============================================================================

plot_df = mine_contribution.sort_values(
    "mean_contribution_percent_of_STILT",
    ascending=True
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

ax.barh(
    plot_df["mine"],
    plot_df[
        "mean_contribution_percent_of_STILT"
    ]
)

ax.set_xlabel(
    "Contribution to mean STILT combined factor (%)"
)

ax.set_ylabel(
    "Mine"
)

ax.set_title(
    "Mean Contribution of Each Mine to\n"
    "Combined STILT Mining Factor — 1–28 February 2026"
)

ax.grid(
    True,
    axis="x",
    alpha=0.3
)

plt.tight_layout()

fig6_png = os.path.join(
    OUTDIR,
    "06_STILT_mean_mine_contributions_Feb2026.png"
)

fig6_pdf = os.path.join(
    OUTDIR,
    "06_STILT_mean_mine_contributions_Feb2026.pdf"
)

plt.savefig(
    fig6_png,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    fig6_pdf,
    bbox_inches="tight"
)

plt.show()
plt.close()


# =============================================================================
# 19. FIGURE 7
#     Mean emission adjustment by mine
# =============================================================================

plot_adj = adjustment_df.sort_values(
    "mean_adjustment_gCH4_s",
    ascending=True
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

ax.barh(
    plot_adj["mine"],
    plot_adj["mean_adjustment_gCH4_s"]
)

ax.axvline(
    0.0,
    linestyle="--",
    linewidth=1.2
)

ax.set_xlabel(
    "Mean STILT emission adjustment (g CH₄/s)"
)

ax.set_ylabel(
    "Mine"
)

ax.set_title(
    "Mean STILT Mine Emission Adjustment Relative to Prior\n"
    "1-28 February 2026"
)

ax.grid(
    True,
    axis="x",
    alpha=0.3
)

plt.tight_layout()

fig7_png = os.path.join(
    OUTDIR,
    "07_STILT_mean_mine_emission_adjustment_Feb2026.png"
)

fig7_pdf = os.path.join(
    OUTDIR,
    "07_STILT_mean_mine_emission_adjustment_Feb2026.pdf"
)

plt.savefig(
    fig7_png,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    fig7_pdf,
    bbox_inches="tight"
)

plt.show()
plt.close()


# =============================================================================
# 20. FINAL SUMMARY
# =============================================================================

print("\n")
print("=" * 100)
print("OUTPUT FILES")
print("=" * 100)

print("\nFull daily diagnostic:")
print(full_csv)

print("\nMine contribution analysis:")
print(mine_csv)

print("\nMine emission adjustment analysis:")
print(adjustment_csv)

print("\nFigures:")

for f in [
    fig1_png,
    fig1_pdf,
    fig2_png,
    fig2_pdf,
    fig3_png,
    fig3_pdf,
    fig4_png,
    fig4_pdf,
    fig5_png,
    fig5_pdf,
    fig6_png,
    fig6_pdf,
    fig7_png,
    fig7_pdf
]:
    print(f)

print("\n")
print("=" * 100)
print("DONE")
print("=" * 100)

