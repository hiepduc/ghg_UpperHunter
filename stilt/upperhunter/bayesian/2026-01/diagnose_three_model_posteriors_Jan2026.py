#!/usr/bin/env python3
"""
===============================================================================
DIAGNOSTIC: STILT + WRF-GHG + FLEXPART-WRF POSTERIORS
Upper Hunter CH4 - January 2026

Purpose
-------
Diagnose differences between the three independent inverse systems.

Focus:
  1. Daily posterior scaling factors
  2. FLEXPART-WRF uncertainty directly from 6000 MCMC samples
  3. STILT and WRF-GHG 95% intervals
  4. Daily observed CH4
  5. FLEXPART posterior prediction versus observations
  6. Identification of days with strong model disagreement

No ensemble weighting is changed by this script.
===============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# USER CONFIGURATION
# =============================================================================

START_DATE = "2026-01-05"
END_DATE   = "2026-01-31"

# -------------------------------------------------------------------------
# Prior mine emissions
# -------------------------------------------------------------------------
PRIOR_FILE = (
    "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter/"
    "UpperHuntercoalmines.csv"
)

# -------------------------------------------------------------------------
# FLEXPART-WRF
# -------------------------------------------------------------------------
FLEXPART_SCALING_FILE = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "hmatrix/daily_20260105_20260131/"
    "bayesian_analytical/mcmc_blocked/"
    "mcmc_daily_scaling.csv"
)

FLEXPART_SAMPLES_FILE = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "hmatrix/daily_20260105_20260131/"
    "bayesian_analytical/mcmc_blocked/"
    "mcmc_samples.npy"
)

FLEXPART_H_FILE = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "hmatrix/daily_20260105_20260131/"
    "H_daily_13mine_1124x351_ppm.npy"
)

OBS_VECTOR_FILE = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "hmatrix/daily_20260105_20260131/"
    "observation_vector.csv"
)

# -------------------------------------------------------------------------
# STILT
# -------------------------------------------------------------------------
STILT_FILE = (
    "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter/"
    "stilt/Bayesian_daily_inversion_Jan2026/"
    "daily_scaling_factors_Jan2026.csv"
)

# -------------------------------------------------------------------------
# WRF-GHG
# -------------------------------------------------------------------------
WRF_FILE = (
    "/mnt/scratch_lustre/duch/runnetzero/"
    "bayesian_daily_MH_5_31Jan2026/"
    "bayesian_daily_posterior.csv"
)

# -------------------------------------------------------------------------
# Picarro observations
# -------------------------------------------------------------------------
PICARRO_FILE = (
    "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter/"
    "tapm_output/Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

# -------------------------------------------------------------------------
# Output
# -------------------------------------------------------------------------
OUTPUT_DIR = (
    "/home/duch/flexpart-wrf/upperhunter_flexpart/"
    "hmatrix/daily_20260105_20260131/"
    "three_model_posterior_diagnostics_Jan2026"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# CONSTANTS
# =============================================================================

MINES = [f"OC{i}" for i in range(1, 14)]

RECEPTORS = [
    "Singleton cleaned",
    "Muswellbrook cleaned",
    "Merriwa cleaned",
]

# Observation column names
OBS_COLS = {
    "Singleton": "Singleton cleaned",
    "Muswellbrook": "Muswellbrook cleaned",
    "Merriwa": "Merriwa cleaned",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def normalize_date(series):
    return pd.to_datetime(series).dt.normalize()


# =============================================================================
# 1. READ PRIOR
# =============================================================================

section("READING PRIOR MINE EMISSIONS")

prior = pd.read_csv(PRIOR_FILE)

prior = prior[prior["Facility_code_name"].isin(MINES)].copy()

prior = (
    prior.set_index("Facility_code_name")
    .reindex(MINES)
    .reset_index()
)

prior_emissions = prior["gCH4/s"].astype(float).values

prior_total = prior_emissions.sum()

weights = prior_emissions / prior_total

print(f"Prior total = {prior_total:.6f} g/s")
print(f"Sum weights = {weights.sum():.10f}")


# =============================================================================
# 2. READ FLEXPART DAILY MEANS
# =============================================================================

section("READING FLEXPART-WRF DAILY POSTERIOR")

fp_daily = pd.read_csv(FLEXPART_SCALING_FILE)

fp_daily["date"] = pd.to_datetime(fp_daily["date"]).dt.normalize()

fp_daily = fp_daily[
    (fp_daily["date"] >= pd.Timestamp(START_DATE)) &
    (fp_daily["date"] <= pd.Timestamp(END_DATE))
].copy()

print(f"FLEXPART daily records = {len(fp_daily)}")

# Recalculate weighted factor from daily mine means
fp_daily["FLEXPART_factor_recalculated"] = (
    fp_daily[MINES].values @ weights
)

if "FLEXPART_factor" in fp_daily.columns:
    maxdiff = np.max(
        np.abs(
            fp_daily["FLEXPART_factor"].values
            - fp_daily["FLEXPART_factor_recalculated"].values
        )
    )

    print(
        "Maximum difference between stored and recalculated "
        f"FLEXPART factor = {maxdiff:.6e}"
    )

fp_daily["FLEXPART_factor"] = (
    fp_daily["FLEXPART_factor_recalculated"]
)


# =============================================================================
# 3. READ FULL FLEXPART MCMC
# =============================================================================

section("READING FULL FLEXPART-WRF MCMC")

samples = np.load(FLEXPART_SAMPLES_FILE)

print(f"MCMC sample shape = {samples.shape}")

if samples.shape[1] != 351:
    raise ValueError(
        f"Expected 351 parameters, found {samples.shape[1]}"
    )

if samples.shape[0] < 100:
    raise ValueError("Too few MCMC samples.")

n_samples = samples.shape[0]

# Parameter ordering:
# day 1: OC1 ... OC13
# day 2: OC1 ... OC13
# ...
# day 27: OC1 ... OC13

n_days = len(fp_daily)

expected_parameters = n_days * 13

if samples.shape[1] != expected_parameters:
    raise ValueError(
        f"Expected {expected_parameters} parameters "
        f"for {n_days} days, found {samples.shape[1]}"
    )

# Reshape:
# (sample, parameter) -> (sample, day, mine)

samples_3d = samples.reshape(
    n_samples,
    n_days,
    13
)

# Prior-emission-weighted aggregate FLEXPART factor
# for every posterior sample and every day
fp_aggregate_samples = np.einsum(
    "sdm,m->sd",
    samples_3d,
    weights
)

# Posterior statistics
fp_mean = np.mean(fp_aggregate_samples, axis=0)
fp_median = np.median(fp_aggregate_samples, axis=0)
fp_std = np.std(fp_aggregate_samples, axis=0, ddof=1)

fp_lower95 = np.percentile(
    fp_aggregate_samples,
    2.5,
    axis=0
)

fp_upper95 = np.percentile(
    fp_aggregate_samples,
    97.5,
    axis=0
)

fp_diag = pd.DataFrame({
    "date": fp_daily["date"].values,
    "FLEXPART_factor": fp_mean,
    "FLEXPART_median": fp_median,
    "FLEXPART_std": fp_std,
    "FLEXPART_lower95": fp_lower95,
    "FLEXPART_upper95": fp_upper95,
})

print("\nFLEXPART posterior aggregate diagnostics:")
print(
    fp_diag[
        [
            "date",
            "FLEXPART_factor",
            "FLEXPART_std",
            "FLEXPART_lower95",
            "FLEXPART_upper95",
        ]
    ].to_string(index=False)
)


# =============================================================================
# 4. READ STILT
# =============================================================================

section("READING STILT POSTERIOR")

stilt = pd.read_csv(STILT_FILE)

stilt["date"] = pd.to_datetime(stilt["date"]).dt.normalize()

stilt = stilt[
    (stilt["date"] >= pd.Timestamp(START_DATE)) &
    (stilt["date"] <= pd.Timestamp(END_DATE))
].copy()

print(f"STILT records = {len(stilt)}")

# Aggregate STILT using prior emission weights
stilt_factor = (
    stilt[MINES].values @ weights
)

# Approximate uncertainty using existing STILT mine std values
stilt_var = np.zeros(len(stilt))

for i, mine in enumerate(MINES):
    std_col = f"{mine}_std"

    if std_col in stilt.columns:
        stilt_var += (
            weights[i] ** 2
            * stilt[std_col].astype(float).values ** 2
        )

stilt_std = np.sqrt(stilt_var)

stilt_diag = pd.DataFrame({
    "date": stilt["date"].values,
    "STILT_factor": stilt_factor,
    "STILT_std": stilt_std,
    "STILT_lower95": stilt_factor - 1.96 * stilt_std,
    "STILT_upper95": stilt_factor + 1.96 * stilt_std,
})


# =============================================================================
# 5. READ WRF-GHG
# =============================================================================

section("READING WRF-GHG POSTERIOR")

wrf = pd.read_csv(WRF_FILE)

wrf["date"] = pd.to_datetime(wrf["date"]).dt.normalize()

wrf = wrf[
    (wrf["date"] >= pd.Timestamp(START_DATE)) &
    (wrf["date"] <= pd.Timestamp(END_DATE))
].copy()

wrf = wrf[
    wrf["sector"].astype(str).str.strip().str.lower()
    == "mining"
].copy()

print(f"WRF-GHG mining records = {len(wrf)}")

wrf_diag = pd.DataFrame({
    "date": wrf["date"].values,
    "WRFGHG_factor":
        wrf["posterior_mean"].astype(float).values,
    "WRFGHG_std":
        wrf["posterior_std"].astype(float).values,
    "WRFGHG_lower95":
        wrf["lower_2.5pct"].astype(float).values,
    "WRFGHG_upper95":
        wrf["upper_97.5pct"].astype(float).values,
})


# =============================================================================
# 6. READ PICARRO OBSERVATIONS
# =============================================================================

section("READING PICARRO OBSERVATIONS")

obs = pd.read_csv(PICARRO_FILE)

print("Observation columns:")
print(obs.columns.tolist())

# Existing Datetime column is used.
obs["datetime"] = pd.to_datetime(
    obs["Datetime"],
    errors="coerce"
)

obs = obs.dropna(subset=["datetime"]).copy()

obs["date"] = obs["datetime"].dt.normalize()

obs = obs[
    (obs["date"] >= pd.Timestamp(START_DATE)) &
    (obs["date"] <= pd.Timestamp(END_DATE))
].copy()

print(f"Observation records = {len(obs)}")

# Daily statistics for each receptor
daily_obs_rows = []

for date, group in obs.groupby("date"):

    row = {"date": date}

    all_values = []

    for receptor, col in OBS_COLS.items():

        if col not in group.columns:
            row[f"{receptor}_mean"] = np.nan
            row[f"{receptor}_N"] = 0
            continue

        values = pd.to_numeric(
            group[col],
            errors="coerce"
        ).dropna()

        row[f"{receptor}_mean"] = (
            values.mean() if len(values) > 0 else np.nan
        )

        row[f"{receptor}_N"] = len(values)

        if len(values) > 0:
            all_values.extend(values.values)

    row["OBS_mean"] = (
        np.mean(all_values)
        if len(all_values) > 0
        else np.nan
    )

    row["OBS_std"] = (
        np.std(all_values, ddof=1)
        if len(all_values) > 1
        else np.nan
    )

    row["OBS_N"] = len(all_values)

    daily_obs_rows.append(row)

obs_daily = pd.DataFrame(daily_obs_rows)


# =============================================================================
# 7. MERGE ALL DIAGNOSTICS
# =============================================================================

section("MERGING DIAGNOSTICS")

diag = fp_diag.merge(
    stilt_diag,
    on="date",
    how="outer"
)

diag = diag.merge(
    wrf_diag,
    on="date",
    how="outer"
)

diag = diag.merge(
    obs_daily,
    on="date",
    how="outer"
)

diag = diag.sort_values("date").reset_index(drop=True)

print(f"Diagnostic days = {len(diag)}")


# =============================================================================
# 8. MODEL MEAN AND DISAGREEMENT
# =============================================================================

diag["three_model_mean"] = (
    diag["STILT_factor"]
    + diag["WRFGHG_factor"]
    + diag["FLEXPART_factor"]
) / 3.0

diag["model_spread"] = np.sqrt(
    (
        (diag["STILT_factor"] -
         diag["three_model_mean"]) ** 2
        +
        (diag["WRFGHG_factor"] -
         diag["three_model_mean"]) ** 2
        +
        (diag["FLEXPART_factor"] -
         diag["three_model_mean"]) ** 2
    ) / 3.0
)

diag["FLEXPART_minus_STILT"] = (
    diag["FLEXPART_factor"]
    - diag["STILT_factor"]
)

diag["FLEXPART_minus_WRFGHG"] = (
    diag["FLEXPART_factor"]
    - diag["WRFGHG_factor"]
)

diag["FLEXPART_distance_from_other_two"] = (
    (
        np.abs(diag["FLEXPART_factor"] -
               diag["STILT_factor"])
        +
        np.abs(diag["FLEXPART_factor"] -
               diag["WRFGHG_factor"])
    ) / 2.0
)


# =============================================================================
# 9. CORRELATIONS
# =============================================================================

section("DAILY CORRELATIONS")

def corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan

    return np.corrcoef(
        x[mask],
        y[mask]
    )[0, 1]


r_fp_obs = corr(
    diag["FLEXPART_factor"].values,
    diag["OBS_mean"].values
)

r_stilt_obs = corr(
    diag["STILT_factor"].values,
    diag["OBS_mean"].values
)

r_wrf_obs = corr(
    diag["WRFGHG_factor"].values,
    diag["OBS_mean"].values
)

r_fp_stilt = corr(
    diag["FLEXPART_factor"].values,
    diag["STILT_factor"].values
)

r_fp_wrf = corr(
    diag["FLEXPART_factor"].values,
    diag["WRFGHG_factor"].values
)

print(f"FLEXPART vs daily observed CH4 : {r_fp_obs:.6f}")
print(f"STILT vs daily observed CH4    : {r_stilt_obs:.6f}")
print(f"WRF-GHG vs daily observed CH4  : {r_wrf_obs:.6f}")
print(f"FLEXPART vs STILT              : {r_fp_stilt:.6f}")
print(f"FLEXPART vs WRF-GHG            : {r_fp_wrf:.6f}")


# =============================================================================
# 10. IDENTIFY STRONG FLEXPART EXCURSIONS
# =============================================================================

section("LARGEST FLEXPART DEVIATIONS")

excursions = diag.sort_values(
    "FLEXPART_distance_from_other_two",
    ascending=False
).copy()

print(
    excursions[
        [
            "date",
            "STILT_factor",
            "WRFGHG_factor",
            "FLEXPART_factor",
            "FLEXPART_lower95",
            "FLEXPART_upper95",
            "OBS_mean",
            "model_spread",
            "FLEXPART_distance_from_other_two",
        ]
    ].head(10).to_string(index=False)
)


# =============================================================================
# 11. CHECK WHETHER OTHER MODELS FALL INSIDE FLEXPART 95% INTERVAL
# =============================================================================

diag["STILT_inside_FLEXPART95"] = (
    (diag["STILT_factor"] >= diag["FLEXPART_lower95"])
    &
    (diag["STILT_factor"] <= diag["FLEXPART_upper95"])
)

diag["WRFGHG_inside_FLEXPART95"] = (
    (diag["WRFGHG_factor"] >= diag["FLEXPART_lower95"])
    &
    (diag["WRFGHG_factor"] <= diag["FLEXPART_upper95"])
)

print("\nOther-model values inside FLEXPART 95% interval:")

print(
    f"STILT    : "
    f"{diag['STILT_inside_FLEXPART95'].sum()} / {len(diag)} days"
)

print(
    f"WRF-GHG  : "
    f"{diag['WRFGHG_inside_FLEXPART95'].sum()} / {len(diag)} days"
)


# =============================================================================
# 12. CONVERT FACTORS TO EMISSIONS
# =============================================================================

for model in ["STILT", "WRFGHG", "FLEXPART"]:
    factor_col = f"{model}_factor"

    if factor_col in diag.columns:
        diag[f"{model}_g_s"] = (
            diag[factor_col] * prior_total
        )

diag["three_model_mean_g_s"] = (
    diag["three_model_mean"] * prior_total
)

diag["model_spread_g_s"] = (
    diag["model_spread"] * prior_total
)


# =============================================================================
# 13. SAVE DIAGNOSTIC CSV
# =============================================================================

diagnostic_file = os.path.join(
    OUTPUT_DIR,
    "three_model_posterior_diagnostics_daily_Jan2026.csv"
)

diag.to_csv(
    diagnostic_file,
    index=False,
    float_format="%.8f"
)

print(f"\nSaved: {diagnostic_file}")


# =============================================================================
# 14. SUMMARY CSV
# =============================================================================

summary = pd.DataFrame({
    "quantity": [
        "Number of days",
        "FLEXPART mean factor",
        "FLEXPART mean std",
        "FLEXPART mean lower95",
        "FLEXPART mean upper95",
        "STILT mean factor",
        "WRF-GHG mean factor",
        "FLEXPART vs observed correlation",
        "STILT vs observed correlation",
        "WRF-GHG vs observed correlation",
        "FLEXPART vs STILT correlation",
        "FLEXPART vs WRF-GHG correlation",
        "STILT inside FLEXPART 95%",
        "WRF-GHG inside FLEXPART 95%",
    ],
    "value": [
        len(diag),
        diag["FLEXPART_factor"].mean(),
        diag["FLEXPART_std"].mean(),
        diag["FLEXPART_lower95"].mean(),
        diag["FLEXPART_upper95"].mean(),
        diag["STILT_factor"].mean(),
        diag["WRFGHG_factor"].mean(),
        r_fp_obs,
        r_stilt_obs,
        r_wrf_obs,
        r_fp_stilt,
        r_fp_wrf,
        diag["STILT_inside_FLEXPART95"].sum(),
        diag["WRFGHG_inside_FLEXPART95"].sum(),
    ]
})

summary_file = os.path.join(
    OUTPUT_DIR,
    "three_model_posterior_diagnostics_summary_Jan2026.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print(f"Saved: {summary_file}")


# =============================================================================
# 15. PLOT: THREE MODEL FACTORS + UNCERTAINTY
# =============================================================================

section("CREATING PLOTS")

dates = diag["date"]

fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(
    dates,
    diag["STILT_factor"],
    marker="o",
    linewidth=2,
    label="STILT-WRF"
)

ax.plot(
    dates,
    diag["WRFGHG_factor"],
    marker="s",
    linewidth=2,
    label="WRF-GHG"
)

ax.plot(
    dates,
    diag["FLEXPART_factor"],
    marker="^",
    linewidth=2,
    label="FLEXPART-WRF"
)

ax.fill_between(
    dates,
    diag["FLEXPART_lower95"],
    diag["FLEXPART_upper95"],
    alpha=0.20,
    label="FLEXPART 95% credible interval"
)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.2,
    label="Prior factor = 1"
)

ax.set_xlabel("Date")
ax.set_ylabel("Mine emission scaling factor")
ax.set_title(
    "Upper Hunter CH4: Three Inverse Models\n"
    "Daily posterior scaling factors"
)

ax.grid(True, alpha=0.3)
ax.legend()
fig.autofmt_xdate()

fig.tight_layout()

plot1 = os.path.join(
    OUTPUT_DIR,
    "01_three_model_factors_with_FLEXPART_95CI_Jan2026.png"
)

fig.savefig(
    plot1,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print(f"Saved: {plot1}")


# =============================================================================
# 16. PLOT: OBSERVED CH4 + MODEL FACTORS
# =============================================================================

fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.plot(
    dates,
    diag["OBS_mean"],
    marker="o",
    linewidth=2,
    label="Observed CH4 daily mean"
)

ax1.set_xlabel("Date")
ax1.set_ylabel("Observed CH4 (ppmv)")

ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()

ax2.plot(
    dates,
    diag["STILT_factor"],
    linestyle="--",
    linewidth=1.8,
    label="STILT factor"
)

ax2.plot(
    dates,
    diag["WRFGHG_factor"],
    linestyle="--",
    linewidth=1.8,
    label="WRF-GHG factor"
)

ax2.plot(
    dates,
    diag["FLEXPART_factor"],
    linestyle="--",
    linewidth=1.8,
    label="FLEXPART factor"
)

ax2.set_ylabel("Emission scaling factor")

ax1.set_title(
    "Upper Hunter CH4: Observed CH4 and Inverse-Model Factors"
)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="best"
)

fig.autofmt_xdate()
fig.tight_layout()

plot2 = os.path.join(
    OUTPUT_DIR,
    "02_observed_CH4_vs_three_model_factors_Jan2026.png"
)

fig.savefig(
    plot2,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print(f"Saved: {plot2}")


# =============================================================================
# 17. PLOT: FLEXPART VS OTHER MODELS
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(
    dates,
    diag["FLEXPART_minus_STILT"],
    marker="o",
    linewidth=2,
    label="FLEXPART − STILT"
)

ax.plot(
    dates,
    diag["FLEXPART_minus_WRFGHG"],
    marker="s",
    linewidth=2,
    label="FLEXPART − WRF-GHG"
)

ax.axhline(
    0.0,
    linestyle="--",
    linewidth=1.2
)

ax.set_xlabel("Date")
ax.set_ylabel("Scaling-factor difference")

ax.set_title(
    "FLEXPART-WRF Difference from STILT-WRF and WRF-GHG"
)

ax.grid(True, alpha=0.3)
ax.legend()

fig.autofmt_xdate()
fig.tight_layout()

plot3 = os.path.join(
    OUTPUT_DIR,
    "03_FLEXPART_difference_from_other_models_Jan2026.png"
)

fig.savefig(
    plot3,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print(f"Saved: {plot3}")


# =============================================================================
# 18. PLOT: MODEL SPREAD
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 6))

ax.bar(
    dates,
    diag["model_spread"]
)

ax.set_xlabel("Date")
ax.set_ylabel("Three-model spread")

ax.set_title(
    "Three-Model Inverse Spread"
)

ax.grid(True, axis="y", alpha=0.3)

fig.autofmt_xdate()
fig.tight_layout()

plot4 = os.path.join(
    OUTPUT_DIR,
    "04_three_model_spread_Jan2026.png"
)

fig.savefig(
    plot4,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print(f"Saved: {plot4}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

section("DIAGNOSTIC COMPLETE")

print(f"Output directory:")
print(OUTPUT_DIR)

print("\nFiles:")
print("  three_model_posterior_diagnostics_daily_Jan2026.csv")
print("  three_model_posterior_diagnostics_summary_Jan2026.csv")
print("  01_three_model_factors_with_FLEXPART_95CI_Jan2026.png")
print("  02_observed_CH4_vs_three_model_factors_Jan2026.png")
print("  03_FLEXPART_difference_from_other_models_Jan2026.png")
print("  04_three_model_spread_Jan2026.png")

print("\nKey correlations:")
print(f"  FLEXPART vs observations = {r_fp_obs:.4f}")
print(f"  STILT vs observations    = {r_stilt_obs:.4f}")
print(f"  WRF-GHG vs observations  = {r_wrf_obs:.4f}")

print("\nLargest FLEXPART disagreement days:")
print(
    excursions[
        [
            "date",
            "STILT_factor",
            "WRFGHG_factor",
            "FLEXPART_factor",
            "OBS_mean",
            "model_spread",
        ]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 90)
print("DONE")
print("=" * 90)

