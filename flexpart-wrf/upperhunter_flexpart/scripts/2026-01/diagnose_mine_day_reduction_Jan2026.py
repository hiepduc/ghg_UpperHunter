#!/usr/bin/env python3
"""
===============================================================================
FLEXPART Upper Hunter CH4
Mine/day attribution of prior -> posterior reduction
January 5-31, 2026

Uses existing:
    H_daily_13mine_1124x351_ppm.npy
    observation_vector.csv
    UpperHuntercoalmines.csv
    mcmc_samples.npy

Unknowns:
    13 mines x 27 days = 351 source-day scaling factors

Calculates:
    1. Prior emission for each mine
    2. Posterior mean emission for each mine
    3. Prior/posterior concentration contribution by mine-day
    4. Prior -> posterior concentration reduction
    5. Mine-level attribution of total reduction
    6. Day-level attribution of total reduction
    7. Largest mine-day reductions
    8. Largest mine-day posterior increases
===============================================================================
"""

import os
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE = os.path.expanduser(
    "~/flexpart-wrf/upperhunter_flexpart/hmatrix/"
    "daily_20260105_20260131"
)

H_FILE = os.path.join(
    BASE,
    "H_daily_13mine_1124x351_ppm.npy"
)

OBS_FILE = os.path.join(
    BASE,
    "observation_vector.csv"
)

PRIOR_FILE = os.path.expanduser(
    "~/flexpart-wrf/emission_UH/RH_upperHunter/"
    "UpperHuntercoalmines.csv"
)

MCMC_FILE = os.path.join(
    BASE,
    "bayesian_analytical",
    "mcmc_blocked",
    "mcmc_samples.npy"
)

OUTDIR = os.path.join(
    BASE,
    "mine_day_reduction_diagnostic_Jan2026"
)

os.makedirs(OUTDIR, exist_ok=True)


# =============================================================================
# SETTINGS
# =============================================================================

BACKGROUND_PPM = 1.85

START_DATE = pd.Timestamp("2026-01-05")
END_DATE   = pd.Timestamp("2026-01-31")

MINES = [f"OC{i}" for i in range(1, 14)]
N_MINE = 13
N_DAY = 27


# =============================================================================
# HEADER
# =============================================================================

print("=" * 90)
print("FLEXPART UPPER HUNTER CH4")
print("MINE/DAY ATTRIBUTION OF PRIOR -> POSTERIOR REDUCTION")
print("January 5-31, 2026")
print("=" * 90)


# =============================================================================
# LOAD H
# =============================================================================

print("\nLoading H matrix...")

H = np.load(H_FILE)

print("H shape:", H.shape)

if H.shape != (1124, 351):
    raise ValueError(
        f"Unexpected H shape {H.shape}; expected (1124, 351)"
    )


# =============================================================================
# LOAD OBSERVATIONS
# =============================================================================

print("\nLoading observations...")

obs = pd.read_csv(OBS_FILE)

print("Observation rows:", len(obs))
print("Columns:", list(obs.columns))

if "datetime_aest" in obs.columns:
    obs["datetime"] = pd.to_datetime(obs["datetime_aest"])
elif "datetime_utc" in obs.columns:
    obs["datetime"] = pd.to_datetime(obs["datetime_utc"])
else:
    raise ValueError("No datetime_aest or datetime_utc column found.")

obs["date"] = obs["datetime"].dt.normalize()

print(
    "Date range:",
    obs["date"].min(),
    "to",
    obs["date"].max()
)


# =============================================================================
# LOAD PRIOR EMISSIONS
# =============================================================================

print("\nLoading prior mine emissions...")

prior_df = pd.read_csv(PRIOR_FILE)

print("Prior columns:")
print(list(prior_df.columns))

# Keep OC1-OC13 only
prior_df["Facility_code_name"] = (
    prior_df["Facility_code_name"]
    .astype(str)
    .str.strip()
)

prior_lookup = {}

for mine in MINES:

    match = prior_df[
        prior_df["Facility_code_name"].str.upper() == mine
    ]

    if len(match) == 0:
        raise ValueError(f"Prior emission for {mine} not found.")

    # g CH4/s -> kg CH4/s
    prior_lookup[mine] = float(match.iloc[0]["gCH4/s"]) / 1000.0


E0_mine = np.array(
    [prior_lookup[mine] for mine in MINES]
)

print("\nPrior emissions:")
for mine, e in zip(MINES, E0_mine):
    print(f"{mine:>4s}  {e:.8f} kg/s")

print(
    f"\nTotal prior emission = "
    f"{E0_mine.sum():.8f} kg/s"
)


# =============================================================================
# LOAD MCMC
# =============================================================================

print("\nLoading MCMC posterior samples...")

samples = np.load(MCMC_FILE)

print("MCMC shape:", samples.shape)

if samples.shape[1] != 351:
    raise ValueError(
        f"Unexpected MCMC parameter dimension: {samples.shape[1]}"
    )

n_samples = samples.shape[0]

print("Posterior samples:", n_samples)


# =============================================================================
# RESHAPE POSTERIOR SAMPLES
# =============================================================================

# Original parameter ordering:
#
# day 1: OC1 ... OC13
# day 2: OC1 ... OC13
# ...
# day 27: OC1 ... OC13
#
# Convert:
# (samples, 351)
# ->
# (samples, day, mine)

samples_3d = samples.reshape(
    n_samples,
    N_DAY,
    N_MINE
)

print(
    "Reshaped samples:",
    samples_3d.shape
)


# =============================================================================
# DAILY POSTERIOR MEAN SCALINGS
# =============================================================================

posterior_scaling = samples_3d.mean(axis=0)

dates = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)


# =============================================================================
# POSTERIOR EMISSION BY MINE/DAY
# =============================================================================

posterior_emission = (
    posterior_scaling
    * E0_mine[None, :]
)

prior_emission = np.broadcast_to(
    E0_mine[None, :],
    (N_DAY, N_MINE)
)


# =============================================================================
# MINE-LEVEL EMISSION SUMMARY
# =============================================================================

mine_rows = []

for j, mine in enumerate(MINES):

    prior_total = E0_mine[j]

    post_daily = posterior_emission[:, j]

    posterior_mean = post_daily.mean()

    reduction = prior_total - posterior_mean

    reduction_pct = (
        100.0 * reduction / prior_total
        if prior_total != 0
        else np.nan
    )

    mine_rows.append({
        "mine": mine,
        "prior_kg_s": prior_total,
        "posterior_mean_kg_s": posterior_mean,
        "reduction_kg_s": reduction,
        "reduction_pct": reduction_pct,
        "mean_scaling": posterior_scaling[:, j].mean(),
        "median_scaling": np.median(
            posterior_scaling[:, j]
        ),
        "min_daily_scaling": posterior_scaling[:, j].min(),
        "max_daily_scaling": posterior_scaling[:, j].max(),
    })


mine_summary = pd.DataFrame(mine_rows)

mine_summary.to_csv(
    os.path.join(
        OUTDIR,
        "mine_emission_reduction_summary.csv"
    ),
    index=False
)


# =============================================================================
# SOURCE-DAY H CONTRIBUTION
# =============================================================================

print("\nCalculating source-day concentration contributions...")


# H has one column per mine/day:
#
# column = day * 13 + mine
#
# For each observation:
#
# contribution = H[:, column] * emission
#
# We need aggregate absolute/mean contributions over observations
# for each source-day.

prior_contribution = np.zeros(
    (len(obs), N_DAY, N_MINE)
)

posterior_contribution = np.zeros(
    (len(obs), N_DAY, N_MINE)
)


for d in range(N_DAY):

    for j in range(N_MINE):

        col = d * N_MINE + j

        h_col = H[:, col]

        prior_contribution[:, d, j] = (
            h_col * E0_mine[j]
        )

        posterior_contribution[:, d, j] = (
            h_col * posterior_emission[d, j]
        )


# =============================================================================
# PRIOR -> POSTERIOR REDUCTION
# =============================================================================

reduction_contribution = (
    prior_contribution
    - posterior_contribution
)


# =============================================================================
# AGGREGATE MINE/DAY CONTRIBUTIONS
# =============================================================================

mine_day_rows = []

for d, date in enumerate(dates):

    for j, mine in enumerate(MINES):

        prior_mean_conc = (
            prior_contribution[:, d, j].mean()
        )

        post_mean_conc = (
            posterior_contribution[:, d, j].mean()
        )

        reduction_mean = (
            reduction_contribution[:, d, j].mean()
        )

        mine_day_rows.append({

            "date": date.strftime("%Y-%m-%d"),

            "mine": mine,

            "prior_emission_kg_s":
                E0_mine[j],

            "posterior_scaling":
                posterior_scaling[d, j],

            "posterior_emission_kg_s":
                posterior_emission[d, j],

            "prior_mean_contribution_ppm":
                prior_mean_conc,

            "posterior_mean_contribution_ppm":
                post_mean_conc,

            "reduction_ppm":
                reduction_mean,

        })


mine_day = pd.DataFrame(mine_day_rows)


mine_day.to_csv(
    os.path.join(
        OUTDIR,
        "mine_day_concentration_reduction.csv"
    ),
    index=False
)


# =============================================================================
# TOTAL MINE REDUCTION CONTRIBUTION
# =============================================================================

mine_total_rows = []

for j, mine in enumerate(MINES):

    prior_c = prior_contribution[:, :, j].mean()

    post_c = posterior_contribution[:, :, j].mean()

    reduction_c = prior_c - post_c

    mine_total_rows.append({

        "mine": mine,

        "prior_mean_contribution_ppm":
            prior_c,

        "posterior_mean_contribution_ppm":
            post_c,

        "reduction_ppm":
            reduction_c,

    })


mine_concentration_summary = pd.DataFrame(
    mine_total_rows
)

total_reduction = (
    mine_concentration_summary["reduction_ppm"].sum()
)

mine_concentration_summary["fraction_of_total_reduction"] = (
    mine_concentration_summary["reduction_ppm"]
    / total_reduction
)

mine_concentration_summary["percent_of_total_reduction"] = (
    100.0
    * mine_concentration_summary["fraction_of_total_reduction"]
)


mine_concentration_summary = (
    mine_concentration_summary
    .sort_values(
        "reduction_ppm",
        ascending=False
    )
)


mine_concentration_summary.to_csv(
    os.path.join(
        OUTDIR,
        "mine_concentration_reduction_attribution.csv"
    ),
    index=False
)


# =============================================================================
# DAY-LEVEL REDUCTION
# =============================================================================

day_rows = []

for d, date in enumerate(dates):

    prior_c = prior_contribution[:, d, :].sum(axis=1).mean()

    post_c = posterior_contribution[:, d, :].sum(axis=1).mean()

    reduction_c = prior_c - post_c

    day_rows.append({

        "date": date.strftime("%Y-%m-%d"),

        "prior_mine_contribution_ppm":
            prior_c,

        "posterior_mine_contribution_ppm":
            post_c,

        "reduction_ppm":
            reduction_c,

    })


day_summary = pd.DataFrame(day_rows)

day_total_reduction = day_summary["reduction_ppm"].sum()

day_summary["fraction_of_total_reduction"] = (
    day_summary["reduction_ppm"]
    / day_total_reduction
)

day_summary["percent_of_total_reduction"] = (
    100.0
    * day_summary["fraction_of_total_reduction"]
)

day_summary.to_csv(
    os.path.join(
        OUTDIR,
        "daily_concentration_reduction_attribution.csv"
    ),
    index=False
)


# =============================================================================
# LARGEST POSITIVE REDUCTIONS
# =============================================================================

largest_reductions = (
    mine_day
    .sort_values(
        "reduction_ppm",
        ascending=False
    )
    .head(50)
)

largest_reductions.to_csv(
    os.path.join(
        OUTDIR,
        "top_50_mine_day_reductions.csv"
    ),
    index=False
)


# =============================================================================
# LARGEST POSTERIOR INCREASES
# =============================================================================

largest_increases = (
    mine_day
    .sort_values(
        "reduction_ppm",
        ascending=True
    )
    .head(50)
)

largest_increases.to_csv(
    os.path.join(
        OUTDIR,
        "top_50_mine_day_increases.csv"
    ),
    index=False
)


# =============================================================================
# DAILY SCALING TABLE
# =============================================================================

scaling_rows = []

for d, date in enumerate(dates):

    row = {
        "date": date.strftime("%Y-%m-%d")
    }

    for j, mine in enumerate(MINES):
        row[mine] = posterior_scaling[d, j]

    row["aggregate_scaling"] = (
        posterior_emission[d, :].sum()
        / E0_mine.sum()
    )

    scaling_rows.append(row)


scaling_df = pd.DataFrame(scaling_rows)

scaling_df.to_csv(
    os.path.join(
        OUTDIR,
        "daily_mine_scalings.csv"
    ),
    index=False
)


# =============================================================================
# PRINT RESULTS
# =============================================================================

print("\n")
print("=" * 90)
print("MINE-LEVEL EMISSION REDUCTION")
print("=" * 90)

print(
    mine_summary[
        [
            "mine",
            "prior_kg_s",
            "posterior_mean_kg_s",
            "reduction_kg_s",
            "reduction_pct",
            "mean_scaling",
        ]
    ].to_string(index=False)
)


print("\n")
print("=" * 90)
print("MINE CONTRIBUTION TO CONCENTRATION REDUCTION")
print("=" * 90)

print(
    mine_concentration_summary[
        [
            "mine",
            "prior_mean_contribution_ppm",
            "posterior_mean_contribution_ppm",
            "reduction_ppm",
            "percent_of_total_reduction",
        ]
    ].to_string(index=False)
)


print("\n")
print("=" * 90)
print("TOP 20 MINE-DAY CONCENTRATION REDUCTIONS")
print("=" * 90)

print(
    largest_reductions[
        [
            "date",
            "mine",
            "posterior_scaling",
            "prior_mean_contribution_ppm",
            "posterior_mean_contribution_ppm",
            "reduction_ppm",
        ]
    ].head(20).to_string(index=False)
)


print("\n")
print("=" * 90)
print("TOP 20 MINE-DAY POSTERIOR INCREASES")
print("=" * 90)

print(
    largest_increases[
        [
            "date",
            "mine",
            "posterior_scaling",
            "prior_mean_contribution_ppm",
            "posterior_mean_contribution_ppm",
            "reduction_ppm",
        ]
    ].head(20).to_string(index=False)
)


print("\n")
print("=" * 90)
print("DAY-LEVEL REDUCTION")
print("=" * 90)

print(
    day_summary[
        [
            "date",
            "prior_mine_contribution_ppm",
            "posterior_mine_contribution_ppm",
            "reduction_ppm",
        ]
    ].to_string(index=False)
)


# =============================================================================
# TOTAL EMISSION CHECK
# =============================================================================

prior_total = E0_mine.sum()

posterior_daily_total = posterior_emission.sum(axis=1).mean()

overall_scaling = (
    posterior_daily_total / prior_total
)

print("\n")
print("=" * 90)
print("TOTAL EMISSION CHECK")
print("=" * 90)

print(
    f"Prior total emission:      "
    f"{prior_total:.8f} kg/s"
)

print(
    f"Posterior mean emission:   "
    f"{posterior_daily_total:.8f} kg/s"
)

print(
    f"Posterior/prior:           "
    f"{overall_scaling:.8f}"
)

print(
    f"Overall reduction:         "
    f"{100*(1-overall_scaling):.3f}%"
)


print("\n")
print("=" * 90)
print("OUTPUT DIRECTORY")
print("=" * 90)

print(OUTDIR)

print("\nDone.")

