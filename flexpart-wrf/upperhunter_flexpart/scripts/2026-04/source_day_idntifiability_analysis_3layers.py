#!/usr/bin/env python3
"""
FLEXPART-WRF MCMC SOURCE/DAY IDENTIFIABILITY ANALYSIS
======================================================

Analyse all 13 mines x 31 days = 390 source/day parameters.

For every parameter, calculate:

    - posterior mean
    - posterior median
    - posterior standard deviation
    - posterior 2.5% and 97.5% credible interval
    - posterior minimum / maximum
    - H maximum sensitivity
    - H sum of absolute sensitivities
    - H L2 norm
    - H sum
    - number of observations with nonzero sensitivity
    - whether parameter is informed by observations
    - prior scaling factor
    - prior emission
    - posterior mean/median/CI emissions
    - posterior/prior emission ratios
    - identifiability class

Input:
    H_flexpart_0_10m_13mine_Apr2026.csv
    FLEXPART_MCMC_posterior_samples_Apr2026.npz
    UpperHuntercoalmines.csv

Output:
    FLEXPART_MCMC_source_day_identifiability_Apr2026.csv

Important:
    H is already defined as the ppmv contribution from the PRIOR
    emission of one source/day.

Therefore:

    CH4_model = background + H @ scaling_factors

and posterior source emission is:

    E_posterior = scaling_factor * E_prior

Do NOT multiply H by prior emissions again.

Current prior used by the Gibbs sampler:

    s_j ~ Normal(1, 1^2), truncated at s_j >= 0

Therefore completely uninformed parameters can have posterior means
around 1.29 because the truncated Normal has a mean above 1.
Those parameters should NOT be interpreted as observationally inferred.
"""

import os
import re
import numpy as np
import pandas as pd


# ============================================================================
# USER SETTINGS
# ============================================================================

BASE_DIR = "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart"

H_FILE = (
    f"{BASE_DIR}/H_matrix_flexpart/2026-04/"
    "H_flexpart_0_10m_13mine_Apr2026.csv"
)

POSTERIOR_FILE = (
    f"{BASE_DIR}/H_matrix_flexpart/2026-04/"
    "MCMC_13mine_daily_Apr2026/"
    "FLEXPART_MCMC_posterior_samples_Apr2026.npz"
)

PRIOR_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/"
    "emission_location/UpperHuntercoalmines.csv"
)

OUTPUT_DIR = (
    f"{BASE_DIR}/H_matrix_flexpart/2026-04/"
    "MCMC_13mine_daily_Apr2026"
)

OUTPUT_FILE = (
    f"{OUTPUT_DIR}/"
    "FLEXPART_MCMC_source_day_identifiability_Apr2026.csv"
)

START_DATE = "2026-04-01"
END_DATE = "2026-04-30"

N_MINES = 13
N_DAYS = 30

PRIOR_SCALING = 1.0

# Number of observations required for the qualitative classes.
# These are descriptive thresholds, not formal statistical tests.
WEAK_MAX = 4
MODERATE_MAX = 19

EPS = 1.0e-15


# ============================================================================
# PRINT HEADER
# ============================================================================

print("=" * 110)
print("FLEXPART-WRF MCMC SOURCE/DAY IDENTIFIABILITY ANALYSIS")
print("=" * 110)

print(f"Period      : {START_DATE} to {END_DATE}")
print(f"Mines       : {N_MINES}")
print(f"Days        : {N_DAYS}")
print(f"Parameters  : {N_MINES * N_DAYS}")
print()


# ============================================================================
# READ PRIOR EMISSIONS
# ============================================================================

print("Reading prior emissions:")
print(PRIOR_FILE)
print()

if not os.path.exists(PRIOR_FILE):
    raise FileNotFoundError(PRIOR_FILE)

prior_df = pd.read_csv(PRIOR_FILE)

required_prior_columns = ["Facility_code_name", "gCH4/s"]

missing = [
    c for c in required_prior_columns
    if c not in prior_df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns in prior file: {missing}"
    )

# --------------------------------------------------------------------------
# Extract OC1 ... OC13
# --------------------------------------------------------------------------

prior_emissions = {}

for mine_number in range(1, N_MINES + 1):

    mine = f"OC{mine_number}"

    matches = prior_df[
        prior_df["Facility_code_name"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.startswith(mine)
    ]

    if len(matches) == 0:
        raise ValueError(
            f"Could not find prior emission for {mine}"
        )

    if len(matches) > 1:
        print(
            f"WARNING: {mine} matched {len(matches)} rows. "
            f"Using first row."
        )

    prior_emissions[mine] = float(
        matches.iloc[0]["gCH4/s"]
    )


print("Prior emissions:")

for mine_number in range(1, N_MINES + 1):

    mine = f"OC{mine_number}"

    print(
        f"  {mine:<5s} "
        f"{prior_emissions[mine]:12.3f} g/s"
    )

total_prior_gps = sum(prior_emissions.values())

print()
print(
    f"Total prior = {total_prior_gps:.3f} g CH4/s"
)
print()


# ============================================================================
# READ H MATRIX
# ============================================================================

print("Reading H matrix:")
print(H_FILE)
print()

if not os.path.exists(H_FILE):
    raise FileNotFoundError(H_FILE)

H_df = pd.read_csv(H_FILE)

print(f"H rows       : {len(H_df)}")


# ============================================================================
# IDENTIFY H PARAMETER COLUMNS
# ============================================================================

metadata_columns = [
    "receptor",
    "release_utc",
    "obs_time_aest",
    "flexpart_file",
]

missing_metadata = [
    c for c in metadata_columns
    if c not in H_df.columns
]

if missing_metadata:
    raise ValueError(
        f"Missing H metadata columns: {missing_metadata}"
    )


# Expected form:
#
#     OC1_20260405
#     OC2_20260405
#     ...
#     OC13_20260405
#     OC1_20260406
#     ...
#
# Use regex so that only actual source/day columns are selected.

parameter_pattern = re.compile(
    r"^OC(1[0-3]|[1-9])_(\d{8})$"
)

parameter_names = []

for col in H_df.columns:

    if parameter_pattern.match(str(col)):
        parameter_names.append(col)


print(f"H parameters : {len(parameter_names)}")

expected_parameters = N_MINES * N_DAYS

if len(parameter_names) != expected_parameters:

    print()
    print("WARNING:")
    print(
        f"Expected {expected_parameters} source/day columns "
        f"but found {len(parameter_names)}."
    )

    print()
    print("First 20 detected parameter columns:")

    for p in parameter_names[:20]:
        print(f"  {p}")

    print()

    if len(parameter_names) != expected_parameters:
        raise ValueError(
            "H matrix does not contain the expected "
            "13 x 30 = 390 source/day parameters."
        )


# ============================================================================
# CHECK PARAMETER DATES
# ============================================================================

expected_dates = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)

expected_date_strings = {
    d.strftime("%Y%m%d")
    for d in expected_dates
}

found_dates = set()

for parameter in parameter_names:

    match = parameter_pattern.match(str(parameter))

    if match is not None:
        date_string = match.group(2)
        found_dates.add(date_string)


unexpected_dates = found_dates - expected_date_strings
missing_dates = expected_date_strings - found_dates

if unexpected_dates:
    print(
        "WARNING: H contains source/day dates outside requested period:"
    )
    for d in sorted(unexpected_dates):
        print(f"  {d}")
    print()

if missing_dates:
    print(
        "WARNING: H is missing requested dates:"
    )
    for d in sorted(missing_dates):
        print(f"  {d}")
    print()


# ============================================================================
# READ H NUMERICAL MATRIX
# ============================================================================

H = H_df[parameter_names].to_numpy(dtype=float)

print(
    f"H numerical shape: {H.shape}"
)

print(
    f"H minimum        : {np.nanmin(H):.8e}"
)

print(
    f"H maximum        : {np.nanmax(H):.8e}"
)

print(
    f"H nonzero values : {np.count_nonzero(np.abs(H) > EPS)}"
)

print()


# ============================================================================
# READ GIBBS POSTERIOR
# ============================================================================

print("Reading Gibbs posterior:")
print(POSTERIOR_FILE)
print()

if not os.path.exists(POSTERIOR_FILE):
    raise FileNotFoundError(POSTERIOR_FILE)

posterior_data = np.load(
    POSTERIOR_FILE,
    allow_pickle=True
)

print(
    "Arrays in NPZ:",
    posterior_data.files
)

if "samples" not in posterior_data.files:
    raise ValueError(
        "Posterior NPZ does not contain 'samples'."
    )

posterior_samples = posterior_data["samples"]

print(
    f"Raw posterior shape: {posterior_samples.shape}"
)


# ============================================================================
# HANDLE POSTERIOR SHAPE
# ============================================================================

# Expected current output:
#
#     (20000, 390)
#
# But also support:
#
#     (chains, samples, parameters)

if posterior_samples.ndim == 3:

    n_chains, n_samples_per_chain, n_parameters = (
        posterior_samples.shape
    )

    print(
        f"Posterior chains : {n_chains}"
    )

    print(
        f"Samples/chain    : {n_samples_per_chain}"
    )

    print(
        f"Parameters       : {n_parameters}"
    )

    posterior_samples = posterior_samples.reshape(
        -1,
        n_parameters
    )

elif posterior_samples.ndim == 2:

    n_samples, n_parameters = posterior_samples.shape

    print(
        f"Samples     : {n_samples}"
    )

    print(
        f"Parameters  : {n_parameters}"
    )

else:

    raise ValueError(
        "Unexpected posterior array dimensions: "
        f"{posterior_samples.ndim}"
    )


if posterior_samples.shape[1] != len(parameter_names):

    raise ValueError(
        "Posterior parameter count does not match H matrix.\n"
        f"Posterior: {posterior_samples.shape[1]}\n"
        f"H        : {len(parameter_names)}"
    )


print(
    f"Combined posterior samples: "
    f"{posterior_samples.shape[0]}"
)

print()


# ============================================================================
# POSTERIOR STATISTICS
# ============================================================================

print("Calculating posterior statistics...")
print()

posterior_mean = np.mean(
    posterior_samples,
    axis=0
)

posterior_median = np.median(
    posterior_samples,
    axis=0
)

posterior_sd = np.std(
    posterior_samples,
    axis=0,
    ddof=1
)

posterior_q025 = np.quantile(
    posterior_samples,
    0.025,
    axis=0
)

posterior_q975 = np.quantile(
    posterior_samples,
    0.975,
    axis=0
)

posterior_min = np.min(
    posterior_samples,
    axis=0
)

posterior_max = np.max(
    posterior_samples,
    axis=0
)


# ============================================================================
# H SENSITIVITY / IDENTIFIABILITY METRICS
# ============================================================================

print("Calculating H sensitivity metrics...")

H_abs = np.abs(H)

H_max_abs = np.max(
    H_abs,
    axis=0
)

H_sum_abs = np.sum(
    H_abs,
    axis=0
)

H_l2 = np.sqrt(
    np.sum(
        H ** 2,
        axis=0
    )
)

H_sum = np.sum(
    H,
    axis=0
)

n_nonzero_obs = np.count_nonzero(
    H_abs > EPS,
    axis=0
)

observation_informed = (
    n_nonzero_obs > 0
)


# ============================================================================
# PRIOR EMISSIONS FOR EACH SOURCE/DAY PARAMETER
# ============================================================================

print("Calculating prior emissions for each source/day...")

prior_emission_parameter_gps = np.zeros(
    len(parameter_names),
    dtype=float
)

mine_parameter = []
date_parameter = []

for i, parameter in enumerate(parameter_names):

    match = parameter_pattern.match(
        str(parameter)
    )

    if match is None:
        raise ValueError(
            f"Cannot parse parameter name: {parameter}"
        )

    mine_number = int(
        match.group(1)
    )

    date_string = match.group(2)

    mine = f"OC{mine_number}"

    mine_parameter.append(mine)
    date_parameter.append(date_string)

    if mine not in prior_emissions:
        raise ValueError(
            f"No prior emission found for {mine}"
        )

    prior_emission_parameter_gps[i] = (
        prior_emissions[mine]
    )


# ============================================================================
# POSTERIOR EMISSIONS
# ============================================================================

# E_posterior = scaling factor * E_prior

posterior_mean_emission_gps = (
    posterior_mean
    * prior_emission_parameter_gps
)

posterior_median_emission_gps = (
    posterior_median
    * prior_emission_parameter_gps
)

posterior_sd_emission_gps = (
    posterior_sd
    * prior_emission_parameter_gps
)

posterior_q025_emission_gps = (
    posterior_q025
    * prior_emission_parameter_gps
)

posterior_q975_emission_gps = (
    posterior_q975
    * prior_emission_parameter_gps
)

posterior_min_emission_gps = (
    posterior_min
    * prior_emission_parameter_gps
)

posterior_max_emission_gps = (
    posterior_max
    * prior_emission_parameter_gps
)


# ============================================================================
# POSTERIOR / PRIOR EMISSION RATIOS
# ============================================================================

# OC5 has only 0.01 g/s prior emission, but it is still a valid
# parameter. We therefore protect against division by zero.

posterior_mean_prior_ratio = np.full(
    len(parameter_names),
    np.nan
)

posterior_median_prior_ratio = np.full(
    len(parameter_names),
    np.nan
)

posterior_q025_prior_ratio = np.full(
    len(parameter_names),
    np.nan
)

posterior_q975_prior_ratio = np.full(
    len(parameter_names),
    np.nan
)

valid_prior = (
    np.abs(prior_emission_parameter_gps) > EPS
)

posterior_mean_prior_ratio[valid_prior] = (
    posterior_mean[valid_prior]
    / PRIOR_SCALING
)

posterior_median_prior_ratio[valid_prior] = (
    posterior_median[valid_prior]
    / PRIOR_SCALING
)

posterior_q025_prior_ratio[valid_prior] = (
    posterior_q025[valid_prior]
    / PRIOR_SCALING
)

posterior_q975_prior_ratio[valid_prior] = (
    posterior_q975[valid_prior]
    / PRIOR_SCALING
)


# ============================================================================
# IDENTIFIABILITY CLASS
# ============================================================================

def classify_identifiability(n_obs):

    if n_obs == 0:
        return "UNINFORMED"

    elif n_obs <= WEAK_MAX:
        return "WEAK"

    elif n_obs <= MODERATE_MAX:
        return "MODERATE"

    else:
        return "INFORMED"


identifiability_class = [
    classify_identifiability(int(n))
    for n in n_nonzero_obs
]


# ============================================================================
# INFORMATION CONTENT METRICS
# ============================================================================

# Posterior/prior SD ratio.
#
# Current prior SD = 1.0 before truncation.
#
# Values substantially below 1 indicate posterior uncertainty has
# contracted relative to the nominal prior.
#
# For UNINFORMED parameters this metric should be interpreted carefully
# because the actual prior is truncated.

prior_sd = 1.0

posterior_prior_sd_ratio = (
    posterior_sd / prior_sd
)


# A simple variance-reduction metric:
#
#     1 - posterior variance / prior variance
#
# Positive values indicate posterior variance reduction.
# Negative values indicate posterior variance larger than nominal
# untruncated prior variance.

posterior_variance_reduction = (
    1.0
    - (posterior_sd ** 2)
    / (prior_sd ** 2)
)


# ============================================================================
# BUILD OUTPUT DATAFRAME
# ============================================================================

result_df = pd.DataFrame({

    "parameter": parameter_names,

    "mine": mine_parameter,

    "date": date_parameter,

    # --------------------------------------------------------------
    # Identifiability
    # --------------------------------------------------------------

    "H_max_abs_ppmv_per_factor":
        H_max_abs,

    "H_sum_abs_ppmv_per_factor":
        H_sum_abs,

    "H_L2_ppmv_per_factor":
        H_l2,

    "H_sum_ppmv_per_factor":
        H_sum,

    "n_observations_nonzero_H":
        n_nonzero_obs,

    "observation_informed":
        observation_informed,

    "identifiability_class":
        identifiability_class,

    # --------------------------------------------------------------
    # Prior
    # --------------------------------------------------------------

    "prior_scaling_factor":
        np.full(
            len(parameter_names),
            PRIOR_SCALING
        ),

    "prior_emission_gps":
        prior_emission_parameter_gps,

    # --------------------------------------------------------------
    # Posterior scaling factor
    # --------------------------------------------------------------

    "posterior_mean":
        posterior_mean,

    "posterior_median":
        posterior_median,

    "posterior_sd":
        posterior_sd,

    "posterior_q025":
        posterior_q025,

    "posterior_q975":
        posterior_q975,

    "posterior_min":
        posterior_min,

    "posterior_max":
        posterior_max,

    # --------------------------------------------------------------
    # Posterior uncertainty relative to prior
    # --------------------------------------------------------------

    "posterior_prior_sd_ratio":
        posterior_prior_sd_ratio,

    "posterior_variance_reduction":
        posterior_variance_reduction,

    # --------------------------------------------------------------
    # Posterior emissions
    # --------------------------------------------------------------

    "posterior_mean_emission_gps":
        posterior_mean_emission_gps,

    "posterior_median_emission_gps":
        posterior_median_emission_gps,

    "posterior_sd_emission_gps":
        posterior_sd_emission_gps,

    "posterior_q025_emission_gps":
        posterior_q025_emission_gps,

    "posterior_q975_emission_gps":
        posterior_q975_emission_gps,

    "posterior_min_emission_gps":
        posterior_min_emission_gps,

    "posterior_max_emission_gps":
        posterior_max_emission_gps,

    # --------------------------------------------------------------
    # Posterior/prior ratios
    # --------------------------------------------------------------

    "posterior_mean_prior_ratio":
        posterior_mean_prior_ratio,

    "posterior_median_prior_ratio":
        posterior_median_prior_ratio,

    "posterior_q025_prior_ratio":
        posterior_q025_prior_ratio,

    "posterior_q975_prior_ratio":
        posterior_q975_prior_ratio,
})


# ============================================================================
# SORT OUTPUT BY DATE THEN MINE
# ============================================================================

result_df["_date_sort"] = pd.to_datetime(
    result_df["date"],
    format="%Y%m%d"
)

result_df["_mine_sort"] = (
    result_df["mine"]
    .str.extract(r"(\d+)")[0]
    .astype(int)
)

result_df = (
    result_df
    .sort_values(
        ["_date_sort", "_mine_sort"]
    )
    .drop(
        columns=[
            "_date_sort",
            "_mine_sort"
        ]
    )
    .reset_index(drop=True)
)


# ============================================================================
# CREATE OUTPUT DIRECTORY
# ============================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================================
# WRITE CSV
# ============================================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    float_format="%.8g"
)

print()
print("=" * 110)
print("OUTPUT")
print("=" * 110)

print(
    f"Saved: {OUTPUT_FILE}"
)

print(
    f"Rows : {len(result_df)}"
)

print(
    f"Cols : {len(result_df.columns)}"
)

print()


# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 110)
print("IDENTIFIABILITY SUMMARY")
print("=" * 110)

class_counts = (
    result_df["identifiability_class"]
    .value_counts()
    .reindex(
        [
            "UNINFORMED",
            "WEAK",
            "MODERATE",
            "INFORMED"
        ],
        fill_value=0
    )
)

print()

for cls, count in class_counts.items():

    print(
        f"{cls:<12s}: "
        f"{count:4d} / {len(result_df)}"
    )

print()


# ============================================================================
# NUMBER OF PARAMETERS WITH ZERO H
# ============================================================================

zero_H = (
    result_df["n_observations_nonzero_H"] == 0
)

print(
    f"Parameters with zero observational sensitivity: "
    f"{zero_H.sum()} / {len(result_df)}"
)

print()


# ============================================================================
# SOURCE/DAY SUMMARY
# ============================================================================

print("=" * 110)
print("SOURCE/DAY SUMMARY")
print("=" * 110)

summary_columns = [
    "parameter",
    "mine",
    "date",
    "posterior_mean",
    "posterior_median",
    "posterior_q025",
    "posterior_q975",
    "H_max_abs_ppmv_per_factor",
    "n_observations_nonzero_H",
    "prior_emission_gps",
    "posterior_mean_emission_gps",
    "posterior_mean_prior_ratio",
    "observation_informed",
    "identifiability_class",
]

print()

print(
    result_df[summary_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

print()


# ============================================================================
# MINE-LEVEL SUMMARY
# ============================================================================

print("=" * 110)
print("MINE-LEVEL SUMMARY")
print("=" * 110)

mine_summary_rows = []

for mine_number in range(1, N_MINES + 1):

    mine = f"OC{mine_number}"

    sub = result_df[
        result_df["mine"] == mine
    ]

    prior_gps = prior_emissions[mine]

    # Daily posterior factors are averaged across days here.
    mean_factor = sub["posterior_mean"].mean()
    median_factor = sub["posterior_median"].mean()

    mean_emission_gps = (
        sub["posterior_mean_emission_gps"].mean()
    )

    median_emission_gps = (
        sub["posterior_median_emission_gps"].mean()
    )

    informative_days = int(
        sub["observation_informed"].sum()
    )

    uninformed_days = int(
        (~sub["observation_informed"]).sum()
    )

    total_nonzero_obs = int(
        sub["n_observations_nonzero_H"].sum()
    )

    mine_summary_rows.append({

        "mine":
            mine,

        "prior_emission_gps":
            prior_gps,

        "mean_daily_scaling":
            mean_factor,

        "mean_of_daily_median_scaling":
            median_factor,

        "mean_daily_posterior_emission_gps":
            mean_emission_gps,

        "mean_daily_median_emission_gps":
            median_emission_gps,

        "mean_emission_prior_ratio":
            mean_emission_gps / prior_gps
            if prior_gps > EPS
            else np.nan,

        "informative_days":
            informative_days,

        "uninformed_days":
            uninformed_days,

        "total_nonzero_H_observations":
            total_nonzero_obs,
    })


mine_summary_df = pd.DataFrame(
    mine_summary_rows
)

print()

print(
    mine_summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

print()


# ============================================================================
# SAVE MINE SUMMARY
# ============================================================================

mine_summary_file = (
    f"{OUTPUT_DIR}/"
    "FLEXPART_MCMC_mine_identifiability_summary_Apr2026.csv"
)

mine_summary_df.to_csv(
    mine_summary_file,
    index=False,
    float_format="%.8g"
)

print(
    f"Mine summary saved: {mine_summary_file}"
)

print()


# ============================================================================
# MOST INFORMATIVE PARAMETERS
# ============================================================================

print("=" * 110)
print("TOP 20 SOURCE/DAY PARAMETERS BY H MAXIMUM SENSITIVITY")
print("=" * 110)

top_columns = [
    "parameter",
    "mine",
    "date",
    "H_max_abs_ppmv_per_factor",
    "n_observations_nonzero_H",
    "posterior_mean",
    "posterior_median",
    "posterior_q025",
    "posterior_q975",
    "posterior_mean_prior_ratio",
    "identifiability_class",
]

top20 = (
    result_df
    .sort_values(
        "H_max_abs_ppmv_per_factor",
        ascending=False
    )
    .head(20)
)

print()

print(
    top20[top_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

print()


# ============================================================================
# ZERO-SENSITIVITY PARAMETERS
# ============================================================================

print("=" * 110)
print("ZERO-SENSITIVITY SOURCE/DAY PARAMETERS")
print("=" * 110)

zero_df = result_df[
    result_df["n_observations_nonzero_H"] == 0
]

if len(zero_df) == 0:

    print()
    print("No completely uninformed source/day parameters.")
    print()

else:

    print()
    print(
        zero_df[
            [
                "parameter",
                "mine",
                "date",
                "posterior_mean",
                "posterior_median",
                "posterior_q025",
                "posterior_q975",
                "prior_emission_gps",
                "posterior_mean_prior_ratio",
                "identifiability_class",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}"
        )
    )

    print()


# ============================================================================
# FINAL CHECKS
# ============================================================================

print("=" * 110)
print("FINAL CHECKS")
print("=" * 110)

print()

print(
    "Expected parameters : "
    f"{expected_parameters}"
)

print(
    "H parameters        : "
    f"{len(parameter_names)}"
)

print(
    "Posterior parameters: "
    f"{posterior_samples.shape[1]}"
)

print(
    "Output rows         : "
    f"{len(result_df)}"
)

print()

if (
    len(parameter_names) == expected_parameters
    and posterior_samples.shape[1] == expected_parameters
    and len(result_df) == expected_parameters
):

    print("✓ Parameter dimensions are consistent.")

else:

    print("✗ Parameter dimension check failed.")

print()

print(
    "Note:"
)

print(
    "  UNINFORMED parameters have no nonzero H sensitivity "
    "to the matched observations."
)

print(
    "  Their posterior is therefore determined by the prior "
    "rather than by the observations."
)

print(
    "  With the current truncated Normal prior N(1,1) for s >= 0, "
    "such parameters can have a posterior mean around 1.29."
)

print(
    "  Do not interpret those values as observationally inferred "
    "emission increases."
)

print()

print("=" * 110)
print("ANALYSIS COMPLETE")
print("=" * 110)

