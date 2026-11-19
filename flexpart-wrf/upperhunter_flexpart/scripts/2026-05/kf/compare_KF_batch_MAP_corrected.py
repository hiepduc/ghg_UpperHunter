#!/usr/bin/env python3

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE = "/home/duch/flexpart-wrf/upperhunter_flexpart"

HFILE = (
    BASE
    + "/H_matrix_flexpart/"
      "H_flexpart_0_10m_13mine_Jan2026.csv"
)

OBSFILE = (
    "/home/duch/flexpart-wrf/emission_UH/"
    "RH_upperHunter/tapm_output/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

KF_DIR = (
    BASE
    + "/H_matrix_flexpart/"
      "KF_13mine_daily_Jan2026"
)

BACKGROUND_PPM = 1.85

PRIOR_MEAN = 1.0
PRIOR_SIGMA = 0.5
PRIOR_VARIANCE = PRIOR_SIGMA ** 2

OBS_SIGMA = 0.05
OBS_VARIANCE = OBS_SIGMA ** 2

CORRELATION_LENGTH_DAYS = 3.0

MINES = [f"OC{i}" for i in range(1, 14)]

START_DATE = pd.Timestamp("2026-01-05")
END_DATE = pd.Timestamp("2026-01-31")


# =============================================================================
# LOAD H MATRIX
# =============================================================================

print("=" * 80)
print("LOADING H MATRIX")
print("=" * 80)

Hdf = pd.read_csv(HFILE)

Hdf["obs_time_aest"] = pd.to_datetime(
    Hdf["obs_time_aest"]
)

Hdf["receptor"] = (
    Hdf["receptor"]
    .astype(str)
    .str.lower()
)

METADATA_COLUMNS = [
    "receptor",
    "release_utc",
    "obs_time_aest",
    "flexpart_file"
]

parameter_names = [
    c for c in Hdf.columns
    if c not in METADATA_COLUMNS
]

print("H shape:", Hdf.shape)
print("Number of parameters:", len(parameter_names))


# =============================================================================
# LOAD PICARRO OBSERVATIONS
# =============================================================================

print()
print("=" * 80)
print("LOADING PICARRO OBSERVATIONS")
print("=" * 80)

obs = pd.read_csv(OBSFILE)

obs["Datetime"] = pd.to_datetime(
    obs["Datetime"],
    dayfirst=True,
    errors="coerce"
)

obs_long = obs.melt(
    id_vars=["Datetime"],
    value_vars=[
        "Singleton cleaned",
        "Muswellbrook cleaned",
        "Merriwa cleaned"
    ],
    var_name="receptor_raw",
    value_name="observation"
)

receptor_map = {
    "Singleton cleaned": "singleton",
    "Muswellbrook cleaned": "muswellbrook",
    "Merriwa cleaned": "merriwa"
}

obs_long["receptor"] = (
    obs_long["receptor_raw"]
    .map(receptor_map)
)

obs_long["obs_time_aest"] = (
    obs_long["Datetime"]
    .dt.floor("h")
)

obs_long = obs_long.dropna(
    subset=["observation"]
)

obs_long = obs_long[
    (obs_long["obs_time_aest"] >= START_DATE)
    &
    (
        obs_long["obs_time_aest"]
        < END_DATE + pd.Timedelta(days=1)
    )
]

# Same duplicate handling as KF
obs_long = (
    obs_long
    .groupby(
        ["receptor", "obs_time_aest"],
        as_index=False
    )["observation"]
    .mean()
)

print("Observations after preprocessing:", len(obs_long))


# =============================================================================
# MATCH H AND OBSERVATIONS
# =============================================================================

print()
print("=" * 80)
print("MATCHING H AND OBSERVATIONS")
print("=" * 80)

merged = pd.merge(
    Hdf,
    obs_long,
    on=[
        "receptor",
        "obs_time_aest"
    ],
    how="inner"
)

merged = (
    merged
    .sort_values(
        ["obs_time_aest", "receptor"]
    )
    .reset_index(drop=True)
)

print("Matched observations:", len(merged))


# =============================================================================
# DATE / PARAMETER STRUCTURE
# =============================================================================

dates = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)

N_DAY = len(dates)
N_MINE = len(MINES)
N_PARAM = N_DAY * N_MINE

print()
print("=" * 80)
print("PARAMETER STRUCTURE")
print("=" * 80)

print("Number of days   :", N_DAY)
print("Number of mines  :", N_MINE)
print("Number parameters:", N_PARAM)

print("First parameters:")
print([
    f"{mine}_{dates[0].strftime('%Y%m%d')}"
    for mine in MINES
])

print("Last parameters:")
print([
    f"{mine}_{dates[-1].strftime('%Y%m%d')}"
    for mine in MINES
])


# =============================================================================
# BUILD FULL 351-COLUMN H MATRIX
#
# State vector ordering:
#
# OC1_20260105 ... OC13_20260105
# OC1_20260106 ... OC13_20260106
# ...
# OC1_20260131 ... OC13_20260131
#
# Each observation only sees the state corresponding to its observation day.
# =============================================================================

print()
print("=" * 80)
print("BUILDING FULL BATCH H MATRIX")
print("=" * 80)

Hfull = np.zeros(
    (len(merged), N_PARAM),
    dtype=np.float64
)

d = np.zeros(
    len(merged),
    dtype=np.float64
)

for row_index, (_, row) in enumerate(
    merged.iterrows()
):

    day = pd.Timestamp(
        row["obs_time_aest"]
    ).normalize()

    day_index = (
        day - START_DATE
    ).days

    for mine_index, mine in enumerate(MINES):

        column_name = (
            f"{mine}_{day.strftime('%Y%m%d')}"
        )

        Hfull[
            row_index,
            day_index * N_MINE + mine_index
        ] = float(row[column_name])

    # Observation anomaly relative to background
    d[row_index] = (
        float(row["observation"])
        - BACKGROUND_PPM
    )

print("Hfull shape:", Hfull.shape)
print("d shape    :", d.shape)

print(
    "Hfull non-zero elements:",
    np.count_nonzero(Hfull)
)


# =============================================================================
# BUILD B
#
# B[d,j] = sigma^2 exp(-|d-j| / L)
#
# Cross-mine covariance = 0.
# =============================================================================

print()
print("=" * 80)
print("BUILDING PRIOR COVARIANCE B")
print("=" * 80)

B = np.zeros(
    (N_PARAM, N_PARAM),
    dtype=np.float64
)

for day_i in range(N_DAY):

    for day_j in range(N_DAY):

        temporal_cov = (
            PRIOR_VARIANCE
            * np.exp(
                -abs(day_i - day_j)
                / CORRELATION_LENGTH_DAYS
            )
        )

        for mine_index in range(N_MINE):

            i = (
                day_i * N_MINE
                + mine_index
            )

            j = (
                day_j * N_MINE
                + mine_index
            )

            B[i, j] = temporal_cov

print("B shape:", B.shape)

print(
    "B diagonal range:",
    B.diagonal().min(),
    B.diagonal().max()
)

print(
    "B[OC1 Jan5, OC1 Jan6] =",
    B[0, 13]
)

print(
    "Expected:",
    PRIOR_VARIANCE
    * np.exp(-1.0 / CORRELATION_LENGTH_DAYS)
)


# =============================================================================
# PRIOR MEAN VECTOR
#
# IMPORTANT:
# This is the term that was accidentally omitted in the previous diagnostic.
#
# The prior is:
#
#       x ~ N(s_b, B)
#
# with:
#
#       s_b = 1.0
#
# Therefore the normal-equation RHS contains:
#
#       B^-1 s_b
#
# =============================================================================

s_b = (
    np.ones(N_PARAM, dtype=np.float64)
    * PRIOR_MEAN
)


# =============================================================================
# BATCH MAP SOLUTION
#
# Posterior precision:
#
#       A = B^-1 + H' R^-1 H
#
# RHS:
#
#       b = B^-1 s_b + H' R^-1 d
#
# MAP:
#
#       x_MAP = A^-1 b
# =============================================================================

print()
print("=" * 80)
print("SOLVING EXACT BATCH MAP")
print("=" * 80)

Rinv = 1.0 / OBS_VARIANCE

print("R variance:", OBS_VARIANCE)
print("R inverse :", Rinv)

# B inverse
Binv = np.linalg.inv(B)

# Posterior precision
A = (
    Binv
    + Rinv * (Hfull.T @ Hfull)
)

# IMPORTANT CORRECT PRIOR TERM
b = (
    Binv @ s_b
    + Rinv * (Hfull.T @ d)
)

print("A shape:", A.shape)
print("b shape:", b.shape)

s_map = np.linalg.solve(
    A,
    b
)

print("MAP solution obtained.")


# =============================================================================
# LOAD STORED KF DAILY STATES
# =============================================================================

print()
print("=" * 80)
print("LOADING STORED KF STATES")
print("=" * 80)

z = np.load(
    KF_DIR + "/KF_daily_states.npz",
    allow_pickle=True
)

kf_states = z["states"]

kf_dates = pd.to_datetime(
    z["dates"]
)

print("KF states shape:", kf_states.shape)
print(
    "KF dates:",
    kf_dates[0],
    "to",
    kf_dates[-1]
)

# Flatten in day-major order
s_kf = kf_states.reshape(-1)


# =============================================================================
# DATE CHECK
# =============================================================================

print()
print("=" * 80)
print("DATE CHECK")
print("=" * 80)

print(
    "Dates identical:",
    np.array_equal(
        dates.values,
        kf_dates.values
    )
)


# =============================================================================
# KF VS MAP
# =============================================================================

print()
print("=" * 80)
print("KF VS BATCH MAP")
print("=" * 80)

difference = (
    s_kf - s_map
)

print(
    "KF mean :",
    f"{s_kf.mean():.12f}"
)

print(
    "MAP mean:",
    f"{s_map.mean():.12f}"
)

print(
    "Maximum absolute difference:",
    f"{np.max(np.abs(difference)):.12e}"
)

print(
    "RMSE difference:",
    f"{np.sqrt(np.mean(difference**2)):.12e}"
)

print(
    "Mean absolute difference:",
    f"{np.mean(np.abs(difference)):.12e}"
)

print(
    "Correlation:",
    f"{np.corrcoef(s_kf, s_map)[0,1]:.12f}"
)


# =============================================================================
# MAP GRADIENT CHECK
#
# For an exact MAP solution:
#
#       A x - b = 0
# =============================================================================

gradient = (
    A @ s_map - b
)

print()
print("=" * 80)
print("MAP GRADIENT CHECK")
print("=" * 80)

print(
    "Gradient L2:",
    f"{np.linalg.norm(gradient):.12e}"
)

print(
    "Gradient max abs:",
    f"{np.max(np.abs(gradient)):.12e}"
)


# =============================================================================
# DAILY COMPARISON
# =============================================================================

print()
print("=" * 80)
print("DAILY KF VS MAP")
print("=" * 80)

print(
    f"{'Date':<12}"
    f"{'KF mean':>14}"
    f"{'MAP mean':>14}"
    f"{'Mean diff':>14}"
    f"{'Max diff':>14}"
)

print("-" * 68)

for day_index, date in enumerate(dates):

    start = day_index * N_MINE
    end = start + N_MINE

    k = s_kf[start:end]
    m = s_map[start:end]

    print(
        f"{date.date()} "
        f"{k.mean():14.8f}"
        f"{m.mean():14.8f}"
        f"{(k-m).mean():+14.8f}"
        f"{np.max(np.abs(k-m)):14.8f}"
    )


# =============================================================================
# MINE-BY-MINE COMPARISON
# =============================================================================

print()
print("=" * 80)
print("MINE-BY-MINE KF VS MAP")
print("=" * 80)

print(
    f"{'Mine':<8}"
    f"{'KF mean':>14}"
    f"{'MAP mean':>14}"
    f"{'Mean diff':>14}"
    f"{'Max diff':>14}"
)

print("-" * 64)

for mine_index, mine in enumerate(MINES):

    k = kf_states[:, mine_index]
    m = s_map[mine_index::N_MINE]

    print(
        f"{mine:<8}"
        f"{k.mean():14.8f}"
        f"{m.mean():14.8f}"
        f"{(k-m).mean():+14.8f}"
        f"{np.max(np.abs(k-m)):14.8f}"
    )


# =============================================================================
# FIRST 10 PARAMETER VALUES
# =============================================================================

print()
print("=" * 80)
print("FIRST 10 PARAMETERS")
print("=" * 80)

for i in range(min(10, N_PARAM)):

    day_index = i // N_MINE
    mine_index = i % N_MINE

    name = (
        f"{MINES[mine_index]}_"
        f"{dates[day_index].strftime('%Y%m%d')}"
    )

    print(
        f"{name:<20}"
        f"KF={s_kf[i]:.10f} "
        f"MAP={s_map[i]:.10f} "
        f"diff={difference[i]:+.10e}"
    )


# =============================================================================
# FINAL INTERPRETATION
# =============================================================================

print()
print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

max_diff = np.max(np.abs(difference))

if max_diff < 1e-10:

    print(
        "RESULT: KF and batch MAP agree to numerical precision."
    )

    print(
        "The sequential KF is mathematically equivalent "
        "to the batch MAP for this model."
    )

else:

    print(
        "RESULT: KF and batch MAP do NOT agree."
    )

    print(
        "The discrepancy is larger than numerical precision "
        "and requires further investigation."
    )

print("=" * 80)

