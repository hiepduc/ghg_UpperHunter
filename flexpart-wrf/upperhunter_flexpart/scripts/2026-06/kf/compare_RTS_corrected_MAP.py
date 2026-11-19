#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE = "/home/duch/flexpart-wrf/upperhunter_flexpart"

H_FILE = os.path.join(
    BASE,
    "H_matrix_flexpart",
    "H_flexpart_0_10m_13mine_Jan2026.csv"
)

PICARRO_FILE = (
    "/home/duch/flexpart-wrf/emission_UH/"
    "RH_upperHunter/tapm_output/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

RTS_DIR = os.path.join(
    BASE,
    "H_matrix_flexpart",
    "KF_13mine_daily_Jan2026"
)

RTS_CSV = os.path.join(
    RTS_DIR,
    "RTS_daily_scaling_factors.csv"
)

RTS_NPZ = os.path.join(
    RTS_DIR,
    "RTS_daily_states.npz"
)

BACKGROUND_PPM = 1.85
PRIOR_MEAN = 1.0
PRIOR_SIGMA = 0.5
PRIOR_VARIANCE = PRIOR_SIGMA ** 2

OBS_SIGMA_PPM = 0.05
OBS_VARIANCE = OBS_SIGMA_PPM ** 2

CORRELATION_LENGTH_DAYS = 3.0

START_DATE = pd.Timestamp("2026-01-05")
END_DATE = pd.Timestamp("2026-01-31")

MINES = [f"OC{i}" for i in range(1, 14)]
N_MINE = len(MINES)
N_DAY = 27
N_PARAM = N_MINE * N_DAY


# ============================================================
# HELPER
# ============================================================

def parameter_names():
    names = []

    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        datestr = day.strftime("%Y%m%d")

        for mine in MINES:
            names.append(f"{mine}_{datestr}")

    return names


# ============================================================
# READ H MATRIX
# ============================================================

print("=" * 90)
print("RTS vs CORRECTED BATCH MAP VALIDATION")
print("=" * 90)

print("\nReading H matrix:")
print(H_FILE)

H_df = pd.read_csv(H_FILE)

metadata_columns = [
    "receptor",
    "release_utc",
    "obs_time_aest",
    "flexpart_file",
]

PARAMETER_NAMES = parameter_names()

missing = [
    c for c in PARAMETER_NAMES
    if c not in H_df.columns
]

if missing:
    raise RuntimeError(
        "Missing H parameter columns:\n"
        + "\n".join(missing)
    )

print(f"H shape: {H_df.shape}")
print(f"Number of parameters: {len(PARAMETER_NAMES)}")


# ============================================================
# PREPARE H METADATA
# ============================================================

H_df["obs_time_aest"] = pd.to_datetime(
    H_df["obs_time_aest"]
)

H_df["obs_time_aest"] = (
    H_df["obs_time_aest"]
    .dt.floor("h")
)

H_df["receptor"] = (
    H_df["receptor"]
    .astype(str)
    .str.lower()
    .str.strip()
)


# ============================================================
# READ PICARRO
# ============================================================

print("\nReading Picarro observations:")
print(PICARRO_FILE)

obs = pd.read_csv(PICARRO_FILE)

obs["Datetime"] = pd.to_datetime(
    obs["Datetime"],
    dayfirst=True,
    errors="coerce"
)

obs = obs.rename(
    columns={
        "Singleton cleaned": "singleton",
        "Muswellbrook cleaned": "muswellbrook",
        "Merriwa cleaned": "merriwa",
    }
)

obs_long = obs.melt(
    id_vars=["Datetime"],
    value_vars=[
        "singleton",
        "muswellbrook",
        "merriwa",
    ],
    var_name="receptor",
    value_name="observation",
)

obs_long["Datetime"] = (
    obs_long["Datetime"]
    .dt.floor("h")
)

obs_long["receptor"] = (
    obs_long["receptor"]
    .astype(str)
    .str.lower()
    .str.strip()
)

obs_long["observation"] = pd.to_numeric(
    obs_long["observation"],
    errors="coerce"
)

obs_long = obs_long.dropna(
    subset=["observation"]
)

obs_long = obs_long[
    (obs_long["Datetime"] >= START_DATE)
    &
    (obs_long["Datetime"] < END_DATE + pd.Timedelta(days=1))
]

# Average duplicate receptor/hour observations
obs_long = (
    obs_long
    .groupby(
        ["receptor", "Datetime"],
        as_index=False
    )["observation"]
    .mean()
)

print(f"Observations after preprocessing: {len(obs_long)}")


# ============================================================
# MATCH H TO OBSERVATIONS
# ============================================================

print("\nMatching H rows to observations...")

H_df["_row_id"] = np.arange(len(H_df))

matched = H_df.merge(
    obs_long,
    left_on=["receptor", "obs_time_aest"],
    right_on=["receptor", "Datetime"],
    how="inner",
)

matched = matched.sort_values(
    ["Datetime", "receptor"]
).reset_index(drop=True)

print(f"Matched observations: {len(matched)}")

if len(matched) == 0:
    raise RuntimeError("No H/observation matches found.")


Hfull = matched[PARAMETER_NAMES].to_numpy(
    dtype=np.float64
)

y = matched["observation"].to_numpy(
    dtype=np.float64
)

d = y - BACKGROUND_PPM

print(f"Hfull shape: {Hfull.shape}")


# ============================================================
# BUILD B
#
# B[(mine,day),(same mine,other day)]
#     = sigma^2 exp(-|d1-d2| / L)
#
# Cross-mine covariance = 0.
# ============================================================

print("\nBuilding prior covariance B...")

rho = np.exp(
    -1.0 / CORRELATION_LENGTH_DAYS
)

B = np.zeros(
    (N_PARAM, N_PARAM),
    dtype=np.float64
)

for day_i in range(N_DAY):

    for mine_i in range(N_MINE):

        i = day_i * N_MINE + mine_i

        for day_j in range(N_DAY):

            j = day_j * N_MINE + mine_i

            B[i, j] = (
                PRIOR_VARIANCE
                * np.exp(
                    -abs(day_i - day_j)
                    / CORRELATION_LENGTH_DAYS
                )
            )

print("B shape:", B.shape)
print("rho =", rho)

print(
    "B[OC1 Jan05, OC1 Jan06] =",
    B[0, N_MINE]
)


# ============================================================
# EXACT BATCH MAP
#
# J(x) =
# 0.5 (x-x_b)^T B^-1 (x-x_b)
# + 0.5 (d-Hx)^T R^-1 (d-Hx)
#
# MAP:
#
# (B^-1 + H^T R^-1 H)x
# =
# B^-1 x_b + H^T R^-1 d
# ============================================================

print("\nComputing exact batch MAP...")

# Prior mean
x_b = np.full(
    N_PARAM,
    PRIOR_MEAN,
    dtype=np.float64
)

# B inverse
Binv = np.linalg.inv(B)

Rinv = 1.0 / OBS_VARIANCE

print("Constructing normal equations...")

A = (
    Binv
    + Rinv * (Hfull.T @ Hfull)
)

b = (
    Binv @ x_b
    + Rinv * (Hfull.T @ d)
)

print("Solving MAP system...")

x_map = np.linalg.solve(
    A,
    b
)

x_map = np.asarray(
    x_map,
    dtype=np.float64
)

# Posterior covariance
print("Computing posterior covariance...")

P_map = np.linalg.inv(A)

P_map = 0.5 * (
    P_map + P_map.T
)


# ============================================================
# VERIFY MAP GRADIENT
# ============================================================

gradient = (
    Binv @ (x_map - x_b)
    + Rinv * (
        Hfull.T @ (
            Hfull @ x_map - d
        )
    )
)

print("\nMAP gradient check:")
print("Gradient L2 norm :", np.linalg.norm(gradient))
print("Gradient max abs :", np.max(np.abs(gradient)))


# ============================================================
# RESHAPE MAP TO DAILY × MINE
# ============================================================

map_daily = x_map.reshape(
    N_DAY,
    N_MINE
)

# ============================================================
# READ RTS DAILY SCALING FACTORS
# ============================================================

print("\n" + "=" * 90)
print("READING RTS RESULT")
print("=" * 90)

print("\nRTS CSV:")
print(RTS_CSV)

if not os.path.exists(RTS_CSV):
    raise FileNotFoundError(
        f"RTS CSV not found:\n{RTS_CSV}"
    )

rts_df = pd.read_csv(RTS_CSV)

print("\nRTS columns:")
print(list(rts_df.columns))

print("\nRTS shape:")
print(rts_df.shape)


# ------------------------------------------------------------
# Dates
# ------------------------------------------------------------

rts_df["date"] = pd.to_datetime(
    rts_df["date"]
)

rts_df = (
    rts_df
    .sort_values("date")
    .reset_index(drop=True)
)

expected_dates = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)

if not np.array_equal(
    rts_df["date"].dt.normalize().values,
    expected_dates.values
):
    raise RuntimeError(
        "RTS dates do not match expected "
        "2026-01-05 to 2026-01-31."
    )


# ------------------------------------------------------------
# Verify all mines exist
# ------------------------------------------------------------

missing_mines = [
    mine
    for mine in MINES
    if mine not in rts_df.columns
]

if missing_mines:
    raise RuntimeError(
        "Missing RTS mine columns:\n"
        + "\n".join(missing_mines)
    )


# ------------------------------------------------------------
# Extract RTS states
# ------------------------------------------------------------

rts_states = rts_df[
    MINES
].to_numpy(
    dtype=np.float64
)

print(
    "\nRTS states shape:",
    rts_states.shape
)

if rts_states.shape != (
    N_DAY,
    N_MINE
):
    raise RuntimeError(
        f"Unexpected RTS shape: "
        f"{rts_states.shape}"
    )


# ============================================================
# DIRECT RTS vs MAP COMPARISON
# ============================================================

difference = (
    rts_states
    - map_daily
)

abs_difference = np.abs(
    difference
)

max_abs = np.max(
    abs_difference
)

rmse = np.sqrt(
    np.mean(
        difference ** 2
    )
)

mae = np.mean(
    abs_difference
)

correlation = np.corrcoef(
    rts_states.ravel(),
    map_daily.ravel()
)[0, 1]


print("\n" + "=" * 90)
print("RTS vs BATCH MAP")
print("=" * 90)

print(
    f"Maximum absolute difference : "
    f"{max_abs:.16e}"
)

print(
    f"RMSE                         : "
    f"{rmse:.16e}"
)

print(
    f"Mean absolute difference     : "
    f"{mae:.16e}"
)

print(
    f"Correlation                  : "
    f"{correlation:.16f}"
)


# ============================================================
# DAY-BY-DAY COMPARISON
# ============================================================

print("\n" + "=" * 90)
print("DAY-BY-DAY MEAN SCALING FACTOR")
print("=" * 90)

print(
    f"{'Date':12s}"
    f"{'RTS mean':>15s}"
    f"{'MAP mean':>15s}"
    f"{'Abs diff':>15s}"
)

print("-" * 60)

for i, date in enumerate(expected_dates):

    rts_mean = np.mean(
        rts_states[i]
    )

    map_mean = np.mean(
        map_daily[i]
    )

    diff_mean = abs(
        rts_mean - map_mean
    )

    print(
        f"{date.strftime('%Y-%m-%d'):12s}"
        f"{rts_mean:15.8f}"
        f"{map_mean:15.8f}"
        f"{diff_mean:15.8e}"
    )


# ============================================================
# MINE-BY-MINE COMPARISON
# ============================================================

print("\n" + "=" * 90)
print("MINE-BY-MINE MEAN")
print("=" * 90)

print(
    f"{'Mine':8s}"
    f"{'RTS':>15s}"
    f"{'MAP':>15s}"
    f"{'Abs diff':>15s}"
)

print("-" * 55)

for m, mine in enumerate(MINES):

    rts_mean = np.mean(
        rts_states[:, m]
    )

    map_mean = np.mean(
        map_daily[:, m]
    )

    diff = abs(
        rts_mean - map_mean
    )

    print(
        f"{mine:8s}"
        f"{rts_mean:15.8f}"
        f"{map_mean:15.8f}"
        f"{diff:15.8e}"
    )


# ============================================================
# PARAMETER-BY-PARAMETER COMPARISON
# ============================================================

print("\n" + "=" * 90)
print("LARGEST PARAMETER DIFFERENCES")
print("=" * 90)

flat_rts = rts_states.ravel()
flat_map = map_daily.ravel()
flat_diff = np.abs(
    flat_rts - flat_map
)

largest = np.argsort(
    flat_diff
)[::-1][:20]

print(
    f"{'Parameter':25s}"
    f"{'RTS':>15s}"
    f"{'MAP':>15s}"
    f"{'Abs diff':>15s}"
)

print("-" * 75)

for idx in largest:

    print(
        f"{PARAMETER_NAMES[idx]:25s}"
        f"{flat_rts[idx]:15.8f}"
        f"{flat_map[idx]:15.8f}"
        f"{flat_diff[idx]:15.8e}"
    )


# ============================================================
# FINAL DAY CHECK
#
# Final-day KF = RTS = batch MAP
# ============================================================

print("\n" + "=" * 90)
print("FINAL-DAY CHECK")
print("=" * 90)

final_difference = (
    rts_states[-1]
    - map_daily[-1]
)

print(
    "Jan 31 maximum RTS-MAP difference:",
    np.max(
        np.abs(final_difference)
    )
)

print("\nRTS Jan 31:")

for i, mine in enumerate(MINES):

    print(
        f"  {mine:5s} "
        f"{rts_states[-1, i]:.10f}"
    )

print("\nMAP Jan 31:")

for i, mine in enumerate(MINES):

    print(
        f"  {mine:5s} "
        f"{map_daily[-1, i]:.10f}"
    )


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison_rows = []

for day_i, date in enumerate(expected_dates):

    for mine_i, mine in enumerate(MINES):

        comparison_rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "mine": mine,
            "RTS": rts_states[day_i, mine_i],
            "MAP": map_daily[day_i, mine_i],
            "difference": difference[day_i, mine_i],
            "abs_difference": abs_difference[day_i, mine_i],
        })

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_file = os.path.join(
    RTS_DIR,
    "RTS_vs_corrected_batch_MAP.csv"
)

comparison_df.to_csv(
    comparison_file,
    index=False
)

print("\nComparison saved to:")
print(comparison_file)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 90)
print("FINAL VALIDATION")
print("=" * 90)

if max_abs < 1e-10:

    print("\nSUCCESS:")
    print(
        "RTS smoother and batch MAP agree "
        "to numerical precision."
    )

    print("\nThis verifies:")
    print("  Forward KF -> RTS smoother")
    print("                    =")
    print("                 Batch MAP / 4D-Var")

else:

    print("\nWARNING:")
    print(
        "RTS and batch MAP do NOT agree "
        "to numerical precision."
    )

    print(
        "Maximum absolute difference:",
        max_abs
    )

print("\nDone.")


