#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares


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

OUTPUT_DIR = os.path.join(
    BASE,
    "H_matrix_flexpart",
    "KF_13mine_daily_Jan2026"
)

RTS_FILE = os.path.join(
    OUTPUT_DIR,
    "RTS_daily_scaling_factors.csv"
)

KF_FILE = os.path.join(
    OUTPUT_DIR,
    "daily_KF_scaling_factors.csv"
)


# ============================================================
# SETTINGS
# ============================================================

BACKGROUND_PPM = 1.85

PRIOR_MEAN = 1.0
PRIOR_SIGMA = 0.5
PRIOR_VARIANCE = PRIOR_SIGMA ** 2

OBS_SIGMA = 0.05
OBS_VARIANCE = OBS_SIGMA ** 2

L_CORR = 3.0

START_DATE = pd.Timestamp("2026-01-05")
END_DATE = pd.Timestamp("2026-01-31")

MINES = [f"OC{i}" for i in range(1, 14)]

N_MINE = 13
N_DAY = 27
N_PARAM = 351


# ============================================================
# PARAMETER ORDER
# ============================================================

PARAMETER_NAMES = []

for day in pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
):

    datestr = day.strftime("%Y%m%d")

    for mine in MINES:
        PARAMETER_NAMES.append(
            f"{mine}_{datestr}"
        )


# ============================================================
# READ H
# ============================================================

print("=" * 90)
print("STABLE RTS vs BATCH MAP VALIDATION")
print("=" * 90)

print("\nReading H:")
print(H_FILE)

H_df = pd.read_csv(H_FILE)

print("H shape:", H_df.shape)

print("\nFirst parameter columns:")
print(
    H_df.columns[4:20].tolist()
)

# Verify exact expected ordering
actual_parameter_columns = [
    c for c in H_df.columns
    if c not in [
        "receptor",
        "release_utc",
        "obs_time_aest",
        "flexpart_file",
    ]
]

print("\nFirst actual H parameter columns:")
print(actual_parameter_columns[:20])


# ============================================================
# CHECK PARAMETER SET
# ============================================================

missing = [
    p
    for p in PARAMETER_NAMES
    if p not in H_df.columns
]

extra = [
    p
    for p in actual_parameter_columns
    if p not in PARAMETER_NAMES
]

if missing:
    print("\nMISSING H PARAMETERS:")
    print(missing)
    raise RuntimeError("Missing H parameters.")

if extra:
    print("\nEXTRA H PARAMETERS:")
    print(extra)
    raise RuntimeError("Unexpected H parameters.")


# ============================================================
# TIME / RECEPTOR
# ============================================================

H_df["obs_time_aest"] = pd.to_datetime(
    H_df["obs_time_aest"]
).dt.floor("h")

H_df["receptor"] = (
    H_df["receptor"]
    .astype(str)
    .str.lower()
    .str.strip()
)


# ============================================================
# READ PICARRO
# ============================================================

print("\nReading Picarro:")

obs = pd.read_csv(
    PICARRO_FILE
)

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
    value_name="observation"
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
    (
        obs_long["Datetime"]
        < END_DATE + pd.Timedelta(days=1)
    )
]

obs_long = (
    obs_long
    .groupby(
        ["receptor", "Datetime"],
        as_index=False
    )["observation"]
    .mean()
)

print(
    "Observations:",
    len(obs_long)
)


# ============================================================
# MATCH
# ============================================================

matched = H_df.merge(
    obs_long,
    left_on=[
        "receptor",
        "obs_time_aest"
    ],
    right_on=[
        "receptor",
        "Datetime"
    ],
    how="inner"
)

matched = (
    matched
    .sort_values(
        ["Datetime", "receptor"]
    )
    .reset_index(drop=True)
)

print(
    "Matched:",
    len(matched)
)


H = matched[
    PARAMETER_NAMES
].to_numpy(
    dtype=np.float64
)

y = matched[
    "observation"
].to_numpy(
    dtype=np.float64
)

d = y - BACKGROUND_PPM


# ============================================================
# BUILD B
# ============================================================

rho = np.exp(
    -1.0 / L_CORR
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
                    / L_CORR
                )
            )


# ============================================================
# CHOLESKY OF B
# ============================================================

print("\nCholesky decomposition of B...")

LB = np.linalg.cholesky(B)

print(
    "B reconstruction error:",
    np.max(
        np.abs(
            B - LB @ LB.T
        )
    )
)


# ============================================================
# STABLE BATCH MAP
#
# Whitened objective:
#
# || LB^-1 (x-xb) ||^2
# +
# || (H x-d)/sigma ||^2
#
# This avoids explicitly forming:
#
# B^-1
# H.T H
#
# ============================================================

print("\nConstructing whitened MAP system...")

xb = np.full(
    N_PARAM,
    PRIOR_MEAN,
    dtype=np.float64
)

# B^{-1/2}(x-xb)
#
# Since B = L L^T,
# solve L z = x-xb

A_prior = solve_triangular(
    LB,
    np.eye(N_PARAM),
    lower=True
)

# Observation block
A_obs = H / OBS_SIGMA

# ------------------------------------------------------------
# Correct observation RHS
#
# x = xb + z
#
# H x = d
#
# therefore:
#
# H z = d - H xb
# ------------------------------------------------------------

observation_rhs = (
    d - H @ xb
)

A_aug = np.vstack([
    A_prior,
    A_obs
])

b_aug = np.concatenate([
    np.zeros(N_PARAM),
    observation_rhs / OBS_SIGMA
])


# ------------------------------------------------------------
# Solve least squares
# ------------------------------------------------------------

print("Solving stable MAP system...")

result = np.linalg.lstsq(
    A_aug,
    b_aug,
    rcond=None
)

z_map = result[0]

x_map = xb + z_map


print(
    "Least-squares residual norm:",
    result[1]
)


# ============================================================
# GRADIENT CHECK
# ============================================================

# Compute B^{-1}(x-xb) without explicitly forming B^-1.

delta = x_map - xb

tmp = solve_triangular(
    LB,
    delta,
    lower=True
)

Binv_delta = solve_triangular(
    LB.T,
    tmp,
    lower=False
)

gradient = (
    Binv_delta
    +
    H.T @ (
        H @ x_map - d
    ) / OBS_VARIANCE
)

print("\nStable MAP gradient:")
print(
    "L2 norm :",
    np.linalg.norm(gradient)
)

print(
    "max abs :",
    np.max(np.abs(gradient))
)


# ============================================================
# RESHAPE
# ============================================================

map_daily = x_map.reshape(
    N_DAY,
    N_MINE
)


# ============================================================
# READ RTS
# ============================================================

rts_df = pd.read_csv(
    RTS_FILE
)

rts_df["date"] = pd.to_datetime(
    rts_df["date"]
)

rts_df = (
    rts_df
    .sort_values("date")
    .reset_index(drop=True)
)

rts = rts_df[
    MINES
].to_numpy(
    dtype=np.float64
)


# ============================================================
# READ KF
# ============================================================

kf_df = pd.read_csv(
    KF_FILE
)

kf_df["date"] = pd.to_datetime(
    kf_df["date"]
)

kf_df = (
    kf_df
    .sort_values("date")
    .reset_index(drop=True)
)

kf = kf_df[
    MINES
].to_numpy(
    dtype=np.float64
)


# ============================================================
# COMPARE RTS vs STABLE MAP
# ============================================================

diff_rts = rts - map_daily

print("\n" + "=" * 90)
print("RTS vs STABLE BATCH MAP")
print("=" * 90)

print(
    "Maximum absolute difference:",
    np.max(np.abs(diff_rts))
)

print(
    "RMSE:",
    np.sqrt(np.mean(diff_rts ** 2))
)

print(
    "MAE:",
    np.mean(np.abs(diff_rts))
)

print(
    "Correlation:",
    np.corrcoef(
        rts.ravel(),
        map_daily.ravel()
    )[0, 1]
)


# ============================================================
# KF vs RTS
# ============================================================

diff_kf_rts = kf - rts

print("\n" + "=" * 90)
print("KF vs RTS")
print("=" * 90)

print(
    "Maximum absolute difference:",
    np.max(np.abs(diff_kf_rts))
)

print(
    "RMSE:",
    np.sqrt(np.mean(diff_kf_rts ** 2))
)


# ============================================================
# FINAL DAY
# ============================================================

print("\n" + "=" * 90)
print("JAN 31 FINAL-DAY TEST")
print("=" * 90)

print(
    f"{'Mine':8s}"
    f"{'KF':>15s}"
    f"{'RTS':>15s}"
    f"{'MAP':>15s}"
)

print("-" * 60)

for i, mine in enumerate(MINES):

    print(
        f"{mine:8s}"
        f"{kf[-1,i]:15.8f}"
        f"{rts[-1,i]:15.8f}"
        f"{map_daily[-1,i]:15.8f}"
    )

print("\nKF vs RTS final day:")
print(
    np.max(
        np.abs(
            kf[-1] - rts[-1]
        )
    )
)

print("KF vs MAP final day:")
print(
    np.max(
        np.abs(
            kf[-1] - map_daily[-1]
        )
    )
)

print("RTS vs MAP final day:")
print(
    np.max(
        np.abs(
            rts[-1] - map_daily[-1]
        )
    )
)


# ============================================================
# DAILY MEANS
# ============================================================

print("\n" + "=" * 90)
print("DAILY MEAN")
print("=" * 90)

print(
    f"{'Date':12s}"
    f"{'KF':>14s}"
    f"{'RTS':>14s}"
    f"{'MAP':>14s}"
)

print("-" * 60)

dates = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)

for i, date in enumerate(dates):

    print(
        f"{date.strftime('%Y-%m-%d'):12s}"
        f"{np.mean(kf[i]):14.8f}"
        f"{np.mean(rts[i]):14.8f}"
        f"{np.mean(map_daily[i]):14.8f}"
    )


# ============================================================
# SAVE
# ============================================================

out = []

for d_i, date in enumerate(dates):

    for m_i, mine in enumerate(MINES):

        out.append({
            "date": date.strftime("%Y-%m-%d"),
            "mine": mine,
            "KF": kf[d_i,m_i],
            "RTS": rts[d_i,m_i],
            "MAP": map_daily[d_i,m_i],
            "RTS_minus_MAP":
                rts[d_i,m_i] - map_daily[d_i,m_i],
            "KF_minus_MAP":
                kf[d_i,m_i] - map_daily[d_i,m_i],
        })

out_df = pd.DataFrame(out)

out_file = os.path.join(
    OUTPUT_DIR,
    "KF_RTS_STABLE_MAP_comparison.csv"
)

out_df.to_csv(
    out_file,
    index=False
)

print("\nSaved:")
print(out_file)

print("\nDone.")

