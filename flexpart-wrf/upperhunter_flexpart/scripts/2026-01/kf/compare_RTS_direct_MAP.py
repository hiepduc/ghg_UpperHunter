#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================================
# PATHS
# =============================================================================

BASE = Path(
    "/home/duch/flexpart-wrf/upperhunter_flexpart"
)

H_FILE = (
    BASE
    / "H_matrix_flexpart"
    / "H_flexpart_0_10m_13mine_Jan2026.csv"
)

PICARRO_FILE = Path(
    "/home/duch/flexpart-wrf/emission_UH/"
    "RH_upperHunter/tapm_output/"
    "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

RTS_FILE = (
    BASE
    / "H_matrix_flexpart"
    / "KF_13mine_daily_Jan2026"
    / "RTS_daily_scaling_factors.csv"
)


# =============================================================================
# SETTINGS
# =============================================================================

START_DATE = pd.Timestamp("2026-01-05")
END_DATE = pd.Timestamp("2026-01-31")

BACKGROUND_PPM = 1.85
PRIOR_MEAN = 1.0
PRIOR_SIGMA = 0.5
CORRELATION_LENGTH_DAYS = 3.0
OBS_SIGMA_PPM = 0.05

MINES = [
    f"OC{i}"
    for i in range(1, 14)
]

N_MINE = len(MINES)
DATES = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D",
)

N_DAY = len(DATES)
N_PARAM = N_MINE * N_DAY


# =============================================================================
# PARAMETER ORDER
# =============================================================================

PARAMETER_NAMES = []

for day in DATES:
    day_string = day.strftime("%Y%m%d")

    for mine in MINES:
        PARAMETER_NAMES.append(
            f"{mine}_{day_string}"
        )

assert len(PARAMETER_NAMES) == N_PARAM


# =============================================================================
# READ H
# =============================================================================

print("=" * 90)
print("READING H MATRIX")
print("=" * 90)

H_df = pd.read_csv(H_FILE)

metadata_columns = [
    "receptor",
    "release_utc",
    "obs_time_aest",
    "flexpart_file",
]

missing = [
    c
    for c in metadata_columns
    if c not in H_df.columns
]

if missing:
    raise RuntimeError(
        f"Missing H metadata columns: {missing}"
    )

missing = [
    p
    for p in PARAMETER_NAMES
    if p not in H_df.columns
]

if missing:
    raise RuntimeError(
        f"Missing H parameter columns: {missing[:10]}"
    )

print(
    f"H rows                     : {len(H_df)}"
)

print(
    f"H parameters               : {len(PARAMETER_NAMES)}"
)


# =============================================================================
# H TIMESTAMP
# =============================================================================

H_df["obs_time_aest"] = pd.to_datetime(
    H_df["obs_time_aest"],
    errors="coerce",
)

H_df["obs_time_aest"] = (
    H_df["obs_time_aest"]
    .dt.floor("h")
)


# Preserve H parameter order exactly.
H_matrix_all = (
    H_df[
        PARAMETER_NAMES
    ]
    .to_numpy(
        dtype=np.float64
    )
)


# =============================================================================
# READ PICARRO
# =============================================================================

print()
print("=" * 90)
print("READING PICARRO OBSERVATIONS")
print("=" * 90)

obs = pd.read_csv(
    PICARRO_FILE
)

obs["Datetime"] = pd.to_datetime(
    obs["Datetime"],
    format="%d/%m/%Y %H:%M",
    errors="coerce",
)

obs["Datetime"] = (
    obs["Datetime"]
    .dt.floor("h")
)

receptor_map = {
    "Singleton cleaned": "singleton",
    "Muswellbrook cleaned": "muswellbrook",
    "Merriwa cleaned": "merriwa",
}

obs_long = []

for column, receptor in receptor_map.items():

    tmp = obs[
        [
            "Datetime",
            column,
        ]
    ].copy()

    tmp = tmp.rename(
        columns={
            "Datetime": "obs_time_aest",
            column: "ch4_ppm",
        }
    )

    tmp["receptor"] = receptor

    tmp = tmp[
        [
            "receptor",
            "obs_time_aest",
            "ch4_ppm",
        ]
    ]

    tmp = tmp.dropna(
        subset=["ch4_ppm"]
    )

    obs_long.append(tmp)


obs_long = pd.concat(
    obs_long,
    ignore_index=True,
)


# Average duplicate receptor/hour observations.
obs_long = (
    obs_long
    .groupby(
        [
            "receptor",
            "obs_time_aest",
        ],
        as_index=False,
    )[
        "ch4_ppm"
    ]
    .mean()
)


obs_long = obs_long[
    (
        obs_long["obs_time_aest"]
        >= START_DATE
    )
    &
    (
        obs_long["obs_time_aest"]
        < END_DATE + pd.Timedelta(days=1)
    )
].copy()


print(
    f"Observations               : "
    f"{len(obs_long)}"
)


# =============================================================================
# ALIGN H AND OBSERVATIONS
# =============================================================================

print()
print("=" * 90)
print("ALIGNING H AND OBSERVATIONS")
print("=" * 90)

H_key = H_df[
    [
        "receptor",
        "obs_time_aest",
    ]
].copy()

H_key["H_row"] = np.arange(
    len(H_key)
)

merged = pd.merge(
    obs_long,
    H_key,
    on=[
        "receptor",
        "obs_time_aest",
    ],
    how="inner",
)


# Sort exactly chronologically.
merged = merged.sort_values(
    [
        "obs_time_aest",
        "receptor",
    ]
).reset_index(
    drop=True
)


if len(merged) == 0:
    raise RuntimeError(
        "No H/observation matches."
    )


print(
    f"Matched observations        : "
    f"{len(merged)}"
)


# =============================================================================
# BUILD H AND d
# =============================================================================

row_indices = (
    merged["H_row"]
    .to_numpy(
        dtype=int
    )
)

H = H_matrix_all[
    row_indices,
    :
]

y = merged[
    "ch4_ppm"
].to_numpy(
    dtype=np.float64
)

d = (
    y
    - BACKGROUND_PPM
)


# =============================================================================
# BUILD PRIOR B
# =============================================================================

print()
print("=" * 90)
print("BUILDING PRIOR COVARIANCE B")
print("=" * 90)

B = np.zeros(
    (
        N_PARAM,
        N_PARAM,
    ),
    dtype=np.float64,
)

sigma2 = PRIOR_SIGMA ** 2
Lcorr = CORRELATION_LENGTH_DAYS

for mine_index in range(N_MINE):

    indices = np.array(
        [
            day_index * N_MINE
            + mine_index
            for day_index in range(N_DAY)
        ],
        dtype=int,
    )

    for i, ii in enumerate(indices):

        for j, jj in enumerate(indices):

            lag = abs(i - j)

            B[ii, jj] = (
                sigma2
                * np.exp(
                    -lag / Lcorr
                )
            )


# =============================================================================
# BUILD R
# =============================================================================

Rinv_diag = np.full(
    len(y),
    1.0 / OBS_SIGMA_PPM**2,
    dtype=np.float64,
)


# =============================================================================
# DIRECT MAP
# =============================================================================

print()
print("=" * 90)
print("SOLVING DIRECT BATCH MAP")
print("=" * 90)

Lb = np.linalg.cholesky(B)


def solve_B(rhs):

    z = np.linalg.solve(
        Lb,
        rhs,
    )

    return np.linalg.solve(
        Lb.T,
        z,
    )


Binv = solve_B(
    np.eye(
        N_PARAM,
        dtype=np.float64,
    )
)

H_weighted = (
    H
    * Rinv_diag[:, None]
)

A = (
    Binv
    + H.T @ H_weighted
)

A = (
    A + A.T
) / 2.0

s_b = np.full(
    N_PARAM,
    PRIOR_MEAN,
    dtype=np.float64,
)

b = (
    solve_B(s_b)
    + H.T @ (
        Rinv_diag * d
    )
)

s_map = np.linalg.solve(
    A,
    b,
)


# =============================================================================
# GRADIENT CHECK
# =============================================================================

gradient = (
    solve_B(
        s_map - s_b
    )
    + H.T @ (
        Rinv_diag
        * (
            H @ s_map - d
        )
    )
)

print(
    f"MAP gradient L2           : "
    f"{np.linalg.norm(gradient):.6e}"
)

print(
    f"MAP gradient max abs      : "
    f"{np.max(np.abs(gradient)):.6e}"
)


# =============================================================================
# READ RTS
# =============================================================================

print()
print("=" * 90)
print("READING RTS SOLUTION")
print("=" * 90)

rts_df = pd.read_csv(
    RTS_FILE
)

print(
    f"RTS shape                 : "
    f"{rts_df.shape}"
)

print(
    f"RTS columns               : "
    f"{list(rts_df.columns)}"
)


# Expected format:
#
# date, state_type, n_observations, OC1 ... OC13
#

rts_df["date"] = pd.to_datetime(
    rts_df["date"]
)

rts_df = rts_df.sort_values(
    "date"
).reset_index(
    drop=True
)


# =============================================================================
# BUILD RTS VECTOR
# =============================================================================

s_rts = np.zeros(
    N_PARAM,
    dtype=np.float64,
)

for day_index, day in enumerate(DATES):

    rows = rts_df[
        rts_df["date"] == day
    ]

    if len(rows) != 1:

        raise RuntimeError(
            f"Expected exactly one RTS row "
            f"for {day.date()}, found {len(rows)}"
        )

    row = rows.iloc[0]

    for mine_index, mine in enumerate(MINES):

        parameter_index = (
            day_index * N_MINE
            + mine_index
        )

        s_rts[parameter_index] = float(
            row[mine]
        )


# =============================================================================
# COMPARISON
# =============================================================================

difference = (
    s_rts
    - s_map
)

abs_difference = np.abs(
    difference
)

print()
print("=" * 90)
print("RTS vs DIRECT MAP")
print("=" * 90)

print(
    f"MAP mean scaling           : "
    f"{np.mean(s_map):.10f}"
)

print(
    f"RTS mean scaling           : "
    f"{np.mean(s_rts):.10f}"
)

print(
    f"MAP total mean             : "
    f"{np.mean(s_map):.10f}"
)

print(
    f"RTS total mean             : "
    f"{np.mean(s_rts):.10f}"
)

print(
    f"Max abs difference         : "
    f"{np.max(abs_difference):.10e}"
)

print(
    f"RMSE difference            : "
    f"{np.sqrt(np.mean(difference**2)):.10e}"
)

print(
    f"Mean absolute difference   : "
    f"{np.mean(abs_difference):.10e}"
)

print(
    f"Correlation                : "
    f"{np.corrcoef(s_map, s_rts)[0,1]:.12f}"
)


# =============================================================================
# DAILY TOTALS
# =============================================================================

# Prior emissions, g CH4/s
prior_emissions = np.array([
    1786.45,
    242.86,
    342.53,
    40.60,
    0.01,
    7.85,
    82.69,
    5.95,
    44.92,
    101.93,
    1.87,
    331.49,
    798.08,
], dtype=float)


map_totals = []
rts_totals = []

for day_index, day in enumerate(DATES):

    sl = slice(
        day_index * N_MINE,
        (day_index + 1) * N_MINE,
    )

    map_totals.append(
        np.sum(
            prior_emissions
            * s_map[sl]
        )
    )

    rts_totals.append(
        np.sum(
            prior_emissions
            * s_rts[sl]
        )
    )


daily_compare = pd.DataFrame({
    "date": DATES,
    "MAP_total_gCH4_s": map_totals,
    "RTS_total_gCH4_s": rts_totals,
})

daily_compare["difference_gCH4_s"] = (
    daily_compare["RTS_total_gCH4_s"]
    - daily_compare["MAP_total_gCH4_s"]
)

daily_compare["abs_difference_gCH4_s"] = (
    np.abs(
        daily_compare["difference_gCH4_s"]
    )
)

print()
print("=" * 90)
print("DAILY TOTAL COMPARISON")
print("=" * 90)

print(
    daily_compare.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =============================================================================
# SAVE FULL PARAMETER COMPARISON
# =============================================================================

comparison = pd.DataFrame({
    "parameter": PARAMETER_NAMES,
    "MAP": s_map,
    "RTS": s_rts,
    "difference_RTS_minus_MAP": difference,
    "abs_difference": abs_difference,
})

comparison["day"] = [
    p.split("_")[1]
    for p in PARAMETER_NAMES
]

comparison["mine"] = [
    p.split("_")[0]
    for p in PARAMETER_NAMES
]

comparison = comparison[
    [
        "day",
        "mine",
        "parameter",
        "MAP",
        "RTS",
        "difference_RTS_minus_MAP",
        "abs_difference",
    ]
]

output_dir = (
    BASE
    / "H_matrix_flexpart"
    / "KF_13mine_daily_Jan2026"
)

output_file = (
    output_dir
    / "RTS_vs_direct_MAP.csv"
)

comparison.to_csv(
    output_file,
    index=False,
)

daily_file = (
    output_dir
    / "RTS_vs_direct_MAP_daily_totals.csv"
)

daily_compare.to_csv(
    daily_file,
    index=False,
)

print()
print(
    f"Saved parameter comparison: "
    f"{output_file}"
)

print(
    f"Saved daily comparison    : "
    f"{daily_file}"
)

