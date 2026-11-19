#!/usr/bin/env python3

"""
Validate FLEXPART-WRF prior forward model against Picarro observations.

H matrix:
    H[i, m,d] = FLEXPART contribution from prior emission of mine m
                on source day d to observation i.

Therefore:
    prior_enhancement = sum(H row)

and:
    prior_total_CH4 = BACKGROUND_PPM + prior_enhancement

The inversion later uses:

    y = H @ f

where f contains 13 x 27 daily emission factors.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE = Path.home() / "flexpart-wrf/upperhunter_flexpart"

H_FILE = BASE / "H_matrix_flexpart/H_flexpart_0_10m_13mine_Jan2026.csv"

FLEX_ROOT = BASE / "run/hourly_obs_3layer"

OBS_FILE = (
    Path.home()
    / "flexpart-wrf/emission_UH/RH_upperHunter/tapm_output/"
      "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

BACKGROUND_PPM = 1.85


# =============================================================================
# RECEPTORS
# =============================================================================

RECEPTORS = {
    "singleton": "Singleton cleaned",
    "muswellbrook": "Muswellbrook cleaned",
    "merriwa": "Merriwa cleaned",
}


# =============================================================================
# HELPERS
# =============================================================================

def parse_case_timestamp(path):
    """
    Case directories are named:

        YYYYMMDD_HHMMSS

    This timestamp is the FLEXPART release/receptor time in UTC.
    """
    m = re.search(r"(\d{8})_(\d{6})", str(path))

    if not m:
        raise ValueError(f"Cannot find timestamp in: {path}")

    return pd.to_datetime(
        m.group(1) + m.group(2),
        format="%Y%m%d%H%M%S",
        utc=True,
    )


def find_flex_files():

    files = sorted(
        FLEX_ROOT.glob("*/*/flxout_d01_*.nc")
    )

    if len(files) != len(H):
        raise ValueError(
            f"Number of FLEXPART files ({len(files)}) "
            f"does not equal H rows ({len(H)})"
        )

    return files


def metrics(obs, mod):

    mask = np.isfinite(obs) & np.isfinite(mod)

    obs = np.asarray(obs)[mask]
    mod = np.asarray(mod)[mask]

    n = len(obs)

    if n == 0:
        return {
            "N": 0,
            "R": np.nan,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan,
            "FAC2": np.nan,
        }

    diff = mod - obs

    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    bias = np.mean(diff)

    if n > 1:
        r = np.corrcoef(obs, mod)[0, 1]
        r2 = r ** 2
    else:
        r = np.nan
        r2 = np.nan

    positive = (obs > 0) & (mod > 0)

    if np.any(positive):
        ratio = mod[positive] / obs[positive]
        fac2 = np.mean((ratio >= 0.5) & (ratio <= 2.0))
    else:
        fac2 = np.nan

    return {
        "N": n,
        "R": r,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "FAC2": fac2,
    }


# =============================================================================
# READ H
# =============================================================================

print("=" * 90)
print("FLEXPART-WRF PRIOR FORWARD VALIDATION")
print("=" * 90)

print("\nReading H matrix:")
print(H_FILE)

H = pd.read_csv(H_FILE)

print(f"H shape: {H.shape}")

# First column is normally the observation index.
# Identify numeric mine/day columns.
mine_day_cols = [
    c for c in H.columns
    if re.match(r"^OC\d+_\d{8}$", str(c))
]

if len(mine_day_cols) != 351:
    raise ValueError(
        f"Expected 351 mine/day columns, found {len(mine_day_cols)}"
    )

print(f"Mine/day columns: {len(mine_day_cols)}")

H_values = H[mine_day_cols].to_numpy(dtype=float)

print(
    f"H min: {np.nanmin(H_values):.6e}\n"
    f"H max: {np.nanmax(H_values):.6e}"
)


# =============================================================================
# RECONSTRUCT FLEXPART ROW ORDER
# =============================================================================

print("\nFinding FLEXPART files...")

flex_files = sorted(
    FLEX_ROOT.glob("*/*/flxout_d01_*.nc")
)

print(f"FLEXPART files: {len(flex_files)}")

if len(flex_files) != len(H_values):
    raise ValueError(
        f"FLEXPART files ({len(flex_files)}) != H rows ({len(H_values)})"
    )


# =============================================================================
# BUILD OBSERVATION METADATA
# =============================================================================

rows = []

for i, fp in enumerate(flex_files):

    parts = fp.parts

    # Expected:
    # .../hourly_obs_3layer/receptor/YYYYMMDD_HHMMSS/flxout...
    receptor = parts[-3].lower()
    case = parts[-2]

    release_utc = parse_case_timestamp(fp)
    release_aest = release_utc + pd.Timedelta(hours=10)

    rows.append({
        "H_row": i,
        "receptor": receptor,
        "case": case,
        "release_utc": release_utc,
        "release_aest": release_aest,
    })


meta = pd.DataFrame(rows)

print("\nObservation time range:")
print("UTC :", meta.release_utc.min(), "to", meta.release_utc.max())
print("AEST:", meta.release_aest.min(), "to", meta.release_aest.max())

print("\nReceptor counts:")
print(meta.receptor.value_counts())


# =============================================================================
# PRIOR FORWARD MODEL
# =============================================================================

# H already contains the prior mine emissions.
#
# Therefore:
#
#       prior enhancement = sum(H row)
#
# NOT:
#
#       H @ prior_emission
#
# because that would multiply the prior emission twice.

meta["prior_enhancement_ppm"] = np.sum(
    H_values,
    axis=1
)

meta["prior_total_ppm"] = (
    BACKGROUND_PPM
    + meta["prior_enhancement_ppm"]
)


print("\nPrior forward-model statistics:")

print(
    meta[
        [
            "prior_enhancement_ppm",
            "prior_total_ppm",
        ]
    ].describe()
)


# =============================================================================
# READ PICARRO
# =============================================================================

print("\nReading Picarro observations:")
print(OBS_FILE)

obs = pd.read_csv(OBS_FILE)

print("Observation columns:")
print(list(obs.columns))


# Try to identify the timestamp column automatically.
time_candidates = [
    c for c in obs.columns
    if any(
        key in str(c).lower()
        for key in [
            "time",
            "date",
            "datetime",
            "timestamp",
        ]
    )
]

print("\nPossible time columns:")
print(time_candidates)


# Use the first suitable datetime column.
# If your CSV has a specific known timestamp column, this can be
# replaced by that column name.
time_col = None

for c in time_candidates:

    test = pd.to_datetime(
        obs[c],
        errors="coerce"
    )

    if test.notna().sum() > 0:
        time_col = c
        break

if time_col is None:
    raise ValueError(
        "Could not identify Picarro timestamp column."
    )

print(f"\nUsing Picarro time column: {time_col}")


obs["time_aest"] = pd.to_datetime(
    obs[time_col],
    errors="coerce"
)


# Picarro timestamps are already AEST.
if getattr(obs["time_aest"].dt, "tz", None) is not None:
    obs["time_aest"] = (
        obs["time_aest"]
        .dt.tz_localize(None)
    )


# =============================================================================
# HOURLY OBSERVATION MATCHING
# =============================================================================

results = []

for receptor, obs_col in RECEPTORS.items():

    if obs_col not in obs.columns:
        print(
            f"\nWARNING: {obs_col} not found. "
            f"Skipping {receptor}."
        )
        continue

    rmeta = meta[
        meta["receptor"].str.lower() == receptor
    ].copy()

    if len(rmeta) == 0:
        continue

    # Match FLEXPART receptor time to Picarro time.
    #
    # FLEXPART release times are exact hourly times.
    # Picarro data may have seconds, so round to nearest hour.
    obs_tmp = obs[
        ["time_aest", obs_col]
    ].copy()

    obs_tmp["time_hour"] = (
        obs_tmp["time_aest"]
        .dt.round("h")
    )

    obs_tmp = obs_tmp.rename(
        columns={obs_col: "obs_total_ppm"}
    )

    # Remove duplicate hourly observations using mean.
    obs_hourly = (
        obs_tmp
        .dropna(subset=["time_hour", "obs_total_ppm"])
        .groupby("time_hour", as_index=False)
        ["obs_total_ppm"]
        .mean()
    )

    rmeta["obs_time_aest"] = rmeta["release_aest"].dt.tz_localize(None)

    merged = rmeta.merge(
        obs_hourly,
        left_on="obs_time_aest",
        right_on="time_hour",
        how="left",
    )

    merged["obs_enhancement_ppm"] = (
        merged["obs_total_ppm"] - BACKGROUND_PPM
    )

    merged["receptor"] = receptor

    results.append(merged)


if not results:
    raise RuntimeError("No receptor observations matched.")


validation = pd.concat(
    results,
    ignore_index=True
)


# =============================================================================
# METRICS
# =============================================================================

print("\n")
print("=" * 90)
print("PRIOR MODEL VS PICARRO")
print("=" * 90)

all_metrics = []

for receptor in RECEPTORS:

    sub = validation[
        validation["receptor"] == receptor
    ].copy()

    sub = sub.dropna(
        subset=[
            "obs_total_ppm",
            "prior_total_ppm",
        ]
    )

    if len(sub) == 0:
        print(f"\n{receptor.upper()}: no matched observations")
        continue

    m = metrics(
        sub["obs_total_ppm"].values,
        sub["prior_total_ppm"].values,
    )

    m["Receptor"] = receptor
    all_metrics.append(m)

    print(f"\n{receptor.upper()}")
    print("-" * 50)

    for key in [
        "N",
        "R",
        "R2",
        "RMSE",
        "MAE",
        "Bias",
        "FAC2",
    ]:
        print(f"{key:10s}: {m[key]}")


# Combined
sub = validation.dropna(
    subset=[
        "obs_total_ppm",
        "prior_total_ppm",
    ]
)

if len(sub) > 0:

    m = metrics(
        sub["obs_total_ppm"].values,
        sub["prior_total_ppm"].values,
    )

    m["Receptor"] = "ALL"

    all_metrics.append(m)

    print("\nALL RECEPTORS")
    print("-" * 50)

    for key in [
        "N",
        "R",
        "R2",
        "RMSE",
        "MAE",
        "Bias",
        "FAC2",
    ]:
        print(f"{key:10s}: {m[key]}")


# =============================================================================
# SAVE
# =============================================================================

out_dir = BASE / "H_matrix_flexpart"
out_dir.mkdir(exist_ok=True)

out_file = out_dir / "FLEXPART_prior_validation_Jan2026.csv"

validation.to_csv(
    out_file,
    index=False
)

metrics_file = (
    out_dir /
    "FLEXPART_prior_validation_metrics_Jan2026.csv"
)

pd.DataFrame(all_metrics).to_csv(
    metrics_file,
    index=False
)

print("\n")
print("=" * 90)
print("VALIDATION COMPLETE")
print("=" * 90)

print(f"Detailed results:")
print(out_file)

print(f"\nMetrics:")
print(metrics_file)

