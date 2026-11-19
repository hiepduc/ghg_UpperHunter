#!/usr/bin/env python3
"""
Test STILT mine contributions for one receptor/observation.

Case
----
Muswellbrook
15 January 2026
22:00 AEST = 12:00 UTC

The script integrates the STILT 24-hour footprint against the
individual OC1-OC13 CH4 emission fields.

Emission field units:
    mol CH4 m-2 s-1

STILT footprint units:
    ppm (umol m-2 s)-1

Therefore:

    contribution_ppm =
        sum(footprint * emission_flux * 1e6)

The result is the CH4 enhancement at the receptor due to each mine
for this particular observation.

Outputs
-------
1. Individual mine contributions (ppm)
2. Total mine contribution (ppm)
3. Percentage contribution of each mine
4. Contribution by each of the 24 backward hours
5. CSV files containing the results
"""

import os
import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# SETTINGS
# =============================================================================

EMISSION_FILE = (
    "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter/"
    "stilt_mine_emissions_d03.nc"
)

FOOTPRINT_FILE = (
    "/mnt/scratch_lustre/duch/stilt/upperhunter/"
    "arl_mine_jan2026/footprints/Muswellbrook/footprints/"
    "202601151200_150.89297_-32.26202_5_foot.nc"
)

OUTPUT_DIR = (
    "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter/stilt"
)

MINE_NAMES = [f"OC{i}" for i in range(1, 14)]

# mol -> umol
MOL_TO_UMOL = 1.0e6


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("STILT MINE CONTRIBUTION TEST")
print("=" * 80)

print()
print("Emission file:")
print(EMISSION_FILE)

print()
print("Footprint:")
print(FOOTPRINT_FILE)


# =============================================================================
# CHECK FILES
# =============================================================================

if not os.path.exists(EMISSION_FILE):
    raise FileNotFoundError(
        f"Emission file not found:\n{EMISSION_FILE}"
    )

if not os.path.exists(FOOTPRINT_FILE):
    raise FileNotFoundError(
        f"Footprint file not found:\n{FOOTPRINT_FILE}"
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# READ DATA
# =============================================================================

print()
print("=" * 80)
print("READING DATA")
print("=" * 80)

emission_ds = xr.open_dataset(EMISSION_FILE)

foot_ds = xr.open_dataset(FOOTPRINT_FILE)


# =============================================================================
# DISPLAY FOOTPRINT INFORMATION
# =============================================================================

print()
print("Footprint dimensions:")
print(foot_ds.dims)

print()
print("Footprint variables:")
print(list(foot_ds.data_vars))

print()
print("Footprint coordinates:")
print(list(foot_ds.coords))


# =============================================================================
# READ FOOTPRINT
# =============================================================================

if "foot" not in foot_ds:
    raise ValueError(
        "The footprint NetCDF does not contain variable 'foot'."
    )

foot = foot_ds["foot"]

print()
print("Footprint shape:")
print(foot.shape)

print()
print("Footprint units:")
print(foot.attrs.get("units", "not specified"))


# =============================================================================
# VERIFY DIMENSIONS
# =============================================================================

if foot.dims != ("time", "lat", "lon"):

    print()
    print("WARNING:")
    print(
        "Footprint dimensions are not exactly "
        "('time', 'lat', 'lon')."
    )

    print("Actual dimensions:", foot.dims)

    # Try to transpose if the same dimensions exist
    if set(foot.dims) == {"time", "lat", "lon"}:
        foot = foot.transpose("time", "lat", "lon")
        print("Footprint transposed to (time, lat, lon).")
    else:
        raise ValueError(
            "Cannot identify time/lat/lon dimensions."
        )


# =============================================================================
# READ COORDINATES
# =============================================================================

foot_lat = np.asarray(
    foot_ds["lat"].values,
    dtype=float
)

foot_lon = np.asarray(
    foot_ds["lon"].values,
    dtype=float
)

print()
print("Footprint grid:")
print(
    "  lat:",
    len(foot_lat),
    foot_lat.min(),
    "to",
    foot_lat.max()
)

print(
    "  lon:",
    len(foot_lon),
    foot_lon.min(),
    "to",
    foot_lon.max()
)

print(
    "  time:",
    foot.sizes["time"]
)


# =============================================================================
# READ EMISSION GRID
# =============================================================================

em_lat = np.asarray(
    emission_ds["lat"].values,
    dtype=float
)

em_lon = np.asarray(
    emission_ds["lon"].values,
    dtype=float
)

print()
print("Emission grid:")
print(
    "  lat:",
    len(em_lat),
    em_lat.min(),
    "to",
    em_lat.max()
)

print(
    "  lon:",
    len(em_lon),
    em_lon.min(),
    "to",
    em_lon.max()
)


# =============================================================================
# VERIFY GRIDS MATCH
# =============================================================================

print()
print("=" * 80)
print("CHECKING FOOTPRINT / EMISSION GRID")
print("=" * 80)

if len(foot_lat) != len(em_lat):
    raise ValueError(
        "Latitude dimensions do not match."
    )

if len(foot_lon) != len(em_lon):
    raise ValueError(
        "Longitude dimensions do not match."
    )

if not np.allclose(
    foot_lat,
    em_lat,
    rtol=0,
    atol=1e-8
):
    raise ValueError(
        "Footprint latitude coordinates do not match "
        "the emission grid."
    )

if not np.allclose(
    foot_lon,
    em_lon,
    rtol=0,
    atol=1e-8
):
    raise ValueError(
        "Footprint longitude coordinates do not match "
        "the emission grid."
    )

print("Latitude grids: MATCH")
print("Longitude grids: MATCH")
print("Grid size:", len(em_lat), "x", len(em_lon))


# =============================================================================
# READ FOOTPRINT ARRAY
# =============================================================================

foot_array = np.asarray(
    foot.values,
    dtype=float
)

print()
print("Footprint statistics:")

print(
    "  shape:",
    foot_array.shape
)

print(
    "  min:",
    np.nanmin(foot_array)
)

print(
    "  max:",
    np.nanmax(foot_array)
)

print(
    "  mean:",
    np.nanmean(foot_array)
)

print(
    "  non-zero:",
    np.count_nonzero(
        np.nan_to_num(foot_array)
    )
)


# =============================================================================
# HANDLE INVALID VALUES
# =============================================================================

foot_array = np.nan_to_num(
    foot_array,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# =============================================================================
# CALCULATE CONTRIBUTION FROM EACH MINE
# =============================================================================

print()
print("=" * 80)
print("MINE CONTRIBUTIONS")
print("=" * 80)

results = []

mine_contributions = {}

for mine in MINE_NAMES:

    if mine not in emission_ds:
        raise ValueError(
            f"{mine} not found in emission file."
        )

    emission = np.asarray(
        emission_ds[mine].values,
        dtype=float
    )

    if emission.shape != (
        len(em_lat),
        len(em_lon)
    ):
        raise ValueError(
            f"{mine} has incorrect shape: "
            f"{emission.shape}"
        )

    # Replace invalid values
    emission = np.nan_to_num(
        emission,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # -------------------------------------------------------------------------
    # STILT footprint is:
    #
    # ppm / (umol m-2 s-1)
    #
    # emission is:
    #
    # mol m-2 s-1
    #
    # Convert mol -> umol by 1e6.
    #
    # The emission field is constant in time, so it is broadcast over
    # the 24 footprint hours.
    # -------------------------------------------------------------------------

    contribution = (
        np.sum(
            foot_array *
            emission[np.newaxis, :, :] *
            MOL_TO_UMOL
        )
    )

    mine_contributions[mine] = contribution

    results.append(
        {
            "mine": mine,
            "contribution_ppm": contribution
        }
    )


# =============================================================================
# TOTAL
# =============================================================================

total_contribution = sum(
    mine_contributions.values()
)

for r in results:

    r["percentage_of_total"] = (
        100.0 *
        r["contribution_ppm"] /
        total_contribution
        if total_contribution != 0
        else np.nan
    )


results_df = pd.DataFrame(results)


# =============================================================================
# PRINT RESULTS
# =============================================================================

print()

print(
    f"{'Mine':<8}"
    f"{'Contribution (ppm)':>22}"
    f"{'% of total':>15}"
)

print("-" * 50)

for _, row in results_df.iterrows():

    print(
        f"{row['mine']:<8}"
        f"{row['contribution_ppm']:>22.8f}"
        f"{row['percentage_of_total']:>15.3f}"
    )

print("-" * 50)

print(
    f"{'TOTAL':<8}"
    f"{total_contribution:>22.8f}"
    f"{100.0:>15.3f}"
)


# =============================================================================
# SORT BY CONTRIBUTION
# =============================================================================

print()
print("=" * 80)
print("MINES RANKED BY CONTRIBUTION")
print("=" * 80)

ranked = results_df.sort_values(
    "contribution_ppm",
    ascending=False
).reset_index(drop=True)

ranked["rank"] = np.arange(
    1,
    len(ranked) + 1
)

print()

print(
    f"{'Rank':<6}"
    f"{'Mine':<8}"
    f"{'Contribution (ppm)':>22}"
    f"{'% of total':>15}"
)

print("-" * 55)

for _, row in ranked.iterrows():

    print(
        f"{int(row['rank']):<6}"
        f"{row['mine']:<8}"
        f"{row['contribution_ppm']:>22.8f}"
        f"{row['percentage_of_total']:>15.3f}"
    )


# =============================================================================
# 24-HOUR BACKWARD CONTRIBUTION
# =============================================================================

print()
print("=" * 80)
print("24-HOUR BACKWARD CONTRIBUTION")
print("=" * 80)

print()
print(
    "Calculating contribution from each footprint time layer..."
)

time_values = foot_ds["time"].values

hourly_records = []

for t in range(len(time_values)):

    time_layer = foot_array[t, :, :]

    record = {
        "time_index": t
    }

    # Convert timestamp to readable representation
    try:
        record["time"] = str(
            pd.Timestamp(time_values[t])
        )
    except Exception:
        record["time"] = str(
            time_values[t]
        )

    total_hour = 0.0

    for mine in MINE_NAMES:

        emission = np.asarray(
            emission_ds[mine].values,
            dtype=float
        )

        emission = np.nan_to_num(
            emission,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        c = np.sum(
            time_layer *
            emission *
            MOL_TO_UMOL
        )

        record[mine] = c

        total_hour += c

    record["OC_TOTAL"] = total_hour

    hourly_records.append(record)


hourly_df = pd.DataFrame(hourly_records)


# =============================================================================
# PRINT HOURLY TOTAL
# =============================================================================

print()

print(
    f"{'Index':<8}"
    f"{'Time':<30}"
    f"{'Mine CH4 (ppm)':>20}"
)

print("-" * 62)

for _, row in hourly_df.iterrows():

    print(
        f"{int(row['time_index']):<8}"
        f"{row['time']:<30}"
        f"{row['OC_TOTAL']:>20.8f}"
    )


# =============================================================================
# CHECK SUM OF HOURLY CONTRIBUTIONS
# =============================================================================

hourly_total = hourly_df["OC_TOTAL"].sum()

print("-" * 62)

print(
    f"{'SUM':<38}"
    f"{hourly_total:>20.8f}"
)

print()
print(
    "Direct integration total :",
    f"{total_contribution:.10f}",
    "ppm"
)

print(
    "Hourly integration total :",
    f"{hourly_total:.10f}",
    "ppm"
)

print(
    "Difference               :",
    f"{hourly_total - total_contribution:.3e}",
    "ppm"
)


# =============================================================================
# SAVE RESULTS
# =============================================================================

print()
print("=" * 80)
print("WRITING RESULTS")
print("=" * 80)

summary_file = os.path.join(
    OUTPUT_DIR,
    "Muswellbrook_20260115_2200AEST_mine_contributions.csv"
)

hourly_file = os.path.join(
    OUTPUT_DIR,
    "Muswellbrook_20260115_2200AEST_hourly_mine_contributions.csv"
)

ranked_file = os.path.join(
    OUTPUT_DIR,
    "Muswellbrook_20260115_2200AEST_mine_contributions_ranked.csv"
)

results_df.to_csv(
    summary_file,
    index=False
)

hourly_df.to_csv(
    hourly_file,
    index=False
)

ranked.to_csv(
    ranked_file,
    index=False
)

print()
print("Summary:")
print(summary_file)

print()
print("24-hour contributions:")
print(hourly_file)

print()
print("Ranked contributions:")
print(ranked_file)


# =============================================================================
# FINAL
# =============================================================================

print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)

print()
print(
    "Total OC1-OC13 contribution =",
    f"{total_contribution:.8f}",
    "ppm"
)

print()

emission_ds.close()
foot_ds.close()

