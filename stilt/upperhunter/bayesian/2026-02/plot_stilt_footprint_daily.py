
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot an accumulated daily STILT backward footprint for a selected receptor.

The script:
1. Reads all hourly STILT footprint files for the selected UTC day.
2. Sums the hourly footprints to produce an accumulated daily footprint.
3. Overlays OC1-OC13 open-cut mine boundaries and labels.
4. Overlays the Natural Earth coastline.
5. Marks the receptor location.
6. Saves a high-resolution PNG figure.

STILT release times are in UTC.
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D


# ============================================================
# USER SETTINGS
# ============================================================

# Month directory containing the STILT footprints
MONTH_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/"
    "upperhunter/arl_mine_2026/2026-02"
)

# Select receptor:
# "Singleton", "Muswellbrook", or "Merriwa"
#RECEPTOR = "Singleton"
RECEPTOR = "Muswellbrook"

# Selected day (UTC)
SELECTED_DATE = "2026-02-01"

# Receptor coordinates
RECEPTORS = {
    "Singleton": {
        "lon": 151.16132,
        "lat": -32.57123,
    },
    "Muswellbrook": {
        "lon": 150.89297,
        "lat": -32.26202,
    },
    "Merriwa": {
        "lon": 150.35767,
        "lat": -32.14097,
    },
}

# Mine shapefile directory
MINE_SHP_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/"
    "emission_location"
)

# Coastline shapefile
COASTLINE_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/"
    "coastline/ne_50m_coastline.shp"
)

# Output directory
OUTPUT_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/"
    "upperhunter/bayesian/2026-02/"
    "STILT_footprint_plots/daily"
)

# Figure settings
FIGURE_SIZE = (12, 10)
FIGURE_DPI = 300

# Logarithmic colour scale percentile limits
VMIN_PERCENTILE = 5
VMAX_PERCENTILE = 99.5


# ============================================================
# CHECK USER SETTINGS
# ============================================================

if RECEPTOR not in RECEPTORS:
    raise ValueError(
        "Invalid receptor. Choose Singleton, Muswellbrook or Merriwa."
    )

date_obj = pd.to_datetime(SELECTED_DATE)
date_string = date_obj.strftime("%Y%m%d")

receptor_lon = RECEPTORS[RECEPTOR]["lon"]
receptor_lat = RECEPTORS[RECEPTOR]["lat"]

FOOTPRINT_DIR = os.path.join(
    MONTH_DIR,
    "footprints",
    RECEPTOR,
    "footprints",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("DAILY STILT BACKWARD FOOTPRINT")
print("=" * 70)

print(f"Selected date : {SELECTED_DATE} UTC")
print(f"Receptor      : {RECEPTOR}")
print(f"Longitude     : {receptor_lon}")
print(f"Latitude      : {receptor_lat}")
print(f"Input folder  : {FOOTPRINT_DIR}")
print()


# ============================================================
# FIND FOOTPRINT FILES FOR THE SELECTED DAY
# ============================================================

file_pattern = os.path.join(
    FOOTPRINT_DIR,
    f"{date_string}*_foot.nc",
)

footprint_files = sorted(glob.glob(file_pattern))

if not footprint_files:
    raise FileNotFoundError(
        f"No footprint files found for {SELECTED_DATE} "
        f"in {FOOTPRINT_DIR}"
    )

print(f"Number of candidate files found: {len(footprint_files)}")
print()


# ============================================================
# READ AND ACCUMULATE DAILY FOOTPRINTS
# ============================================================

daily_footprints = []

reference_lon = None
reference_lat = None

release_times = []

for i, filename in enumerate(footprint_files, start=1):

    print(
        f"Reading {i:03d}/{len(footprint_files):03d}: "
        f"{os.path.basename(filename)}"
    )

    with xr.open_dataset(filename, decode_times=True) as ds:

        # Read release time
        release_time = pd.Timestamp(ds["time"].values[0])

        if release_time.tzinfo is None:
            release_time = release_time.tz_localize("UTC")
        else:
            release_time = release_time.tz_convert("UTC")

        # Retain only files whose release time belongs to the
        # selected UTC calendar day.
        if release_time.strftime("%Y%m%d") != date_string:
            warnings.warn(
                f"Skipping file with unexpected release time: "
                f"{release_time}"
            )
            continue

        release_times.append(release_time)

        # Read coordinates
        lon = ds["lon"].values
        lat = ds["lat"].values

        # Check that all files use the same grid
        if reference_lon is None:
            reference_lon = lon.copy()
            reference_lat = lat.copy()
        else:
            if (
                not np.allclose(reference_lon, lon)
                or not np.allclose(reference_lat, lat)
            ):
                raise ValueError(
                    f"Grid mismatch detected in {filename}"
                )

        # Read footprint
        foot = ds["foot"].isel(time=0).values.astype(float)

        # Remove invalid values
        fill_value = ds["foot"].attrs.get("_FillValue", -1.0)

        foot[~np.isfinite(foot)] = np.nan
        foot[foot == fill_value] = np.nan
        foot[foot <= 0] = np.nan

        daily_footprints.append(foot)


if not daily_footprints:
    raise ValueError(
        f"No valid footprint files were available for "
        f"{SELECTED_DATE} UTC."
    )


# ============================================================
# CALCULATE ACCUMULATED DAILY FOOTPRINT
# ============================================================

print()
print("=" * 70)
print("ACCUMULATING DAILY FOOTPRINTS")
print("=" * 70)

footprint_stack = np.stack(daily_footprints, axis=0)

# Sum all hourly footprints.
# Locations with no valid footprint contribution are set to NaN.
daily_foot = np.nansum(footprint_stack, axis=0)
daily_foot[np.sum(np.isfinite(footprint_stack), axis=0) == 0] = np.nan

# Remove zero and negative values
daily_foot[daily_foot <= 0] = np.nan

valid_values = daily_foot[np.isfinite(daily_foot)]

if valid_values.size == 0:
    raise ValueError("The accumulated daily footprint is empty.")

print(f"Valid hourly footprints : {len(daily_footprints)}")
print(f"First release time      : {min(release_times)}")
print(f"Last release time       : {max(release_times)}")
print(f"Grid dimensions         : {daily_foot.shape}")
print(f"Minimum positive value  : {np.nanmin(valid_values):.6e}")
print(f"Maximum value           : {np.nanmax(valid_values):.6e}")
print()


# ============================================================
# DETERMINE LOGARITHMIC COLOUR SCALE
# ============================================================

vmin = np.nanpercentile(valid_values, VMIN_PERCENTILE)
vmax = np.nanpercentile(valid_values, VMAX_PERCENTILE)

if vmin <= 0:
    vmin = np.nanmin(valid_values)

if vmax <= vmin:
    vmax = np.nanmax(valid_values)

if vmax <= vmin:
    raise ValueError(
        "Unable to establish a valid logarithmic colour scale."
    )

print(f"Colour scale minimum: {vmin:.6e}")
print(f"Colour scale maximum: {vmax:.6e}")
print()


# ============================================================
# READ MINE SHAPEFILES
# ============================================================

print("=" * 70)
print("READING OPEN-CUT MINE SHAPEFILES")
print("=" * 70)

mine_geometries = []

for mine_number in range(1, 14):

    mine_name = f"OC{mine_number}"

    shp_file = os.path.join(
        MINE_SHP_DIR,
        f"{mine_name}.shp",
    )

    if not os.path.exists(shp_file):
        warnings.warn(
            f"Shapefile not found: {shp_file}"
        )
        continue

    mine_gdf = gpd.read_file(shp_file)

    if mine_gdf.empty:
        warnings.warn(
            f"No geometry found in {shp_file}"
        )
        continue

    if mine_gdf.crs is None:
        raise ValueError(
            f"Missing CRS information in {shp_file}"
        )

    mine_gdf = mine_gdf.to_crs(epsg=4326)

    mine_geometries.append(
        (mine_name, mine_gdf)
    )

    print(f"Loaded {mine_name}")


# ============================================================
# READ COASTLINE
# ============================================================

print()
print("Reading coastline shapefile...")

if not os.path.exists(COASTLINE_FILE):
    raise FileNotFoundError(
        f"Coastline shapefile not found: {COASTLINE_FILE}"
    )

coastline = gpd.read_file(COASTLINE_FILE)

if coastline.crs is None:
    raise ValueError(
        "Coastline shapefile does not have CRS information."
    )

coastline = coastline.to_crs(epsg=4326)

print(f"Coastline features loaded: {len(coastline)}")


# ============================================================
# PLOT DAILY FOOTPRINT
# ============================================================

print()
print("=" * 70)
print("PLOTTING DAILY FOOTPRINT")
print("=" * 70)

fig, ax = plt.subplots(
    figsize=FIGURE_SIZE
)

# Daily accumulated footprint
mesh = ax.pcolormesh(
    reference_lon,
    reference_lat,
    daily_foot,
    shading="auto",
    cmap="turbo",
    norm=LogNorm(
        vmin=vmin,
        vmax=vmax,
    ),
    zorder=1,
)

# Coastline
coastline.plot(
    ax=ax,
    color="darkblue",
    linewidth=1.0,
    zorder=4,
)

# Mine boundaries and labels
for mine_name, mine_gdf in mine_geometries:

    mine_gdf.boundary.plot(
        ax=ax,
        color="black",
        linewidth=1.5,
        zorder=5,
    )

    for geometry in mine_gdf.geometry:

        if geometry is None or geometry.is_empty:
            continue

        label_point = geometry.representative_point()

        ax.text(
            label_point.x,
            label_point.y,
            mine_name,
            fontsize=8,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            zorder=8,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.65,
                pad=1.5,
            ),
        )


# Receptor location
ax.scatter(
    receptor_lon,
    receptor_lat,
    marker="*",
    s=220,
    color="red",
    edgecolor="black",
    linewidth=0.8,
    zorder=12,
)


# ============================================================
# AXES AND COLOURBAR
# ============================================================

ax.set_xlim(
    np.nanmin(reference_lon),
    np.nanmax(reference_lon),
)

ax.set_ylim(
    np.nanmin(reference_lat),
    np.nanmax(reference_lat),
)

ax.set_xlabel(
    "Longitude (degrees east)",
    fontsize=12,
)

ax.set_ylabel(
    "Latitude (degrees north)",
    fontsize=12,
)

ax.set_title(
    f"Daily STILT Backward Footprint - {RECEPTOR}\n"
    f"{SELECTED_DATE} UTC "
    f"({len(daily_footprints)} releases accumulated)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.4,
)

# Colourbar
cbar = fig.colorbar(
    mesh,
    ax=ax,
    pad=0.02,
    shrink=0.85,
)

cbar.set_label(
    "Accumulated STILT footprint",
    fontsize=11,
)


# ============================================================
# LEGEND
# ============================================================

legend_elements = [
    Line2D(
        [0], [0],
        marker="*",
        color="w",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=13,
        label=f"{RECEPTOR} receptor",
    ),

    Line2D(
        [0], [0],
        color="black",
        linewidth=1.5,
        label="Open-cut mine boundaries",
    ),

    Line2D(
        [0], [0],
        color="darkblue",
        linewidth=1.0,
        label="Coastline",
    ),
]

ax.legend(
    handles=legend_elements,
    loc="upper right",
    fontsize=9,
    frameon=True,
    framealpha=0.9,
)


# ============================================================
# SAVE FIGURE
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    f"{RECEPTOR}_{SELECTED_DATE}_"
    "daily_STILT_footprint_mines_coastline.png",
)

plt.tight_layout()

plt.savefig(
    output_file,
    dpi=FIGURE_DPI,
    bbox_inches="tight",
)

print()
print("=" * 70)
print("DAILY FOOTPRINT PLOT COMPLETED")
print("=" * 70)

print(f"Output file: {output_file}")

plt.show()


