
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot daily accumulated STILT backward footprints for all three
Upper Hunter receptors:

    1. Singleton
    2. Muswellbrook
    3. Merriwa

The script:
- Reads all hourly footprint files for the selected UTC day.
- Accumulates the hourly footprints separately for each receptor.
- Plots the three daily footprints in a 1 x 3 panel figure.
- Overlays OC1-OC13 mine boundaries and labels.
- Overlays Natural Earth coastline.
- Marks each receptor location.
- Uses a separate logarithmic colour scale for each panel.
- Saves a high-resolution PNG figure.

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

# Month directory containing STILT footprints
MONTH_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/"
    "upperhunter/arl_mine_2026/2026-02"
)

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
FIGURE_SIZE = (21, 8)
FIGURE_DPI = 300

# Logarithmic colour scale percentile limits
VMIN_PERCENTILE = 5
VMAX_PERCENTILE = 99.5

# Use the same geographic extent for all three panels
COMMON_EXTENT = True

# Add mine labels
SHOW_MINE_LABELS = True


# ============================================================
# INITIAL SETTINGS
# ============================================================

date_obj = pd.to_datetime(SELECTED_DATE)
date_string = date_obj.strftime("%Y%m%d")

receptor_names = [
    "Singleton",
    "Muswellbrook",
    "Merriwa",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("DAILY STILT BACKWARD FOOTPRINTS - THREE RECEPTORS")
print("=" * 80)

print(f"Selected date: {SELECTED_DATE} UTC")
print()


# ============================================================
# FUNCTION: READ AND ACCUMULATE DAILY FOOTPRINT
# ============================================================

def read_daily_footprint(receptor):

    print("-" * 70)
    print(f"Processing receptor: {receptor}")
    print("-" * 70)

    footprint_dir = os.path.join(
        MONTH_DIR,
        "footprints",
        receptor,
        "footprints",
    )

    file_pattern = os.path.join(
        footprint_dir,
        f"{date_string}*_foot.nc",
    )

    footprint_files = sorted(
        glob.glob(file_pattern)
    )

    if not footprint_files:
        raise FileNotFoundError(
            f"No footprint files found for {receptor} "
            f"on {SELECTED_DATE} in {footprint_dir}"
        )

    print(f"Candidate files found: {len(footprint_files)}")

    footprint_list = []
    release_times = []

    reference_lon = None
    reference_lat = None

    for i, filename in enumerate(
        footprint_files,
        start=1,
    ):

        print(
            f"Reading {i:03d}/{len(footprint_files):03d}: "
            f"{os.path.basename(filename)}"
        )

        with xr.open_dataset(
            filename,
            decode_times=True,
        ) as ds:

            release_time = pd.Timestamp(
                ds["time"].values[0]
            )

            if release_time.tzinfo is None:
                release_time = release_time.tz_localize("UTC")
            else:
                release_time = release_time.tz_convert("UTC")

            if release_time.strftime("%Y%m%d") != date_string:
                warnings.warn(
                    f"Skipping unexpected release time: "
                    f"{release_time}"
                )
                continue

            release_times.append(release_time)

            lon = ds["lon"].values
            lat = ds["lat"].values

            if reference_lon is None:
                reference_lon = lon.copy()
                reference_lat = lat.copy()

            else:
                if (
                    not np.allclose(reference_lon, lon)
                    or not np.allclose(reference_lat, lat)
                ):
                    raise ValueError(
                        f"Grid mismatch detected for {receptor}: "
                        f"{filename}"
                    )

            foot = ds["foot"].isel(time=0).values.astype(float)

            fill_value = ds["foot"].attrs.get(
                "_FillValue",
                -1.0,
            )

            foot[~np.isfinite(foot)] = np.nan
            foot[foot == fill_value] = np.nan
            foot[foot <= 0] = np.nan

            footprint_list.append(foot)

    if not footprint_list:
        raise ValueError(
            f"No valid footprints found for {receptor} "
            f"on {SELECTED_DATE}"
        )

    # Stack hourly footprints
    footprint_stack = np.stack(
        footprint_list,
        axis=0,
    )

    # Accumulate hourly footprints
    daily_foot = np.nansum(
        footprint_stack,
        axis=0,
    )

    # Preserve missing grid cells as NaN
    valid_count = np.sum(
        np.isfinite(footprint_stack),
        axis=0,
    )

    daily_foot[valid_count == 0] = np.nan
    daily_foot[daily_foot <= 0] = np.nan

    valid_values = daily_foot[
        np.isfinite(daily_foot)
    ]

    if valid_values.size == 0:
        raise ValueError(
            f"Empty daily footprint for {receptor}"
        )

    print()
    print(f"Valid releases: {len(release_times)}")
    print(f"First release : {min(release_times)}")
    print(f"Last release  : {max(release_times)}")
    print(f"Grid size     : {daily_foot.shape}")
    print(
        f"Maximum value : {np.nanmax(valid_values):.6e}"
    )
    print()

    return {
        "footprint": daily_foot,
        "lon": reference_lon,
        "lat": reference_lat,
        "release_times": release_times,
        "n_releases": len(release_times),
    }


# ============================================================
# READ ALL THREE RECEPTORS
# ============================================================

daily_data = {}

for receptor in receptor_names:

    daily_data[receptor] = read_daily_footprint(
        receptor
    )


# ============================================================
# CHECK GRID CONSISTENCY
# ============================================================

print("=" * 80)
print("CHECKING RECEPTOR GRIDS")
print("=" * 80)

base_lon = daily_data["Singleton"]["lon"]
base_lat = daily_data["Singleton"]["lat"]

for receptor in receptor_names[1:]:

    lon = daily_data[receptor]["lon"]
    lat = daily_data[receptor]["lat"]

    if (
        not np.allclose(base_lon, lon)
        or not np.allclose(base_lat, lat)
    ):
        raise ValueError(
            f"The footprint grid for {receptor} differs "
            "from the Singleton grid."
        )

print("All three receptors have matching footprint grids.")


# ============================================================
# CHECK NUMBER OF HOURLY RELEASES
# ============================================================

print()
print("Number of valid hourly releases:")

for receptor in receptor_names:

    print(
        f"{receptor:15s}: "
        f"{daily_data[receptor]['n_releases']}"
    )

release_counts = [
    daily_data[r]["n_releases"]
    for r in receptor_names
]

if len(set(release_counts)) > 1:

    warnings.warn(
        "The three receptors have different numbers of "
        "valid hourly releases. Interpret their accumulated "
        "footprint magnitudes with caution."
    )

else:

    print("All three receptors have equal release counts.")


# ============================================================
# READ MINE SHAPEFILES
# ============================================================

print()
print("=" * 80)
print("READING MINE SHAPEFILES")
print("=" * 80)

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
            f"Missing CRS in {shp_file}"
        )

    mine_gdf = mine_gdf.to_crs(
        epsg=4326
    )

    mine_geometries.append(
        (mine_name, mine_gdf)
    )

    print(f"Loaded {mine_name}")


# ============================================================
# READ COASTLINE
# ============================================================

print()
print("Reading coastline...")

if not os.path.exists(COASTLINE_FILE):
    raise FileNotFoundError(
        f"Coastline shapefile not found: {COASTLINE_FILE}"
    )

coastline = gpd.read_file(
    COASTLINE_FILE
)

if coastline.crs is None:
    raise ValueError(
        "Coastline shapefile has no CRS information."
    )

coastline = coastline.to_crs(
    epsg=4326
)

print(
    f"Coastline features loaded: {len(coastline)}"
)


# ============================================================
# DETERMINE COMMON MAP EXTENT
# ============================================================

if COMMON_EXTENT:

    lon_min = np.nanmin(base_lon)
    lon_max = np.nanmax(base_lon)

    lat_min = np.nanmin(base_lat)
    lat_max = np.nanmax(base_lat)

else:

    lon_min = None
    lon_max = None
    lat_min = None
    lat_max = None


# ============================================================
# PLOT THREE DAILY FOOTPRINTS
# ============================================================

print()
print("=" * 80)
print("PLOTTING THREE RECEPTOR FOOTPRINTS")
print("=" * 80)

fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=FIGURE_SIZE,
    sharex=True,
    sharey=True,
)

for ax, receptor in zip(
    axes,
    receptor_names,
):

    print(f"Plotting {receptor}...")

    data = daily_data[receptor]

    daily_foot = data["footprint"]
    lon = data["lon"]
    lat = data["lat"]

    receptor_lon = RECEPTORS[receptor]["lon"]
    receptor_lat = RECEPTORS[receptor]["lat"]

    valid_values = daily_foot[
        np.isfinite(daily_foot)
    ]

    # Separate logarithmic scale for each receptor
    vmin = np.nanpercentile(
        valid_values,
        VMIN_PERCENTILE,
    )

    vmax = np.nanpercentile(
        valid_values,
        VMAX_PERCENTILE,
    )

    if vmin <= 0:
        vmin = np.nanmin(valid_values)

    if vmax <= vmin:
        vmax = np.nanmax(valid_values)

    if vmax <= vmin:
        raise ValueError(
            f"Invalid colour scale for {receptor}"
        )

    # --------------------------------------------------------
    # Footprint
    # --------------------------------------------------------

    mesh = ax.pcolormesh(
        lon,
        lat,
        daily_foot,
        shading="auto",
        cmap="turbo",
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax,
        ),
        zorder=1,
    )

    # --------------------------------------------------------
    # Coastline
    # --------------------------------------------------------

    coastline.plot(
        ax=ax,
        color="darkblue",
        linewidth=1.0,
        zorder=4,
    )

    # --------------------------------------------------------
    # Mine boundaries and labels
    # --------------------------------------------------------

    for mine_name, mine_gdf in mine_geometries:

        mine_gdf.boundary.plot(
            ax=ax,
            color="black",
            linewidth=1.5,
            zorder=5,
        )

        if SHOW_MINE_LABELS:

            for geometry in mine_gdf.geometry:

                if geometry is None or geometry.is_empty:
                    continue

                label_point = geometry.representative_point()

                ax.text(
                    label_point.x,
                    label_point.y,
                    mine_name,
                    fontsize=7,
                    fontweight="bold",
                    color="black",
                    ha="center",
                    va="center",
                    zorder=8,
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.65,
                        pad=1.2,
                    ),
                )

    # --------------------------------------------------------
    # Receptor location
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_title(
        f"{receptor}\n"
        f"{data['n_releases']} hourly releases",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax.set_xlabel(
        "Longitude (degrees east)",
        fontsize=10,
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.4,
    )

    if COMMON_EXTENT:

        ax.set_xlim(
            lon_min,
            lon_max,
        )

        ax.set_ylim(
            lat_min,
            lat_max,
        )

    # --------------------------------------------------------
    # Individual colourbar
    # --------------------------------------------------------

    cbar = fig.colorbar(
        mesh,
        ax=ax,
        pad=0.025,
        shrink=0.82,
    )

    cbar.set_label(
        "Accumulated footprint",
        fontsize=9,
    )


# ============================================================
# COMMON FIGURE LABELS AND LEGEND
# ============================================================

axes[0].set_ylabel(
    "Latitude (degrees north)",
    fontsize=11,
)

fig.suptitle(
    f"Daily STILT Backward Footprints - Upper Hunter\n"
    f"{SELECTED_DATE} UTC",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

legend_elements = [

    Line2D(
        [0], [0],
        marker="*",
        color="w",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=13,
        label="Receptor location",
    ),

    Line2D(
        [0], [0],
        color="black",
        linewidth=1.5,
        label="OC1-OC13 mine boundaries",
    ),

    Line2D(
        [0], [0],
        color="darkblue",
        linewidth=1.0,
        label="Coastline",
    ),
]

fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=3,
    fontsize=10,
    frameon=True,
    bbox_to_anchor=(0.5, 0.015),
)

plt.tight_layout(
    rect=[0, 0.08, 1, 0.91]
)


# ============================================================
# SAVE FIGURE
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    f"UpperHunter_{SELECTED_DATE}_"
    "daily_STILT_footprints_3receptors.png",
)

plt.savefig(
    output_file,
    dpi=FIGURE_DPI,
    bbox_inches="tight",
)

print()
print("=" * 80)
print("DAILY THREE-RECEPTOR FOOTPRINT PLOT COMPLETED")
print("=" * 80)

print(f"Output file: {output_file}")

plt.show()


