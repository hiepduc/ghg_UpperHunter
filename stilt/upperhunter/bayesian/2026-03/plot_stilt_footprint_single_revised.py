
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot an individual hourly STILT surface influence footprint
with Upper Hunter open-cut mine boundaries and coastline.

Input:
    STILT footprint NetCDF file
    OC1.shp through OC13.shp
    Natural Earth coastline shapefile

Output:
    High-resolution PNG map

Footprint:
    foot(time, lat, lon)

All vector layers are transformed to EPSG:4326.
"""

import os
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

# STILT footprint NetCDF file
INPUT_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/"
    "upperhunter/arl_mine_2026/2026-03/footprints/"
    "Merriwa/footprints/"
    "202603010100_150.35767_-32.14097_5_foot.nc"
)

# Mine shapefile directory
MINE_SHAPEFILE_DIR = (
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
    "upperhunter/bayesian/2026-03/"
    "STILT_footprint_plots"
)

# Receptor information
RECEPTOR_NAME = "Merriwa"
RECEPTOR_LON = 150.35767
RECEPTOR_LAT = -32.14097

# Mine settings
MINE_IDS = range(1, 14)

MINE_EDGE_COLOR = "black"
MINE_LINEWIDTH = 1.5
MINE_LABEL_COLOR = "black"
MINE_LABEL_SIZE = 9

# Coastline settings
COASTLINE_COLOR = "darkblue"
COASTLINE_LINEWIDTH = 1.0

# Footprint colour scale
VMIN = None
VMAX = None

VMIN_PERCENTILE = 1.0
VMAX_PERCENTILE = 99.5

# Figure settings
FIGURE_WIDTH = 12
FIGURE_HEIGHT = 10
FIGURE_DPI = 300


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CHECK INPUT FILES
# ============================================================

if not os.path.isfile(INPUT_FILE):
    raise FileNotFoundError(
        f"STILT footprint file not found:\n{INPUT_FILE}"
    )

if not os.path.isfile(COASTLINE_FILE):
    raise FileNotFoundError(
        f"Coastline shapefile not found:\n{COASTLINE_FILE}"
    )


# ============================================================
# READ STILT NETCDF
# ============================================================

print("=" * 70)
print("READING STILT FOOTPRINT")
print("=" * 70)

print("Input file:")
print(INPUT_FILE)

ds = xr.open_dataset(
    INPUT_FILE,
    decode_times=True
)

print("\nNetCDF dataset:")
print(ds)


# ============================================================
# READ FOOTPRINT AND COORDINATES
# ============================================================

if "foot" not in ds.variables:
    raise KeyError("Variable 'foot' not found.")

if "lon" not in ds.variables or "lat" not in ds.variables:
    raise KeyError("Longitude or latitude coordinate is missing.")

if "time" not in ds.variables:
    raise KeyError("Time coordinate is missing.")

foot_da = ds["foot"]

foot = foot_da.isel(time=0).values

lon = ds["lon"].values
lat = ds["lat"].values

if foot.ndim != 2:
    raise ValueError(
        f"Expected a 2D footprint, received {foot.shape}"
    )

print("\nFootprint dimensions:", foot.shape)

print(
    f"Longitude range: {np.nanmin(lon):.5f} "
    f"to {np.nanmax(lon):.5f}"
)

print(
    f"Latitude range: {np.nanmin(lat):.5f} "
    f"to {np.nanmax(lat):.5f}"
)


# ============================================================
# HANDLE INVALID AND FILL VALUES
# ============================================================

fill_value = foot_da.attrs.get("_FillValue", -1.0)

foot = np.asarray(foot, dtype=float)

foot[~np.isfinite(foot)] = np.nan
foot[foot <= 0] = np.nan
foot[foot == fill_value] = np.nan

valid_values = foot[np.isfinite(foot)]

if valid_values.size == 0:
    ds.close()
    raise ValueError(
        "No valid positive footprint values were found."
    )

print("\nFootprint statistics:")
print(f"Minimum: {valid_values.min():.6e}")
print(f"Maximum: {valid_values.max():.6e}")
print(f"Mean:    {valid_values.mean():.6e}")
print(f"Median:  {np.median(valid_values):.6e}")
print(f"Valid cells: {valid_values.size}")


# ============================================================
# RELEASE TIME
# ============================================================

release_time = pd.Timestamp(ds["time"].values[0])

if release_time.tzinfo is None:
    release_time = release_time.tz_localize("UTC")
else:
    release_time = release_time.tz_convert("UTC")

release_string = release_time.strftime(
    "%Y-%m-%d %H:%M UTC"
)

print("\nRelease time:", release_string)


# ============================================================
# DETERMINE COLOUR SCALE
# ============================================================

if VMIN is None:
    vmin = np.nanpercentile(
        valid_values,
        VMIN_PERCENTILE
    )
else:
    vmin = VMIN

if VMAX is None:
    vmax = np.nanpercentile(
        valid_values,
        VMAX_PERCENTILE
    )
else:
    vmax = VMAX

if vmin <= 0:
    vmin = valid_values.min()

if vmax <= vmin:
    vmax = valid_values.max()

if vmax <= vmin:
    ds.close()
    raise ValueError("Invalid logarithmic colour scale.")

print("\nColour scale:")
print(f"Minimum: {vmin:.6e}")
print(f"Maximum: {vmax:.6e}")


# ============================================================
# READ MINE SHAPEFILES
# ============================================================

print("\n" + "=" * 70)
print("READING UPPER HUNTER MINE SHAPEFILES")
print("=" * 70)

mine_data = {}

for mine_number in MINE_IDS:

    mine_name = f"OC{mine_number}"

    shapefile = os.path.join(
        MINE_SHAPEFILE_DIR,
        f"{mine_name}.shp"
    )

    if not os.path.isfile(shapefile):
        warnings.warn(
            f"Shapefile not found: {shapefile}"
        )
        continue

    print(f"Reading {mine_name}")

    gdf = gpd.read_file(shapefile)

    if gdf.empty:
        warnings.warn(
            f"{mine_name} shapefile contains no features."
        )
        continue

    if gdf.crs is None:
        raise ValueError(
            f"{mine_name} has no defined CRS."
        )

    gdf = gdf.to_crs(epsg=4326)

    gdf = gdf[
        gdf.geometry.notna() &
        ~gdf.geometry.is_empty
    ].copy()

    if gdf.empty:
        continue

    mine_data[mine_name] = gdf

    print(
        f"  Features: {len(gdf)}"
    )

    print(
        f"  Geometry types: "
        f"{', '.join(gdf.geom_type.unique())}"
    )

print(
    f"\nSuccessfully loaded "
    f"{len(mine_data)} of 13 mine shapefiles."
)

if len(mine_data) == 0:
    ds.close()
    raise RuntimeError(
        "No mine shapefiles were successfully loaded."
    )


# ============================================================
# READ NATURAL EARTH COASTLINE
# ============================================================

print("\n" + "=" * 70)
print("READING NATURAL EARTH COASTLINE")
print("=" * 70)

coastline = gpd.read_file(
    COASTLINE_FILE
)

if coastline.empty:
    ds.close()
    raise ValueError(
        "Coastline shapefile contains no features."
    )

if coastline.crs is None:
    ds.close()
    raise ValueError(
        "Coastline shapefile has no defined CRS."
    )

print("Original coastline CRS:", coastline.crs)

# Transform coastline to geographic coordinates.
coastline = coastline.to_crs(epsg=4326)

print("Transformed coastline CRS:", coastline.crs)
print("Coastline features:", len(coastline))


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
)


# ============================================================
# PLOT STILT FOOTPRINT
# ============================================================

mesh = ax.pcolormesh(
    lon,
    lat,
    foot,
    shading="auto",
    cmap="turbo",
    norm=LogNorm(
        vmin=vmin,
        vmax=vmax
    ),
    rasterized=True,
    zorder=1
)


# ============================================================
# PLOT COASTLINE
# ============================================================

print("\nPlotting coastline")

coastline.plot(
    ax=ax,
    color=COASTLINE_COLOR,
    linewidth=COASTLINE_LINEWIDTH,
    zorder=4
)


# ============================================================
# PLOT MINE BOUNDARIES AND LABELS
# ============================================================

print("\nPlotting mine boundaries")

for mine_name, gdf in mine_data.items():

    # Polygon boundaries
    polygon_mask = gdf.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    )

    if polygon_mask.any():

        gdf.loc[polygon_mask].boundary.plot(
            ax=ax,
            color=MINE_EDGE_COLOR,
            linewidth=MINE_LINEWIDTH,
            zorder=5
        )

    # Point geometries
    point_mask = gdf.geom_type.isin(
        ["Point", "MultiPoint"]
    )

    if point_mask.any():

        gdf.loc[point_mask].plot(
            ax=ax,
            color=MINE_EDGE_COLOR,
            markersize=35,
            marker="^",
            edgecolor="white",
            linewidth=0.5,
            zorder=7
        )

    # Line geometries
    line_mask = gdf.geom_type.isin(
        ["LineString", "MultiLineString"]
    )

    if line_mask.any():

        gdf.loc[line_mask].plot(
            ax=ax,
            color=MINE_EDGE_COLOR,
            linewidth=MINE_LINEWIDTH,
            zorder=5
        )

    # Mine labels
    for geometry in gdf.geometry:

        label_point = geometry.representative_point()

        ax.annotate(
            mine_name,
            xy=(
                label_point.x,
                label_point.y
            ),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=MINE_LABEL_SIZE,
            fontweight="bold",
            color=MINE_LABEL_COLOR,
            ha="left",
            va="bottom",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.7,
                pad=1.2
            ),
            zorder=9
        )

    print(f"  Plotted {mine_name}")


# ============================================================
# PLOT RECEPTOR LOCATION
# ============================================================

ax.plot(
    RECEPTOR_LON,
    RECEPTOR_LAT,
    marker="*",
    color="red",
    markersize=18,
    markeredgecolor="black",
    markeredgewidth=0.9,
    linestyle="none",
    label=f"{RECEPTOR_NAME} receptor",
    zorder=12
)


# ============================================================
# AXIS LABELS AND TITLE
# ============================================================

ax.set_xlabel(
    "Longitude (degrees east)",
    fontsize=12
)

ax.set_ylabel(
    "Latitude (degrees north)",
    fontsize=12
)

ax.set_title(
    f"STILT Surface Influence Footprint\n"
    f"{RECEPTOR_NAME} | Release: {release_string}",
    fontsize=14,
    pad=15
)

ax.tick_params(
    axis="both",
    labelsize=10
)


# ============================================================
# GRID
# ============================================================

ax.grid(
    True,
    linestyle="--",
    linewidth=0.6,
    alpha=0.4,
    zorder=2
)


# ============================================================
# LEGEND
# ============================================================

legend_elements = [

    Line2D(
        [0],
        [0],
        marker="*",
        color="red",
        markeredgecolor="black",
        markersize=14,
        linestyle="none",
        label=f"{RECEPTOR_NAME} receptor"
    ),

    Line2D(
        [0],
        [0],
        color="black",
        linewidth=1.5,
        label="Open-cut mine boundaries"
    ),

    Line2D(
        [0],
        [0],
        color=COASTLINE_COLOR,
        linewidth=1.5,
        label="Coastline"
    )
]

ax.legend(
    handles=legend_elements,
    loc="upper right",
    fontsize=10,
    frameon=True
)


# ============================================================
# COLOUR BAR
# ============================================================

cbar = fig.colorbar(
    mesh,
    ax=ax,
    shrink=0.88,
    pad=0.03
)

cbar.set_label(
    "Surface influence footprint "
    "(ppm / (µmol m⁻² s))",
    fontsize=11
)

cbar.ax.tick_params(
    labelsize=9
)


# ============================================================
# MAP EXTENT
# ============================================================

ax.set_xlim(
    np.nanmin(lon),
    np.nanmax(lon)
)

ax.set_ylim(
    np.nanmin(lat),
    np.nanmax(lat)
)


# ============================================================
# SAVE FIGURE
# ============================================================

output_filename = os.path.basename(
    INPUT_FILE
).replace(
    "_foot.nc",
    "_footprint_with_mines_coastline.png"
)

output_file = os.path.join(
    OUTPUT_DIR,
    output_filename
)

plt.tight_layout()

plt.savefig(
    output_file,
    dpi=FIGURE_DPI,
    bbox_inches="tight"
)

print("\n" + "=" * 70)
print("STILT FOOTPRINT WITH MINES AND COASTLINE COMPLETED")
print("=" * 70)

print("Mine shapefiles plotted:", len(mine_data))
print("Coastline features:", len(coastline))
print("Output directory:", OUTPUT_DIR)
print("Output figure:", output_file)


# ============================================================
# DISPLAY FIGURE
# ============================================================

plt.show()

ds.close()


