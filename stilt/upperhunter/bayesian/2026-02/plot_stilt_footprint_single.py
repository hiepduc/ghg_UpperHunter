
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot an individual hourly STILT surface influence footprint.

Input:
    STILT footprint NetCDF file

NetCDF variables:
    lon(lon)              Longitude at cell centres
    lat(lat)              Latitude at cell centres
    time(time)            UTC release time
    foot(time, lat, lon)  STILT surface influence footprint

Output:
    High-resolution PNG footprint map

The footprint is plotted using a logarithmic colour scale.
The receptor location is marked with a red star.
"""

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/"
    "upperhunter/arl_mine_2026/2026-02/footprints/"
    "Merriwa/footprints/"
    "202602010100_150.35767_-32.14097_5_foot.nc"
)

OUTPUT_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/"
    "upperhunter/bayesian/2026-02/"
    "STILT_footprint_plots"
)

RECEPTOR_NAME = "Merriwa"

RECEPTOR_LON = 150.35767
RECEPTOR_LAT = -32.14097

# Colour scale limits.
# None means automatically determined from the footprint data.
VMIN = None
VMAX = None

# Percentiles used for automatic colour limits.
VMIN_PERCENTILE = 1.0
VMAX_PERCENTILE = 99.5

# Figure settings
FIGURE_WIDTH = 11
FIGURE_HEIGHT = 9
FIGURE_DPI = 300


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not os.path.isfile(INPUT_FILE):
    raise FileNotFoundError(
        f"STILT footprint file not found:\n{INPUT_FILE}"
    )


# ============================================================
# READ NETCDF FILE
# ============================================================

print("=" * 65)
print("READING STILT FOOTPRINT")
print("=" * 65)

print("Input file:")
print(INPUT_FILE)

# Decode the NetCDF time coordinate automatically.
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
    raise KeyError("Variable 'foot' was not found in NetCDF file.")

if "lon" not in ds.variables or "lat" not in ds.variables:
    raise KeyError("Longitude or latitude coordinate is missing.")

if "time" not in ds.variables:
    raise KeyError("Time coordinate is missing.")

foot_da = ds["foot"]

# Select the first time record.
foot = foot_da.isel(time=0).values

lon = ds["lon"].values
lat = ds["lat"].values

# Ensure footprint is two-dimensional.
if foot.ndim != 2:
    raise ValueError(
        f"Expected a 2D footprint after selecting time, "
        f"but received shape {foot.shape}"
    )

print("\nFootprint dimensions:", foot.shape)

print(
    "Longitude range: "
    f"{np.nanmin(lon):.5f} to {np.nanmax(lon):.5f}"
)

print(
    "Latitude range: "
    f"{np.nanmin(lat):.5f} to {np.nanmax(lat):.5f}"
)


# ============================================================
# HANDLE INVALID AND FILL VALUES
# ============================================================

fill_value = foot_da.attrs.get("_FillValue", -1.0)

# Convert to floating-point array.
foot = np.asarray(foot, dtype=float)

# Exclude missing, negative, zero and fill values.
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
print(f"Valid grid cells: {valid_values.size}")


# ============================================================
# READ RELEASE TIME
# ============================================================

# Xarray decodes the NetCDF time coordinate.
# Pandas converts it into a Python datetime object.

release_time = pd.Timestamp(ds["time"].values[0])

# Convert to UTC for display.
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

# Ensure valid positive limits for LogNorm.
if vmin <= 0:
    vmin = valid_values.min()

if vmax <= vmin:
    vmax = valid_values.max()

if vmax <= vmin:
    ds.close()
    raise ValueError(
        "Unable to determine valid logarithmic colour limits."
    )

print("\nColour scale:")
print(f"Minimum: {vmin:.6e}")
print(f"Maximum: {vmax:.6e}")


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
)


# ============================================================
# PLOT FOOTPRINT
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
    rasterized=True
)


# ============================================================
# PLOT RECEPTOR LOCATION
# ============================================================

ax.plot(
    RECEPTOR_LON,
    RECEPTOR_LAT,
    marker="*",
    color="red",
    markersize=17,
    markeredgecolor="black",
    markeredgewidth=0.9,
    linestyle="none",
    label=f"{RECEPTOR_NAME} receptor",
    zorder=10
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
# GRID AND LEGEND
# ============================================================

ax.grid(
    True,
    linestyle="--",
    linewidth=0.6,
    alpha=0.4
)

ax.legend(
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
# PLOT EXTENT
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
    "_footprint.png"
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

print("\n" + "=" * 65)
print("FOOTPRINT PLOT COMPLETED")
print("=" * 65)

print("Output directory:")
print(OUTPUT_DIR)

print("\nOutput figure:")
print(output_file)


# ============================================================
# DISPLAY FIGURE
# ============================================================

plt.show()

# ============================================================
# CLOSE DATASET
# ============================================================

ds.close()

