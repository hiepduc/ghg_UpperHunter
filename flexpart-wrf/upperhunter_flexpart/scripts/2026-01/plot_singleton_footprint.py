#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from netCDF4 import Dataset
import shapefile

# ============================================================
# FILES
# ============================================================

run_dir = os.path.expanduser(
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart/run/2026-01/hourly_obs_3layer/singleton/20260121_150000"
)

foot_file = os.path.join(
    run_dir, "flxout_d01_20260121_140000.nc"
)

header_file = os.path.join(
    run_dir, "header_d01.nc"
)

mine_dir = os.path.expanduser(
    "/mnt/scratch_lustre/nzem_meim_scratch/data/emission_location"
)

coast_file = os.path.expanduser(
    "/mnt/scratch_lustre/nzem_meim_scratch/data/coastline/ne_50m_coastline"
)

output_png = os.path.join(
    run_dir,
    "FLEXPART_WRF_Singleton_20260121_1400_coast.png"
)

# ============================================================
# RECEPTOR
# ============================================================

receptor_lon = 151.16132
receptor_lat = -32.57123

# ============================================================
# READ FLEXPART-WRF GRID
# ============================================================

with Dataset(header_file) as nc:

    lon = np.asarray(nc.variables["XLONG"][:])
    lat = np.asarray(nc.variables["XLAT"][:])

    lon_corner = np.asarray(
        nc.variables["XLONG_CORNER"][:]
    )

    lat_corner = np.asarray(
        nc.variables["XLAT_CORNER"][:]
    )

# ============================================================
# READ FOOTPRINT
# ============================================================

with Dataset(foot_file) as nc:

    conc = np.asarray(
        nc.variables["CONC"][0, 0, 0, :, :, :]
    )

    times = nc.variables["Times"][:]

    time_string = "".join(
        x.decode() if hasattr(x, "decode") else str(x)
        for x in times[0]
    )

# ============================================================
# CHECK DIMENSIONS
# ============================================================

print("Grid center shape :", lon.shape)
print("Grid corner shape :", lon_corner.shape)
print("CONC shape        :", conc.shape)
print("Output time       :", time_string)

# ============================================================
# COMBINE VERTICAL LEVELS
# ============================================================

foot = np.sum(conc, axis=0)

print()
print("Footprint statistics:")
print("  min       =", np.nanmin(foot))
print("  max       =", np.nanmax(foot))
print("  nonzero   =", np.count_nonzero(foot))

# ============================================================
# CONSTRUCT CELL EDGES
#
# XLONG_CORNER / XLAT_CORNER are the lower-left corners
# of the 147 x 165 cells.
#
# Therefore construct a 148 x 166 edge grid.
# ============================================================

ny, nx = lon_corner.shape

lon_edge = np.empty((ny + 1, nx + 1))
lat_edge = np.empty((ny + 1, nx + 1))

# Existing lower-left corners
lon_edge[:-1, :-1] = lon_corner
lat_edge[:-1, :-1] = lat_corner

# Last column: extrapolate eastward
lon_edge[:-1, -1] = (
    lon_corner[:, -1]
    + (lon_corner[:, -1] - lon_corner[:, -2])
)

lat_edge[:-1, -1] = (
    lat_corner[:, -1]
    + (lat_corner[:, -1] - lat_corner[:, -2])
)

# Last row: extrapolate northward
lon_edge[-1, :-1] = (
    lon_corner[-1, :]
    + (lon_corner[-1, :] - lon_corner[-2, :])
)

lat_edge[-1, :-1] = (
    lat_corner[-1, :]
    + (lat_corner[-1, :] - lat_corner[-2, :])
)

# Upper-right corner
lon_edge[-1, -1] = (
    lon_corner[-1, -1]
    + (lon_corner[-1, -1] - lon_corner[-2, -1])
    + (lon_corner[-1, -1] - lon_corner[-1, -2])
    - (lon_corner[-2, -1] - lon_corner[-2, -2])
)

lat_edge[-1, -1] = (
    lat_corner[-1, -1]
    + (lat_corner[-1, -1] - lat_corner[-2, -1])
    + (lat_corner[-1, -1] - lat_corner[-1, -2])
    - (lat_corner[-2, -1] - lat_corner[-2, -2])
)

print()
print("Edge grid shape   :", lon_edge.shape)

# ============================================================
# LOG-SCALE RANGE
# ============================================================

positive = foot[foot > 0]

if len(positive) == 0:
    raise RuntimeError(
        "No positive FLEXPART-WRF footprint values found."
    )

vmin = np.percentile(positive, 5)
vmax = np.percentile(positive, 99.5)

if vmin <= 0:
    vmin = positive.min()

if vmax <= vmin:
    vmax = positive.max()

print()
print("Log plot range:")
print("  vmin =", vmin)
print("  vmax =", vmax)

foot_plot = np.ma.masked_where(
    foot <= 0,
    foot
)

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 9)
)

# FLEXPART-WRF footprint
pcm = ax.pcolormesh(
    lon_edge,
    lat_edge,
    foot_plot,
    shading="flat",
    norm=LogNorm(
        vmin=vmin,
        vmax=vmax
    ),
    zorder=1
)

# ============================================================
# COASTLINE
# ============================================================

print()
print("Reading coastline...")

try:

    coast = shapefile.Reader(coast_file)

    for shape in coast.shapes():

        points = np.asarray(shape.points)

        if len(points) < 2:
            continue

        parts = list(shape.parts) + [len(points)]

        for p in range(len(parts) - 1):

            start = parts[p]
            end = parts[p + 1]

            xy = points[start:end]

            if len(xy) < 2:
                continue

            # Only plot segments near the FLEXPART domain
            if (
                np.max(xy[:, 0]) < np.nanmin(lon) - 0.2
                or np.min(xy[:, 0]) > np.nanmax(lon) + 0.2
                or np.max(xy[:, 1]) < np.nanmin(lat) - 0.2
                or np.min(xy[:, 1]) > np.nanmax(lat) + 0.2
            ):
                continue

            ax.plot(
                xy[:, 0],
                xy[:, 1],
                linewidth=1.2,
                linestyle="-",
                zorder=5
            )

    print("Coastline plotted.")

except Exception as e:

    print("WARNING: could not read coastline:")
    print(e)

# ============================================================
# MINE POLYGONS
# ============================================================

print()
print("Reading mine polygons...")

for i in range(1, 14):

    shp_base = os.path.join(
        mine_dir,
        f"OC{i}"
    )

    shp_file = shp_base + ".shp"

    if not os.path.exists(shp_file):

        print("WARNING: missing", shp_file)

        continue

    try:

        sf = shapefile.Reader(shp_base)

    except Exception as e:

        print("WARNING: cannot read", shp_file)
        print(e)

        continue

    print(
        f"Reading OC{i}: "
        f"{len(sf.shapes())} polygon(s)"
    )

    for shape in sf.shapes():

        points = np.asarray(
            shape.points
        )

        if len(points) < 2:
            continue

        parts = list(shape.parts) + [
            len(points)
        ]

        for p in range(len(parts) - 1):

            start = parts[p]
            end = parts[p + 1]

            xy = points[start:end]

            if len(xy) < 2:
                continue

            ax.plot(
                xy[:, 0],
                xy[:, 1],
                linewidth=1.5,
                linestyle="-",
                zorder=10
            )

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    shapes = sf.shapes()

    if len(shapes) > 0:

        pts = np.asarray(
            shapes[0].points
        )

        if len(pts) > 0:

            cx = np.mean(pts[:, 0])
            cy = np.mean(pts[:, 1])

            ax.text(
                cx,
                cy,
                f"OC{i}",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=20
            )

# ============================================================
# SINGLETON RECEPTOR
# ============================================================

ax.plot(
    receptor_lon,
    receptor_lat,
    marker="*",
    markersize=18,
    markeredgecolor="black",
    markerfacecolor="red",
    zorder=30
)

ax.text(
    receptor_lon + 0.015,
    receptor_lat + 0.015,
    "Singleton",
    fontsize=12,
    fontweight="bold",
    zorder=30
)

# ============================================================
# DOMAIN
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
# LABELS
# ============================================================

ax.set_xlabel(
    "Longitude (°E)",
    fontsize=12
)

ax.set_ylabel(
    "Latitude (°N)",
    fontsize=12
)

ax.set_title(
    "FLEXPART-WRF Backward Footprint — Singleton\n"
    f"{time_string} UTC | 24 h backward",
    fontsize=14
)

cbar = fig.colorbar(
    pcm,
    ax=ax,
    pad=0.02
)

cbar.set_label(
    "CONC (s m$^3$ kg$^{-1}$)",
    fontsize=11
)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.4,
    zorder=2
)

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

plt.savefig(
    output_png,
    dpi=250,
    bbox_inches="tight"
)

print()
print("Saved:")
print(output_png)

plt.show()

