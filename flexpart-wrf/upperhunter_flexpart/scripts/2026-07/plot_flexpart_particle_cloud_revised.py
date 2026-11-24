
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot FLEXPART-WRF particle-position clouds from partposit_* files.

For the current FLEXPART-WRF output:
    x,y are metres on the WRF d03 grid
    dx = dy = 1000 m
    grid = 165 x 147

The particle coordinates are converted to lon/lat using bilinear
interpolation of XLONG/XLAT from header_d01.nc.

The plot shows all available particle clouds at selected backward times.
The receptor simulation date/time is obtained from the RUN_DIR name.
"""

from pathlib import Path
from datetime import datetime

import struct
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import shapefile


# ============================================================
# USER SETTINGS
# ============================================================

RUN_DIR = Path(
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/"
    "upperhunter_flexpart/run/2026-07/hourly_obs_3layer/"
    "singleton/20260716_180000"
)

HEADER = RUN_DIR / "header_d01.nc"

MINE_DIR = Path(
    "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter"
)

OUTPUT_PNG = RUN_DIR / "FLEXPART_particle_cloud_Singleton.png"


# ------------------------------------------------------------
# Particle files to plot
#
# The timestamp is extracted automatically from each filename.
# The backward time is calculated relative to the receptor time.
# ------------------------------------------------------------

PARTICLE_FILES = [
    RUN_DIR / "partposit_20260716140000",
    RUN_DIR / "partposit_20260716150000",
    RUN_DIR / "partposit_20260716160000",
    RUN_DIR / "partposit_20260716170000",
]


RECEPTOR_LON = 151.16132
RECEPTOR_LAT = -32.57123

DX = 1000.0
DY = 1000.0


# ------------------------------------------------------------
# Particle record format for nspec = 1
#
# integer*4 npoint
# real*4    x
# real*4    y
# real*4    z
# integer*4 itramem
# real*4    topo
# real*4    pvi
# real*4    qvi
# real*4    rhoi
# real*4    hmixi
# real*4    tri
# real*4    tti
# real*4    xmass1
#
# = 2 integers + 11 real*4 = 52 bytes
# ------------------------------------------------------------

HEADER_SIZE = 12
RECORD_SIZE = 52
RECORD_FMT = "<ifffi8f"


# ============================================================
# RECEPTOR DATE/TIME
# ============================================================

# Expected directory format: YYYYMMDD_HHMMSS

try:
    receptor_datetime = datetime.strptime(
        RUN_DIR.name,
        "%Y%m%d_%H%M%S"
    )

except ValueError:

    raise ValueError(
        "Cannot extract receptor date/time from RUN_DIR name: "
        f"{RUN_DIR.name}\n"
        "Expected format: YYYYMMDD_HHMMSS"
    )


print("=" * 75)
print("FLEXPART-WRF PARTICLE CLOUD PLOT")
print("=" * 75)

print("Run directory:", RUN_DIR)
print(
    "Receptor simulation time:",
    receptor_datetime.strftime("%d %B %Y %H:%M:%S")
)


# ============================================================
# READ HEADER GRID
# ============================================================

print("\nReading FLEXPART-WRF header:", HEADER)

if not HEADER.exists():
    raise FileNotFoundError(f"Header file not found: {HEADER}")

with netCDF4.Dataset(HEADER) as nc:

    xlong = np.asarray(
        nc.variables["XLONG"][:],
        dtype=float
    )

    xlat = np.asarray(
        nc.variables["XLAT"][:],
        dtype=float
    )

# Remove singleton dimensions if present

xlong = np.squeeze(xlong)
xlat = np.squeeze(xlat)

if xlong.ndim != 2 or xlat.ndim != 2:
    raise ValueError(
        f"Expected 2-D XLONG/XLAT arrays; "
        f"got {xlong.shape} and {xlat.shape}"
    )

if xlong.shape != xlat.shape:
    raise ValueError(
        f"XLONG and XLAT shapes differ: "
        f"{xlong.shape} and {xlat.shape}"
    )

print("Grid shape:", xlong.shape)

print(
    "Longitude range:",
    np.nanmin(xlong),
    "to",
    np.nanmax(xlong)
)

print(
    "Latitude range:",
    np.nanmin(xlat),
    "to",
    np.nanmax(xlat)
)


# ============================================================
# PARTICLE READER
# ============================================================

def read_particle_file(filename):

    filename = Path(filename)

    particles = []

    with open(filename, "rb") as f:

        # ----------------------------------------------------
        # First Fortran sequential-unformatted record
        #
        # int32 itime
        # int32 numpart
        # int32 outgrid_option
        #
        # Surrounded by 4-byte record-length markers.
        # ----------------------------------------------------

        marker = f.read(4)

        if len(marker) != 4:
            return None, []

        record_len = struct.unpack("<i", marker)[0]

        if record_len != HEADER_SIZE:
            raise RuntimeError(
                f"Unexpected header record length "
                f"{record_len} in {filename}"
            )

        header = f.read(record_len)

        end_marker = f.read(4)

        if len(header) != HEADER_SIZE:
            raise RuntimeError(
                f"Incomplete header in {filename}"
            )

        if len(end_marker) != 4:
            raise RuntimeError(
                f"Missing header end marker in {filename}"
            )

        if struct.unpack("<i", end_marker)[0] != record_len:
            raise RuntimeError(
                f"Header Fortran record markers do not match "
                f"in {filename}"
            )

        itime, numpart, outgrid_option = struct.unpack(
            "<iii",
            header
        )

        # ----------------------------------------------------
        # Particle records
        # ----------------------------------------------------

        for ip in range(numpart):

            marker = f.read(4)

            if len(marker) != 4:
                raise RuntimeError(
                    f"Missing particle record marker "
                    f"{ip + 1}/{numpart} in {filename}"
                )

            record_len = struct.unpack("<i", marker)[0]

            if record_len != RECORD_SIZE:
                raise RuntimeError(
                    f"Unexpected particle record length "
                    f"{record_len}; expected {RECORD_SIZE} "
                    f"in {filename}"
                )

            data = f.read(record_len)

            end_marker = f.read(4)

            if len(data) != RECORD_SIZE:
                raise RuntimeError(
                    f"Incomplete particle record "
                    f"{ip + 1}/{numpart} in {filename}"
                )

            if len(end_marker) != 4:
                raise RuntimeError(
                    f"Missing particle end marker "
                    f"for particle {ip + 1} in {filename}"
                )

            if struct.unpack("<i", end_marker)[0] != record_len:
                raise RuntimeError(
                    f"Particle Fortran record markers do not "
                    f"match for particle {ip + 1} in {filename}"
                )

            values = struct.unpack(RECORD_FMT, data)

            npoint = values[0]
            x = values[1]
            y = values[2]
            z = values[3]

            particles.append(
                {
                    "npoint": npoint,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )

    return (itime, numpart, outgrid_option), particles


# ============================================================
# X/Y METRES -> LON/LAT
# ============================================================

def xy_to_lonlat(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    xi = x / DX
    yj = y / DY

    lon_out = np.full(xi.shape, np.nan)
    lat_out = np.full(yj.shape, np.nan)

    ny, nx = xlong.shape

    valid = (
        np.isfinite(xi)
        & np.isfinite(yj)
        & (xi >= 0)
        & (xi <= nx - 1)
        & (yj >= 0)
        & (yj <= ny - 1)
    )

    if not np.any(valid):
        return lon_out, lat_out

    xv = xi[valid]
    yv = yj[valid]

    # Keep interpolation inside grid

    i0 = np.floor(xv).astype(int)
    j0 = np.floor(yv).astype(int)

    i0 = np.minimum(i0, nx - 2)
    j0 = np.minimum(j0, ny - 2)

    i1 = i0 + 1
    j1 = j0 + 1

    fx = xv - i0
    fy = yv - j0

    # Bilinear interpolation

    lon = (
        xlong[j0, i0] * (1 - fx) * (1 - fy)
        + xlong[j1, i0] * (1 - fx) * fy
        + xlong[j0, i1] * fx * (1 - fy)
        + xlong[j1, i1] * fx * fy
    )

    lat = (
        xlat[j0, i0] * (1 - fx) * (1 - fy)
        + xlat[j1, i0] * (1 - fx) * fy
        + xlat[j0, i1] * fx * (1 - fy)
        + xlat[j1, i1] * fx * fy
    )

    lon_out[valid] = lon
    lat_out[valid] = lat

    return lon_out, lat_out


# ============================================================
# READ PARTICLES
# ============================================================

all_clouds = []

print("\nReading particle files")
print("=" * 75)

for filename in PARTICLE_FILES:

    if not filename.exists():

        print("MISSING:", filename)
        continue

    # Extract particle-output timestamp from filename

    try:

        timestamp_string = filename.name.replace(
            "partposit_", ""
        )

        particle_datetime = datetime.strptime(
            timestamp_string,
            "%Y%m%d%H%M%S"
        )

    except ValueError:

        print(
            "WARNING: Cannot extract timestamp from:",
            filename.name
        )
        continue

    # Calculate backward time relative to receptor time

    time_difference = (
        receptor_datetime - particle_datetime
    ).total_seconds() / 3600.0

    if time_difference < 0:

        print(
            "WARNING: Particle timestamp is later than "
            "receptor time:",
            filename.name
        )
        continue

    label = f"-{time_difference:g} h"

    # Read binary particle file

    info, particles = read_particle_file(filename)

    if info is None:

        print("WARNING: Could not read:", filename)
        continue

    itime, numpart, outgrid_option = info

    if len(particles) == 0:

        print(
            f"WARNING: No particles in {filename.name}"
        )
        continue

    x = np.array(
        [p["x"] for p in particles],
        dtype=float
    )

    y = np.array(
        [p["y"] for p in particles],
        dtype=float
    )

    z = np.array(
        [p["z"] for p in particles],
        dtype=float
    )

    lon, lat = xy_to_lonlat(x, y)

    valid = (
        np.isfinite(lon)
        & np.isfinite(lat)
    )

    print(
        f"{label:7s} "
        f"{particle_datetime.strftime('%Y-%m-%d %H:%M')}  "
        f"{filename.name}  "
        f"itime={itime:7d}  "
        f"particles={numpart:5d}  "
        f"valid={np.sum(valid):5d}"
    )

    if np.any(valid):

        print(
            f"         lon = {np.nanmin(lon):.5f}"
            f" to {np.nanmax(lon):.5f}"
        )

        print(
            f"         lat = {np.nanmin(lat):.5f}"
            f" to {np.nanmax(lat):.5f}"
        )

    all_clouds.append(
        {
            "label": label,
            "datetime": particle_datetime,
            "lon": lon[valid],
            "lat": lat[valid],
            "z": z[valid],
        }
    )


if len(all_clouds) == 0:

    raise RuntimeError(
        "No valid particle clouds were read. "
        "Check the particle file paths and file contents."
    )


print("\nNumber of particle clouds loaded:", len(all_clouds))


# ============================================================
# READ MINE SHAPEFILES
# ============================================================

print("\nReading mine polygons")
print("=" * 75)

mine_shapes = []

for n in range(1, 14):

    shp = MINE_DIR / f"OC{n}.shp"

    if not shp.exists():

        print("Missing:", shp)
        continue

    sf = shapefile.Reader(str(shp))

    for shape in sf.shapes():

        pts = np.asarray(shape.points)

        if pts.ndim == 2 and pts.shape[1] >= 2:

            mine_shapes.append(
                (f"OC{n}", pts[:, 0], pts[:, 1])
            )

    sf.close()

print("Mine polygons loaded:", len(mine_shapes))


# ============================================================
# DETERMINE MAP EXTENT
# ============================================================

lon_min = np.nanmin(xlong)
lon_max = np.nanmax(xlong)

lat_min = np.nanmin(xlat)
lat_max = np.nanmax(xlat)

lon_margin = 0.03
lat_margin = 0.03


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 10)
)


# ------------------------------------------------------------
# Particle clouds
#
# IMPORTANT:
# Do not use zip(all_clouds, markers) with a fixed-length
# marker list. That was the reason the fourth cloud was
# not plotted.
#
# Use a loop over all_clouds and cycle through the available
# Matplotlib colours and markers.
# ------------------------------------------------------------

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

markers = ["o", "^", "s", "D", "v", "P", "X", "*"]

for i, cloud in enumerate(all_clouds):

    color = color_cycle[i % len(color_cycle)]
    marker = markers[i % len(markers)]

    ax.scatter(
        cloud["lon"],
        cloud["lat"],
        s=12,
        alpha=0.45,
        color=color,
        marker=marker,
        label=(
            f"{cloud['label']} "
            f"({cloud['datetime'].strftime('%H:%M')})"
        ),
        rasterized=True,
        zorder=3,
    )


# ------------------------------------------------------------
# Mine polygons
# ------------------------------------------------------------

for name, lon, lat in mine_shapes:

    ax.plot(
        lon,
        lat,
        linewidth=1.4,
        color="black",
        zorder=5,
    )

    # Label near polygon centroid

    cx = np.mean(lon)
    cy = np.mean(lat)

    ax.text(
        cx,
        cy,
        name,
        fontsize=8,
        ha="center",
        va="center",
        fontweight="bold",
        zorder=6,
    )


# ------------------------------------------------------------
# Receptor
# ------------------------------------------------------------

ax.scatter(
    RECEPTOR_LON,
    RECEPTOR_LAT,
    marker="*",
    s=250,
    color="red",
    edgecolor="black",
    linewidth=1.2,
    label="Singleton receptor",
    zorder=20,
)

ax.text(
    RECEPTOR_LON + 0.015,
    RECEPTOR_LAT + 0.015,
    "Singleton receptor",
    fontsize=10,
    fontweight="bold",
    color="black",
    zorder=21,
)


# ------------------------------------------------------------
# Plot formatting
# ------------------------------------------------------------

ax.set_xlim(
    lon_min - lon_margin,
    lon_max + lon_margin
)

ax.set_ylim(
    lat_min - lat_margin,
    lat_max + lat_margin
)

ax.set_xlabel(
    "Longitude (°E)",
    fontsize=12
)

ax.set_ylabel(
    "Latitude (°N)",
    fontsize=12
)


# ------------------------------------------------------------
# Dynamic title
# ------------------------------------------------------------

receptor_time_text = receptor_datetime.strftime(
    "%d %B %Y, %H:%M"
)

snapshot_text = ", ".join(
    [
        (
            f"{cloud['label']} "
            f"({cloud['datetime'].strftime('%H:%M')})"
        )
        for cloud in all_clouds
    ]
)

ax.set_title(
    "FLEXPART-WRF Backward Particle Position Clouds\n"
    f"Singleton receptor — {receptor_time_text}\n"
    f"Backward snapshots: {snapshot_text}",
    fontsize=13,
    fontweight="bold",
    pad=15,
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend(
    loc="upper right",
    frameon=True,
    fontsize=9,
)


# ============================================================
# SAVE FIGURE
# ============================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

print("\n" + "=" * 75)
print("PLOT SUMMARY")
print("=" * 75)

print(
    "Receptor time:",
    receptor_datetime.strftime("%Y-%m-%d %H:%M:%S")
)

print("Particle clouds plotted:", len(all_clouds))

for cloud in all_clouds:

    print(
        f"  {cloud['label']:7s} "
        f"{cloud['datetime'].strftime('%Y-%m-%d %H:%M')} "
        f"particles={len(cloud['lon'])}"
    )

print("\nSaved figure:")
print(OUTPUT_PNG)

plt.show()

