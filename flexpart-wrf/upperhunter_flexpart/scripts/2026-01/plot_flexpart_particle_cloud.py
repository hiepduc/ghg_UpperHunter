#!/usr/bin/env python3

"""
Plot FLEXPART-WRF particle-position clouds from partposit_* files.

This does NOT modify FLEXPART-WRF source code.

For the current FLEXPART-WRF output:
    x,y are metres on the WRF d03 grid
    dx = dy = 1000 m
    grid = 165 x 147

The particle coordinates are converted to lon/lat using bilinear
interpolation of XLONG/XLAT from header_d01.nc.

The plot shows particle clouds at selected backward times.
"""

from pathlib import Path
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import netCDF4
import shapefile


# ============================================================
# USER SETTINGS
# ============================================================

RUN_DIR = Path(
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart/run/2026-01/hourly_obs_3layer/singleton/20260116_180000"
)

HEADER = RUN_DIR / "header_d01.nc"

MINE_DIR = Path(
    "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter"
)

OUTPUT_PNG = RUN_DIR / "FLEXPART_particle_cloud_Singleton.png"

# Particle files to plot.
# These correspond to:
#   -3 h = 20260109230000
#   -2 h = 20260110000000
#   -1 h = 20260110010000
PARTICLE_FILES = [
    ("-3 h", RUN_DIR / "partposit_20260116150000"),
    ("-2 h", RUN_DIR / "partposit_20260116160000"),
    ("-1 h", RUN_DIR / "partposit_20260116170000"),
]

RECEPTOR_LON = 151.16132
RECEPTOR_LAT = -32.57123

DX = 1000.0
DY = 1000.0

# Particle record format for nspec = 1:
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
HEADER_SIZE = 12
RECORD_SIZE = 52
RECORD_FMT = "<ifffi8f"


# ============================================================
# READ HEADER GRID
# ============================================================

print("Reading FLEXPART-WRF header:", HEADER)

with netCDF4.Dataset(HEADER) as nc:
    xlong = np.asarray(nc.variables["XLONG"][:], dtype=float)
    xlat = np.asarray(nc.variables["XLAT"][:], dtype=float)

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
        # record:
        #   int32 itime
        #   int32 numpart
        #   int32 outgrid_option
        #
        # surrounded by 4-byte record-length markers.
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
                    f"{ip+1}/{numpart} in {filename}"
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
                    f"{ip+1}/{numpart} in {filename}"
                )

            if struct.unpack("<i", end_marker)[0] != record_len:
                raise RuntimeError(
                    f"Particle Fortran record markers do not "
                    f"match for particle {ip+1} in {filename}"
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
    lat_out = np.full(xi.shape, np.nan)

    nx = xlong.shape[1]
    ny = xlong.shape[0]

    valid = (
        (xi >= 0)
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
print("=" * 70)

for label, filename in PARTICLE_FILES:

    if not filename.exists():
        print("MISSING:", filename)
        continue

    info, particles = read_particle_file(filename)

    itime, numpart, outgrid_option = info

    x = np.array([p["x"] for p in particles], dtype=float)
    y = np.array([p["y"] for p in particles], dtype=float)
    z = np.array([p["z"] for p in particles], dtype=float)

    lon, lat = xy_to_lonlat(x, y)

    valid = np.isfinite(lon) & np.isfinite(lat)

    print(
        f"{label:5s}  "
        f"{filename.name}  "
        f"itime={itime:7d}  "
        f"particles={numpart:4d}  "
        f"valid={np.sum(valid):4d}"
    )

    if np.any(valid):

        print(
            f"       lon = {np.nanmin(lon):.5f}"
            f" to {np.nanmax(lon):.5f}"
        )
        print(
            f"       lat = {np.nanmin(lat):.5f}"
            f" to {np.nanmax(lat):.5f}"
        )

    all_clouds.append(
        {
            "label": label,
            "lon": lon[valid],
            "lat": lat[valid],
            "z": z[valid],
        }
    )


# ============================================================
# READ MINE SHAPEFILES
# ============================================================

print("\nReading mine polygons")
print("=" * 70)

mine_shapes = []

for n in range(1, 14):

    shp = MINE_DIR / f"OC{n}.shp"

    if not shp.exists():
        print("Missing:", shp)
        continue

    sf = shapefile.Reader(str(shp))

    for shape in sf.shapes():

        pts = np.asarray(shape.points)

        if pts.shape[1] >= 2:
            mine_shapes.append(
                (f"OC{n}", pts[:, 0], pts[:, 1])
            )

    sf.close()

print("Mine polygons loaded:", len(mine_shapes))


# ============================================================
# DETERMINE MAP EXTENT
# ============================================================

# Use the full WRF d03 geographic domain, with a small margin.

lon_min = np.nanmin(xlong)
lon_max = np.nanmax(xlong)
lat_min = np.nanmin(xlat)
lat_max = np.nanmax(xlat)

lon_margin = 0.03
lat_margin = 0.03


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(12, 10))

# ------------------------------------------------------------
# Particle clouds
# ------------------------------------------------------------

markers = ["o", "o", "o"]

for cloud, marker in zip(all_clouds, markers):

    ax.scatter(
        cloud["lon"],
        cloud["lat"],
        s=10,
        alpha=0.35,
        marker=marker,
        label=cloud["label"],
        rasterized=True,
    )


# ------------------------------------------------------------
# Mine polygons
# ------------------------------------------------------------

for name, lon, lat in mine_shapes:

    ax.plot(
        lon,
        lat,
        linewidth=1.4,
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
    )


# ------------------------------------------------------------
# Receptor
# ------------------------------------------------------------

ax.scatter(
    RECEPTOR_LON,
    RECEPTOR_LAT,
    marker="*",
    s=220,
    edgecolor="black",
    linewidth=1.2,
    zorder=20,
)

ax.text(
    RECEPTOR_LON + 0.015,
    RECEPTOR_LAT + 0.015,
    "Singleton receptor",
    fontsize=10,
    fontweight="bold",
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

ax.set_xlabel("Longitude (°E)", fontsize=12)
ax.set_ylabel("Latitude (°N)", fontsize=12)

ax.set_title(
    "FLEXPART-WRF Backward Particle Position Clouds\n"
    "Singleton receptor — 24 h backward simulation",
    fontsize=14,
    fontweight="bold",
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend(
    loc="upper right",
    frameon=True,
    fontsize=10,
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

print("\nSaved:")
print(OUTPUT_PNG)

plt.show()

