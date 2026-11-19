#!/usr/bin/env python3

"""
FLEXPART-WRF:
    24-h integrated CONC footprint + backward particle clouds

Singleton receptor:
    151.16132 E
    -32.57123 N
    5 m AGL

Release:
    2026-02-10 02:00 UTC
    = 12:00 AEST

Backward duration:
    24 h

No FLEXPART-WRF source code is modified.
"""

from pathlib import Path
import struct

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import netCDF4
import shapefile


# ============================================================
# PATHS
# ============================================================

RUN_DIR = Path(
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart/run/2026-02/hourly_obs_3layer/singleton/20260216_180000"
)

HEADER = RUN_DIR / "header_d01.nc"

FOOTPRINT = RUN_DIR / "flxout_d01_20260216_170000.nc"

MINE_DIR = Path(
    "/mnt/scratch_lustre/nzem_meim_scratch/data/emission_location"
)

OUTPUT_PNG = (
    RUN_DIR /
    "FLEXPART_footprint_plus_particle_cloud_Singleton.png"
)


# ============================================================
# RECEPTOR
# ============================================================

RECEPTOR_LON = 151.16132
RECEPTOR_LAT = -32.57123


# ============================================================
# GRID
# ============================================================

DX = 1000.0
DY = 1000.0


# ============================================================
# PARTICLE FILES
# ============================================================

PARTICLE_FILES = [
    (
        "-3 h",
        RUN_DIR / "partposit_20260216150000"
    ),
    (
        "-2 h",
        RUN_DIR / "partposit_20260216160000"
    ),
    (
        "-1 h",
        RUN_DIR / "partposit_20260216170000"
    ),
]


# ============================================================
# PARTICLE BINARY FORMAT
# ============================================================

HEADER_SIZE = 12
RECORD_SIZE = 52

RECORD_FMT = "<ifffi8f"


# ============================================================
# READ WRF/FLEXPART GRID
# ============================================================

print("=" * 80)
print("READING FLEXPART-WRF GRID")
print("=" * 80)

with netCDF4.Dataset(HEADER) as nc:

    xlong = np.asarray(
        nc.variables["XLONG"][:],
        dtype=float
    )

    xlat = np.asarray(
        nc.variables["XLAT"][:],
        dtype=float
    )

print("Grid:", xlong.shape)

print(
    "Longitude:",
    np.nanmin(xlong),
    "to",
    np.nanmax(xlong)
)

print(
    "Latitude:",
    np.nanmin(xlat),
    "to",
    np.nanmax(xlat)
)


# ============================================================
# READ FOOTPRINT
# ============================================================

print("\n" + "=" * 80)
print("READING CONC FOOTPRINT")
print("=" * 80)

with netCDF4.Dataset(FOOTPRINT) as nc:

    conc = np.asarray(
        nc.variables["CONC"][:],
        dtype=float
    )

print("CONC shape:", conc.shape)

print(
    "CONC min:",
    np.nanmin(conc)
)

print(
    "CONC max:",
    np.nanmax(conc)
)

print(
    "CONC sum:",
    np.nansum(conc)
)


# ============================================================
# INTEGRATE FOOTPRINT
# ============================================================
#
# CONC dimensions:
#
# Time,
# ageclass,
# releases,
# bottom_top,
# south_north,
# west_east
#
# Sum over:
#   Time
#   ageclass
#   releases
#   bottom_top
#
# leaving:
#   south_north
#   west_east
#
# ============================================================

footprint = np.nansum(
    conc,
    axis=(0, 1, 2, 3)
)

print("\nIntegrated footprint shape:", footprint.shape)

print(
    "Integrated footprint min:",
    np.nanmin(footprint)
)

print(
    "Integrated footprint max:",
    np.nanmax(footprint)
)

print(
    "Integrated footprint sum:",
    np.nansum(footprint)
)

nonzero = np.isfinite(footprint) & (footprint > 0)

print(
    "Non-zero footprint cells:",
    np.sum(nonzero)
)


# ============================================================
# PARTICLE READER
# ============================================================

def read_particle_file(filename):

    particles = []

    with open(filename, "rb") as f:

        # -------------------------------
        # Header Fortran record
        # -------------------------------

        marker = f.read(4)

        if len(marker) != 4:
            return None, []

        record_len = struct.unpack(
            "<i",
            marker
        )[0]

        if record_len != HEADER_SIZE:
            raise RuntimeError(
                f"Unexpected header record length "
                f"{record_len} in {filename}"
            )

        data = f.read(record_len)

        end_marker = f.read(4)

        if len(data) != HEADER_SIZE:
            raise RuntimeError(
                f"Incomplete header in {filename}"
            )

        if struct.unpack("<i", end_marker)[0] != record_len:
            raise RuntimeError(
                f"Header record markers do not match "
                f"in {filename}"
            )

        itime, numpart, outgrid_option = struct.unpack(
            "<iii",
            data
        )

        # -------------------------------
        # Particle records
        # -------------------------------

        for ip in range(numpart):

            marker = f.read(4)

            record_len = struct.unpack(
                "<i",
                marker
            )[0]

            if record_len != RECORD_SIZE:
                raise RuntimeError(
                    f"Unexpected particle record "
                    f"length {record_len}"
                )

            data = f.read(record_len)

            end_marker = f.read(4)

            if len(data) != RECORD_SIZE:
                raise RuntimeError(
                    f"Incomplete particle record"
                )

            if struct.unpack("<i", end_marker)[0] != record_len:
                raise RuntimeError(
                    f"Particle record markers do not match"
                )

            values = struct.unpack(
                RECORD_FMT,
                data
            )

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

    return (
        itime,
        numpart,
        outgrid_option
    ), particles


# ============================================================
# X/Y -> LON/LAT
# ============================================================

def xy_to_lonlat(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    xi = x / DX
    yj = y / DY

    lon_out = np.full(
        xi.shape,
        np.nan
    )

    lat_out = np.full(
        xi.shape,
        np.nan
    )

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

    i0 = np.floor(xv).astype(int)
    j0 = np.floor(yv).astype(int)

    i0 = np.minimum(
        i0,
        nx - 2
    )

    j0 = np.minimum(
        j0,
        ny - 2
    )

    i1 = i0 + 1
    j1 = j0 + 1

    fx = xv - i0
    fy = yv - j0

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
# READ PARTICLE CLOUDS
# ============================================================

print("\n" + "=" * 80)
print("READING PARTICLE CLOUDS")
print("=" * 80)

clouds = []

for label, filename in PARTICLE_FILES:

    info, particles = read_particle_file(
        filename
    )

    itime, numpart, outgrid_option = info

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

    lon, lat = xy_to_lonlat(
        x,
        y
    )

    valid = (
        np.isfinite(lon)
        & np.isfinite(lat)
    )

    print(
        f"{label:5s} "
        f"itime={itime:7d} "
        f"particles={numpart:4d} "
        f"valid={np.sum(valid):4d}"
    )

    clouds.append(
        {
            "label": label,
            "lon": lon[valid],
            "lat": lat[valid],
            "z": z[valid],
        }
    )


# ============================================================
# READ MINE POLYGONS
# ============================================================

print("\n" + "=" * 80)
print("READING MINE POLYGONS")
print("=" * 80)

mine_shapes = []

for n in range(1, 14):

    shp = MINE_DIR / f"OC{n}.shp"

    sf = shapefile.Reader(
        str(shp)
    )

    for shape in sf.shapes():

        pts = np.asarray(
            shape.points
        )

        if len(pts) >= 2:

            mine_shapes.append(
                (
                    f"OC{n}",
                    pts[:, 0],
                    pts[:, 1]
                )
            )

    sf.close()

print(
    "Mine polygons:",
    len(mine_shapes)
)


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(13, 10)
)


# ============================================================
# FOOTPRINT
# ============================================================

positive = (
    np.isfinite(footprint)
    & (footprint > 0)
)

# Mask zero values because LogNorm cannot display zero.
foot_plot = np.ma.masked_where(
    ~positive,
    footprint
)

# Use WRF cell centers.
mesh = ax.pcolormesh(
    xlong,
    xlat,
    foot_plot,
    shading="auto",
    norm=LogNorm(
        vmin=np.nanmin(
            footprint[positive]
        ),
        vmax=np.nanmax(
            footprint[positive]
        )
    ),
    alpha=0.75,
)

cbar = plt.colorbar(
    mesh,
    ax=ax,
    pad=0.02
)

cbar.set_label(
    "24-h integrated FLEXPART-WRF CONC sensitivity\n"
    "(sum over time and model layers)",
    fontsize=11
)


# ============================================================
# MINE POLYGONS
# ============================================================

for name, lon, lat in mine_shapes:

    ax.plot(
        lon,
        lat,
        linewidth=1.5,
        zorder=10
    )

    ax.text(
        np.mean(lon),
        np.mean(lat),
        name,
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=11
    )


# ============================================================
# PARTICLE CLOUDS
# ============================================================

for cloud in clouds:

    ax.scatter(
        cloud["lon"],
        cloud["lat"],
        s=16,
        alpha=0.45,
        zorder=15,
        label=cloud["label"]
    )


# ============================================================
# RECEPTOR
# ============================================================

ax.scatter(
    RECEPTOR_LON,
    RECEPTOR_LAT,
    marker="*",
    s=300,
    edgecolor="black",
    linewidth=1.5,
    zorder=30
)

ax.text(
    RECEPTOR_LON + 0.018,
    RECEPTOR_LAT + 0.018,
    "Singleton",
    fontsize=11,
    fontweight="bold",
    zorder=31
)


# ============================================================
# LEGEND
# ============================================================

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=7,
        label="-1 h particle positions"
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=7,
        label="-2 h particle positions"
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=7,
        label="-3 h particle positions"
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        linestyle="None",
        markersize=14,
        markeredgecolor="black",
        label="Singleton receptor"
    ),
]

ax.legend(
    handles=legend_elements,
    loc="upper right",
    fontsize=10,
    frameon=True
)


# ============================================================
# MAP EXTENT
# ============================================================

lon_min = np.nanmin(xlong)
lon_max = np.nanmax(xlong)
lat_min = np.nanmin(xlat)
lat_max = np.nanmax(xlat)

ax.set_xlim(
    lon_min - 0.03,
    lon_max + 0.03
)

ax.set_ylim(
    lat_min - 0.03,
    lat_max + 0.03
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
    "FLEXPART-WRF Singleton Backward Transport\n"
    "24-h integrated CONC footprint + particle positions\n"
    "Release: 10 Feb 2026 02:00 UTC (12:00 AEST), 5 m AGL",
    fontsize=14,
    fontweight="bold"
)

ax.grid(
    True,
    alpha=0.25
)


# ============================================================
# SAVE
# ============================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_PNG)

plt.show()

