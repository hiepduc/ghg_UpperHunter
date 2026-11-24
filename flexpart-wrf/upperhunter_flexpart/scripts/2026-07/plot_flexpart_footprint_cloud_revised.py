#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot FLEXPART-WRF integrated concentration footprint and backward
particle positions for the Singleton receptor.

The receptor time is extracted from the run directory name:
    YYYYMMDD_HHMMSS

FLEXPART timestamps are in UTC.

The script:
    1. Reads the FLEXPART-WRF CONC field from the flxout NetCDF file.
    2. Integrates CONC over time, species, age classes and vertical levels.
    3. Reads XLONG and XLAT from header_d01.nc.
    4. Reads particle positions from partposit_* binary files.
    5. Converts particle x/y positions to longitude and latitude.
    6. Reads mine boundaries OC1–OC13.
    7. Plots the integrated footprint, particle clouds and Singleton receptor.
"""

from pathlib import Path
from datetime import datetime, timedelta
import struct

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from netCDF4 import Dataset
import shapefile


# ============================================================================
# USER SETTINGS
# ============================================================================

RUN_DIR = Path(
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/"
    "upperhunter_flexpart/run/2026-07/hourly_obs_3layer/"
    "singleton/20260716_180000"
)

HEADER = RUN_DIR / "header_d01.nc"

FOOTPRINT = RUN_DIR / "flxout_d01_20260716_170000.nc"

MINE_DIR = Path(
    "/mnt/scratch_lustre/nzem_meim_scratch/data/emission_location"
)

OUTPUT_PNG = RUN_DIR / (
    "FLEXPART_footprint_plus_particle_cloud_Singleton.png"
)

# Particle snapshots.
PARTICLE_FILES = [
    ("-3 h", RUN_DIR / "partposit_20260716150000"),
    ("-2 h", RUN_DIR / "partposit_20260716160000"),
    ("-1 h", RUN_DIR / "partposit_20260716170000"),
]

# Singleton receptor coordinates.
RECEPTOR_LON = 151.16132
RECEPTOR_LAT = -32.57123

# FLEXPART backward simulation period.
BACKWARD_HOURS = 24

# FLEXPART timestamps are UTC.
TIME_ZONE_LABEL = "UTC"

# WRF horizontal grid spacing.
DX = 1000.0
DY = 1000.0

# Particle binary format.
HEADER_SIZE = 12
RECORD_SIZE = 52
RECORD_FMT = "<ifffi8f"

# Plot settings.
FIGSIZE = (14, 11)
DPI = 300

FOOTPRINT_ALPHA = 0.75

# Lower percentile used for logarithmic colour scaling.
LOG_MIN_PERCENTILE = 5

# Number of mine shapefiles.
N_MINES = 13


# ============================================================================
# RECEPTOR TIME AND SIMULATION PERIOD
# ============================================================================

try:
    RECEPTOR_TIME = datetime.strptime(
        RUN_DIR.name,
        "%Y%m%d_%H%M%S"
    )
except ValueError as exc:
    raise ValueError(
        "Cannot extract receptor time from RUN_DIR.name. "
        "Expected YYYYMMDD_HHMMSS, e.g. 20260716_180000."
    ) from exc

SIMULATION_START = RECEPTOR_TIME - timedelta(
    hours=BACKWARD_HOURS
)

RECEPTOR_TIME_TEXT = RECEPTOR_TIME.strftime(
    "%d %B %Y, %H:%M"
)

SIMULATION_START_TEXT = SIMULATION_START.strftime(
    "%d %B %Y, %H:%M"
)


# ============================================================================
# READ WRF LONGITUDE AND LATITUDE
# ============================================================================

def read_wrf_grid(header_file):
    """
    Read XLONG and XLAT from the FLEXPART-WRF header NetCDF file.
    """

    if not header_file.exists():
        raise FileNotFoundError(
            f"WRF header file not found: {header_file}"
        )

    print("\nReading WRF longitude/latitude grid:")
    print(header_file)

    with Dataset(header_file, "r") as nc:
        xlong = np.asarray(
            nc.variables["XLONG"][:]
        ).squeeze()

        xlat = np.asarray(
            nc.variables["XLAT"][:]
        ).squeeze()

    if xlong.ndim != 2 or xlat.ndim != 2:
        raise ValueError(
            "Expected two-dimensional XLONG and XLAT arrays. "
            f"Received XLONG={xlong.shape}, XLAT={xlat.shape}"
        )

    print("WRF grid shape:", xlong.shape)

    return xlong, xlat


# ============================================================================
# CONVERT PARTICLE X/Y COORDINATES TO LONGITUDE/LATITUDE
# ============================================================================

def xy_to_lonlat(x, y, xlong, xlat):
    """
    Convert FLEXPART particle x/y coordinates in metres to longitude
    and latitude using bilinear interpolation of the WRF grid.

    Assumes:
        x = 0 and y = 0 correspond to the first WRF grid point.
        DX = DY = 1000 metres.
    """

    ny, nx = xlong.shape

    grid_x = np.asarray(x, dtype=float) / DX
    grid_y = np.asarray(y, dtype=float) / DY

    # Keep coordinates within the WRF grid.
    grid_x = np.clip(grid_x, 0.0, nx - 1.001)
    grid_y = np.clip(grid_y, 0.0, ny - 1.001)

    i0 = np.floor(grid_x).astype(int)
    j0 = np.floor(grid_y).astype(int)

    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)

    fx = grid_x - i0
    fy = grid_y - j0

    # Longitude interpolation.
    lon00 = xlong[j0, i0]
    lon10 = xlong[j0, i1]
    lon01 = xlong[j1, i0]
    lon11 = xlong[j1, i1]

    lon = (
        lon00 * (1.0 - fx) * (1.0 - fy)
        + lon10 * fx * (1.0 - fy)
        + lon01 * (1.0 - fx) * fy
        + lon11 * fx * fy
    )

    # Latitude interpolation.
    lat00 = xlat[j0, i0]
    lat10 = xlat[j0, i1]
    lat01 = xlat[j1, i0]
    lat11 = xlat[j1, i1]

    lat = (
        lat00 * (1.0 - fx) * (1.0 - fy)
        + lat10 * fx * (1.0 - fy)
        + lat01 * (1.0 - fx) * fy
        + lat11 * fx * fy
    )

    return lon, lat


# ============================================================================
# READ FLEXPART PARTICLE POSITION FILE
# ============================================================================

def read_particle_file(filename):
    """
    Read particle positions from a FLEXPART-WRF partposit binary file.

    Assumed binary format:
        Header: 12 bytes
        Fortran record marker: 4 bytes
        Particle payload: 52 bytes
        Fortran trailing record marker: 4 bytes

    Particle payload:
        <ifffi8f
    """

    if not filename.exists():
        raise FileNotFoundError(
            f"Particle file not found: {filename}"
        )

    print(f"\nReading particle file: {filename.name}")

    data = filename.read_bytes()

    if len(data) <= HEADER_SIZE:
        raise ValueError(
            f"Particle file is too small: {filename}"
        )

    particle_x = []
    particle_y = []
    particle_z = []

    offset = HEADER_SIZE

    while offset + 4 <= len(data):

        record_length = struct.unpack_from(
            "<i",
            data,
            offset
        )[0]

        if record_length != RECORD_SIZE:
            offset += 1
            continue

        payload_start = offset + 4
        payload_end = payload_start + RECORD_SIZE
        trailing_marker_end = payload_end + 4

        if trailing_marker_end > len(data):
            break

        trailing_length = struct.unpack_from(
            "<i",
            data,
            payload_end
        )[0]

        if trailing_length != RECORD_SIZE:
            offset += 1
            continue

        values = struct.unpack_from(
            RECORD_FMT,
            data,
            payload_start
        )

        # Particle record:
        # values[0] = particle identifier
        # values[1] = x coordinate
        # values[2] = y coordinate
        # values[3] = z coordinate

        particle_x.append(values[1])
        particle_y.append(values[2])
        particle_z.append(values[3])

        offset = trailing_marker_end

    if not particle_x:
        raise ValueError(
            f"No particle records found in {filename}. "
            "Check the binary record format and HEADER_SIZE."
        )

    particle_x = np.asarray(
        particle_x,
        dtype=float
    )

    particle_y = np.asarray(
        particle_y,
        dtype=float
    )

    particle_z = np.asarray(
        particle_z,
        dtype=float
    )

    print("Number of particles:", len(particle_x))

    print(
        "X range (m):",
        np.nanmin(particle_x),
        np.nanmax(particle_x)
    )

    print(
        "Y range (m):",
        np.nanmin(particle_y),
        np.nanmax(particle_y)
    )

    return particle_x, particle_y, particle_z


# ============================================================================
# READ FLEXPART FOOTPRINT
# ============================================================================

def read_footprint(footprint_file):
    """
    Read FLEXPART-WRF CONC and integrate over:
        time
        species
        age classes
        vertical levels

    Expected CONC dimensions:
        (time, species, ageclass, level, latitude, longitude)
    """

    if not footprint_file.exists():
        raise FileNotFoundError(
            f"Footprint file not found: {footprint_file}"
        )

    print("\nReading FLEXPART footprint:")
    print(footprint_file)

    with Dataset(footprint_file, "r") as nc:

        if "CONC" not in nc.variables:
            raise KeyError(
                f"CONC variable not found in {footprint_file}"
            )

        conc = np.asarray(
            nc.variables["CONC"][:],
            dtype=float
        )

    print("CONC shape:", conc.shape)

    if conc.ndim != 6:
        raise ValueError(
            "Expected six-dimensional CONC array "
            "(time, species, ageclass, level, y, x). "
            f"Received shape: {conc.shape}"
        )

    # Integrate over time, species, age classes and vertical levels.
    footprint = np.nansum(
        conc,
        axis=(0, 1, 2, 3)
    )

    footprint = np.asarray(
        footprint,
        dtype=float
    )

    footprint[~np.isfinite(footprint)] = np.nan
    footprint[footprint <= 0.0] = np.nan

    print("Integrated footprint shape:", footprint.shape)

    print(
        "Non-zero grid cells:",
        np.count_nonzero(np.isfinite(footprint))
    )

    if not np.any(np.isfinite(footprint)):
        raise ValueError(
            "Integrated footprint contains no positive values."
        )

    print(
        "Footprint minimum:",
        np.nanmin(footprint)
    )

    print(
        "Footprint maximum:",
        np.nanmax(footprint)
    )

    return footprint


# ============================================================================
# READ MINE SHAPEFILES
# ============================================================================

def read_mine_shapefiles(mine_directory):
    """
    Read mine polygon shapefiles OC1.shp through OC13.shp.

    Returns:
        [(mine_name, polygon_parts), ...]
    """

    mines = []

    print("\nReading mine shapefiles:")

    for mine_number in range(1, N_MINES + 1):

        mine_name = f"OC{mine_number}"

        shp_file = (
            mine_directory / f"{mine_name}.shp"
        )

        if not shp_file.exists():
            print(
                f"WARNING: {shp_file.name} not found; skipping."
            )
            continue

        try:
            reader = shapefile.Reader(
                str(shp_file)
            )

            polygon_parts = []

            for shape_record in reader.shapeRecords():

                shape = shape_record.shape

                points = np.asarray(
                    shape.points,
                    dtype=float
                )

                if points.size == 0:
                    continue

                part_indices = list(shape.parts)
                part_indices.append(len(points))

                for k in range(len(part_indices) - 1):

                    start = part_indices[k]
                    end = part_indices[k + 1]

                    polygon = points[start:end]

                    if len(polygon) >= 2:
                        polygon_parts.append(polygon)

            if polygon_parts:

                mines.append(
                    (mine_name, polygon_parts)
                )

                print(
                    f"{mine_name}: "
                    f"{len(polygon_parts)} polygon part(s)"
                )

        except Exception as exc:
            print(
                f"WARNING: Could not read {shp_file}: {exc}"
            )

    return mines


# ============================================================================
# MAIN PLOTTING FUNCTION
# ============================================================================

def main():

    print("=" * 75)
    print("FLEXPART-WRF FOOTPRINT + PARTICLE CLOUD")
    print("=" * 75)

    print("Run directory:", RUN_DIR)

    print(
        "Receptor time:",
        RECEPTOR_TIME_TEXT,
        TIME_ZONE_LABEL
    )

    print(
        "Backward simulation start:",
        SIMULATION_START_TEXT,
        TIME_ZONE_LABEL
    )

    print(
        "Backward period:",
        BACKWARD_HOURS,
        "hours"
    )

    # ------------------------------------------------------------------------
    # Read WRF grid.
    # ------------------------------------------------------------------------

    xlong, xlat = read_wrf_grid(HEADER)

    # ------------------------------------------------------------------------
    # Read integrated footprint.
    # ------------------------------------------------------------------------

    footprint = read_footprint(FOOTPRINT)

    if footprint.shape != xlong.shape:
        raise ValueError(
            "Footprint and WRF grid dimensions do not match:\n"
            f"Footprint: {footprint.shape}\n"
            f"WRF grid:  {xlong.shape}"
        )

    # ------------------------------------------------------------------------
    # Read mine polygons.
    # ------------------------------------------------------------------------

    mines = read_mine_shapefiles(MINE_DIR)

    # ------------------------------------------------------------------------
    # Read particle files and convert positions to lon/lat.
    # ------------------------------------------------------------------------

    clouds = []

    for label, particle_file in PARTICLE_FILES:

        px, py, pz = read_particle_file(
            particle_file
        )

        plon, plat = xy_to_lonlat(
            px,
            py,
            xlong,
            xlat
        )

        valid = (
            np.isfinite(plon)
            & np.isfinite(plat)
            & (plon >= np.nanmin(xlong) - 0.1)
            & (plon <= np.nanmax(xlong) + 0.1)
            & (plat >= np.nanmin(xlat) - 0.1)
            & (plat <= np.nanmax(xlat) + 0.1)
        )

        clouds.append({
            "label": label,
            "filename": particle_file.name,
            "lon": plon[valid],
            "lat": plat[valid],
            "z": pz[valid]
        })

        print(
            f"{label}: "
            f"{np.count_nonzero(valid)} "
            "particles inside plotting domain"
        )

    # ------------------------------------------------------------------------
    # Create figure.
    # ------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # ------------------------------------------------------------------------
    # Plot integrated footprint.
    # ------------------------------------------------------------------------

    positive_values = footprint[
        np.isfinite(footprint)
        & (footprint > 0.0)
    ]

    vmin = np.nanpercentile(
        positive_values,
        LOG_MIN_PERCENTILE
    )

    vmax = np.nanmax(
        positive_values
    )

    if not np.isfinite(vmin) or vmin <= 0.0:
        vmin = np.nanmin(positive_values)

    if vmax <= vmin:
        vmax = vmin * 10.0

    footprint_plot = ax.pcolormesh(
        xlong,
        xlat,
        footprint,
        shading="auto",
        cmap="viridis",
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax
        ),
        alpha=FOOTPRINT_ALPHA,
        zorder=1
    )

    cbar = fig.colorbar(
        footprint_plot,
        ax=ax,
        orientation="vertical",
        pad=0.02,
        shrink=0.85
    )

    cbar.set_label(
        "24-hour integrated FLEXPART-WRF CONC",
        fontsize=11
    )

    # ------------------------------------------------------------------------
    # Plot mine outlines and labels.
    # ------------------------------------------------------------------------

    for mine_name, polygon_parts in mines:

        for polygon in polygon_parts:

            ax.plot(
                polygon[:, 0],
                polygon[:, 1],
                color="black",
                linewidth=1.0,
                zorder=4
            )

        all_points = np.vstack(
            polygon_parts
        )

        mine_lon = np.nanmean(
            all_points[:, 0]
        )

        mine_lat = np.nanmean(
            all_points[:, 1]
        )

        ax.text(
            mine_lon,
            mine_lat,
            mine_name,
            fontsize=7,
            color="black",
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.65,
                "pad": 1
            }
        )

    # ------------------------------------------------------------------------
    # Plot particle clouds.
    # ------------------------------------------------------------------------

    cloud_styles = [
        {
            "color": "red",
            "marker": "o",
            "size": 13,
            "alpha": 0.45
        },
        {
            "color": "deepskyblue",
            "marker": "o",
            "size": 13,
            "alpha": 0.45
        },
        {
            "color": "orange",
            "marker": "o",
            "size": 13,
            "alpha": 0.45
        },
        {
            "color": "magenta",
            "marker": "o",
            "size": 13,
            "alpha": 0.45
        },
        {
            "color": "lime",
            "marker": "o",
            "size": 13,
            "alpha": 0.45
        }
    ]

    particle_legend_handles = []

    for i, cloud in enumerate(clouds):

        style = cloud_styles[
            i % len(cloud_styles)
        ]

        ax.scatter(
            cloud["lon"],
            cloud["lat"],
            s=style["size"],
            c=style["color"],
            marker=style["marker"],
            alpha=style["alpha"],
            edgecolors="none",
            zorder=6
        )

        particle_legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="none",
                markerfacecolor=style["color"],
                markeredgecolor="none",
                markersize=7,
                label=cloud["label"]
            )
        )

    # ------------------------------------------------------------------------
    # Plot Singleton receptor.
    # ------------------------------------------------------------------------

    ax.scatter(
        RECEPTOR_LON,
        RECEPTOR_LAT,
        s=180,
        marker="*",
        color="red",
        edgecolor="black",
        linewidth=1.2,
        zorder=10
    )

    receptor_legend_handle = Line2D(
        [0],
        [0],
        marker="*",
        color="none",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=13,
        label="Singleton receptor"
    )

    # ------------------------------------------------------------------------
    # Set map extent.
    # ------------------------------------------------------------------------

    lon_min = np.nanmin(xlong)
    lon_max = np.nanmax(xlong)

    lat_min = np.nanmin(xlat)
    lat_max = np.nanmax(xlat)

    margin = 0.03

    ax.set_xlim(
        lon_min - margin,
        lon_max + margin
    )

    ax.set_ylim(
        lat_min - margin,
        lat_max + margin
    )

    # ------------------------------------------------------------------------
    # Axis labels and title.
    # ------------------------------------------------------------------------

    ax.set_xlabel(
        "Longitude",
        fontsize=12
    )

    ax.set_ylabel(
        "Latitude",
        fontsize=12
    )

    ax.set_title(
        "FLEXPART-WRF Singleton Backward Simulation\n"
        f"Receptor time: {RECEPTOR_TIME_TEXT} UTC\n"
        f"Backward period: {SIMULATION_START_TEXT} to "
        f"{RECEPTOR_TIME_TEXT} UTC "
        f"({BACKWARD_HOURS} hours)",
        fontsize=13,
        fontweight="bold",
        pad=15
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.4,
        zorder=0
    )

    # ------------------------------------------------------------------------
    # Legend.
    # ------------------------------------------------------------------------

    mine_legend_handle = Line2D(
        [0],
        [0],
        color="black",
        linewidth=1.5,
        label="Open-cut mine boundaries"
    )

    legend_handles = (
        [
            mine_legend_handle,
            receptor_legend_handle
        ]
        + particle_legend_handles
    )

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=9,
        frameon=True,
        framealpha=0.9
    )

    # ------------------------------------------------------------------------
    # Save figure.
    # ------------------------------------------------------------------------

    plt.tight_layout()

    OUTPUT_PNG.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_PNG,
        dpi=DPI,
        bbox_inches="tight"
    )

    print("\n" + "=" * 75)
    print("PLOT SAVED")
    print("=" * 75)

    print(OUTPUT_PNG)

    plt.show()


# ============================================================================
# EXECUTE
# ============================================================================

if __name__ == "__main__":
    main()

