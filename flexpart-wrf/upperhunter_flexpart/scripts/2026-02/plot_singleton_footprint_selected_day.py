
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot all available hourly FLEXPART-WRF footprints for Singleton
on a selected day, saving one PNG per footprint.

Preserves:
  - Existing FLEXPART-WRF grid and corner-grid handling
  - Vertical sum of CONC
  - Logarithmic footprint colour scale
  - Coastline
  - OC1-OC13 mine polygons and labels
  - Singleton receptor marker

Time labels in filenames are taken from the footprint file.
"""

import os
import glob
import numpy as np
import matplotlib

# HPC batch plotting: save figures without requiring a display.
#matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from netCDF4 import Dataset
import shapefile


# ============================================================
# USER SETTINGS
# ============================================================

# Select the day to plot: YYYYMMDD
SELECTED_DAY = "20260215"

BASE_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/"
    "flexpart-wrf/upperhunter_flexpart/run/2026-02/"
    "hourly_obs_3layer/singleton"
)

MINE_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/"
    "data/emission_location"
)

COAST_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/"
    "data/coastline/ne_50m_coastline"
)

# Receptor location
RECEPTOR_LON = 151.16132
RECEPTOR_LAT = -32.57123

# Set True to include footprints whose CONC is entirely zero.
PLOT_ZERO_FOOTPRINTS = False

# Optional: use one common colour scale for all footprints that day.
# Recommended for comparing footprint strength between hours.
USE_COMMON_LOG_SCALE = True

# Percentiles used to define the common/daily logarithmic scale.
VMIN_PERCENTILE = 5.0
VMAX_PERCENTILE = 99.5

# Output directory; does not overwrite your existing single plot.
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    f"plots_{SELECTED_DAY}"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER: DECODE NETCDF TIME
# ============================================================

def decode_time(times):
    """Convert a NetCDF Times row to a readable string."""
    row = times[0]
    return "".join(
        x.decode() if hasattr(x, "decode") else str(x)
        for x in row
    ).strip().replace("\x00", "")


# ============================================================
# HELPER: READ GRID AND CONSTRUCT CELL EDGES
# ============================================================

def read_grid(header_file):
    with Dataset(header_file) as nc:
        lon = np.asarray(nc.variables["XLONG"][:]).squeeze()
        lat = np.asarray(nc.variables["XLAT"][:]).squeeze()

        lon_corner = np.asarray(
            nc.variables["XLONG_CORNER"][:]
        ).squeeze()

        lat_corner = np.asarray(
            nc.variables["XLAT_CORNER"][:]
        ).squeeze()

    print("Grid center shape :", lon.shape)
    print("Grid corner shape :", lon_corner.shape)

    # The corner arrays contain the lower-left corners
    # of the grid cells. Add the final east and north edges.
    ny, nx = lon_corner.shape

    lon_edge = np.empty((ny + 1, nx + 1))
    lat_edge = np.empty((ny + 1, nx + 1))

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

    print("Edge grid shape   :", lon_edge.shape)

    return lon, lat, lon_edge, lat_edge


# ============================================================
# HELPER: READ FOOTPRINT
# ============================================================

def read_footprint(foot_file):
    with Dataset(foot_file) as nc:
        # Same vertical summation as the original script.
        conc = np.asarray(
            nc.variables["CONC"][0, 0, 0, :, :, :]
        )

        time_string = decode_time(nc.variables["Times"][:])

    # Sum the vertical levels.
    foot = np.sum(conc, axis=0)

    # Avoid plotting invalid values.
    foot = np.where(np.isfinite(foot), foot, 0.0)
    foot = np.maximum(foot, 0.0)

    return foot, time_string


# ============================================================
# HELPER: READ COASTLINE ONCE
# ============================================================

def read_coastline(lon, lat):
    print("\nReading coastline...")

    coast_segments = []

    try:
        coast = shapefile.Reader(COAST_FILE)

        xmin = np.nanmin(lon) - 0.2
        xmax = np.nanmax(lon) + 0.2
        ymin = np.nanmin(lat) - 0.2
        ymax = np.nanmax(lat) + 0.2

        for shape in coast.shapes():
            points = np.asarray(shape.points)

            if len(points) < 2:
                continue

            parts = list(shape.parts) + [len(points)]

            for p in range(len(parts) - 1):
                xy = points[parts[p]:parts[p + 1]]

                if len(xy) < 2:
                    continue

                if (
                    np.max(xy[:, 0]) < xmin
                    or np.min(xy[:, 0]) > xmax
                    or np.max(xy[:, 1]) < ymin
                    or np.min(xy[:, 1]) > ymax
                ):
                    continue

                coast_segments.append(xy)

        print(f"Coastline plotted: {len(coast_segments)} segments")

    except Exception as exc:
        print("WARNING: could not read coastline:")
        print(exc)

    return coast_segments


# ============================================================
# HELPER: READ MINE POLYGONS ONCE
# ============================================================

def read_mines():
    print("\nReading mine polygons...")

    mine_segments = []
    mine_labels = []

    for i in range(1, 14):
        shp_base = os.path.join(MINE_DIR, f"OC{i}")
        shp_file = shp_base + ".shp"

        if not os.path.exists(shp_file):
            print("WARNING: missing", shp_file)
            continue

        try:
            sf = shapefile.Reader(shp_base)
            shapes = sf.shapes()

        except Exception as exc:
            print("WARNING: cannot read", shp_file)
            print(exc)
            continue

        print(f"Reading OC{i}: {len(shapes)} polygon(s)")

        for shape in shapes:
            points = np.asarray(shape.points)

            if len(points) < 2:
                continue

            parts = list(shape.parts) + [len(points)]

            for p in range(len(parts) - 1):
                xy = points[parts[p]:parts[p + 1]]

                if len(xy) >= 2:
                    mine_segments.append((i, xy))

        # Keep the original script's first-polygon label placement.
        if len(shapes) > 0:
            pts = np.asarray(shapes[0].points)

            if len(pts) > 0:
                mine_labels.append(
                    (
                        i,
                        float(np.mean(pts[:, 0])),
                        float(np.mean(pts[:, 1]))
                    )
                )
    return mine_segments, mine_labels


# ============================================================
# HELPER: PLOT ONE FOOTPRINT
# ============================================================

def plot_one(
    foot,
    time_string,
    lon,
    lat,
    lon_edge,
    lat_edge,
    coast_segments,
    mine_segments,
    mine_labels,
    vmin,
    vmax,
    output_png,
):
    positive = foot[foot > 0]

    if positive.size == 0 and not PLOT_ZERO_FOOTPRINTS:
        print(f"  SKIP {time_string}: footprint is all zero")
        return False

    # LogNorm requires a positive lower limit.
    if positive.size == 0:
        # Nothing positive to display; draw an empty masked field.
        vmin_plot, vmax_plot = 1.0, 10.0
    else:
        vmin_plot, vmax_plot = vmin, vmax

    foot_plot = np.ma.masked_where(foot <= 0, foot)

    fig, ax = plt.subplots(figsize=(11, 9))

    pcm = ax.pcolormesh(
        lon_edge,
        lat_edge,
        foot_plot,
        shading="flat",
        norm=LogNorm(vmin=vmin_plot, vmax=vmax_plot),
        zorder=1,
    )

    # Coastline
    for xy in coast_segments:
        ax.plot(
            xy[:, 0], xy[:, 1],
            linewidth=1.2,
            linestyle="-",
            zorder=5,
        )

    # Mine polygon boundaries
    for mine_id, xy in mine_segments:
        ax.plot(
            xy[:, 0], xy[:, 1],
            linewidth=1.5,
            linestyle="-",
            zorder=10,
        )

    # Mine labels
    for mine_id, cx, cy in mine_labels:
        ax.text(
            cx, cy, f"OC{mine_id}",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=20,
        )

    # Singleton receptor
    ax.plot(
        RECEPTOR_LON,
        RECEPTOR_LAT,
        marker="*",
        markersize=18,
        markeredgecolor="black",
        markerfacecolor="red",
        zorder=30,
    )

    ax.text(
        RECEPTOR_LON + 0.015,
        RECEPTOR_LAT + 0.015,
        "Singleton",
        fontsize=12,
        fontweight="bold",
        zorder=30,
    )

    ax.set_xlim(np.nanmin(lon), np.nanmax(lon))
    ax.set_ylim(np.nanmin(lat), np.nanmax(lat))

    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)

    ax.set_title(
        "FLEXPART-WRF Backward Footprint — Singleton\n"
        f"{time_string} UTC | 24 h backward",
        fontsize=14,
    )

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label("CONC (s m$^3$ kg$^{-1}$)", fontsize=11)

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.4,
        zorder=2,
    )

    plt.tight_layout()
    plt.savefig(output_png, dpi=250, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(
        f"  Saved: {output_png}\n"
        f"    min={np.nanmin(foot):.6g}, "
        f"max={np.nanmax(foot):.6g}, "
        f"nonzero={np.count_nonzero(foot)}"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():
    day_dir = os.path.join(BASE_DIR, f"{SELECTED_DAY}_*")

    run_dirs = sorted(
        d for d in glob.glob(day_dir)
        if os.path.isdir(d)
    )

    # Find the matching footprint files in the run directories.
    footprint_files = []

    for run_dir in run_dirs:
        pattern = os.path.join(
            run_dir,
            f"flxout_d01_{SELECTED_DAY}_*.nc"
        )
        footprint_files.extend(glob.glob(pattern))

    footprint_files = sorted(set(footprint_files))

    if not footprint_files:
        raise FileNotFoundError(
            f"No footprint files found for {SELECTED_DAY}.\n"
            f"Searched under: {os.path.join(BASE_DIR, SELECTED_DAY + '_*')}\n"
            "Check the selected date and actual directory/file names."
        )

    print("=" * 72)
    print("SINGLETON DAILY FOOTPRINT PLOTTING")
    print("=" * 72)
    print("Selected day :", SELECTED_DAY)
    print("Files found  :", len(footprint_files))
    print("Output dir   :", OUTPUT_DIR)

    # Use the first footprint's run directory for the grid header.
    first_run_dir = os.path.dirname(footprint_files[0])
    header_file = os.path.join(first_run_dir, "header_d01.nc")

    if not os.path.exists(header_file):
        raise FileNotFoundError(
            f"Grid header not found: {header_file}"
        )

    lon, lat, lon_edge, lat_edge = read_grid(header_file)

    # Read each footprint once. This allows a common colour scale
    # to be calculated for the selected day.
    footprints = []
    all_positive = []

    print("\nReading daily footprint files...")

    for foot_file in footprint_files:
        try:
            foot, time_string = read_footprint(foot_file)

            if foot.shape != lon.shape:
                print(
                    f"  SKIP shape mismatch: {foot_file}\n"
                    f"    footprint={foot.shape}, grid={lon.shape}"
                )
                continue

            npositive = np.count_nonzero(foot > 0)
            print(
                f"  Read {os.path.basename(foot_file)}"
                f" | time={time_string}"
                f" | nonzero={npositive}"
            )

            footprints.append((foot_file, foot, time_string))

            positive = foot[foot > 0]
            if positive.size:
                all_positive.append(positive)

        except Exception as exc:
            print(f"  WARNING: failed to read {foot_file}")
            print(f"    {exc}")

    if not footprints:
        raise RuntimeError(
            f"No readable footprint files for {SELECTED_DAY}."
        )

    if not all_positive:
        if not PLOT_ZERO_FOOTPRINTS:
            raise RuntimeError(
                "All readable footprints have zero values. "
                "Set PLOT_ZERO_FOOTPRINTS=True to save empty maps."
            )
        common_vmin, common_vmax = 1.0, 10.0
    else:
        positive_all = np.concatenate(all_positive)

        common_vmin = float(
            np.percentile(positive_all, VMIN_PERCENTILE)
        )
        common_vmax = float(
            np.percentile(positive_all, VMAX_PERCENTILE)
        )

        if common_vmin <= 0:
            common_vmin = float(positive_all.min())

        if common_vmax <= common_vmin:
            common_vmax = float(positive_all.max())

        if common_vmax <= common_vmin:
            common_vmax = common_vmin * 10.0

    print("\nDaily logarithmic colour scale:")
    print("  vmin =", common_vmin)
    print("  vmax =", common_vmax)

    coast_segments = read_coastline(lon, lat)
    mine_segments, mine_labels = read_mines()

    print("\nPlotting footprints...")

    saved = 0
    skipped = 0

    for foot_file, foot, time_string in footprints:
        # Use the actual output timestamp where possible.
        # Strip punctuation so the filename is filesystem-friendly.
        time_tag = "".join(
            ch for ch in time_string if ch.isdigit()
        )

        if not time_tag:
            time_tag = os.path.splitext(
                os.path.basename(foot_file)
            )[0].replace("flxout_d01_", "")

        output_png = os.path.join(
            OUTPUT_DIR,
            f"FLEXPART_WRF_Singleton_{time_tag}_coast.png"
        )

        # Use a shared colour scale by default, enabling comparison
        # of footprint strengths across hours of the selected day.
        if USE_COMMON_LOG_SCALE:
            vmin, vmax = common_vmin, common_vmax
        else:
            positive = foot[foot > 0]

            if positive.size == 0:
                vmin, vmax = 1.0, 10.0
            else:
                vmin = float(np.percentile(positive, VMIN_PERCENTILE))
                vmax = float(np.percentile(positive, VMAX_PERCENTILE))

                if vmin <= 0:
                    vmin = float(positive.min())
                if vmax <= vmin:
                    vmax = float(positive.max())
                if vmax <= vmin:
                    vmax = vmin * 10.0

        did_save = plot_one(
            foot,
            time_string,
            lon,
            lat,
            lon_edge,
            lat_edge,
            coast_segments,
            mine_segments,
            mine_labels,
            vmin,
            vmax,
            output_png,
        )

        if did_save:
            saved += 1
        else:
            skipped += 1

    print("\n" + "=" * 72)
    print("DAILY FOOTPRINT PLOTTING COMPLETE")
    print("=" * 72)
    print("Selected day       :", SELECTED_DAY)
    print("Readable files     :", len(footprints))
    print("PNG files saved    :", saved)
    print("All-zero skipped   :", skipped)
    print("Output directory   :", OUTPUT_DIR)


if __name__ == "__main__":
    main()

