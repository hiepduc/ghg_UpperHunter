#!/usr/bin/env python3

"""
===============================================================================
STILT BACKWARD TRAJECTORY PLOT
===============================================================================

Plots STILT backward particle trajectories for a selected receptor/time.

The STILT RDS structure is:

    x
     |
     +-- file
     +-- receptor
     +-- particle
     |     +-- time
     |     +-- indx
     |     +-- long
     |     +-- lati
     |     +-- zagl
     |     +-- foot
     |     +-- ...
     |
     +-- params

The particle data are extracted with native R and passed to Python.

===============================================================================
"""

import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd


# =============================================================================
# USER SETTINGS
# =============================================================================

# -------------------------------------------------------------------------
# Select STILT receptor time
# -------------------------------------------------------------------------

DATE_TIME = "202601051200"

# -------------------------------------------------------------------------
# Receptor
# -------------------------------------------------------------------------

RECEPTOR_NAME = "Muswellbrook"

RECEPTOR_LON = 150.89297
RECEPTOR_LAT = -32.26202

# -------------------------------------------------------------------------
# STILT trajectory directory
# -------------------------------------------------------------------------

TRAJ_DIR = (
    "/mnt/scratch_lustre/duch/stilt/upperhunter/"
    "arl_mine_jan2026/footprints/Muswellbrook/particles"
)

# -------------------------------------------------------------------------
# Mine shapefiles
# -------------------------------------------------------------------------

MINE_DIR = (
    "/home/duch/flexpart-wrf/emission_UH/"
    "RH_upperHunter"
)

# -------------------------------------------------------------------------
# Output
# -------------------------------------------------------------------------

OUTPUT_DIR = (
    "/mnt/scratch_lustre/duch/stilt/upperhunter/"
    "arl_mine_jan2026/footprints/Muswellbrook/"
    "trajectory_plots"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =============================================================================
# TRAJECTORY FILE
# =============================================================================

traj_file = os.path.join(
    TRAJ_DIR,
    f"{DATE_TIME}_150.89297_-32.26202_5_traj.rds"
)

if not os.path.exists(traj_file):

    raise FileNotFoundError(
        f"\nTrajectory file does not exist:\n{traj_file}"
    )


print("=" * 75)
print("STILT BACKWARD TRAJECTORY PLOT")
print("=" * 75)

print()
print("Trajectory:")
print(traj_file)


# =============================================================================
# EXTRACT particle DATA USING R
# =============================================================================

print()
print("Extracting x$particle using R...")


tmp_csv = tempfile.NamedTemporaryFile(
    suffix=".csv",
    delete=False
)

tmp_csv.close()

tmp_csv_path = tmp_csv.name


r_code = r'''
args <- commandArgs(trailingOnly=TRUE)

rds_file <- args[1]
csv_file <- args[2]

# ------------------------------------------------------------
# Read STILT RDS
# ------------------------------------------------------------

x <- readRDS(rds_file)

# ------------------------------------------------------------
# Extract particle dataframe
# ------------------------------------------------------------

particle <- x$particle

if (!is.data.frame(particle)) {

    stop(
        "x$particle is not a data.frame"
    )
}

cat("Particle dataframe:\n")
print(dim(particle))

cat("\nColumns:\n")
print(names(particle))

# ------------------------------------------------------------
# Write only trajectory information needed by Python
# ------------------------------------------------------------

write.csv(
    particle,
    csv_file,
    row.names = FALSE
)

cat("\nCSV written:\n")
cat(csv_file)
cat("\n")
'''


result = subprocess.run(
    [
        "Rscript",
        "-e",
        r_code,
        traj_file,
        tmp_csv_path
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)


print()
print(result.stdout)


if result.returncode != 0:

    raise RuntimeError(
        "\nR failed to extract x$particle"
    )


# =============================================================================
# READ PARTICLES IN PYTHON
# =============================================================================

particle = pd.read_csv(
    tmp_csv_path
)

# Remove temporary CSV
try:
    os.remove(tmp_csv_path)
except Exception:
    pass


print()
print("=" * 75)
print("PARTICLE DATA")
print("=" * 75)

print()
print("Shape:")
print(particle.shape)

print()
print("Columns:")
print(list(particle.columns))


# =============================================================================
# CHECK REQUIRED COLUMNS
# =============================================================================

required = [
    "time",
    "indx",
    "long",
    "lati"
]

missing = [
    c for c in required
    if c not in particle.columns
]

if missing:

    raise RuntimeError(
        f"Missing required columns: {missing}"
    )


# =============================================================================
# CLEAN DATA
# =============================================================================

for col in [
    "time",
    "indx",
    "long",
    "lati",
    "zagl"
]:

    if col in particle.columns:

        particle[col] = pd.to_numeric(
            particle[col],
            errors="coerce"
        )


particle = particle.dropna(
    subset=[
        "time",
        "indx",
        "long",
        "lati"
    ]
)


print()
print(
    "Valid trajectory records:",
    len(particle)
)

print(
    "Number of particles:",
    particle["indx"].nunique()
)

print(
    "Backward time range:",
    particle["time"].min(),
    "to",
    particle["time"].max(),
    "hours"
)


# =============================================================================
# LOAD MINE POLYGONS
# =============================================================================

print()
print("=" * 75)
print("LOADING UPPER HUNTER MINES")
print("=" * 75)

mines = []

for i in range(1, 14):

    filename = os.path.join(
        MINE_DIR,
        f"OC{i}.shp"
    )

    if not os.path.exists(filename):

        print(
            f"WARNING: OC{i}.shp not found"
        )

        continue

    gdf = gpd.read_file(filename)

    if gdf.crs is not None:

        gdf = gdf.to_crs(
            "EPSG:4326"
        )

    mines.append(
        (f"OC{i}", gdf)
    )

    print(
        f"Loaded OC{i}"
    )


# =============================================================================
# CREATE PLOT
# =============================================================================

fig, ax = plt.subplots(
    figsize=(12, 10)
)


# =============================================================================
# PLOT ALL PARTICLES
# =============================================================================

for particle_id, group in particle.groupby(
    "indx",
    sort=False
):

    # STILT time is backward time.
    # Sort from receptor backwards.
    group = group.sort_values(
        "time",
        ascending=True
    )

    ax.plot(
        group["long"].values,
        group["lati"].values,
        linewidth=0.5,
        alpha=0.20
    )


# =============================================================================
# PLOT MINE POLYGONS
# =============================================================================

for mine_name, gdf in mines:

    gdf.boundary.plot(
        ax=ax,
        linewidth=1.5
    )

    # Compatible with older GeoPandas
    centroid = (
        gdf.geometry
        .unary_union
        .centroid
    )

    ax.text(
        centroid.x,
        centroid.y,
        mine_name,
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center"
    )


# =============================================================================
# PLOT RECEPTOR
# =============================================================================

ax.plot(
    RECEPTOR_LON,
    RECEPTOR_LAT,
    marker="*",
    markersize=24,
    markeredgecolor="black",
    markerfacecolor="white",
    markeredgewidth=1.5,
    zorder=20
)

ax.text(
    RECEPTOR_LON + 0.015,
    RECEPTOR_LAT + 0.015,
    RECEPTOR_NAME,
    fontsize=12,
    fontweight="bold",
    zorder=20
)


# =============================================================================
# TITLE
# =============================================================================

dt = pd.to_datetime(
    DATE_TIME,
    format="%Y%m%d%H%M",
    utc=True
)

time_string = dt.strftime(
    "%Y-%m-%d %H:%M UTC"
)

n_particles = particle["indx"].nunique()

title = (
    f"STILT Backward Trajectories\n"
    f"{RECEPTOR_NAME} — {time_string}\n"
    f"{n_particles} particles"
)

ax.set_title(
    title,
    fontsize=15,
    fontweight="bold"
)


# =============================================================================
# AXES
# =============================================================================

ax.set_xlabel(
    "Longitude",
    fontsize=12
)

ax.set_ylabel(
    "Latitude",
    fontsize=12
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.3
)

ax.set_aspect(
    "equal",
    adjustable="box"
)


# =============================================================================
# SAVE
# =============================================================================

output_file = os.path.join(
    OUTPUT_DIR,
    f"{DATE_TIME}_{RECEPTOR_NAME}_trajectories.png"
)

plt.tight_layout()

fig.savefig(
    output_file,
    dpi=250,
    bbox_inches="tight"
)

print()
print("=" * 75)
print("PLOT SAVED")
print("=" * 75)

print()
print(output_file)


# =============================================================================
# DISPLAY
# =============================================================================

plt.show()

