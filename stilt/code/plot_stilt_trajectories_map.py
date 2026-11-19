#!/usr/bin/env python3

"""
Plot STILT backward trajectories for one receptor/time.

Adds:
    - Mine polygons OC1-OC13
    - Receptor
    - Singleton
    - Merriwa
    - Stockton (or Newcastle)
    - Australian coastline
    - Time-coloured trajectories

The STILT .rds file contains:
    x$particle

Because pyreadr cannot reliably read the STILT list structure,
Rscript is used to extract x$particle to CSV first.
"""

import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

# Cartopy for coastline
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ============================================================
# USER SETTINGS
# ============================================================

DATE_TIME = "202601051200"

TRAJ_DIR = (
    "/mnt/scratch_lustre/duch/stilt/upperhunter/"
    "arl_mine_jan2026/footprints/Muswellbrook/particles"
)

MINE_DIR = (
    "/home/duch/flexpart-wrf/emission_UH/"
    "RH_upperHunter"
)

OUTPUT_DIR = (
    "/mnt/scratch_lustre/duch/stilt/upperhunter/"
    "arl_mine_jan2026/footprints/Muswellbrook/"
    "trajectory_plots"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# RECEPTOR
# ============================================================

RECEPTOR_LON = 150.89297
RECEPTOR_LAT = -32.26202
RECEPTOR_NAME = "Muswellbrook"


# ============================================================
# OTHER PLACES
# ============================================================

PLACES = {
    "Muswellbrook": (150.89297, -32.26202),
    "Singleton":    (151.16132, -32.57123),
    "Merriwa":      (150.35767, -32.14097),

    # Choose ONE of these:
    "Stockton":     (151.7810, -32.9070),
    # "Newcastle": (151.7817, -32.9283),
}


# ============================================================
# FIND RDS FILE
# ============================================================

rds_file = os.path.join(
    TRAJ_DIR,
    f"{DATE_TIME}_{RECEPTOR_LON:.5f}_{RECEPTOR_LAT:.5f}_5_traj.rds"
)

if not os.path.exists(rds_file):
    raise FileNotFoundError(
        f"Trajectory file not found:\n{rds_file}"
    )

print("Trajectory file:")
print(rds_file)


# ============================================================
# READ STILT RDS USING R
# ============================================================

with tempfile.NamedTemporaryFile(
    suffix=".csv",
    delete=False
) as tmp:

    csv_file = tmp.name


r_code = r'''
args <- commandArgs(trailingOnly = TRUE)

rds_file <- args[1]
csv_file <- args[2]

x <- readRDS(rds_file)

if (!is.list(x)) {
    stop("RDS object is not a list")
}

if (!"particle" %in% names(x)) {
    stop("RDS object does not contain $particle")
}

particle <- x$particle

if (!is.data.frame(particle)) {
    stop("x$particle is not a data.frame")
}

write.csv(
    particle,
    csv_file,
    row.names = FALSE
)
'''

cmd = [
    "Rscript",
    "-e",
    r_code,
    rds_file,
    csv_file
]

print("\nExtracting particle data using R...")

subprocess.run(
    cmd,
    check=True
)


# ============================================================
# READ PARTICLE DATA
# ============================================================

particle = pd.read_csv(csv_file)

os.remove(csv_file)

print("\nParticle data:")
print(particle.head())

print("\nNumber of rows:", len(particle))
print("Number of particles:", particle["indx"].nunique())
print("Time range:",
      particle["time"].min(),
      "to",
      particle["time"].max())


# ============================================================
# MINE SHAPEFILES
# ============================================================

mine_files = []

for i in range(1, 14):

    shp = os.path.join(
        MINE_DIR,
        f"OC{i}.shp"
    )

    if os.path.exists(shp):
        mine_files.append((i, shp))
    else:
        print(f"WARNING: {shp} not found")


# ============================================================
# CREATE MAP
# ============================================================

fig = plt.figure(
    figsize=(12, 10)
)

ax = plt.axes(
    projection=ccrs.PlateCarree()
)


# ============================================================
# MAP EXTENT
# ============================================================

# Slightly larger than the STILT footprint grid so that
# Singleton, Merriwa and the coast are visible.

ax.set_extent(
    [
        150.20,   # west
        152.05,   # east
        -33.25,   # south
        -31.85    # north
    ],
    crs=ccrs.PlateCarree()
)


# ============================================================
# COASTLINE / MAP FEATURES
# ============================================================

# Ocean
ax.add_feature(
    cfeature.OCEAN,
    zorder=0
)

# Land
ax.add_feature(
    cfeature.LAND,
    zorder=0
)

# Coastline
ax.add_feature(
    cfeature.COASTLINE,
    linewidth=1.2,
    zorder=2
)

# Optional borders
ax.add_feature(
    cfeature.BORDERS,
    linewidth=0.5,
    edgecolor="gray",
    zorder=2
)


# ============================================================
# GRID
# ============================================================

gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.5,
    linestyle="--",
    alpha=0.5
)

gl.top_labels = False
gl.right_labels = False

gl.xlabel_style = {
    "size": 10
}

gl.ylabel_style = {
    "size": 10
}


# ============================================================
# PLOT TRAJECTORIES
# ============================================================

print("\nPlotting trajectories...")

# Use a continuous colour map for backward time
cmap = plt.cm.turbo

tmin = particle["time"].min()
tmax = particle["time"].max()

norm = plt.Normalize(
    vmin=tmin,
    vmax=tmax
)


# Each indx = one STILT particle
for particle_id, df in particle.groupby("indx"):

    df = df.sort_values("time")

    # Remove invalid coordinates
    df = df.dropna(
        subset=["long", "lati"]
    )

    if len(df) < 2:
        continue

    # Plot each trajectory as coloured segments.
    # A simple approach is to colour the entire trajectory
    # according to its mean backward time.

    mean_time = df["time"].mean()

    ax.plot(
        df["long"],
        df["lati"],
        linewidth=0.45,
        alpha=0.18,
        color=cmap(norm(mean_time)),
        transform=ccrs.PlateCarree(),
        zorder=3
    )


# ============================================================
# PLOT MINE POLYGONS
# ============================================================

print("\nPlotting mines...")

for mine_id, shp in mine_files:

    try:

        gdf = gpd.read_file(shp)

        # Make sure CRS is geographic
        if gdf.crs is not None:
            gdf = gdf.to_crs("EPSG:4326")

        # Mine boundary
        gdf.boundary.plot(
            ax=ax,
            linewidth=1.4,
            color="black",
            zorder=5
        )

        # Mine centroid
        centroid = gdf.geometry.unary_union.centroid

        ax.text(
            centroid.x,
            centroid.y,
            f"OC{mine_id}",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            transform=ccrs.PlateCarree(),
            zorder=6,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.65,
                pad=1.5
            )
        )

    except Exception as e:

        print(
            f"WARNING: Could not plot OC{mine_id}: {e}"
        )


# ============================================================
# PLOT PLACES
# ============================================================

print("\nPlotting places...")

for name, (lon, lat) in PLACES.items():

    if name == RECEPTOR_NAME:

        # Receptor
        ax.plot(
            lon,
            lat,
            marker="*",
            markersize=17,
            markeredgecolor="black",
            markerfacecolor="white",
            markeredgewidth=1.2,
            transform=ccrs.PlateCarree(),
            zorder=10
        )

        ax.text(
            lon + 0.035,
            lat + 0.015,
            name,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
            transform=ccrs.PlateCarree(),
            zorder=11,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
                pad=2
            )
        )

    else:

        # Other towns
        ax.plot(
            lon,
            lat,
            marker="o",
            markersize=6,
            markeredgecolor="black",
            markerfacecolor="white",
            markeredgewidth=1.0,
            transform=ccrs.PlateCarree(),
            zorder=10
        )

        # Adjust labels depending on location
        if name == "Merriwa":

            dx = 0.025
            dy = 0.015

        elif name == "Singleton":

            dx = 0.025
            dy = -0.015

        elif name == "Stockton":

            dx = -0.025
            dy = 0.015

        else:

            dx = 0.025
            dy = 0.015

        ax.text(
            lon + dx,
            lat + dy,
            name,
            fontsize=10,
            fontweight="bold",
            ha="left" if dx > 0 else "right",
            va="bottom",
            transform=ccrs.PlateCarree(),
            zorder=11,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
                pad=1.5
            )
        )

# ============================================================
# TITLE
# ============================================================

from datetime import datetime, timedelta

# STILT run time is UTC
dt_utc = datetime.strptime(
    DATE_TIME,
    "%Y%m%d%H%M"
)

# AEST = UTC + 10
dt_aest = dt_utc + timedelta(hours=10)

n_particles = particle["indx"].nunique()

# Determine backward duration
backward_hours = abs(particle["time"].min())

title = (
    "STILT Backward Trajectories\n"
    f"{RECEPTOR_NAME} — "
    f"{dt_aest.strftime('%d %b %Y %H:%M')} AEST\n"
    f"{n_particles} particles • "
    f"5 m AGL • "
    f"{backward_hours:.0f} h backward"
)

# Use figure-level title rather than ax.set_title()
fig.suptitle(
    title,
    fontsize=17,
    fontweight="bold",
    y=0.98
)

# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    f"{DATE_TIME}_{RECEPTOR_NAME}_trajectories.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

print("\nSaved:")
print(output_file)

plt.show()


