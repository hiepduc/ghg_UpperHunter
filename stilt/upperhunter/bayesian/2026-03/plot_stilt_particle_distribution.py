
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot STILT particle trajectories and particle distributions.

Figures:
1. Backward particle trajectories.
2. Spatial density of particle positions.
3. Particle positions at selected backward ages.
4. Vertical particle distribution versus backward age.

The script reads nested STILT RDS files using rdata.

Time convention:
    time = -1  : 1 hour before receptor release
    time = -60 : 60 hours before receptor release

All release times are UTC.
"""

import os
import re
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import rdata


# ============================================================
# USER SETTINGS
# ============================================================

RDS_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/"
    "upperhunter/arl_mine_2026/2026-03/footprints/"
    "Merriwa/particles/"
    "202603010100_150.35767_-32.14097_5_traj.rds"
)

OUTPUT_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/"
    "upperhunter/bayesian/2026-03/"
    "STILT_particle_plots"
)

MINE_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/"
    "emission_location"
)

COASTLINE_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/"
    "coastline/ne_50m_coastline.shp"
)

# Selected backward ages
SELECTED_AGES = [6, 12, 24, 36, 48, 60]

# Map padding in degrees
MAP_PADDING = 0.025

# Hexbin resolution
HEXBIN_GRIDSIZE = 100

# Mine labels
SHOW_MINE_LABELS = True

# Figure resolution
DPI = 400

# Large figure sizes for Figures 1 and 2
MAP_FIGSIZE = (22, 18)

# Figure sizes for Figures 3 and 4
AGE_FIGSIZE = (19, 13)
VERTICAL_FIGSIZE = (18, 12)

# Trajectory appearance
TRAJECTORY_LINEWIDTH = 0.85
TRAJECTORY_ALPHA = 0.45

# Particle density appearance
DENSITY_MINCNT = 1

# Map background appearance
COASTLINE_LINEWIDTH = 0.9
MINE_LINEWIDTH = 1.4


# ============================================================
# GENERAL SETTINGS
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

receptor_name = Path(RDS_FILE).parents[1].name

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.titlesize": 22,
    "axes.linewidth": 1.2,
    "savefig.dpi": DPI,
})


# ============================================================
# READ STILT RDS FILE
# ============================================================

print("=" * 70)
print("READING STILT PARTICLE DATA")
print("=" * 70)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rds_data = rdata.read_rds(RDS_FILE)

if not isinstance(rds_data, dict):
    raise TypeError("The RDS object is not a dictionary.")

if "particle" not in rds_data:
    raise KeyError("Cannot find 'particle' in the RDS object.")

particle = rds_data["particle"].copy()

if not isinstance(particle, pd.DataFrame):
    particle = pd.DataFrame(particle)

particle.columns = [str(c) for c in particle.columns]

required_columns = [
    "time", "indx", "long", "lati", "zagl", "foot"
]

missing = [
    c for c in required_columns
    if c not in particle.columns
]

if missing:
    raise ValueError(
        f"Missing required particle columns: {missing}"
    )

particle = particle.replace(
    [np.inf, -np.inf], np.nan
)

for column in required_columns:
    particle[column] = pd.to_numeric(
        particle[column], errors="coerce"
    )

particle = particle.dropna(
    subset=[
        "time", "indx", "long",
        "lati", "zagl"
    ]
).copy()

particle["age_hours"] = -particle["time"]

if (particle["age_hours"] < 0).any():
    raise ValueError(
        "Positive time values were found. "
        "Check the STILT time convention."
    )

particle["indx"] = particle["indx"].astype(int)

particle = particle.sort_values(
    ["indx", "age_hours"]
).reset_index(drop=True)

print(f"\nInput file: {RDS_FILE}")
print(f"Number of records: {len(particle):,}")
print(
    f"Number of particles: "
    f"{particle['indx'].nunique():,}"
)
print(
    f"Backward simulation: "
    f"{particle['age_hours'].min():.0f} to "
    f"{particle['age_hours'].max():.0f} hours"
)


# ============================================================
# EXTRACT RECEPTOR INFORMATION
# ============================================================

receptor = rds_data.get("receptor", {})

receptor_lat = float(
    np.asarray(receptor["lati"]).flatten()[0]
)

receptor_lon = float(
    np.asarray(receptor["long"]).flatten()[0]
)

receptor_height = float(
    np.asarray(receptor["zagl"]).flatten()[0]
)

basename = os.path.basename(RDS_FILE)

match = re.match(r"(\d{12})_", basename)

if match:
    release_time = datetime.strptime(
        match.group(1), "%Y%m%d%H%M"
    )
else:
    raise ValueError(
        "Could not extract release time from RDS filename."
    )

print(f"\nReceptor: {receptor_name}")
print(f"Receptor longitude: {receptor_lon:.5f}")
print(f"Receptor latitude:  {receptor_lat:.5f}")
print(f"Receptor height:    {receptor_height:.1f} m")
print(
    "Release time (UTC): "
    + release_time.strftime("%Y-%m-%d %H:%M")
)


# ============================================================
# LOAD MINE SHAPEFILES
# ============================================================

print("\n" + "=" * 70)
print("LOADING MINE BOUNDARIES")
print("=" * 70)

mine_layers = []

for i in range(1, 14):

    mine_name = f"OC{i}"

    shp_file = os.path.join(
        MINE_DIR, f"{mine_name}.shp"
    )

    if not os.path.exists(shp_file):
        print(f"WARNING: Missing {shp_file}")
        continue

    try:
        mine = gpd.read_file(shp_file)

        if mine.empty:
            continue

        if mine.crs is not None:
            mine = mine.to_crs("EPSG:4326")
        else:
            print(
                f"WARNING: {mine_name} has no CRS. "
                "Assuming longitude/latitude."
            )

        mine_layers.append((mine_name, mine))

        print(f"Loaded {mine_name}")

    except Exception as exc:
        print(
            f"WARNING: Could not read {shp_file}: {exc}"
        )


# ============================================================
# LOAD COASTLINE
# ============================================================

coastline = None

if os.path.exists(COASTLINE_FILE):

    try:
        coastline = gpd.read_file(COASTLINE_FILE)

        if coastline.crs is not None:
            coastline = coastline.to_crs("EPSG:4326")
        else:
            print(
                "WARNING: Coastline CRS is undefined. "
                "Assuming longitude/latitude."
            )

        print("\nCoastline loaded.")

    except Exception as exc:
        print(
            f"WARNING: Could not read coastline: {exc}"
        )

else:
    print("\nWARNING: Coastline shapefile not found.")


# ============================================================
# DEFINE MAP EXTENT
# ============================================================

lon_min = min(
    particle["long"].min(),
    receptor_lon
) - MAP_PADDING

lon_max = max(
    particle["long"].max(),
    receptor_lon
) + MAP_PADDING

lat_min = min(
    particle["lati"].min(),
    receptor_lat
) - MAP_PADDING

lat_max = max(
    particle["lati"].max(),
    receptor_lat
) + MAP_PADDING

# Geographic aspect correction:
# One degree of longitude is shorter than one degree
# of latitude at the receptor latitude.

mean_lat = np.deg2rad(
    (lat_min + lat_max) / 2.0
)

longitude_scale = np.cos(mean_lat)

# Preserve the geographic extent while allowing the
# map axes to display longitude and latitude correctly.


# ============================================================
# CLIP MAP LAYERS TO THE DISPLAY EXTENT
# ============================================================

map_bbox = (
    lon_min,
    lat_min,
    lon_max,
    lat_max
)

if coastline is not None:

    try:
        coastline_map = gpd.clip(
            coastline,
            gpd.GeoDataFrame(
                geometry=gpd.GeoSeries.from_bbox(map_bbox),
                crs="EPSG:4326"
            )
        )
    except Exception:
        coastline_map = coastline
else:
    coastline_map = None

mine_map_layers = []

for mine_name, mine in mine_layers:

    try:
        clipped_mine = gpd.clip(
            mine,
            gpd.GeoDataFrame(
                geometry=gpd.GeoSeries.from_bbox(map_bbox),
                crs="EPSG:4326"
            )
        )

        if not clipped_mine.empty:
            mine_map_layers.append(
                (mine_name, clipped_mine)
            )

    except Exception:
        mine_map_layers.append(
            (mine_name, mine)
        )


# ============================================================
# FUNCTION: ADD MAP BACKGROUND
# ============================================================

def add_map_background(
    ax,
    show_labels=True,
    show_receptor=True
):

    # Coastline
    if coastline_map is not None and not coastline_map.empty:

        coastline_map.plot(
            ax=ax,
            color="none",
            edgecolor="grey",
            linewidth=COASTLINE_LINEWIDTH,
            zorder=1
        )

    # Mine boundaries
    for mine_name, mine in mine_map_layers:

        mine.boundary.plot(
            ax=ax,
            color="black",
            linewidth=MINE_LINEWIDTH,
            zorder=3
        )

        if show_labels and SHOW_MINE_LABELS:

            for geom in mine.geometry:

                if geom is None or geom.is_empty:
                    continue

                point = geom.representative_point()

                ax.text(
                    point.x,
                    point.y,
                    mine_name,
                    fontsize=12,
                    color="black",
                    ha="center",
                    va="center",
                    zorder=5,
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.8,
                        pad=2
                    )
                )

    # Receptor
    if show_receptor:

        ax.scatter(
            receptor_lon,
            receptor_lat,
            marker="*",
            s=350,
            color="red",
            edgecolor="black",
            linewidth=1.2,
            label="Receptor",
            zorder=10
        )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # Correct geographic aspect ratio.
    ax.set_aspect(
        1.0 / longitude_scale,
        adjustable="box"
    )

    ax.set_xlabel(
        "Longitude (°E)",
        fontsize=19,
        labelpad=10
    )

    ax.set_ylabel(
        "Latitude (°)",
        fontsize=19,
        labelpad=10
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.8,
        alpha=0.6
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=16,
        width=1.2,
        length=6
    )


# ============================================================
# FUNCTION: SAVE FIGURE
# ============================================================

def save_figure(fig, filename):

    output_file = os.path.join(
        OUTPUT_DIR,
        filename
    )

    fig.savefig(
        output_file,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.3,
        facecolor="white"
    )

    print(f"Saved: {output_file}")


# ============================================================
# FIGURE 1: BACKWARD PARTICLE TRAJECTORIES
# ============================================================

print("\nCreating Figure 1: Particle trajectories")

fig, ax = plt.subplots(
    figsize=MAP_FIGSIZE,
    constrained_layout=True
)

cmap = plt.cm.turbo

norm = Normalize(
    vmin=0,
    vmax=60
)

# Plot all particle trajectories
for particle_id, group in particle.groupby("indx"):

    group = group.sort_values("age_hours")

    if len(group) < 2:
        continue

    points = group[
        ["long", "lati"]
    ].to_numpy()

    segments = np.stack(
        [points[:-1], points[1:]],
        axis=1
    )

    segment_age = (
        group["age_hours"].to_numpy()[:-1]
        + group["age_hours"].to_numpy()[1:]
    ) / 2.0

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=TRAJECTORY_LINEWIDTH,
        alpha=TRAJECTORY_ALPHA,
        zorder=2
    )

    lc.set_array(segment_age)

    ax.add_collection(lc)

add_map_background(ax)

ax.set_title(
    f"STILT Backward Particle Trajectories - {receptor_name}\n"
    f"Release: {release_time:%d %b %Y %H:%M} UTC",
    fontsize=23,
    fontweight="bold",
    pad=22
)

sm = ScalarMappable(
    cmap=cmap,
    norm=norm
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    fraction=0.045,
    pad=0.035,
    shrink=0.85
)

cbar.set_label(
    "Backward trajectory age (hours)",
    fontsize=18,
    labelpad=12
)

cbar.ax.tick_params(
    labelsize=15,
    width=1.2,
    length=5
)

save_figure(
    fig,
    "01_particle_trajectories.png"
)

plt.show()
plt.close(fig)


# ============================================================
# FIGURE 2: PARTICLE SPATIAL DENSITY
# ============================================================

print("\nCreating Figure 2: Particle density")

fig, ax = plt.subplots(
    figsize=MAP_FIGSIZE,
    constrained_layout=True
)

hb = ax.hexbin(
    particle["long"],
    particle["lati"],
    gridsize=HEXBIN_GRIDSIZE,
    mincnt=DENSITY_MINCNT,
    bins="log",
    cmap="inferno",
    linewidths=0.25,
    edgecolors="none",
    alpha=0.95,
    zorder=2
)

add_map_background(ax)

ax.set_title(
    f"STILT Particle Spatial Density - {receptor_name}\n"
    f"{particle['indx'].nunique():,} particles, "
    f"1-60 hours backward",
    fontsize=23,
    fontweight="bold",
    pad=22
)

cbar = fig.colorbar(
    hb,
    ax=ax,
    fraction=0.045,
    pad=0.035,
    shrink=0.85
)

cbar.set_label(
    "Particle count per hexagonal bin (log scale)",
    fontsize=18,
    labelpad=12
)

cbar.ax.tick_params(
    labelsize=15,
    width=1.2,
    length=5
)

save_figure(
    fig,
    "02_particle_spatial_density.png"
)

plt.show()
plt.close(fig)


# ============================================================
# FIGURE 3: PARTICLE DISTRIBUTION AT SELECTED AGES
# ============================================================

print("\nCreating Figure 3: Selected backward ages")

fig, axes = plt.subplots(
    2,
    3,
    figsize=AGE_FIGSIZE,
    sharex=True,
    sharey=True
)

axes = axes.flatten()

age_colours = plt.cm.turbo(
    np.linspace(
        0.05,
        0.95,
        len(SELECTED_AGES)
    )
)

for ax, age, colour in zip(
    axes,
    SELECTED_AGES,
    age_colours
):

    subset = particle[
        np.isclose(
            particle["age_hours"],
            age
        )
    ]

    ax.scatter(
        subset["long"],
        subset["lati"],
        s=22,
        color=colour,
        alpha=0.75,
        edgecolors="none",
        zorder=4
    )

    add_map_background(
        ax,
        show_labels=False
    )

    ax.set_title(
        f"{age} hours backward\n"
        f"{len(subset)} particles",
        fontsize=15,
        fontweight="bold"
    )

    ax.tick_params(labelsize=11)

fig.suptitle(
    f"Evolution of STILT Particle Distribution - {receptor_name}\n"
    f"Release: {release_time:%d %b %Y %H:%M} UTC",
    fontsize=20,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.94]
)

save_figure(
    fig,
    "03_particle_distribution_by_age.png"
)

plt.show()
plt.close(fig)


# ============================================================
# FIGURE 4: VERTICAL PARTICLE DISTRIBUTION
# ============================================================

print("\nCreating Figure 4: Vertical particle distribution")

fig, ax = plt.subplots(
    figsize=VERTICAL_FIGSIZE,
    constrained_layout=True
)

age_bins = np.arange(
    0,
    61,
    1
)

height_bins = np.arange(
    0,
    1601,
    25
)

hist = ax.hist2d(
    particle["age_hours"],
    particle["zagl"],
    bins=[
        age_bins,
        height_bins
    ],
    cmap="viridis",
    norm=colors.LogNorm(vmin=1),
    cmin=1
)

ax.set_xlabel(
    "Backward trajectory age (hours)",
    fontsize=18
)

ax.set_ylabel(
    "Height above ground level (m)",
    fontsize=18
)

ax.set_title(
    f"Vertical Distribution of STILT Particles - {receptor_name}",
    fontsize=22,
    fontweight="bold",
    pad=18
)

ax.set_xlim(0, 60)
ax.set_ylim(0, 1600)

ax.tick_params(
    labelsize=15
)

ax.grid(
    True,
    linestyle=":",
    linewidth=0.7,
    alpha=0.5
)

cbar = fig.colorbar(
    hist[3],
    ax=ax,
    fraction=0.045,
    pad=0.035
)

cbar.set_label(
    "Number of particle positions (log scale)",
    fontsize=17
)

cbar.ax.tick_params(labelsize=14)

save_figure(
    fig,
    "04_particle_vertical_distribution.png"
)

plt.show()
plt.close(fig)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("ALL FOUR FIGURES COMPLETED")
print("=" * 70)

for filename in sorted(os.listdir(OUTPUT_DIR)):

    if filename.endswith(".png"):
        print(
            os.path.join(
                OUTPUT_DIR,
                filename
            )
        )

print("\nOutput directory:")
print(OUTPUT_DIR)

