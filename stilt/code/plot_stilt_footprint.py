#!/usr/bin/env python3

"""
===============================================================================
INTERACTIVE STILT FOOTPRINT DISPLAY
===============================================================================

Displays each STILT footprint layer separately.

For each footprint:
    - STILT footprint is plotted
    - 13 Upper Hunter mine polygons are overlaid
    - Mine names are labelled
    - Singleton receptor is shown
    - Footprint is displayed on logarithmic scale
    - Press a key or click in the plot window to continue

The STILT NetCDF variable has dimensions:

    foot(time, lat, lon)

===============================================================================
"""

import os
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

FOOT_FILE = (
    "/mnt/scratch_lustre/duch/stilt/upperhunter/"
    "test_UH_ARL_Jan10/out_oldgood/by-id/"
    "202601100200_151.16132_-32.57123_5_foot.nc"
)

FOOT_FILE = "/mnt/scratch_lustre/duch/stilt/upperhunter/test_UH_ARL_Jan10/out_oldgood/by-id/202601100200_151.16132_-32.57123_5/202601100200_151.16132_-32.57123_5_foot.nc"

MINE_DIR = (
    "/home/duch/flexpart-wrf/emission_UH/"
    "RH_upperHunter"
)

OUTPUT_DIR = (
    "/mnt/scratch_lustre/duch/stilt/upperhunter/"
    "test_UH_ARL_Jan10/plots_python"
)

# Singleton receptor
RECEPTOR_LON = 151.16132
RECEPTOR_LAT = -32.57123

# Mine shapefiles
MINE_FILES = [
    os.path.join(MINE_DIR, f"OC{i}.shp")
    for i in range(1, 14)
]


# =============================================================================
# CREATE OUTPUT DIRECTORY
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# LOAD STILT FOOTPRINT
# =============================================================================

print("=" * 60)
print("Loading STILT footprint")
print("=" * 60)

ds = xr.open_dataset(FOOT_FILE)

print(ds)

lon = ds["lon"].values
lat = ds["lat"].values
time = ds["time"].values

foot = ds["foot"]

print()
print("Footprint dimensions:")
print("    ", foot.dims)
print("Footprint shape:")
print("    ", foot.shape)

print()
print("Longitude:")
print("    ", lon.min(), "to", lon.max())

print("Latitude:")
print("    ", lat.min(), "to", lat.max())

print("Number of footprint times:")
print("    ", len(time))


# =============================================================================
# LOAD MINE SHAPEFILES
# =============================================================================

print()
print("=" * 60)
print("Loading mine shapefiles")
print("=" * 60)

mines = []

for i, filename in enumerate(MINE_FILES, start=1):

    if not os.path.exists(filename):
        print(f"WARNING: {filename} does not exist")
        continue

    gdf = gpd.read_file(filename)

    # Ensure geographic coordinates
    if gdf.crs is not None:
        gdf = gdf.to_crs("EPSG:4326")

    mines.append((f"OC{i}", gdf))

    print(f"Loaded OC{i}: {filename}")


print()
print(f"Number of mines loaded: {len(mines)}")


# =============================================================================
# CALCULATE CELL EDGES
# =============================================================================

# STILT lon/lat values are cell centres.
# pcolormesh needs cell edges.

dx = np.median(np.diff(lon))
dy = np.median(np.diff(lat))

lon_edges = np.concatenate(
    (
        [lon[0] - dx / 2],
        (lon[:-1] + lon[1:]) / 2,
        [lon[-1] + dx / 2]
    )
)

lat_edges = np.concatenate(
    (
        [lat[0] - dy / 2],
        (lat[:-1] + lat[1:]) / 2,
        [lat[-1] + dy / 2]
    )
)


# =============================================================================
# FOOTPRINT PLOT FUNCTION
# =============================================================================

def draw_footprint(z, title, output_file=None):

    """
    Draw one STILT footprint.

    z must have dimensions:

        lat x lon

    i.e.

        (133, 178)
    """

    print()
    print("Footprint array shape:", z.shape)

    # -------------------------------------------------------------------------
    # Check dimensions
    # -------------------------------------------------------------------------

    expected_shape = (len(lat), len(lon))

    if z.shape != expected_shape:

        raise ValueError(
            f"Unexpected footprint shape {z.shape}. "
            f"Expected {expected_shape} = (lat, lon)."
        )

    # -------------------------------------------------------------------------
    # Remove zero values before logarithm
    # -------------------------------------------------------------------------

    z_plot = np.where(z > 0, z, np.nan)

    positive = z[z > 0]

    if len(positive) > 0:

        print("Footprint minimum positive:",
              positive.min())

        print("Footprint maximum:",
              positive.max())

    else:

        print("WARNING: footprint contains no positive values")

    # -------------------------------------------------------------------------
    # Log10 footprint
    # -------------------------------------------------------------------------

    logz = np.full_like(z, np.nan, dtype=float)

    mask = z > 0

    logz[mask] = np.log10(z[mask])

    # -------------------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 9))

    # -------------------------------------------------------------------------
    # Plot footprint
    # -------------------------------------------------------------------------

    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        logz,
        shading="flat",
        cmap="viridis"
    )

    cbar = fig.colorbar(mesh, ax=ax)

    cbar.set_label(
        r"log$_{10}$(STILT footprint)"
    )

    # -------------------------------------------------------------------------
    # Plot mine polygons
    # -------------------------------------------------------------------------

    for mine_name, gdf in mines:

        gdf.boundary.plot(
            ax=ax,
            linewidth=1.5,
            edgecolor="red"
        )

        # Mine centroid
        #centroid = gdf.geometry.union_all().centroid
        centroid = gdf.geometry.unary_union.centroid

        ax.text(
            centroid.x,
            centroid.y,
            mine_name,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center"
        )

    # -------------------------------------------------------------------------
    # Plot receptor
    # -------------------------------------------------------------------------

    ax.plot(
        RECEPTOR_LON,
        RECEPTOR_LAT,
        marker="*",
        markersize=18,
        markeredgecolor="black",
        markerfacecolor="white",
        zorder=10
    )

    ax.text(
        RECEPTOR_LON + 0.015,
        RECEPTOR_LAT + 0.015,
        "Singleton",
        fontsize=11,
        fontweight="bold",
        zorder=10
    )

    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.set_title(title)

    ax.set_xlim(lon_edges[0], lon_edges[-1])
    ax.set_ylim(lat_edges[0], lat_edges[-1])

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Save PNG if requested
    # -------------------------------------------------------------------------

    if output_file is not None:

        fig.savefig(
            output_file,
            dpi=200,
            bbox_inches="tight"
        )

        print("Saved:")
        print("   ", output_file)

    return fig


# =============================================================================
# INTERACTIVE DISPLAY
# =============================================================================

print()
print("=" * 60)
print("INTERACTIVE STILT FOOTPRINT DISPLAY")
print("=" * 60)

print()
print("Each footprint will be displayed separately.")
print("Click the plot or press a key to continue.")
print()


# Make sure matplotlib uses interactive mode
plt.ion()


# =============================================================================
# LOOP THROUGH FOOTPRINT TIMES
# =============================================================================

for k in range(len(time)):

    print()
    print("=" * 60)
    print(f"Footprint {k + 1} of {len(time)}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Convert time to readable string
    # -------------------------------------------------------------------------

    timestamp = np.datetime_as_string(
        time[k],
        unit="s"
    )

    print("Time:", timestamp, "UTC")

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # xarray reads the NetCDF dimensions as:
    #
    #     time, lat, lon
    #
    # Therefore select:
    #
    #     foot[k, :, :]
    #
    # This produces:
    #
    #     lat x lon = 133 x 178
    # -------------------------------------------------------------------------

    z = foot.isel(time=k).values

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    title = (
        "STILT footprint\n"
        f"Singleton — {timestamp} UTC"
    )

    fig = draw_footprint(
        z,
        title
    )

    plt.show(block=False)

    # -------------------------------------------------------------------------
    # Wait for user interaction
    #
    # Any key press or mouse click in the figure continues.
    # -------------------------------------------------------------------------

    print()
    print("Click the plot or press a key to continue...")

    while True:

        result = plt.waitforbuttonpress(
            timeout=0.1
        )

        if result is not None:
            break

    # -------------------------------------------------------------------------
    # Close current figure
    # -------------------------------------------------------------------------

    plt.close(fig)


# =============================================================================
# CLOSE DATASET
# =============================================================================

ds.close()

print()
print("=" * 60)
print("Finished displaying all footprints")
print("=" * 60)

