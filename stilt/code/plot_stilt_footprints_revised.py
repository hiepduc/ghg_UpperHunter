#!/usr/bin/env python3

"""
===============================================================================
STILT HOURLY + COMBINED FOOTPRINT PLOTS OVER UPPER HUNTER MINES
===============================================================================

For one STILT receptor observation time this script:

    1. Reads the corresponding *_foot.nc file.
    2. Plots every footprint time layer separately.
    3. Uses LOG10 scaling for the footprint.
    4. Uses a COMMON logarithmic colour scale for all hourly plots.
    5. Plots OC1-OC13 mine shapefiles.
    6. Adds Muswellbrook, Singleton, Merriwa and Stockton.
    7. Adds coastline.
    8. Creates a combined footprint by summing all time layers.
    9. Uses a separate logarithmic colour scale for the combined footprint.
   10. Saves the combined footprint as a NetCDF file.

Expected STILT NetCDF dimensions:

    foot(time, lat, lon)

xarray will read this as:

    (time, lat, lon)

===============================================================================
"""


# =============================================================================
# IMPORTS
# =============================================================================

import os
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =============================================================================
# USER SETTINGS
# =============================================================================

# -----------------------------------------------------------------------------
# STILT receptor observation/run time
#
# This is UTC.
#
# Example:
#
# 202601051200
#
# = 05 January 2026 12:00 UTC
# = 05 January 2026 22:00 AEST
# -----------------------------------------------------------------------------

DATE_TIME = "202601051200"


# -----------------------------------------------------------------------------
# Receptor
# -----------------------------------------------------------------------------

RECEPTOR_NAME = "Muswellbrook"

RECEPTOR_LON = 150.89297
RECEPTOR_LAT = -32.26202

RECEPTOR_ZAGL = 5


# =============================================================================
# DIRECTORIES
# =============================================================================

FOOTPRINT_DIR = (
    "/mnt/scratch_lustre/duch/stilt/upperhunter/"
    "arl_mine_jan2026/footprints/Muswellbrook/footprints"
)


MINE_DIR = (
    "/home/duch/flexpart-wrf/emission_UH/"
    "RH_upperHunter"
)


OUTPUT_DIR = (
    "/mnt/scratch_lustre/duch/stilt/upperhunter/"
    "arl_mine_jan2026/footprints/Muswellbrook/"
    "footprint_plots"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =============================================================================
# PLACES
# =============================================================================
#
# Coordinates are longitude, latitude.
#
# You can replace Stockton with Newcastle if preferred.
# =============================================================================

PLACES = {

    "Muswellbrook":
        (150.89297, -32.26202),

    "Singleton":
        (151.16132, -32.57123),

    "Merriwa":
        (150.35767, -32.14097),

    "Stockton":
        (151.7810, -32.9070),

    # Alternatively:
    #
    # "Newcastle":
    #     (151.7817, -32.9283),
}


# =============================================================================
# MAP EXTENT
# =============================================================================
#
# Expanded eastward so the coastline and Stockton are visible.
# =============================================================================

MAP_EXTENT = [
    150.20,       # west
    152.05,       # east
    -33.25,       # south
    -31.85        # north
]


# =============================================================================
# INPUT FOOTPRINT FILE
# =============================================================================

foot_file = os.path.join(
    FOOTPRINT_DIR,
    f"{DATE_TIME}_"
    f"{RECEPTOR_LON:.5f}_"
    f"{RECEPTOR_LAT:.5f}_"
    f"{RECEPTOR_ZAGL}_foot.nc"
)


if not os.path.exists(foot_file):

    raise FileNotFoundError(
        "\n"
        "============================================================\n"
        "FOOTPRINT FILE NOT FOUND\n"
        "============================================================\n"
        f"{foot_file}\n"
    )


# =============================================================================
# PRINT INFORMATION
# =============================================================================

print()
print("=" * 80)
print("STILT FOOTPRINT ANALYSIS")
print("=" * 80)

print()
print("Input footprint:")
print(foot_file)


# =============================================================================
# READ NETCDF
# =============================================================================

ds = xr.open_dataset(
    foot_file
)


print()
print("NetCDF structure:")
print(ds)


# =============================================================================
# CHECK FOOT VARIABLE
# =============================================================================

if "foot" not in ds:

    raise ValueError(
        "Variable 'foot' was not found in the NetCDF file."
    )


foot = ds["foot"]


# =============================================================================
# CHECK DIMENSIONS
# =============================================================================

print()
print("Original footprint dimensions:")
print(foot.dims)

print()
print("Original footprint shape:")
print(foot.shape)


required_dims = [
    "time",
    "lat",
    "lon"
]


for dim in required_dims:

    if dim not in foot.dims:

        raise ValueError(
            f"Required dimension '{dim}' "
            "was not found."
        )


# =============================================================================
# FORCE DIMENSION ORDER
# =============================================================================
#
# This guarantees:
#
#     time, lat, lon
#
# regardless of the order stored in the NetCDF.
# =============================================================================

foot = foot.transpose(
    "time",
    "lat",
    "lon"
)


# =============================================================================
# GRID INFORMATION
# =============================================================================

lon = foot["lon"].values
lat = foot["lat"].values

ntime = foot.sizes["time"]


print()
print("=" * 80)
print("FOOTPRINT GRID")
print("=" * 80)

print(
    f"Longitude: {len(lon)} points "
    f"({lon.min():.5f} to {lon.max():.5f})"
)

print(
    f"Latitude : {len(lat)} points "
    f"({lat.min():.5f} to {lat.max():.5f})"
)

print(
    f"Time layers: {ntime}"
)


# =============================================================================
# TIME INFORMATION
# =============================================================================

print()
print("NetCDF time coordinate:")

try:

    for k in range(ntime):

        print(
            f"  {k:3d}: {foot['time'].values[k]}"
        )

except Exception as e:

    print(
        "Could not print time coordinate:",
        e
    )


# =============================================================================
# LOAD MINE SHAPEFILES
# =============================================================================

print()
print("=" * 80)
print("LOADING MINE SHAPEFILES")
print("=" * 80)


mine_files = []


for i in range(1, 14):

    shp = os.path.join(
        MINE_DIR,
        f"OC{i}.shp"
    )

    if os.path.exists(shp):

        mine_files.append(
            (i, shp)
        )

        print(
            f"OC{i}: {shp}"
        )

    else:

        print(
            f"WARNING: OC{i}.shp not found"
        )


print()
print(
    f"Found {len(mine_files)} "
    "mine shapefiles."
)


# =============================================================================
# READ MINE SHAPEFILES ONCE
# =============================================================================

mine_data = []


for mine_id, shp in mine_files:

    try:

        gdf = gpd.read_file(
            shp
        )

        # Convert to WGS84 if required
        if gdf.crs is not None:

            gdf = gdf.to_crs(
                "EPSG:4326"
            )

        mine_data.append(
            (
                mine_id,
                gdf
            )
        )

    except Exception as e:

        print(
            f"WARNING: Could not read "
            f"OC{mine_id}: {e}"
        )


# =============================================================================
# FUNCTION: CREATE BASE MAP
# =============================================================================

def create_map():

    fig = plt.figure(
        figsize=(12, 10)
    )

    ax = plt.axes(
        projection=ccrs.PlateCarree()
    )

    # -------------------------------------------------------------------------
    # Map extent
    # -------------------------------------------------------------------------

    ax.set_extent(
        MAP_EXTENT,
        crs=ccrs.PlateCarree()
    )

    # -------------------------------------------------------------------------
    # Land
    # -------------------------------------------------------------------------

    ax.add_feature(
        cfeature.LAND,
        zorder=0
    )

    # -------------------------------------------------------------------------
    # Ocean
    # -------------------------------------------------------------------------

    ax.add_feature(
        cfeature.OCEAN,
        zorder=0
    )

    # -------------------------------------------------------------------------
    # Coastline
    # -------------------------------------------------------------------------

    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=1.2,
        zorder=2
    )

    # -------------------------------------------------------------------------
    # Country/state boundaries
    # -------------------------------------------------------------------------

    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.5,
        edgecolor="gray",
        zorder=2
    )

    # -------------------------------------------------------------------------
    # Grid
    # -------------------------------------------------------------------------

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

    return fig, ax


# =============================================================================
# FUNCTION: PLOT MINES
# =============================================================================

def plot_mines(ax):

    for mine_id, gdf in mine_data:

        # ---------------------------------------------------------------------
        # Mine boundary
        # ---------------------------------------------------------------------

        gdf.boundary.plot(
            ax=ax,
            linewidth=1.4,
            color="black",
            zorder=6
        )

        # ---------------------------------------------------------------------
        # Mine centroid
        # ---------------------------------------------------------------------

        try:

            # Compatible with older GeoPandas
            centroid = (
                gdf.geometry
                .unary_union
                .centroid
            )

            ax.text(
                centroid.x,
                centroid.y,
                f"OC{mine_id}",
                fontsize=8.5,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                transform=ccrs.PlateCarree(),
                zorder=7,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.70,
                    pad=1.2
                )
            )

        except Exception as e:

            print(
                f"WARNING: centroid "
                f"OC{mine_id}: {e}"
            )


# =============================================================================
# FUNCTION: PLOT PLACES
# =============================================================================

def plot_places(ax):

    for name, (lon_p, lat_p) in PLACES.items():

        # ---------------------------------------------------------------------
        # Receptor
        # ---------------------------------------------------------------------

        if name == RECEPTOR_NAME:

            ax.plot(
                lon_p,
                lat_p,
                marker="*",
                markersize=18,
                markeredgecolor="black",
                markerfacecolor="white",
                markeredgewidth=1.3,
                transform=ccrs.PlateCarree(),
                zorder=12
            )

            ax.text(
                lon_p + 0.035,
                lat_p + 0.015,
                name,
                fontsize=11,
                fontweight="bold",
                ha="left",
                va="bottom",
                transform=ccrs.PlateCarree(),
                zorder=13,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.80,
                    pad=2
                )
            )

        # ---------------------------------------------------------------------
        # Other towns
        # ---------------------------------------------------------------------

        else:

            ax.plot(
                lon_p,
                lat_p,
                marker="o",
                markersize=6,
                markeredgecolor="black",
                markerfacecolor="white",
                markeredgewidth=1.0,
                transform=ccrs.PlateCarree(),
                zorder=12
            )

            # -------------------------------------------------------------
            # Label offsets
            # -------------------------------------------------------------

            if name == "Merriwa":

                dx = 0.025
                dy = 0.015

            elif name == "Singleton":

                dx = 0.025
                dy = -0.015

            elif name == "Stockton":

                dx = -0.025
                dy = 0.015

            elif name == "Newcastle":

                dx = -0.025
                dy = 0.015

            else:

                dx = 0.025
                dy = 0.015

            ax.text(
                lon_p + dx,
                lat_p + dy,
                name,
                fontsize=10,
                fontweight="bold",
                ha=(
                    "left"
                    if dx > 0
                    else "right"
                ),
                va="bottom",
                transform=ccrs.PlateCarree(),
                zorder=13,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.80,
                    pad=1.5
                )
            )


# =============================================================================
# FUNCTION: GET LOG COLOUR SCALE
# =============================================================================

def get_log_limits(
    z,
    low_percentile=1.0,
    high_percentile=99.5
):

    z = np.asarray(
        z,
        dtype=float
    )

    positive = z[
        np.isfinite(z) &
        (z > 0)
    ]

    if len(positive) == 0:

        return None, None

    log_values = np.log10(
        positive
    )

    vmin = np.percentile(
        log_values,
        low_percentile
    )

    vmax = np.percentile(
        log_values,
        high_percentile
    )

    return vmin, vmax


# =============================================================================
# FUNCTION: PLOT FOOTPRINT
# =============================================================================

def plot_footprint(
    z,
    title,
    output_file,
    vmin_log=None,
    vmax_log=None
):

    # -------------------------------------------------------------------------
    # Create map
    # -------------------------------------------------------------------------

    fig, ax = create_map()

    # -------------------------------------------------------------------------
    # Convert to float
    # -------------------------------------------------------------------------

    z = np.asarray(
        z,
        dtype=float
    )

    # -------------------------------------------------------------------------
    # Remove invalid and non-positive values
    # -------------------------------------------------------------------------

    z[
        ~np.isfinite(z)
    ] = np.nan

    z[
        z <= 0
    ] = np.nan

    # -------------------------------------------------------------------------
    # Log10 transformation
    # -------------------------------------------------------------------------

    logz = np.log10(
        z
    )

    # -------------------------------------------------------------------------
    # Find finite values
    # -------------------------------------------------------------------------

    finite = logz[
        np.isfinite(logz)
    ]

    if len(finite) == 0:

        print(
            "WARNING: Footprint contains "
            "no positive values."
        )

        plt.close(
            fig
        )

        return


    # -------------------------------------------------------------------------
    # If limits weren't supplied, calculate them
    # -------------------------------------------------------------------------

    if vmin_log is None:

        vmin_log = np.percentile(
            finite,
            1
        )


    if vmax_log is None:

        vmax_log = np.percentile(
            finite,
            99.5
        )


    # -------------------------------------------------------------------------
    # Plot footprint
    # -------------------------------------------------------------------------

    mesh = ax.pcolormesh(
        lon,
        lat,
        logz,
        shading="auto",
        cmap="turbo",
        vmin=vmin_log,
        vmax=vmax_log,
        transform=ccrs.PlateCarree(),
        zorder=3
    )


    # -------------------------------------------------------------------------
    # Mine polygons
    # -------------------------------------------------------------------------

    plot_mines(
        ax
    )


    # -------------------------------------------------------------------------
    # Towns / receptor
    # -------------------------------------------------------------------------

    plot_places(
        ax
    )


    # -------------------------------------------------------------------------
    # Colourbar
    # -------------------------------------------------------------------------

    cbar = plt.colorbar(
        mesh,
        ax=ax,
        pad=0.02,
        shrink=0.80
    )


    cbar.set_label(
        "log₁₀ STILT footprint\n"
        "ppm (µmol⁻¹ m² s)",
        fontsize=11
    )


    # -------------------------------------------------------------------------
    # Use integer powers of 10 where possible
    # -------------------------------------------------------------------------

    tick_min = int(
        np.floor(
            vmin_log
        )
    )

    tick_max = int(
        np.ceil(
            vmax_log
        )
    )

    ticks = np.arange(
        tick_min,
        tick_max + 1
    )


    # Avoid ticks outside actual range
    ticks = ticks[
        (ticks >= vmin_log) &
        (ticks <= vmax_log)
    ]


    if len(ticks) > 0:

        cbar.set_ticks(
            ticks
        )

        cbar.set_ticklabels(
            [
                rf"$10^{{{int(t)}}}$"
                for t in ticks
            ]
        )


    # -------------------------------------------------------------------------
    # Title
    # -------------------------------------------------------------------------

    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
        y=0.98
    )


    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.25
    )


    print(
        f"Saved: {output_file}"
    )


    plt.show()

    plt.close(
        fig
    )


# =============================================================================
# COMMON LOG SCALE FOR ALL HOURLY FOOTPRINTS
# =============================================================================

print()
print("=" * 80)
print("CALCULATING COMMON LOGARITHMIC SCALE")
print("=" * 80)


all_values = []


for k in range(ntime):

    z = foot.isel(
        time=k
    ).values.astype(
        float
    )

    positive = z[
        np.isfinite(z) &
        (z > 0)
    ]

    if len(positive) > 0:

        all_values.append(
            positive
        )


if len(all_values) > 0:

    all_values = np.concatenate(
        all_values
    )

    log_values = np.log10(
        all_values
    )

    # -------------------------------------------------------------------------
    # Use percentiles so extreme individual pixels don't dominate
    # -------------------------------------------------------------------------

    global_vmin_log = np.percentile(
        log_values,
        1.0
    )

    global_vmax_log = np.percentile(
        log_values,
        99.5
    )

else:

    global_vmin_log = None
    global_vmax_log = None


print()
print(
    "COMMON HOURLY LOG10 SCALE:"
)

print(
    f"    vmin = {global_vmin_log}"
)

print(
    f"    vmax = {global_vmax_log}"
)


# =============================================================================
# PLOT EACH FOOTPRINT TIME LAYER
# =============================================================================

print()
print("=" * 80)
print("PLOTTING INDIVIDUAL FOOTPRINT LAYERS")
print("=" * 80)


for k in range(ntime):

    # -------------------------------------------------------------------------
    # Footprint
    # -------------------------------------------------------------------------

    z = foot.isel(
        time=k
    ).values


    # -------------------------------------------------------------------------
    # Get actual time from NetCDF
    # -------------------------------------------------------------------------

    try:

        time_value = (
            foot["time"]
            .values[k]
        )

        # Convert numpy datetime64
        time_seconds = (
            time_value
            .astype(
                "datetime64[s]"
            )
        )

        time_dt = (
            time_seconds
            .astype(datetime)
        )

        time_string_utc = (
            time_dt.strftime(
                "%d %b %Y %H:%M UTC"
            )
        )

        # AEST
        time_dt_aest = (
            time_dt +
            timedelta(hours=10)
        )

        time_string_aest = (
            time_dt_aest.strftime(
                "%d %b %Y %H:%M AEST"
            )
        )

    except Exception as e:

        print(
            "WARNING: Could not convert "
            f"NetCDF time: {e}"
        )

        time_string_utc = (
            f"Layer {k + 1}"
        )

        time_string_aest = ""


    # -------------------------------------------------------------------------
    # Determine nominal backward layer
    # -------------------------------------------------------------------------

    #
    # We display the actual NetCDF time rather than assuming
    # that layer k = exactly k hours backward.
    #

    title = (
        "STILT Backward Footprint\n"
        f"{RECEPTOR_NAME} — "
        f"Observation "
        f"{datetime.strptime(DATE_TIME, '%Y%m%d%H%M').strftime('%d %b %Y %H:%M')} UTC\n"
        f"Layer {k + 1}/{ntime} — "
        f"{time_string_utc} "
        f"({time_string_aest})"
    )


    # -------------------------------------------------------------------------
    # Output filename
    # -------------------------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{DATE_TIME}_"
        f"{RECEPTOR_NAME}_"
        f"footprint_layer_{k + 1:02d}.png"
    )


    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    plot_footprint(
        z=z,
        title=title,
        output_file=output_file,
        vmin_log=global_vmin_log,
        vmax_log=global_vmax_log
    )


# =============================================================================
# COMBINED FOOTPRINT
# =============================================================================

print()
print("=" * 80)
print("CALCULATING COMBINED FOOTPRINT")
print("=" * 80)


# -----------------------------------------------------------------------------
# Read all footprint layers
#
# dimensions:
#
#     time, lat, lon
# -----------------------------------------------------------------------------

foot_all = foot.values.astype(
    float
)


# -----------------------------------------------------------------------------
# Replace invalid values with zero
# -----------------------------------------------------------------------------

foot_all[
    ~np.isfinite(
        foot_all
    )
] = 0.0


# -----------------------------------------------------------------------------
# Negative footprint values are not meaningful for this visualization
# -----------------------------------------------------------------------------

foot_all[
    foot_all < 0
] = 0.0


# -----------------------------------------------------------------------------
# Sum all backward-time layers
# -----------------------------------------------------------------------------

combined = np.sum(
    foot_all,
    axis=0
)


print()
print("Combined footprint statistics:")

print(
    f"    Minimum = {combined.min():.6e}"
)

print(
    f"    Maximum = {combined.max():.6e}"
)

print(
    f"    Sum     = {combined.sum():.6e}"
)


# =============================================================================
# COMBINED LOG SCALE
# =============================================================================

combined_vmin_log, combined_vmax_log = (
    get_log_limits(
        combined,
        low_percentile=1.0,
        high_percentile=99.5
    )
)


print()
print(
    "COMBINED LOG10 SCALE:"
)

print(
    f"    vmin = {combined_vmin_log}"
)

print(
    f"    vmax = {combined_vmax_log}"
)


# =============================================================================
# COMBINED FOOTPRINT TITLE
# =============================================================================

dt_utc = datetime.strptime(
    DATE_TIME,
    "%Y%m%d%H%M"
)


dt_aest = (
    dt_utc +
    timedelta(hours=10)
)


combined_title = (
    "STILT Combined Backward Footprint\n"
    f"{RECEPTOR_NAME} — "
    f"{dt_utc.strftime('%d %b %Y %H:%M')} UTC "
    f"({dt_aest.strftime('%d %b %Y %H:%M')} AEST)\n"
    f"{ntime} backward-time layers summed • "
    f"{RECEPTOR_ZAGL} m AGL"
)


# =============================================================================
# COMBINED FOOTPRINT PNG
# =============================================================================

combined_file = os.path.join(
    OUTPUT_DIR,
    f"{DATE_TIME}_"
    f"{RECEPTOR_NAME}_"
    f"footprint_COMBINED.png"
)


plot_footprint(
    z=combined,
    title=combined_title,
    output_file=combined_file,
    vmin_log=combined_vmin_log,
    vmax_log=combined_vmax_log
)


# =============================================================================
# SAVE COMBINED FOOTPRINT AS NETCDF
# =============================================================================

combined_nc = os.path.join(
    OUTPUT_DIR,
    f"{DATE_TIME}_"
    f"{RECEPTOR_NAME}_"
    f"footprint_COMBINED.nc"
)


combined_da = xr.DataArray(
    combined,
    coords={
        "lat": lat,
        "lon": lon
    },
    dims=[
        "lat",
        "lon"
    ],
    name="footprint_combined"
)


# -----------------------------------------------------------------------------
# NetCDF attributes
# -----------------------------------------------------------------------------

combined_da.attrs[
    "description"
] = (
    "Combined STILT backward footprint; "
    "sum of all available time layers"
)

combined_da.attrs[
    "receptor"
] = RECEPTOR_NAME

combined_da.attrs[
    "receptor_longitude"
] = RECEPTOR_LON

combined_da.attrs[
    "receptor_latitude"
] = RECEPTOR_LAT

combined_da.attrs[
    "receptor_zagl_m"
] = RECEPTOR_ZAGL

combined_da.attrs[
    "observation_time_utc"
] = DATE_TIME

combined_da.attrs[
    "number_of_time_layers"
] = ntime

combined_da.attrs[
    "aggregation"
] = (
    "sum over time dimension"
)


# -----------------------------------------------------------------------------
# Write NetCDF
# -----------------------------------------------------------------------------

combined_da.to_netcdf(
    combined_nc
)


print()
print(
    "Saved combined NetCDF:"
)

print(
    combined_nc
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 80)
print("STILT FOOTPRINT PLOTTING COMPLETE")
print("=" * 80)

print()
print(
    f"Receptor        : {RECEPTOR_NAME}"
)

print(
    f"Observation UTC : "
    f"{dt_utc.strftime('%Y-%m-%d %H:%M')}"
)

print(
    f"Observation AEST: "
    f"{dt_aest.strftime('%Y-%m-%d %H:%M')}"
)

print(
    f"Time layers     : {ntime}"
)

print(
    f"Mine shapefiles : {len(mine_data)}"
)

print()
print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Combined PNG:"
)

print(
    combined_file
)

print()
print(
    "Combined NetCDF:"
)

print(
    combined_nc
)

print()


