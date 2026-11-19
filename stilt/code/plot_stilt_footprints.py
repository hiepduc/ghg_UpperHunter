#!/usr/bin/env python3

"""
===============================================================================
STILT FOOTPRINT PLOTS OVER UPPER HUNTER MINE SHAPEFILES
===============================================================================

For one STILT receptor observation time:

    1. Read the corresponding *_foot.nc file.
    2. Plot every backward-time footprint layer separately.
    3. Overlay OC1-OC13 mine shapefiles.
    4. Add Muswellbrook, Singleton, Merriwa and Stockton.
    5. Add coastline.
    6. Create a combined footprint by summing all backward-time layers.

The STILT NetCDF footprint is expected to contain:

    foot(time, lat, lon)

xarray reads this as:

    (time, lat, lon)

The footprint is plotted on a regular lon/lat grid.

===============================================================================
"""

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

# Observation / receptor time in UTC
#
# Example:
# 202601051200 = 5 January 2026 12:00 UTC
#
DATE_TIME = "202601051200"


# Receptor
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

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# OTHER PLACES
# =============================================================================

PLACES = {

    "Muswellbrook":
        (150.89297, -32.26202),

    "Singleton":
        (151.16132, -32.57123),

    "Merriwa":
        (150.35767, -32.14097),

    # Use Stockton
    "Stockton":
        (151.7810, -32.9070),

    # Alternatively use Newcastle:
    # "Newcastle":
    #     (151.7817, -32.9283),
}


# =============================================================================
# MAP EXTENT
# =============================================================================

MAP_EXTENT = [
    150.20,     # west
    152.05,     # east
    -33.25,     # south
    -31.85      # north
]


# =============================================================================
# INPUT NETCDF
# =============================================================================

foot_file = os.path.join(
    FOOTPRINT_DIR,
    f"{DATE_TIME}_{RECEPTOR_LON:.5f}_{RECEPTOR_LAT:.5f}_"
    f"{RECEPTOR_ZAGL}_foot.nc"
)

if not os.path.exists(foot_file):

    raise FileNotFoundError(
        "\nFootprint file not found:\n"
        f"{foot_file}\n"
    )

print("=" * 80)
print("STILT FOOTPRINT ANALYSIS")
print("=" * 80)

print("\nInput file:")
print(foot_file)


# =============================================================================
# READ FOOTPRINT
# =============================================================================

ds = xr.open_dataset(foot_file)

print("\nNetCDF:")
print(ds)

if "foot" not in ds:

    raise ValueError(
        "Variable 'foot' was not found in the NetCDF file."
    )

foot = ds["foot"]


# =============================================================================
# CHECK DIMENSIONS
# =============================================================================

print("\nFootprint dimensions:")
print(foot.dims)

print("\nFootprint shape:")
print(foot.shape)


# Expected:
#
# ('time', 'lat', 'lon')
#
# shape:
#
# (ntime, nlat, nlon)

if not all(
    dim in foot.dims
    for dim in ["time", "lat", "lon"]
):

    raise ValueError(
        "Expected footprint dimensions containing "
        "'time', 'lat' and 'lon'."
    )


# Make sure the dimensions are ordered as:
#
# time, lat, lon

foot = foot.transpose(
    "time",
    "lat",
    "lon"
)


# =============================================================================
# COORDINATES
# =============================================================================

lon = foot["lon"].values
lat = foot["lat"].values

ntime = foot.sizes["time"]


print("\nGrid:")
print(f"    longitude: {len(lon)}")
print(f"    latitude : {len(lat)}")
print(f"    time     : {ntime}")

print("\nLongitude range:")
print(lon.min(), "to", lon.max())

print("\nLatitude range:")
print(lat.min(), "to", lat.max())


# =============================================================================
# TIME INFORMATION
# =============================================================================

print("\nNetCDF time coordinate:")

try:

    print(foot["time"].values)

except Exception:

    print("Unable to display time coordinate.")


# =============================================================================
# MINE SHAPEFILES
# =============================================================================

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

    else:

        print(
            f"WARNING: OC{i}.shp not found"
        )


print(
    f"\nFound {len(mine_files)} mine shapefiles."
)


# =============================================================================
# LOAD MINE DATA ONCE
# =============================================================================

mine_data = []

for mine_id, shp in mine_files:

    try:

        gdf = gpd.read_file(shp)

        if gdf.crs is not None:

            gdf = gdf.to_crs(
                "EPSG:4326"
            )

        mine_data.append(
            (mine_id, gdf)
        )

    except Exception as e:

        print(
            f"WARNING: Could not read OC{mine_id}: {e}"
        )


# =============================================================================
# FUNCTION: PLOT MAP BASE
# =============================================================================

def create_map():

    fig = plt.figure(
        figsize=(12, 10)
    )

    ax = plt.axes(
        projection=ccrs.PlateCarree()
    )

    # ---------------------------------------------------------
    # Extent
    # ---------------------------------------------------------

    ax.set_extent(
        MAP_EXTENT,
        crs=ccrs.PlateCarree()
    )

    # ---------------------------------------------------------
    # Land and ocean
    # ---------------------------------------------------------

    ax.add_feature(
        cfeature.LAND,
        zorder=0
    )

    ax.add_feature(
        cfeature.OCEAN,
        zorder=0
    )

    # ---------------------------------------------------------
    # Coastline
    # ---------------------------------------------------------

    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=1.2,
        zorder=2
    )

    # ---------------------------------------------------------
    # State/country boundaries
    # ---------------------------------------------------------

    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.5,
        edgecolor="gray",
        zorder=2
    )

    # ---------------------------------------------------------
    # Grid
    # ---------------------------------------------------------

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

        # -----------------------------------------------------
        # Mine boundary
        # -----------------------------------------------------

        gdf.boundary.plot(
            ax=ax,
            linewidth=1.4,
            color="black",
            zorder=6
        )

        # -----------------------------------------------------
        # Mine centroid
        # -----------------------------------------------------

        try:

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
                f"WARNING: centroid OC{mine_id}: {e}"
            )


# =============================================================================
# FUNCTION: PLOT PLACES
# =============================================================================

def plot_places(ax):

    for name, (lon_p, lat_p) in PLACES.items():

        # -----------------------------------------------------
        # Receptor
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Other towns
        # -----------------------------------------------------

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

            # Label positions

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
                ha="left" if dx > 0 else "right",
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
# FOOTPRINT PLOTTING FUNCTION
# =============================================================================

def plot_footprint(
    z,
    title,
    output_file,
    vmax=None
):

    fig, ax = create_map()

    # ---------------------------------------------------------
    # Positive footprint only
    # ---------------------------------------------------------

    z = np.asarray(
        z,
        dtype=float
    )

    z[z <= 0] = np.nan

    # ---------------------------------------------------------
    # Log10 footprint
    # ---------------------------------------------------------

    logz = np.log10(z)

    # ---------------------------------------------------------
    # Determine colour range
    # ---------------------------------------------------------

    finite = logz[
        np.isfinite(logz)
    ]

    if len(finite) == 0:

        print(
            "WARNING: footprint contains no positive values."
        )

        plt.close(fig)

        return

    vmin = np.percentile(
        finite,
        1
    )

    if vmax is None:

        vmax_plot = np.percentile(
            finite,
            99.5
        )

    else:

        vmax_plot = vmax

    # ---------------------------------------------------------
    # Footprint
    # ---------------------------------------------------------

    mesh = ax.pcolormesh(
        lon,
        lat,
        logz,
        shading="auto",
        cmap="turbo",
        vmin=vmin,
        vmax=vmax_plot,
        transform=ccrs.PlateCarree(),
        zorder=3
    )

    # ---------------------------------------------------------
    # Mine polygons
    # ---------------------------------------------------------

    plot_mines(ax)

    # ---------------------------------------------------------
    # Places
    # ---------------------------------------------------------

    plot_places(ax)

    # ---------------------------------------------------------
    # Colourbar
    # ---------------------------------------------------------

    cbar = plt.colorbar(
        mesh,
        ax=ax,
        pad=0.02,
        shrink=0.80
    )

    cbar.set_label(
        "log10 Footprint\n"
        "ppm (µmol⁻¹ m² s)",
        fontsize=11
    )

    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------

    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

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

    plt.close(fig)


# =============================================================================
# DETERMINE A COMMON COLOUR SCALE
# =============================================================================

print("\nCalculating common colour scale...")

all_values = []

for k in range(ntime):

    z = foot.isel(
        time=k
    ).values

    z = np.asarray(
        z,
        dtype=float
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

    global_vmax = np.percentile(
        np.log10(all_values),
        99.5
    )

else:

    global_vmax = None


# =============================================================================
# PLOT EACH BACKWARD HOUR
# =============================================================================

print("\n")
print("=" * 80)
print("PLOTTING INDIVIDUAL BACKWARD-TIME FOOTPRINTS")
print("=" * 80)


for k in range(ntime):

    # ---------------------------------------------------------
    # Footprint layer
    # ---------------------------------------------------------

    z = foot.isel(
        time=k
    ).values

    # ---------------------------------------------------------
    # Time label
    # ---------------------------------------------------------

    try:

        time_value = foot["time"].values[k]

        # Convert numpy datetime64 if possible
        time_dt = pd_timestamp = (
            np.datetime64(time_value)
        )

        # Convert to Python datetime
        time_dt = (
            pd_timestamp
            .astype("datetime64[s]")
            .astype(datetime)
        )

        time_string = time_dt.strftime(
            "%d %b %Y %H:%M UTC"
        )

    except Exception:

        time_string = (
            f"backward layer {k}"
        )


    # ---------------------------------------------------------
    # Calculate backward hour from observation
    # ---------------------------------------------------------

    backward_hour = k

    # If the NetCDF time coordinate is chronological
    # backward from the receptor, k=0 is normally the
    # earliest stored layer. Therefore we don't assume
    # the order here.
    #
    # We display the layer number as well.

    title = (
        "STILT Backward Footprint\n"
        f"{RECEPTOR_NAME} — "
        f"{DATE_TIME[:4]}-"
        f"{DATE_TIME[4:6]}-"
        f"{DATE_TIME[6:8]} "
        f"{DATE_TIME[8:10]}:"
        f"{DATE_TIME[10:]} UTC\n"
        f"Footprint layer {k + 1}/{ntime} — "
        f"{time_string}"
    )


    output_file = os.path.join(
        OUTPUT_DIR,
        f"{DATE_TIME}_{RECEPTOR_NAME}_"
        f"footprint_layer_{k + 1:02d}.png"
    )


    plot_footprint(
        z=z,
        title=title,
        output_file=output_file,
        vmax=global_vmax
    )


# =============================================================================
# COMBINED FOOTPRINT
# =============================================================================

print("\n")
print("=" * 80)
print("CALCULATING COMBINED FOOTPRINT")
print("=" * 80)


# Convert to numpy array:
#
# dimensions:
#
# time x lat x lon

foot_all = foot.values.astype(
    float
)


# Replace invalid values
foot_all[
    ~np.isfinite(foot_all)
] = 0.0


# Replace negative values
foot_all[
    foot_all < 0
] = 0.0


# Sum all backward-time layers
combined = np.sum(
    foot_all,
    axis=0
)


print("\nCombined footprint:")
print(
    "min =",
    combined.min()
)

print(
    "max =",
    combined.max()
)

print(
    "sum =",
    combined.sum()
)


# =============================================================================
# PLOT COMBINED FOOTPRINT
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
    f"{dt_aest.strftime('%d %b %Y %H:%M')} AEST "
    f"({DATE_TIME[8:10]}:{DATE_TIME[10:]} UTC)\n"
    f"{ntime} backward-time layers summed • "
    f"{RECEPTOR_ZAGL} m AGL"
)


combined_file = os.path.join(
    OUTPUT_DIR,
    f"{DATE_TIME}_{RECEPTOR_NAME}_"
    f"footprint_COMBINED.png"
)


plot_footprint(
    z=combined,
    title=combined_title,
    output_file=combined_file
)


# =============================================================================
# SAVE COMBINED FOOTPRINT AS NETCDF
# =============================================================================

combined_nc = os.path.join(
    OUTPUT_DIR,
    f"{DATE_TIME}_{RECEPTOR_NAME}_"
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

combined_da.attrs[
    "description"
] = (
    "Combined STILT backward footprint; "
    "sum of all time layers"
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

combined_da.to_netcdf(
    combined_nc
)


print(
    "\nSaved combined NetCDF:"
)

print(
    combined_nc
)


# =============================================================================
# SUMMARY
# =============================================================================

print("\n")
print("=" * 80)
print("DONE")
print("=" * 80)

print(
    f"\nNumber of footprint layers: {ntime}"
)

print(
    "\nIndividual plots:"
)

print(
    OUTPUT_DIR
)

print(
    "\nCombined plot:"
)

print(
    combined_file
)

print(
    "\nCombined NetCDF:"
)

print(
    combined_nc
)

print()

