#!/usr/bin/env python3

"""
======================================================================
STEP 1
Allocate Upper Hunter OC mine CH4 emissions onto the ACTUAL WRF GRIDS
======================================================================

INPUT
-----

Inventory:
    UpperHuntercoalmines.csv

Mine polygons:
    OC1.shp ... OC13.shp

ACTUAL WRF-Chem grids:
    wrfinput_d01
    wrfinput_d02
    wrfinput_d03

IMPORTANT
---------

This version uses the ACTUAL WRF-Chem input grids.

It does NOT use:
    geo_em.d01.nc
    geo_em.d02.nc
    geo_em.d03.nc

WRF grids are obtained from:

    XLAT
    XLONG

in:

    wrfinput_d01
    wrfinput_d02
    wrfinput_d03

OUTPUT
------

wrfchem_allocation_d01.csv
wrfchem_allocation_d02.csv
wrfchem_allocation_d03.csv

Diagnostic plots:

wrfchem_allocation_d01.png
wrfchem_allocation_d02.png
wrfchem_allocation_d03.png

Units in final allocation:

    mol CH4 km^-2 hr^-1

METHOD
------

1. Read mine CH4 inventory in g CH4/s.
2. Convert each mine emission to mol CH4/hr.
3. Read mine polygon.
4. Read ACTUAL WRF-Chem grid from wrfinput.
5. Construct approximate grid-cell polygons from XLAT/XLONG
   grid-cell centres.
6. Calculate intersection area between each mine polygon
   and each WRF grid cell.
7. Allocate each mine's total emission according to the
   intersection-area fraction.
8. Convert cell emission to:

       mol CH4 km^-2 hr^-1

9. Check conservation for every mine.
10. Save allocation CSV and diagnostic plot.

======================================================================
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from shapely.geometry import box

import matplotlib.pyplot as plt

from pyproj import CRS


# ====================================================================
# USER SETTINGS
# ====================================================================

INVENTORY_FILE = "/mnt/scratch_lustre/nzem_meim_scratch/data/emission_location/UpperHuntercoalmines.csv"

POLYGON_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/"
    "emission_location"
)

AREA_CRS = "EPSG:28356"
GEOGRAPHIC_CRS = "EPSG:4326"


# --------------------------------------------------------------------
# ACTUAL WRF-CHEM INPUT FILES
# --------------------------------------------------------------------

WRF_INPUT_FILES = {
    "d01": "wrfinput_d01",
    "d02": "wrfinput_d02",
    "d03": "wrfinput_d03",
}


# --------------------------------------------------------------------
# MINE NAMES
# --------------------------------------------------------------------

MINE_NAMES = [
    "OC1",
    "OC2",
    "OC3",
    "OC4",
    "OC5",
    "OC6",
    "OC7",
    "OC8",
    "OC9",
    "OC10",
    "OC11",
    "OC12",
    "OC13",
]


# --------------------------------------------------------------------
# CH4 MOLECULAR WEIGHT
# --------------------------------------------------------------------

CH4_MW = 16.04       # g/mol


# --------------------------------------------------------------------
# SMALL NUMERICAL TOLERANCE
# --------------------------------------------------------------------

AREA_TOLERANCE = 1.0e-8


# ====================================================================
# PRINT HEADER
# ====================================================================

print()
print("=" * 80)
print("STEP 1 - WRF-CHEM OC MINE CH4 GRID ALLOCATION")
print("=" * 80)

print()
print("IMPORTANT:")
print("  Allocation is based on ACTUAL wrfinput grids.")
print("  geo_em files are NOT used.")

print()
print("WRF input files:")

for domain, filename in WRF_INPUT_FILES.items():

    print(
        f"  {domain}: {filename}"
    )


# ====================================================================
# CHECK INPUT FILES
# ====================================================================

print()
print("=" * 80)
print("CHECKING INPUT FILES")
print("=" * 80)


if not os.path.exists(INVENTORY_FILE):

    raise FileNotFoundError(
        "\nInventory file not found:\n"
        + INVENTORY_FILE
    )


for domain, filename in WRF_INPUT_FILES.items():

    if not os.path.exists(filename):

        raise FileNotFoundError(
            f"\nWRF input file not found for {domain}:\n"
            f"{filename}"
        )


if not os.path.isdir(POLYGON_DIR):

    raise FileNotFoundError(
        "\nPolygon directory not found:\n"
        + POLYGON_DIR
    )


print()
print("All required WRF input files found.")
print("Inventory found.")
print("Polygon directory found.")


# ====================================================================
# HELPER: FIND SHAPEFILE
# ====================================================================

def find_shapefile(mine):

    exact = os.path.join(
        POLYGON_DIR,
        mine + ".shp"
    )

    if os.path.exists(exact):

        return exact


    files = glob.glob(
        os.path.join(
            POLYGON_DIR,
            "*.shp"
        )
    )


    for f in files:

        base = os.path.splitext(
            os.path.basename(f)
        )[0]

        if base.lower() == mine.lower():

            return f


    return None


# ====================================================================
# READ INVENTORY
# ====================================================================

print()
print("=" * 80)
print("READING CH4 INVENTORY")
print("=" * 80)

print()
print(
    "Inventory:",
    INVENTORY_FILE
)


df_inv = pd.read_csv(
    INVENTORY_FILE
)


print()
print("Inventory columns:")

for c in df_inv.columns:

    print(
        "  ",
        c
    )


# ====================================================================
# FIND EMISSION COLUMN
# ====================================================================

emission_column = None


for c in df_inv.columns:

    if c.strip().lower() == "gch4/s":

        emission_column = c

        break


if emission_column is None:

    raise RuntimeError(
        "\nCould not find 'gCH4/s' column in inventory."
    )


# ====================================================================
# FIND FACILITY COLUMN
# ====================================================================

facility_column = None


for c in df_inv.columns:

    if (
        c.strip().lower()
        ==
        "facility_code_name"
    ):

        facility_column = c

        break


if facility_column is None:

    raise RuntimeError(
        "\nCould not find Facility_code_name column."
    )


# ====================================================================
# READ MINE INVENTORY
# ====================================================================

print()
print("=" * 80)
print("OC MINE EMISSIONS")
print("=" * 80)


inventory = {}


for mine in MINE_NAMES:

    rows = df_inv[
        df_inv[facility_column]
        .astype(str)
        .str.strip()
        .str.upper()
        ==
        mine.upper()
    ]


    if len(rows) == 0:

        print(
            f"WARNING: {mine} not found in inventory."
        )

        continue


    # If multiple rows exist, sum them.

    rate = rows[
        emission_column
    ].astype(float).sum()


    inventory[mine] = rate


    print(
        f"{mine:5s} "
        f"{rate:15.6f} g/s"
    )


# ====================================================================
# TOTAL INVENTORY
# ====================================================================

total_inventory_gs = sum(
    inventory.values()
)


print()
print(
    "Total OC CH4 emission =",
    total_inventory_gs,
    "g/s"
)


# ====================================================================
# CONVERT TO MOL/HR
# ====================================================================

print()
print("=" * 80)
print("EMISSION CONVERSION")
print("=" * 80)


print(
    "CH4 molecular weight =",
    CH4_MW,
    "g/mol"
)


print()
print(
    "Conversion:"
)

print(
    "  mol/hr = g/s × 3600 / 16.04"
)


inventory_mol_hr = {}


for mine, rate_gs in inventory.items():

    mol_hr = (
        rate_gs
        * 3600.0
        / CH4_MW
    )


    inventory_mol_hr[mine] = mol_hr


    print(
        f"{mine:5s} "
        f"{mol_hr:15.6f} mol/hr"
    )


# ====================================================================
# READ MINE POLYGONS
# ====================================================================

print()
print("=" * 80)
print("READING MINE POLYGONS")
print("=" * 80)


mine_polygons = {}


for mine in MINE_NAMES:

    shp = find_shapefile(
        mine
    )


    if shp is None:

        print(
            f"WARNING: {mine} shapefile not found."
        )

        continue


    print()
    print(
        mine,
        ":",
        shp
    )


    gdf = gpd.read_file(
        shp
    )


    print(
        "  features =",
        len(gdf)
    )


    print(
        "  CRS =",
        gdf.crs
    )


    if gdf.crs is None:

        raise RuntimeError(
            f"{mine} shapefile has no CRS."
        )


    # ---------------------------------------------------------------
    # Convert to geographic CRS
    # ---------------------------------------------------------------

    gdf = gdf.to_crs(
        GEOGRAPHIC_CRS
    )


    # ---------------------------------------------------------------
    # Project to metre-based CRS
    # ---------------------------------------------------------------

    gdf_area = gdf.to_crs(
        AREA_CRS
    )


    # ---------------------------------------------------------------
    # Merge multipart features
    # ---------------------------------------------------------------

    geometry = (
        gdf_area.geometry
        .unary_union
    )


    if geometry.is_empty:

        raise RuntimeError(
            f"{mine} geometry is empty."
        )


    area_m2 = geometry.area


    print(
        "  polygon area =",
        area_m2,
        "m2"
    )


    # ---------------------------------------------------------------
    # Save geometry in EPSG:28356
    # ---------------------------------------------------------------

    mine_polygons[mine] = geometry


# ====================================================================
# READ ACTUAL WRF GRID
# ====================================================================

def read_wrf_grid(
    filename
):

    print()
    print("=" * 80)
    print(
        "READING ACTUAL WRF GRID:",
        filename
    )
    print("=" * 80)


    nc = xr.open_dataset(
        filename
    )


    # ---------------------------------------------------------------
    # WRF input uses XLAT / XLONG
    # ---------------------------------------------------------------

    if "XLAT" not in nc:

        raise RuntimeError(
            f"{filename} does not contain XLAT."
        )


    if "XLONG" not in nc:

        raise RuntimeError(
            f"{filename} does not contain XLONG."
        )


    lat = nc["XLAT"].values
    lon = nc["XLONG"].values


    # ---------------------------------------------------------------
    # Remove Time dimension
    # ---------------------------------------------------------------

    if lat.ndim == 3:

        lat = lat[0]


    if lon.ndim == 3:

        lon = lon[0]


    lat = np.asarray(
        lat,
        dtype=float
    )


    lon = np.asarray(
        lon,
        dtype=float
    )


    nc.close()


    if lat.shape != lon.shape:

        raise RuntimeError(
            f"\nXLAT and XLONG shapes differ in "
            f"{filename}:\n"
            f"XLAT  = {lat.shape}\n"
            f"XLONG = {lon.shape}"
        )


    print()
    print(
        "Grid shape:",
        lat.shape
    )


    print()
    print(
        "Latitude:"
    )

    print(
        "  min =",
        np.nanmin(lat)
    )

    print(
        "  max =",
        np.nanmax(lat)
    )


    print()
    print(
        "Longitude:"
    )

    print(
        "  min =",
        np.nanmin(lon)
    )

    print(
        "  max =",
        np.nanmax(lon)
    )


    print()
    print(
        "Grid centre:"
    )

    print(
        "  lat =",
        np.nanmean(lat)
    )

    print(
        "  lon =",
        np.nanmean(lon)
    )


    return lat, lon


# ====================================================================
# BUILD WRF GRID POLYGONS
# ====================================================================

def build_grid_polygons(
    lat,
    lon
):

    """
    Construct approximate WRF mass-grid cell polygons.

    XLAT / XLONG represent cell centres.

    Cell boundaries are estimated halfway between
    neighbouring centre coordinates.

    The resulting polygons are initially in geographic
    coordinates and then projected to EPSG:28356.
    """

    ny, nx = lat.shape


    grid_polygons = []


    for j in range(ny):

        for i in range(nx):

            # -------------------------------------------------------
            # Longitude boundaries
            # -------------------------------------------------------

            if i == 0:

                dx_right = (
                    lon[j, i + 1]
                    - lon[j, i]
                )

                lon_w = (
                    lon[j, i]
                    - 0.5 * dx_right
                )

            elif i == nx - 1:

                dx_left = (
                    lon[j, i]
                    - lon[j, i - 1]
                )

                lon_w = (
                    lon[j, i]
                    - 0.5 * dx_left
                )

            else:

                dx_left = (
                    lon[j, i]
                    - lon[j, i - 1]
                )

                lon_w = (
                    lon[j, i]
                    - 0.5 * dx_left
                )


            if i == nx - 1:

                dx_left = (
                    lon[j, i]
                    - lon[j, i - 1]
                )

                lon_e = (
                    lon[j, i]
                    + 0.5 * dx_left
                )

            else:

                dx_right = (
                    lon[j, i + 1]
                    - lon[j, i]
                )

                lon_e = (
                    lon[j, i]
                    + 0.5 * dx_right
                )


            # -------------------------------------------------------
            # Latitude boundaries
            # -------------------------------------------------------

            if j == 0:

                dy_north = (
                    lat[j + 1, i]
                    - lat[j, i]
                )

                lat_s = (
                    lat[j, i]
                    - 0.5 * dy_north
                )

            elif j == ny - 1:

                dy_south = (
                    lat[j, i]
                    - lat[j - 1, i]
                )

                lat_s = (
                    lat[j, i]
                    - 0.5 * dy_south
                )

            else:

                dy_south = (
                    lat[j, i]
                    - lat[j - 1, i]
                )

                lat_s = (
                    lat[j, i]
                    - 0.5 * dy_south
                )


            if j == ny - 1:

                dy_south = (
                    lat[j, i]
                    - lat[j - 1, i]
                )

                lat_n = (
                    lat[j, i]
                    + 0.5 * dy_south
                )

            else:

                dy_north = (
                    lat[j + 1, i]
                    - lat[j, i]
                )

                lat_n = (
                    lat[j, i]
                    + 0.5 * dy_north
                )


            # -------------------------------------------------------
            # Create geographic polygon
            # -------------------------------------------------------

            poly = box(
                lon_w,
                lat_s,
                lon_e,
                lat_n
            )


            grid_polygons.append(
                (
                    j,
                    i,
                    poly
                )
            )


    return grid_polygons


# ====================================================================
# ALLOCATE ONE DOMAIN
# ====================================================================

def allocate_domain(
    domain,
    wrf_file
):

    print()
    print("#" * 80)
    print(
        f"ALLOCATING CH4 TO ACTUAL WRF {domain.upper()}"
    )
    print("#" * 80)


    # ----------------------------------------------------------------
    # Read actual WRF grid
    # ----------------------------------------------------------------

    lat, lon = read_wrf_grid(
        wrf_file
    )


    ny, nx = lat.shape


    # ----------------------------------------------------------------
    # Build grid polygons
    # ----------------------------------------------------------------

    print()
    print(
        "Building actual WRF grid polygons..."
    )


    grid_polygons = (
        build_grid_polygons(
            lat,
            lon
        )
    )


    print(
        "Number of grid cells:",
        len(grid_polygons)
    )


    # ----------------------------------------------------------------
    # Convert WRF grid to GeoDataFrame
    # ----------------------------------------------------------------

    records = []


    for j, i, poly in grid_polygons:

        records.append(
            {
                "j": j,
                "i": i,
                "geometry": poly,
            }
        )


    gdf_grid = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=GEOGRAPHIC_CRS
    )


    # ----------------------------------------------------------------
    # Project WRF grid to metre-based CRS
    # ----------------------------------------------------------------

    gdf_grid = gdf_grid.to_crs(
        AREA_CRS
    )


    # ----------------------------------------------------------------
    # Emission accumulator
    # ----------------------------------------------------------------

    allocation = np.zeros(
        (ny, nx),
        dtype=float
    )


    # ----------------------------------------------------------------
    # Mine-by-mine allocation
    # ----------------------------------------------------------------

    mine_totals = {}


    for mine in MINE_NAMES:

        if mine not in mine_polygons:

            continue


        if mine not in inventory_mol_hr:

            continue


        print()
        print(
            "Processing",
            mine
        )


        mine_geom = (
            mine_polygons[mine]
        )


        mine_area = (
            mine_geom.area
        )


        if mine_area <= 0:

            print(
                "WARNING:",
                mine,
                "has zero area."
            )

            continue


        # ------------------------------------------------------------
        # Mine bounding box
        # ------------------------------------------------------------

        mine_bounds = (
            gpd.GeoSeries(
                [mine_geom],
                crs=AREA_CRS
            ).total_bounds
        )


        # ------------------------------------------------------------
        # Candidate WRF cells
        # ------------------------------------------------------------

        possible = gdf_grid.cx[
            mine_bounds[0]:
            mine_bounds[2],
            mine_bounds[1]:
            mine_bounds[3]
        ]


        print(
            "Candidate cells:",
            len(possible)
        )


        total_allocated = 0.0
        cells_used = 0


        # ------------------------------------------------------------
        # Loop through candidate cells
        # ------------------------------------------------------------

        for idx, row in possible.iterrows():

            cell_geom = (
                row.geometry
            )


            if not cell_geom.intersects(
                mine_geom
            ):

                continue


            intersection = (
                cell_geom.intersection(
                    mine_geom
                )
            )


            intersection_area = (
                intersection.area
            )


            if (
                intersection_area
                <=
                AREA_TOLERANCE
            ):

                continue


            # --------------------------------------------------------
            # Fraction of mine area in this cell
            # --------------------------------------------------------

            fraction = (
                intersection_area
                /
                mine_area
            )


            # --------------------------------------------------------
            # Total mine emission
            # --------------------------------------------------------

            mine_mol_hr = (
                inventory_mol_hr[mine]
            )


            cell_mol_hr = (
                mine_mol_hr
                *
                fraction
            )


            # --------------------------------------------------------
            # WRF cell area
            #
            # Convert m2 -> km2
            # --------------------------------------------------------

            cell_area_km2 = (
                cell_geom.area
                /
                1.0e6
            )


            if cell_area_km2 <= 0:

                continue


            # --------------------------------------------------------
            # Convert to WRF-Chem emission flux
            #
            # mol/hr
            #
            # ->
            #
            # mol km^-2 hr^-1
            # --------------------------------------------------------

            cell_flux = (
                cell_mol_hr
                /
                cell_area_km2
            )


            j = int(
                row["j"]
            )

            i = int(
                row["i"]
            )


            allocation[j, i] += (
                cell_flux
            )


            total_allocated += (
                cell_mol_hr
            )


            cells_used += 1


        # ------------------------------------------------------------
        # Store conservation information
        # ------------------------------------------------------------

        mine_totals[mine] = {

            "expected_mol_hr":
                inventory_mol_hr[mine],

            "allocated_mol_hr":
                total_allocated,

            "cells":
                cells_used,
        }


        print(
            f"{mine}: "
            f"{cells_used} cells"
        )


        print(
            "  expected mol/hr =",
            inventory_mol_hr[mine]
        )


        print(
            "  allocated mol/hr =",
            total_allocated
        )


        if inventory_mol_hr[mine] != 0:

            error = (
                total_allocated
                /
                inventory_mol_hr[mine]
                -
                1.0
            ) * 100.0


            print(
                "  conservation error =",
                error,
                "%"
            )


    # =================================================================
    # SAVE CSV
    # =================================================================

    rows = []


    for j in range(ny):

        for i in range(nx):

            value = allocation[
                j,
                i
            ]


            if value <= 0:

                continue


            rows.append(
                {
                    "domain":
                        domain,

                    "j":
                        j,

                    "i":
                        i,

                    "lat":
                        lat[j, i],

                    "lon":
                        lon[j, i],

                    "E_CH4_mol_km2_hr":
                        value,
                }
            )


    result = pd.DataFrame(
        rows
    )


    output_csv = (
        f"wrfchem_allocation_{domain}.csv"
    )


    result.to_csv(
        output_csv,
        index=False
    )


    print()
    print(
        "Saved:",
        output_csv
    )


    # =================================================================
    # DIAGNOSTICS
    # =================================================================

    total_grid_flux = (
        np.sum(allocation)
    )


    print()
    print(
        "Non-zero WRF cells:",
        len(result)
    )


    print(
        "Maximum E_CH4:",
        np.max(allocation)
    )


    print(
        "Sum of E_CH4 grid values:",
        total_grid_flux,
        "mol km^-2 hr^-1"
    )


    # =================================================================
    # RETURN
    # =================================================================

    return {

        "lat":
            lat,

        "lon":
            lon,

        "allocation":
            allocation,

        "mine_totals":
            mine_totals,

        "csv":
            output_csv,
    }


# ====================================================================
# PROCESS THREE DOMAINS
# ====================================================================

print()
print("=" * 80)
print("PROCESSING ACTUAL WRF-CHEM DOMAINS")
print("=" * 80)


results = {}


for domain in [
    "d01",
    "d02",
    "d03"
]:

    results[domain] = (
        allocate_domain(
            domain,
            WRF_INPUT_FILES[domain]
        )
    )


# ====================================================================
# CREATE DIAGNOSTIC PLOTS
# ====================================================================

print()
print("=" * 80)
print("CREATING DIAGNOSTIC PLOTS")
print("=" * 80)


for domain in [
    "d01",
    "d02",
    "d03"
]:

    result = results[
        domain
    ]


    lat = result[
        "lat"
    ]


    lon = result[
        "lon"
    ]


    allocation = result[
        "allocation"
    ]


    fig, ax = plt.subplots(
        figsize=(11, 9)
    )


    positive = (
        allocation > 0
    )


    if np.any(positive):

        values = allocation[
            positive
        ]


        # ------------------------------------------------------------
        # Log scale where possible
        # ------------------------------------------------------------

        if np.min(values) > 0:

            norm = (
                plt.matplotlib.colors.LogNorm(
                    vmin=np.min(values),
                    vmax=np.max(values)
                )
            )

        else:

            norm = None


        mesh = ax.scatter(
            lon[positive],
            lat[positive],
            c=values,
            s=20,
            cmap="turbo",
            norm=norm,
        )


        cbar = fig.colorbar(
            mesh,
            ax=ax
        )


        cbar.set_label(
            "E_CH4 "
            "(mol km$^{-2}$ hr$^{-1}$)"
        )


    # =================================================================
    # PLOT MINE POLYGONS
    # =================================================================

    for mine in MINE_NAMES:

        if mine not in mine_polygons:

            continue


        geom = mine_polygons[
            mine
        ]


        # ------------------------------------------------------------
        # Convert polygon back to geographic coordinates
        # ------------------------------------------------------------

        gdf_tmp = gpd.GeoSeries(
            [geom],
            crs=AREA_CRS
        ).to_crs(
            GEOGRAPHIC_CRS
        )


        gdf_tmp.boundary.plot(
            ax=ax,
            color="black",
            linewidth=1.0
        )


        centroid = (
            gdf_tmp.iloc[0]
            .centroid
        )


        ax.text(
            centroid.x,
            centroid.y,
            mine,
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="center"
        )


    # =================================================================
    # GRID EXTENT
    # =================================================================

    ax.set_xlim(
        np.nanmin(lon),
        np.nanmax(lon)
    )


    ax.set_ylim(
        np.nanmin(lat),
        np.nanmax(lat)
    )


    ax.set_xlabel(
        "Longitude (°E)"
    )


    ax.set_ylabel(
        "Latitude (°S)"
    )


    ax.set_title(
        f"WRF-Chem CH₄ allocation - "
        f"{domain.upper()} "
        f"(actual wrfinput grid)"
    )


    ax.grid(
        True,
        linestyle="--",
        alpha=0.4
    )


    output_png = (
        f"wrfchem_allocation_{domain}.png"
    )


    plt.tight_layout()


    plt.savefig(
        output_png,
        dpi=300
    )


    plt.show()

    plt.close()


    print(
        "Saved:",
        output_png
    )


# ====================================================================
# FINAL CONSERVATION SUMMARY
# ====================================================================

print()
print("=" * 80)
print("FINAL CONSERVATION CHECK")
print("=" * 80)


for domain in [
    "d01",
    "d02",
    "d03"
]:

    result = results[
        domain
    ]


    print()
    print(
        domain.upper()
    )


    for mine, values in (
        result[
            "mine_totals"
        ].items()
    ):

        expected = values[
            "expected_mol_hr"
        ]


        allocated = values[
            "allocated_mol_hr"
        ]


        if expected != 0:

            error = (
                allocated
                /
                expected
                -
                1.0
            ) * 100.0

        else:

            error = 0.0


        print(
            f"{mine:5s} "
            f"expected={expected:.6f} "
            f"allocated={allocated:.6f} "
            f"error={error:.4f}%"
        )


# ====================================================================
# FINISHED
# ====================================================================

print()
print("=" * 80)
print("STEP 1 COMPLETED")
print("=" * 80)


print()
print(
    "The three allocation files are:"
)


print(
    "  wrfchem_allocation_d01.csv"
)


print(
    "  wrfchem_allocation_d02.csv"
)


print(
    "  wrfchem_allocation_d03.csv"
)


print()
print(
    "The allocation is based on:"
)


print(
    "  wrfinput_d01"
)


print(
    "  wrfinput_d02"
)


print(
    "  wrfinput_d03"
)


print()
print(
    "geo_em files are NOT used."
)


print()
print("=" * 80)

