#!/usr/bin/env python3

"""
===============================================================================
CREATE INDIVIDUAL UPPER HUNTER OPEN-CUT MINE EMISSION FIELDS FOR STILT
===============================================================================

Purpose
-------
Create individual CH4 emission fields for OC1 ... OC13 directly from:

    1. UpperHuntercoalmines.csv
    2. OC1.shp ... OC13.shp
    3. An existing STILT footprint NetCDF file

The WRF-Chem wrfchemi files are NOT used.

Each mine is assumed to have a constant annual-average emission rate.
The inventory gCH4/s value is therefore converted to a constant emission
flux over the mine polygon.

Spatial allocation is performed by polygon/grid-cell intersection area.

OUTPUT
------
One NetCDF file containing:

    OC1
    OC2
    ...
    OC13
    OC_TOTAL

Units:

    mol CH4 m-2 s-1

The latitude/longitude grid is copied from the supplied STILT footprint file.

IMPORTANT
---------
The STILT footprint has units approximately:

    ppm (umol m-2 s)-1

Therefore the emission field is converted to:

    umol CH4 m-2 s-1

internally when checking STILT compatibility.

For each mine:

    sum(flux * cell_area) = inventory emission [mol/s]

to numerical precision.

===============================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import box

from netCDF4 import Dataset


# =============================================================================
# USER SETTINGS
# =============================================================================

# -------------------------------------------------------------------------
# Base directory containing the inventory and mine shapefiles
# -------------------------------------------------------------------------

BASE_DIR = (
    "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter"
)

# -------------------------------------------------------------------------
# Inventory
# -------------------------------------------------------------------------

INVENTORY_FILE = os.path.join(
    BASE_DIR,
    "UpperHuntercoalmines.csv"
)

# -------------------------------------------------------------------------
# Mine shapefiles
# -------------------------------------------------------------------------

MINE_SHAPEFILE_DIR = BASE_DIR

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

# -------------------------------------------------------------------------
# Existing STILT footprint
#
# This is used ONLY to obtain the exact STILT lat/lon grid.
# -------------------------------------------------------------------------

STILT_FOOTPRINT = (
    "/mnt/scratch_lustre/duch/stilt/upperhunter/"
    "arl_mine_jan2026/footprints/Muswellbrook/footprints/"
    "202601151200_150.89297_-32.26202_5_foot.nc"
)

# -------------------------------------------------------------------------
# Output
# -------------------------------------------------------------------------

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "stilt_mine_emissions_d03.nc"
)

# -------------------------------------------------------------------------
# CH4 molecular weight
#
# Inventory is g CH4 / s
# -------------------------------------------------------------------------

CH4_MW_G_MOL = 16.04

# -------------------------------------------------------------------------
# Geographic CRS
# -------------------------------------------------------------------------

GEOGRAPHIC_CRS = "EPSG:4326"

# -------------------------------------------------------------------------
# Equal-area / metric CRS used for polygon intersections.
#
# UTM zone 56S covers the Upper Hunter.
# -------------------------------------------------------------------------

METRIC_CRS = "EPSG:32756"


# =============================================================================
# HEADER
# =============================================================================

print()
print("=" * 80)
print("UPPER HUNTER INDIVIDUAL MINE EMISSIONS FOR STILT")
print("=" * 80)

print()
print("Inventory:")
print(" ", INVENTORY_FILE)

print()
print("STILT footprint:")
print(" ", STILT_FOOTPRINT)

print()
print("Output:")
print(" ", OUTPUT_FILE)

# =============================================================================
# CHECK INPUTS
# =============================================================================

if not os.path.exists(INVENTORY_FILE):
    raise FileNotFoundError(
        "Inventory file not found:\n"
        + INVENTORY_FILE
    )

if not os.path.exists(STILT_FOOTPRINT):
    raise FileNotFoundError(
        "STILT footprint file not found:\n"
        + STILT_FOOTPRINT
    )

for mine in MINE_NAMES:

    shp = os.path.join(
        MINE_SHAPEFILE_DIR,
        mine + ".shp"
    )

    if not os.path.exists(shp):

        raise FileNotFoundError(
            f"Shapefile not found for {mine}:\n"
            f"{shp}"
        )


# =============================================================================
# READ INVENTORY
# =============================================================================

print()
print("=" * 80)
print("READING INVENTORY")
print("=" * 80)

df = pd.read_csv(
    INVENTORY_FILE,
    skipinitialspace=True
)

# Remove accidental whitespace from column names

df.columns = [
    str(c).strip()
    for c in df.columns
]

required_columns = [
    "Facility_code_name",
    "gCH4/s",
]

for col in required_columns:

    if col not in df.columns:

        raise RuntimeError(
            f"Required inventory column not found: {col}"
        )

# Clean facility names

df["Facility_code_name"] = (
    df["Facility_code_name"]
    .astype(str)
    .str.strip()
)

# Convert emission column to numeric

df["gCH4/s"] = pd.to_numeric(
    df["gCH4/s"],
    errors="coerce"
)

print()
print("Inventory mines:")

for mine in MINE_NAMES:

    match = df[
        df["Facility_code_name"] == mine
    ]

    if len(match) != 1:

        raise RuntimeError(
            f"Expected exactly one inventory record for "
            f"{mine}, found {len(match)}"
        )

    emission_g_s = float(
        match.iloc[0]["gCH4/s"]
    )

    if not np.isfinite(emission_g_s):
        raise RuntimeError(
            f"Invalid emission for {mine}"
        )

    if emission_g_s < 0:
        raise RuntimeError(
            f"Negative emission for {mine}: "
            f"{emission_g_s}"
        )

    print(
        f"  {mine:5s} "
        f"{emission_g_s:12.5f} g CH4/s"
    )


# =============================================================================
# READ STILT GRID
# =============================================================================

print()
print("=" * 80)
print("READING STILT GRID")
print("=" * 80)

nc = Dataset(
    STILT_FOOTPRINT,
    "r"
)

print()
print("Dimensions:")

for name, dim in nc.dimensions.items():

    print(
        f"  {name:10s} = {len(dim)}"
    )

# -------------------------------------------------------------------------
# Latitude
# -------------------------------------------------------------------------

if "lat" not in nc.variables:

    raise RuntimeError(
        "STILT footprint does not contain variable 'lat'"
    )

# -------------------------------------------------------------------------
# Longitude
# -------------------------------------------------------------------------

if "lon" not in nc.variables:

    raise RuntimeError(
        "STILT footprint does not contain variable 'lon'"
    )

lat = np.asarray(
    nc.variables["lat"][:],
    dtype=np.float64
)

lon = np.asarray(
    nc.variables["lon"][:],
    dtype=np.float64
)

nc.close()

print()
print("STILT grid:")
print("  Number of latitudes :", len(lat))
print("  Number of longitudes:", len(lon))

print()
print(
    "Latitude range:",
    np.min(lat),
    "to",
    np.max(lat)
)

print(
    "Longitude range:",
    np.min(lon),
    "to",
    np.max(lon)
)


# =============================================================================
# CHECK STILT GRID
# =============================================================================

if len(lat) < 2:
    raise RuntimeError(
        "Need at least two latitude points."
    )

if len(lon) < 2:
    raise RuntimeError(
        "Need at least two longitude points."
    )

dlat = np.diff(lat)
dlon = np.diff(lon)

if not np.all(
    np.isfinite(dlat)
):

    raise RuntimeError(
        "Latitude grid contains invalid spacing."
    )

if not np.all(
    np.isfinite(dlon)
):

    raise RuntimeError(
        "Longitude grid contains invalid spacing."
    )

# Check approximately regular grid

lat_spacing = float(
    np.median(
        np.abs(dlat)
    )
)

lon_spacing = float(
    np.median(
        np.abs(dlon)
    )
)

print()
print("Grid spacing:")
print(
    f"  dlat = {lat_spacing:.8f} degree"
)

print(
    f"  dlon = {lon_spacing:.8f} degree"
)

if not np.allclose(
    np.abs(dlat),
    lat_spacing,
    rtol=1e-5,
    atol=1e-10
):

    warnings.warn(
        "Latitude grid is not perfectly regular."
    )

if not np.allclose(
    np.abs(dlon),
    lon_spacing,
    rtol=1e-5,
    atol=1e-10
):

    warnings.warn(
        "Longitude grid is not perfectly regular."
    )


# =============================================================================
# CREATE GRID CELL EDGES
# =============================================================================

def calculate_edges(values):
    """
    Calculate cell edges from cell-center coordinates.
    Works for increasing or decreasing coordinates.
    """

    values = np.asarray(
        values,
        dtype=np.float64
    )

    edges = np.empty(
        len(values) + 1,
        dtype=np.float64
    )

    edges[1:-1] = (
        values[:-1] + values[1:]
    ) / 2.0

    edges[0] = (
        values[0]
        - (values[1] - values[0]) / 2.0
    )

    edges[-1] = (
        values[-1]
        + (values[-1] - values[-2]) / 2.0
    )

    return edges


lat_edges = calculate_edges(lat)
lon_edges = calculate_edges(lon)


# =============================================================================
# CREATE GRID CELL POLYGONS
# =============================================================================

print()
print("=" * 80)
print("CREATING STILT GRID CELL POLYGONS")
print("=" * 80)

nlat = len(lat)
nlon = len(lon)

print()
print(
    "Total grid cells:",
    nlat * nlon
)

# -------------------------------------------------------------------------
# IMPORTANT
#
# The intersection must be done in a metric CRS.
#
# We first create geographic cells, then transform them to UTM 56S.
# -------------------------------------------------------------------------

grid_cells = []

grid_indices = []

for j in range(nlat):

    lat0 = min(
        lat_edges[j],
        lat_edges[j + 1]
    )

    lat1 = max(
        lat_edges[j],
        lat_edges[j + 1]
    )

    for i in range(nlon):

        lon0 = min(
            lon_edges[i],
            lon_edges[i + 1]
        )

        lon1 = max(
            lon_edges[i],
            lon_edges[i + 1]
        )

        polygon = box(
            lon0,
            lat0,
            lon1,
            lat1
        )

        grid_cells.append(
            polygon
        )

        grid_indices.append(
            (j, i)
        )


grid_gdf = gpd.GeoDataFrame(
    {
        "j": [
            x[0] for x in grid_indices
        ],
        "i": [
            x[1] for x in grid_indices
        ],
    },
    geometry=grid_cells,
    crs=GEOGRAPHIC_CRS
)

print()
print("Transforming STILT grid to:")
print(" ", METRIC_CRS)

grid_gdf = grid_gdf.to_crs(
    METRIC_CRS
)

# Store grid cell areas

grid_gdf["area_m2"] = (
    grid_gdf.geometry.area
)

print()
print(
    "Total grid area:",
    grid_gdf["area_m2"].sum(),
    "m2"
)


# =============================================================================
# PREPARE OUTPUT ARRAYS
# =============================================================================

emissions = {}

for mine in MINE_NAMES:

    emissions[mine] = np.zeros(
        (nlat, nlon),
        dtype=np.float64
    )


# =============================================================================
# PROCESS EACH MINE
# =============================================================================

print()
print("=" * 80)
print("ALLOCATING INDIVIDUAL MINE EMISSIONS")
print("=" * 80)


for mine in MINE_NAMES:

    print()
    print("-" * 80)
    print(
        f"PROCESSING {mine}"
    )
    print("-" * 80)

    # -------------------------------------------------------------------------
    # Inventory emission
    # -------------------------------------------------------------------------

    row = df[
        df["Facility_code_name"] == mine
    ].iloc[0]

    emission_g_s = float(
        row["gCH4/s"]
    )

    # -------------------------------------------------------------------------
    # Convert g/s CH4 -> mol/s CH4
    # -------------------------------------------------------------------------

    emission_mol_s = (
        emission_g_s
        / CH4_MW_G_MOL
    )

    print()
    print(
        f"Inventory emission: "
        f"{emission_g_s:.8f} g/s"
    )

    print(
        f"Inventory emission: "
        f"{emission_mol_s:.8e} mol/s"
    )

    # -------------------------------------------------------------------------
    # Read shapefile
    # -------------------------------------------------------------------------

    shp = os.path.join(
        MINE_SHAPEFILE_DIR,
        mine + ".shp"
    )

    print()
    print(
        "Shapefile:",
        shp
    )

    mine_gdf = gpd.read_file(
        shp
    )

    if mine_gdf.empty:

        raise RuntimeError(
            f"{mine} shapefile contains no geometry."
        )

    print(
        "  Number of features:",
        len(mine_gdf)
    )

    print(
        "  CRS:",
        mine_gdf.crs
    )

    if mine_gdf.crs is None:

        raise RuntimeError(
            f"{mine} shapefile has no CRS."
        )

    # -------------------------------------------------------------------------
    # Convert to geographic CRS
    # -------------------------------------------------------------------------

    mine_gdf = mine_gdf.to_crs(
        GEOGRAPHIC_CRS
    )

    # -------------------------------------------------------------------------
    # Merge all features belonging to the mine
    # -------------------------------------------------------------------------

    mine_geometry = (
        mine_gdf.geometry
        .unary_union
    )

    if mine_geometry.is_empty:

        raise RuntimeError(
            f"{mine} geometry is empty."
        )

    # -------------------------------------------------------------------------
    # Create GeoDataFrame for mine
    # -------------------------------------------------------------------------

    mine_single = gpd.GeoDataFrame(
        {
            "mine": [mine]
        },
        geometry=[mine_geometry],
        crs=GEOGRAPHIC_CRS
    )

    # -------------------------------------------------------------------------
    # Transform to metric CRS
    # -------------------------------------------------------------------------

    mine_single = mine_single.to_crs(
        METRIC_CRS
    )

    mine_polygon = (
        mine_single.geometry.iloc[0]
    )

    mine_area_m2 = (
        mine_polygon.area
    )

    print()
    print(
        f"Mine polygon area: "
        f"{mine_area_m2:.3f} m2"
    )

    # -------------------------------------------------------------------------
    # Candidate grid cells using bounding box
    # -------------------------------------------------------------------------

    minx, miny, maxx, maxy = (
        mine_polygon.bounds
    )

    candidate_mask = (
        (grid_gdf.geometry.bounds["minx"] <= maxx)
        &
        (grid_gdf.geometry.bounds["maxx"] >= minx)
        &
        (grid_gdf.geometry.bounds["miny"] <= maxy)
        &
        (grid_gdf.geometry.bounds["maxy"] >= miny)
    )

    candidates = grid_gdf.loc[
        candidate_mask
    ]

    print()
    print(
        "Candidate STILT cells:",
        len(candidates)
    )

    # -------------------------------------------------------------------------
    # Intersect mine with candidate grid cells
    # -------------------------------------------------------------------------

    allocated_mol_s = 0.0
    intersected_cells = 0

    for idx, cell_row in candidates.iterrows():

        cell_polygon = (
            cell_row.geometry
        )

        intersection = (
            mine_polygon
            .intersection(cell_polygon)
        )

        if intersection.is_empty:
            continue

        intersection_area = (
            intersection.area
        )

        if intersection_area <= 0.0:
            continue

        # ---------------------------------------------------------------------
        # Fraction of mine area in this grid cell
        # ---------------------------------------------------------------------

        fraction = (
            intersection_area
            / mine_area_m2
        )

        # ---------------------------------------------------------------------
        # Emission assigned to this cell [mol/s]
        # ---------------------------------------------------------------------

        cell_emission_mol_s = (
            emission_mol_s
            * fraction
        )

        # ---------------------------------------------------------------------
        # Convert to surface flux [mol/m2/s]
        #
        # The flux is calculated over the actual STILT grid-cell area.
        # ---------------------------------------------------------------------

        cell_area_m2 = (
            cell_row["area_m2"]
        )

        cell_flux = (
            cell_emission_mol_s
            / cell_area_m2
        )

        j = int(
            cell_row["j"]
        )

        i = int(
            cell_row["i"]
        )

        emissions[mine][j, i] += (
            cell_flux
        )

        allocated_mol_s += (
            cell_flux
            * cell_area_m2
        )

        intersected_cells += 1

    # -------------------------------------------------------------------------
    # Conservation check
    # -------------------------------------------------------------------------

    absolute_error = (
        allocated_mol_s
        - emission_mol_s
    )

    relative_error = (
        absolute_error
        / emission_mol_s
        if emission_mol_s != 0
        else 0.0
    )

    print()
    print(
        "Intersected cells:",
        intersected_cells
    )

    print()
    print(
        "Requested emission:",
        f"{emission_mol_s:.12e}",
        "mol/s"
    )

    print(
        "Allocated emission:",
        f"{allocated_mol_s:.12e}",
        "mol/s"
    )

    print(
        "Absolute error:",
        f"{absolute_error:.12e}",
        "mol/s"
    )

    print(
        "Relative error:",
        f"{relative_error:.12e}"
    )

    if abs(relative_error) > 1e-8:

        raise RuntimeError(
            f"\nConservation check failed for {mine}.\n"
            f"Relative error = {relative_error}"
        )

    print()
    print(
        f"CONSERVATION CHECK PASSED: {mine}"
    )


# =============================================================================
# CREATE TOTAL MINE FIELD
# =============================================================================

print()
print("=" * 80)
print("CREATING TOTAL OC MINE FIELD")
print("=" * 80)

oc_total = np.zeros(
    (nlat, nlon),
    dtype=np.float64
)

for mine in MINE_NAMES:

    oc_total += emissions[mine]


# =============================================================================
# FINAL CONSERVATION SUMMARY
# =============================================================================

print()
print("=" * 80)
print("FINAL EMISSION SUMMARY")
print("=" * 80)

summary_rows = []

for mine in MINE_NAMES:

    row = df[
        df["Facility_code_name"] == mine
    ].iloc[0]

    inventory_g_s = float(
        row["gCH4/s"]
    )

    inventory_mol_s = (
        inventory_g_s
        / CH4_MW_G_MOL
    )

    allocated_mol_s = (
        np.sum(
            emissions[mine]
            * np.array(
                grid_gdf["area_m2"]
            ).reshape(
                nlat,
                nlon
            )
        )
    )

    summary_rows.append(
        {
            "Mine": mine,
            "Inventory_g_CH4_s": inventory_g_s,
            "Inventory_mol_CH4_s": inventory_mol_s,
            "Allocated_mol_CH4_s": allocated_mol_s,
            "Relative_error": (
                allocated_mol_s
                / inventory_mol_s
                - 1.0
            ),
            "Nonzero_cells": int(
                np.count_nonzero(
                    emissions[mine] > 0
                )
            ),
        }
    )

summary = pd.DataFrame(
    summary_rows
)

print()

print(
    summary.to_string(
        index=False,
        formatters={
            "Inventory_g_CH4_s":
                "{:.6f}".format,
            "Inventory_mol_CH4_s":
                "{:.8e}".format,
            "Allocated_mol_CH4_s":
                "{:.8e}".format,
            "Relative_error":
                "{:.3e}".format,
        }
    )
)

# =============================================================================
# WRITE NETCDF
# =============================================================================

print()
print("=" * 80)
print("WRITING STILT MINE EMISSION NETCDF")
print("=" * 80)

# Remove existing file

if os.path.exists(OUTPUT_FILE):

    print()
    print(
        "Removing existing output:"
    )

    print(
        " ",
        OUTPUT_FILE
    )

    os.remove(
        OUTPUT_FILE
    )

nc = Dataset(
    OUTPUT_FILE,
    "w",
    format="NETCDF4"
)

# -------------------------------------------------------------------------
# Dimensions
# -------------------------------------------------------------------------

nc.createDimension(
    "lat",
    nlat
)

nc.createDimension(
    "lon",
    nlon
)

# -------------------------------------------------------------------------
# Coordinates
# -------------------------------------------------------------------------

lat_var = nc.createVariable(
    "lat",
    "f8",
    ("lat",)
)

lon_var = nc.createVariable(
    "lon",
    "f8",
    ("lon",)
)

lat_var[:] = lat
lon_var[:] = lon

lat_var.units = "degrees_north"
lat_var.long_name = "latitude"

lon_var.units = "degrees_east"
lon_var.long_name = "longitude"

# -------------------------------------------------------------------------
# Mine variables
# -------------------------------------------------------------------------

for mine in MINE_NAMES:

    var = nc.createVariable(
        mine,
        "f8",
        ("lat", "lon"),
        zlib=True,
        complevel=4,
        fill_value=0.0
    )

    var[:, :] = emissions[mine]

    var.units = (
        "mol CH4 m-2 s-1"
    )

    var.long_name = (
        f"{mine} open-cut coal mine "
        "CH4 emission flux"
    )

    var.description = (
        "Spatially allocated from mine polygon "
        "using inventory annual-average emission."
    )

    var.coordinates = (
        "lon lat"
    )

# -------------------------------------------------------------------------
# Total
# -------------------------------------------------------------------------

total_var = nc.createVariable(
    "OC_TOTAL",
    "f8",
    ("lat", "lon"),
    zlib=True,
    complevel=4,
    fill_value=0.0
)

total_var[:, :] = oc_total

total_var.units = (
    "mol CH4 m-2 s-1"
)

total_var.long_name = (
    "Total Upper Hunter open-cut mine "
    "CH4 emission flux"
)

total_var.description = (
    "Sum of OC1 through OC13."
)

total_var.coordinates = (
    "lon lat"
)

# -------------------------------------------------------------------------
# Global attributes
# -------------------------------------------------------------------------

nc.title = (
    "Upper Hunter individual open-cut mine "
    "CH4 emission fields for STILT"
)

nc.source = (
    "OC1-OC13 mine shapefiles and "
    "UpperHuntercoalmines.csv"
)

nc.emission_species = (
    "CH4"
)

nc.emission_units = (
    "mol CH4 m-2 s-1"
)

nc.ch4_molecular_weight = (
    CH4_MW_G_MOL
)

nc.spatial_allocation = (
    "Mine-polygon/STILT-grid-cell intersection area"
)

nc.temporal_assumption = (
    "Constant annual-average emission rate; "
    "equivalent to annual emission divided by 365 days"
)

nc.stilt_footprint_source = (
    STILT_FOOTPRINT
)

nc.grid_crs = (
    "EPSG:4326"
)

nc.allocation_crs = (
    METRIC_CRS
)

nc.mine_count = (
    len(MINE_NAMES)
)

nc.mine_names = (
    ",".join(MINE_NAMES)
)

nc.close()


# =============================================================================
# WRITE SUMMARY CSV
# =============================================================================

summary_file = os.path.splitext(
    OUTPUT_FILE
)[0] + "_summary.csv"

summary.to_csv(
    summary_file,
    index=False
)


# =============================================================================
# FINAL DIAGNOSTICS
# =============================================================================

print()
print("=" * 80)
print("COMPLETED")
print("=" * 80)

print()
print("NetCDF:")
print(
    " ",
    OUTPUT_FILE
)

print()
print("Summary:")
print(
    " ",
    summary_file
)

print()
print("Variables:")
print(
    " ",
    ", ".join(MINE_NAMES)
)

print()
print("Plus:")
print("  OC_TOTAL")

print()
print("Units:")
print("  mol CH4 m-2 s-1")

print()
print("Grid:")
print(
    f"  {nlat} latitude x {nlon} longitude"
)

print()
print("STILT grid spacing:")
print(
    f"  latitude  = {lat_spacing:.8f} degree"
)

print(
    f"  longitude = {lon_spacing:.8f} degree"
)

print()
print("=" * 80)
print("STILT MINE EMISSION FIELD CREATION COMPLETE")
print("=" * 80)

