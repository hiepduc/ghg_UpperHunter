#!/usr/bin/env python3
"""
Create individual Upper Hunter open-cut coal mine CH4 emission fields
on the STILT footprint grid.

Purpose
-------
Create one constant spatial CH4 emission field for each mine OC1-OC13.

The mine emission rate is assumed constant throughout the year:

    annual-average emission = inventory gCH4/s

Therefore:

    mol CH4/s = (gCH4/s) / molecular_weight

The total mine emission is spatially distributed over the mine polygon.
For each STILT grid cell:

    cell_emission_mol_s =
        mine_total_mol_s * intersection_area / mine_area

    flux =
        cell_emission_mol_s / STILT_cell_area

Units of output:
    mol CH4 m-2 s-1

This is appropriate for integration with a STILT footprint whose units are:

    ppm (umol m-2 s)-1

Therefore, when calculating the STILT concentration contribution:

    contribution_ppm =
        sum(footprint * flux_mol_m2_s * 1e6)

because:

    1 mol = 1e6 umol

Important
---------
The output contains separate variables:

    OC1 ... OC13

and:

    OC_TOTAL

No temporal dimension is used because the emissions are constant in time.

The STILT footprint is used ONLY to obtain the receptor grid.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box


# =============================================================================
# USER SETTINGS
# =============================================================================

BASE_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/arl_mine_2026/2026-01/footprints/Muswellbrook/footprints/plot_mines"
   # "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter"
)

INVENTORY_FILE = os.path.join(
    BASE_DIR,
    "UpperHuntercoalmines.csv"
)

FOOTPRINT_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/arl_mine_2026/2026-01/"
    "footprints/Muswellbrook/footprints/"
    "202601151200_150.89297_-32.26202_5_foot.nc"
)

OUTPUT_NC = os.path.join(
    BASE_DIR,
    "stilt_mine_emissions_d03.nc"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "stilt_mine_emissions_summary.csv"
)

MINE_NAMES = [f"OC{i}" for i in range(1, 14)]

# Molecular weight of CH4
MW_CH4 = 16.04       # g/mol

# STILT grid is geographic, but intersection areas must be calculated
# in a projected CRS.
UTM_CRS = "EPSG:32756"      # UTM zone 56S


# =============================================================================
# FUNCTIONS
# =============================================================================

def get_cell_edges(values):
    """
    Convert regularly spaced cell-center coordinates into cell edges.

    Example:
        centers = [1, 2, 3]

    gives:
        edges = [0.5, 1.5, 2.5, 3.5]
    """

    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        raise ValueError("Need at least two grid coordinates.")

    diffs = np.diff(values)

    # Check regular spacing
    if not np.allclose(diffs, diffs[0], rtol=0, atol=1e-8):
        raise ValueError(
            "STILT grid coordinates are not regularly spaced."
        )

    d = diffs[0]

    edges = np.empty(len(values) + 1, dtype=float)

    edges[1:-1] = 0.5 * (values[:-1] + values[1:])

    edges[0] = values[0] - d / 2.0
    edges[-1] = values[-1] + d / 2.0

    return edges


def make_grid_cells(lat, lon):
    """
    Create STILT grid-cell polygons in EPSG:4326.

    Returns
    -------
    cells : list
        List of tuples:
            (j, i, polygon)
    """

    lat_edges = get_cell_edges(lat)
    lon_edges = get_cell_edges(lon)

    cells = []

    for j in range(len(lat)):
        for i in range(len(lon)):

            polygon = box(
                lon_edges[i],
                lat_edges[j],
                lon_edges[i + 1],
                lat_edges[j + 1]
            )

            cells.append((j, i, polygon))

    return cells, lat_edges, lon_edges


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("UPPER HUNTER STILT MINE EMISSION FIELD")
print("=" * 80)

print()
print("Inventory:")
print(INVENTORY_FILE)

print()
print("STILT footprint:")
print(FOOTPRINT_FILE)

print()
print("Output:")
print(OUTPUT_NC)


# =============================================================================
# READ STILT FOOTPRINT GRID
# =============================================================================

print()
print("=" * 80)
print("READING STILT GRID")
print("=" * 80)

ds = xr.open_dataset(FOOTPRINT_FILE)

lat = np.asarray(ds["lat"].values, dtype=float)
lon = np.asarray(ds["lon"].values, dtype=float)

print("lat:")
print("  number =", len(lat))
print("  min    =", lat.min())
print("  max    =", lat.max())

print("lon:")
print("  number =", len(lon))
print("  min    =", lon.min())
print("  max    =", lon.max())

print("lat spacing =", np.diff(lat).min(), np.diff(lat).max())
print("lon spacing =", np.diff(lon).min(), np.diff(lon).max())

cells, lat_edges, lon_edges = make_grid_cells(lat, lon)

print()
print("Number of STILT cells =", len(cells))

print()
print("STILT cell bounds:")
print("  longitude:", lon_edges[0], "to", lon_edges[-1])
print("  latitude :", lat_edges[0], "to", lat_edges[-1])


# =============================================================================
# READ INVENTORY
# =============================================================================

print()
print("=" * 80)
print("READING MINE INVENTORY")
print("=" * 80)

inv = pd.read_csv(INVENTORY_FILE)

inv["Facility_code_name"] = (
    inv["Facility_code_name"]
    .astype(str)
    .str.strip()
)

inv["gCH4/s"] = pd.to_numeric(
    inv["gCH4/s"],
    errors="coerce"
)

print()
print("Inventory mine emissions:")

for mine in MINE_NAMES:

    row = inv.loc[
        inv["Facility_code_name"] == mine
    ]

    if len(row) != 1:
        raise ValueError(
            f"Expected exactly one inventory record for {mine}, "
            f"found {len(row)}"
        )

    rate = float(row.iloc[0]["gCH4/s"])

    mol_s = rate / MW_CH4

    print(
        f"  {mine:4s} : "
        f"{rate:12.4f} g/s  "
        f"{mol_s:12.6f} mol/s"
    )


# =============================================================================
# CREATE GEOGRAPHIC GRID DATAFRAME
# =============================================================================

print()
print("=" * 80)
print("CREATING STILT GRID")
print("=" * 80)

grid_records = []

for j, i, polygon in cells:

    grid_records.append(
        {
            "j": j,
            "i": i,
            "geometry": polygon
        }
    )

grid = gpd.GeoDataFrame(
    grid_records,
    geometry="geometry",
    crs="EPSG:4326"
)

# Project grid to UTM for accurate area calculations
grid_utm = grid.to_crs(UTM_CRS)

grid_utm["cell_area_m2"] = grid_utm.geometry.area

print()
print("Grid CRS:", grid_utm.crs)

print(
    "Cell area range:",
    grid_utm["cell_area_m2"].min(),
    "to",
    grid_utm["cell_area_m2"].max(),
    "m2"
)


# =============================================================================
# PROCESS EACH MINE
# =============================================================================

mine_fields = {}

summary = []

for mine in MINE_NAMES:

    print()
    print("=" * 80)
    print(f"PROCESSING {mine}")
    print("=" * 80)

    shp = os.path.join(
        BASE_DIR,
        f"{mine}.shp"
    )

    if not os.path.exists(shp):
        raise FileNotFoundError(shp)

    # -------------------------------------------------------------------------
    # Read mine polygon
    # -------------------------------------------------------------------------

    mine_gdf = gpd.read_file(shp)

    if mine_gdf.empty:
        raise ValueError(f"{mine} shapefile contains no features.")

    if mine_gdf.crs is None:
        raise ValueError(
            f"{mine} shapefile has no CRS."
        )

    print("Shapefile CRS:", mine_gdf.crs)
    print("Features:", len(mine_gdf))

    # Convert to WGS84 first
    mine_wgs84 = mine_gdf.to_crs("EPSG:4326")

    # Combine all features into one geometry
    mine_geom_wgs84 = mine_wgs84.geometry.unary_union

    # Project to UTM
    mine_geom_utm = gpd.GeoSeries(
        [mine_geom_wgs84],
        crs="EPSG:4326"
    ).to_crs(UTM_CRS).iloc[0]

    mine_area_m2 = mine_geom_utm.area

    print(
        "Mine area =",
        f"{mine_area_m2:.3f}",
        "m2"
    )

    # -------------------------------------------------------------------------
    # Inventory emission
    # -------------------------------------------------------------------------

    row = inv.loc[
        inv["Facility_code_name"] == mine
    ]

    rate_g_s = float(row.iloc[0]["gCH4/s"])

    total_mol_s = rate_g_s / MW_CH4

    print(
        "Inventory emission =",
        f"{rate_g_s:.6f}",
        "g/s"
    )

    print(
        "Inventory emission =",
        f"{total_mol_s:.9f}",
        "mol/s"
    )

    # -------------------------------------------------------------------------
    # Check mine is inside STILT domain
    # -------------------------------------------------------------------------

    grid_union_utm = grid_utm.geometry.unary_union

    outside_area = mine_geom_utm.difference(
        grid_union_utm
    ).area

    inside_area = mine_geom_utm.intersection(
        grid_union_utm
    ).area

    fraction_inside = (
        inside_area / mine_area_m2
        if mine_area_m2 > 0
        else 0.0
    )

    print(
        "Fraction of mine inside STILT grid =",
        f"{fraction_inside:.12f}"
    )

    if fraction_inside < 0.999999:

        raise RuntimeError(
            f"{mine} is not completely inside the STILT grid. "
            f"Only {fraction_inside:.8f} of the mine is inside."
        )

    # -------------------------------------------------------------------------
    # Find intersecting STILT cells
    # -------------------------------------------------------------------------

    # Spatial index query makes this much faster than intersecting
    # every mine against every grid cell.
    candidate_idx = list(
        grid_utm.sindex.query(
            mine_geom_utm,
            predicate="intersects"
        )
    )

    print(
        "Candidate/intersecting cells =",
        len(candidate_idx)
    )

    if len(candidate_idx) == 0:

        raise RuntimeError(
            f"{mine}: zero intersecting STILT cells."
        )

    # -------------------------------------------------------------------------
    # Allocate emission to cells
    # -------------------------------------------------------------------------

    field = np.zeros(
        (len(lat), len(lon)),
        dtype=np.float64
    )

    allocated_mol_s = 0.0

    nonzero_cells = 0

    for idx in candidate_idx:

        cell = grid_utm.iloc[idx]

        intersection = mine_geom_utm.intersection(
            cell.geometry
        )

        if intersection.is_empty:
            continue

        intersection_area = intersection.area

        if intersection_area <= 0:
            continue

        # Fraction of mine area contained in this cell
        fraction = (
            intersection_area /
            mine_area_m2
        )

        # Emission assigned to this grid cell
        cell_mol_s = (
            total_mol_s *
            fraction
        )

        # Convert to surface flux
        cell_area_m2 = cell["cell_area_m2"]

        flux = (
            cell_mol_s /
            cell_area_m2
        )

        j = int(cell["j"])
        i = int(cell["i"])

        field[j, i] += flux

        allocated_mol_s += cell_mol_s

        if flux > 0:
            nonzero_cells += 1

    # -------------------------------------------------------------------------
    # Conservation check
    # -------------------------------------------------------------------------

    relative_error = (
        (allocated_mol_s - total_mol_s)
        / total_mol_s
        if total_mol_s != 0
        else 0.0
    )

    print()
    print("Allocation:")
    print(
        "  allocated emission =",
        f"{allocated_mol_s:.12f}",
        "mol/s"
    )

    print(
        "  expected emission  =",
        f"{total_mol_s:.12f}",
        "mol/s"
    )

    print(
        "  relative error     =",
        f"{relative_error:.3e}"
    )

    print(
        "  non-zero cells     =",
        nonzero_cells
    )

    print(
        "  min flux           =",
        f"{field.min():.6e}",
        "mol m-2 s-1"
    )

    print(
        "  max flux           =",
        f"{field.max():.6e}",
        "mol m-2 s-1"
    )

    if not np.isclose(
        allocated_mol_s,
        total_mol_s,
        rtol=1e-8,
        atol=1e-20
    ):
        raise RuntimeError(
            f"{mine}: emission conservation failed."
        )

    if np.count_nonzero(field) == 0:

        raise RuntimeError(
            f"{mine}: output field contains no non-zero cells."
        )

    # Save
    mine_fields[mine] = field

    summary.append(
        {
            "mine": mine,
            "inventory_gCH4_s": rate_g_s,
            "inventory_molCH4_s": total_mol_s,
            "mine_area_m2": mine_area_m2,
            "nonzero_cells": np.count_nonzero(field),
            "max_flux_mol_m2_s": field.max(),
            "sum_cell_emission_mol_s": allocated_mol_s,
            "relative_error": relative_error
        }
    )


# =============================================================================
# TOTAL FIELD
# =============================================================================

print()
print("=" * 80)
print("CREATING TOTAL MINE FIELD")
print("=" * 80)

total_field = np.zeros(
    (len(lat), len(lon)),
    dtype=np.float64
)

for mine in MINE_NAMES:

    total_field += mine_fields[mine]

print(
    "Total non-zero cells =",
    np.count_nonzero(total_field)
)

print(
    "Total maximum flux =",
    f"{total_field.max():.6e}",
    "mol m-2 s-1"
)


# =============================================================================
# WRITE NETCDF
# =============================================================================

print()
print("=" * 80)
print("WRITING NETCDF")
print("=" * 80)

data_vars = {}

for mine in MINE_NAMES:

    # IMPORTANT:
    # Do not use zero as _FillValue.
    # Zero is a legitimate emission value.
    data_vars[mine] = (
        ("lat", "lon"),
        mine_fields[mine]
    )

data_vars["OC_TOTAL"] = (
    ("lat", "lon"),
    total_field
)

out = xr.Dataset(
    data_vars=data_vars,
    coords={
        "lat": lat,
        "lon": lon
    }
)

out["lat"].attrs = {
    "units": "degrees_north",
    "long_name": "latitude"
}

out["lon"].attrs = {
    "units": "degrees_east",
    "long_name": "longitude"
}

for mine in MINE_NAMES:

    out[mine].attrs = {
        "long_name":
            f"{mine} CH4 surface emission flux",
        "units":
            "mol CH4 m-2 s-1",
        "description":
            "Constant annual-average mine emission distributed over mine polygon"
    }

out["OC_TOTAL"].attrs = {
    "long_name":
        "Total OC1-OC13 CH4 surface emission flux",
    "units":
        "mol CH4 m-2 s-1",
    "description":
        "Sum of individual open-cut mine emission fields"
}

out.attrs = {
    "description":
        "Upper Hunter individual open-cut coal mine CH4 emissions for STILT",
    "emission_assumption":
        "Constant annual-average emission rate throughout the year",
    "molecular_weight_CH4":
        MW_CH4,
    "grid_source":
        "STILT footprint grid",
    "projection_for_area":
        UTM_CRS,
    "source_inventory":
        INVENTORY_FILE
}

out.to_netcdf(
    OUTPUT_NC,
    format="NETCDF4"
)

print()
print("Written:")
print(OUTPUT_NC)


# =============================================================================
# WRITE SUMMARY CSV
# =============================================================================

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print()
print("Summary:")
print(OUTPUT_CSV)

print()
print(summary_df.to_string(index=False))


# =============================================================================
# FINAL CHECK
# =============================================================================

print()
print("=" * 80)
print("FINAL CHECK")
print("=" * 80)

check = xr.open_dataset(OUTPUT_NC)

for mine in MINE_NAMES + ["OC_TOTAL"]:

    arr = check[mine].values

    print(
        f"{mine:8s} "
        f"min={np.nanmin(arr):.6e} "
        f"max={np.nanmax(arr):.6e} "
        f"nonzero={np.count_nonzero(arr)}"
    )

check.close()
ds.close()

print()
print("=" * 80)
print("DONE")
print("=" * 80)

