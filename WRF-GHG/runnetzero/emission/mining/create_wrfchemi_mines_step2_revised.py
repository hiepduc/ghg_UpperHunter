#!/usr/bin/env python3

"""
===============================================================================
STEP 2
Create hourly WRF-Chem CH4 emission files from allocated OC MINE emissions
===============================================================================

INPUT
-----
Step-1 allocation files:

    wrfchem_allocation_d01.csv
    wrfchem_allocation_d02.csv
    wrfchem_allocation_d03.csv

Actual WRF input files:

    wrfinput_d01
    wrfinput_d02
    wrfinput_d03

OUTPUT
------
Directory:

    mines_OC/

Files:

    wrfchemi_d01_YYYY-MM-DD_HH:MM:SS
    wrfchemi_d02_YYYY-MM-DD_HH:MM:SS
    wrfchemi_d03_YYYY-MM-DD_HH:MM:SS

VARIABLE
--------
    E_CH4

UNITS
-----
    mol km^-2 hr^-1

IMPORTANT
---------
The spatial allocation has already been performed by STEP 1.

This program does NOT perform any spatial regridding.

The actual WRF grid is obtained directly from wrfinput_d01/d02/d03.

===============================================================================
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from netCDF4 import Dataset


# =============================================================================
# USER SETTINGS
# =============================================================================

ALLOCATION_FILES = {
    "d01": "wrfchem_allocation_d01.csv",
    "d02": "wrfchem_allocation_d02.csv",
    "d03": "wrfchem_allocation_d03.csv",
}

WRF_INPUT_FILES = {
    "d01": "wrfinput_d01",
    "d02": "wrfinput_d02",
    "d03": "wrfinput_d03",
}

OUTPUT_DIR = "mines_OC"

START_DATE = datetime(
    2026, 1, 1, 0, 0
)

END_DATE = datetime(
    2026, 2, 28, 23, 0
)


# =============================================================================
# HEADER
# =============================================================================

print()
print("=" * 80)
print("STEP 2 - OC MINE CH4 -> WRF-CHEM")
print("=" * 80)

print()
print("Simulation:")
print(
    " ",
    START_DATE.strftime("%Y-%m-%d %H:%M:%S"),
    "to",
    END_DATE.strftime("%Y-%m-%d %H:%M:%S")
)

print()
print("Output directory:")
print(" ", os.path.abspath(OUTPUT_DIR))


# =============================================================================
# CHECK INPUT FILES
# =============================================================================

print()
print("=" * 80)
print("CHECKING INPUT FILES")
print("=" * 80)

for domain in ["d01", "d02", "d03"]:

    allocation_file = ALLOCATION_FILES[domain]
    wrf_file = WRF_INPUT_FILES[domain]

    print()
    print(domain.upper())

    print("  Allocation:")
    print("   ", os.path.abspath(allocation_file))

    if not os.path.exists(allocation_file):

        raise FileNotFoundError(
            "\nAllocation file not found:\n"
            + os.path.abspath(allocation_file)
        )

    print("  WRF input:")
    print("   ", os.path.abspath(wrf_file))

    if not os.path.exists(wrf_file):

        raise FileNotFoundError(
            "\nWRF input file not found:\n"
            + os.path.abspath(wrf_file)
        )


# =============================================================================
# CREATE OUTPUT DIRECTORY
# =============================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print()
print("Output directory ready:")
print(" ", os.path.abspath(OUTPUT_DIR))


# =============================================================================
# READ WRF GRID AND METADATA
# =============================================================================

def read_wrf_grid(filename):

    """
    Read actual WRF mass-grid dimensions, coordinates and relevant
    WRF global attributes from wrfinput.
    """

    print()
    print("-" * 80)
    print("Reading WRF grid and metadata:")
    print(" ", filename)

    nc = Dataset(
        filename,
        "r"
    )

    # -------------------------------------------------------------------------
    # Dimensions
    # -------------------------------------------------------------------------

    if "south_north" not in nc.dimensions:
        raise RuntimeError(
            f"south_north dimension not found in {filename}"
        )

    if "west_east" not in nc.dimensions:
        raise RuntimeError(
            f"west_east dimension not found in {filename}"
        )

    ny = len(
        nc.dimensions["south_north"]
    )

    nx = len(
        nc.dimensions["west_east"]
    )

    # -------------------------------------------------------------------------
    # Coordinates
    # -------------------------------------------------------------------------

    if "XLAT" not in nc.variables:
        raise RuntimeError(
            f"XLAT not found in {filename}"
        )

    if "XLONG" not in nc.variables:
        raise RuntimeError(
            f"XLONG not found in {filename}"
        )

    lat = np.asarray(
        nc.variables["XLAT"][0, :, :],
        dtype=np.float64
    )

    lon = np.asarray(
        nc.variables["XLONG"][0, :, :],
        dtype=np.float64
    )

    # -------------------------------------------------------------------------
    # Check coordinates
    # -------------------------------------------------------------------------

    if lat.shape != (ny, nx):

        raise RuntimeError(
            f"XLAT shape {lat.shape} does not match "
            f"WRF dimensions ({ny}, {nx})"
        )

    if lon.shape != (ny, nx):

        raise RuntimeError(
            f"XLONG shape {lon.shape} does not match "
            f"WRF dimensions ({ny}, {nx})"
        )

    # -------------------------------------------------------------------------
    # Read WRF global attributes
    # -------------------------------------------------------------------------

    wrf_attributes = {}

    important_attributes = [
        "TITLE",
        "START_DATE",
        "SIMULATION_START_DATE",

        "WEST-EAST_GRID_DIMENSION",
        "SOUTH-NORTH_GRID_DIMENSION",
        "BOTTOM-TOP_GRID_DIMENSION",

        "DX",
        "DY",

        "CEN_LAT",
        "CEN_LON",

        "TRUELAT1",
        "TRUELAT2",

        "MOAD_CEN_LAT",

        "STAND_LON",

        "MAP_PROJ",
        "MAP_PROJ_CHAR",

        "POLE_LAT",
        "POLE_LON",

        "GMT",
        "JULYR",
        "JULDAY",

        "MMINLU",
        "ISWATER",
        "ISICE",
        "ISURBAN",
        "ISOILWATER",

        "GRID_ID",
        "PARENT_ID",
        "I_PARENT_START",
        "J_PARENT_START",
        "PARENT_GRID_RATIO",
    ]

    for name in important_attributes:

        if name in nc.ncattrs():

            value = nc.getncattr(name)

            wrf_attributes[name] = value

    nc.close()

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    print()
    print("  WRF grid:")
    print("    south_north =", ny)
    print("    west_east   =", nx)

    print()
    print("  Latitude:")
    print(
        "   ",
        np.nanmin(lat),
        "to",
        np.nanmax(lat)
    )

    print()
    print("  Longitude:")
    print(
        "   ",
        np.nanmin(lon),
        "to",
        np.nanmax(lon)
    )

    print()
    print("  Important WRF metadata:")

    for name in [
        "DX",
        "DY",
        "CEN_LAT",
        "CEN_LON",
        "TRUELAT1",
        "TRUELAT2",
        "STAND_LON",
        "MAP_PROJ",
    ]:

        if name in wrf_attributes:

            print(
                f"    {name} = {wrf_attributes[name]}"
            )

    return (
        ny,
        nx,
        lat,
        lon,
        wrf_attributes
    )


# =============================================================================
# READ STEP-1 ALLOCATION
# =============================================================================

def read_allocation(
    filename,
    domain,
    ny,
    nx
):

    """
    Read Step-1 mine allocation and construct a complete
    WRF E_CH4 grid.

    CSV contains only non-zero cells.
    """

    print()
    print("-" * 80)
    print("Reading mine allocation:")
    print(" ", filename)

    df = pd.read_csv(
        filename
    )

    print()
    print(
        "  Allocation records:",
        len(df)
    )

    # -------------------------------------------------------------------------
    # Required columns
    # -------------------------------------------------------------------------

    required_columns = [
        "j",
        "i",
        "E_CH4_mol_km2_hr"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise RuntimeError(
                "\nRequired column not found: "
                + column
                + "\nFile: "
                + filename
            )

    # -------------------------------------------------------------------------
    # Check domain
    # -------------------------------------------------------------------------

    if "domain" in df.columns:

        domains = (
            df["domain"]
            .astype(str)
            .str.lower()
            .unique()
        )

        print()
        print(
            "  Domain values:",
            domains
        )

        if domain.lower() not in domains:

            raise RuntimeError(
                f"\nAllocation file {filename} does not contain "
                f"expected domain {domain}.\n"
                f"Found: {domains}"
            )

    # -------------------------------------------------------------------------
    # Create complete grid
    # -------------------------------------------------------------------------

    emission = np.zeros(
        (ny, nx),
        dtype=np.float64
    )

    # -------------------------------------------------------------------------
    # Insert allocation
    # -------------------------------------------------------------------------

    for _, row in df.iterrows():

        j = int(row["j"])
        i = int(row["i"])

        value = float(
            row["E_CH4_mol_km2_hr"]
        )

        if j < 0 or j >= ny:

            raise RuntimeError(
                f"\nInvalid j index {j} for {domain}.\n"
                f"Valid range: 0 to {ny - 1}"
            )

        if i < 0 or i >= nx:

            raise RuntimeError(
                f"\nInvalid i index {i} for {domain}.\n"
                f"Valid range: 0 to {nx - 1}"
            )

        if not np.isfinite(value):

            continue

        if value < 0.0:

            raise RuntimeError(
                f"\nNegative CH4 emission at "
                f"j={j}, i={i}: {value}"
            )

        emission[j, i] += value

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    emission[
        ~np.isfinite(emission)
    ] = 0.0

    positive = emission > 0.0

    print()
    print("  E_CH4 diagnostics:")

    print(
        "    Grid shape:",
        emission.shape
    )

    print(
        "    Non-zero cells:",
        np.count_nonzero(positive)
    )

    print(
        "    Minimum:",
        np.min(emission)
    )

    print(
        "    Maximum:",
        np.max(emission)
    )

    print(
        "    Mean:",
        np.mean(emission)
    )

    print(
        "    Sum:",
        np.sum(emission),
        "mol km^-2 hr^-1"
    )

    return emission


# =============================================================================
# CREATE WRF-CHEM EMISSION FILE
# =============================================================================

def create_wrfchemi(
    output_file,
    date_time,
    lat,
    lon,
    emission,
    wrf_attributes,
):

    """
    Create one WRF-Chem hourly anthropogenic CH4 emission file.
    """

    ny, nx = lat.shape

    print()
    print("Creating:")
    print(" ", output_file)

    # -------------------------------------------------------------------------
    # Create NetCDF
    # -------------------------------------------------------------------------

    nc = Dataset(
        output_file,
        "w",
        format="NETCDF4"
    )

    # -------------------------------------------------------------------------
    # Dimensions
    #
    # Time is UNLIMITED, which is the normal WRF-Chem convention.
    # -------------------------------------------------------------------------

    nc.createDimension(
        "Time",
        None
    )

    nc.createDimension(
        "DateStrLen",
        19
    )

    nc.createDimension(
        "emissions_zdim_stag",
        1
    )

    nc.createDimension(
        "south_north",
        ny
    )

    nc.createDimension(
        "west_east",
        nx
    )

    # =========================================================================
    # COPY IMPORTANT WRF GLOBAL ATTRIBUTES
    # =========================================================================

    for name, value in wrf_attributes.items():

        try:

            nc.setncattr(
                name,
                value
            )

        except Exception as exc:

            print(
                f"WARNING: Could not copy global attribute "
                f"{name}: {exc}"
            )

    # =========================================================================
    # OVERRIDE / ADD EMISSION-SPECIFIC GLOBAL ATTRIBUTES
    # =========================================================================

    nc.setncattr(
        "TITLE",
        "WRF-Chem Anthropogenic CH4 Emissions"
    )

    nc.setncattr(
        "Title",
        "Upper Hunter Open-Cut Mine CH4 emissions"
    )

    nc.setncattr(
        "Domain",
        "WRF-Chem " + output_file.split("/")[-1]
    )

    nc.setncattr(
        "History",
        "Created from Step-1 mine allocation using actual WRF grid"
    )

    nc.setncattr(
        "Author",
        "WRF-Chem CH4 emission converter"
    )

    nc.setncattr(
        "Emission_source",
        "Upper Hunter open-cut coal mines"
    )

    nc.setncattr(
        "Emission_units",
        "mol km^-2 hr^-1"
    )

    # =========================================================================
    # TIMES
    # =========================================================================

    times_var = nc.createVariable(
        "Times",
        "S1",
        (
            "Time",
            "DateStrLen"
        )
    )

    times_var.description = "Model times"

    times_string = date_time.strftime(
        "%Y-%m-%d_%H:%M:%S"
    )

    times_var[0, :] = np.array(
        list(times_string),
        dtype="S1"
    )

    # =========================================================================
    # XLAT
    # =========================================================================

    lat_var = nc.createVariable(
        "XLAT",
        "f4",
        (
            "Time",
            "south_north",
            "west_east"
        ),
        fill_value=False
    )

    lat_var.MemoryOrder = "XY "
    lat_var.description = (
        "LATITUDE, SOUTH IS NEGATIVE"
    )
    lat_var.units = "degree north"
    lat_var.stagger = ""
    lat_var.FieldType = 104

    lat_var[0, :, :] = lat.astype(
        np.float32
    )

    # =========================================================================
    # XLONG
    # =========================================================================

    lon_var = nc.createVariable(
        "XLONG",
        "f4",
        (
            "Time",
            "south_north",
            "west_east"
        ),
        fill_value=False
    )

    lon_var.MemoryOrder = "XY "
    lon_var.description = (
        "LONGITUDE, WEST IS NEGATIVE"
    )
    lon_var.units = "degree east"
    lon_var.stagger = ""
    lon_var.FieldType = 104

    lon_var[0, :, :] = lon.astype(
        np.float32
    )

    # =========================================================================
    # E_CH4
    # =========================================================================

    evar = nc.createVariable(
        "E_CH4",
        "f4",
        (
            "Time",
            "emissions_zdim_stag",
            "south_north",
            "west_east"
        ),
        fill_value=False
    )

    evar.MemoryOrder = "XYZ"

    evar.description = "EMISSIONS"

    evar.units = (
        "mol km^-2 hr^-1"
    )

    evar.stagger = "Z"

    evar.coordinates = "XLONG XLAT"

    evar.FieldType = 104

    evar[0, 0, :, :] = emission.astype(
        np.float32
    )

    # =========================================================================
    # CLOSE
    # =========================================================================

    nc.close()


# =============================================================================
# READ ALL DOMAINS
# =============================================================================

domain_data = {}

for domain in ["d01", "d02", "d03"]:

    print()
    print()
    print("=" * 80)
    print(
        f"PROCESSING DOMAIN {domain.upper()}"
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Read actual WRF grid
    # -------------------------------------------------------------------------

    (
        ny,
        nx,
        lat,
        lon,
        wrf_attributes
    ) = read_wrf_grid(
        WRF_INPUT_FILES[domain]
    )

    # -------------------------------------------------------------------------
    # Read Step-1 allocation
    # -------------------------------------------------------------------------

    emission = read_allocation(
        ALLOCATION_FILES[domain],
        domain,
        ny,
        nx
    )

    # -------------------------------------------------------------------------
    # Store
    # -------------------------------------------------------------------------

    domain_data[domain] = {
        "ny": ny,
        "nx": nx,
        "lat": lat,
        "lon": lon,
        "emission": emission,
        "wrf_attributes": wrf_attributes,
    }


# =============================================================================
# CREATE HOURLY FILES
# =============================================================================

print()
print()
print("=" * 80)
print("CREATING HOURLY WRF-CHEM MINE EMISSION FILES")
print("=" * 80)

for domain in ["d01", "d02", "d03"]:

    print()
    print("#" * 80)
    print(
        f"CREATING FILES FOR {domain.upper()}"
    )
    print("#" * 80)

    lat = domain_data[domain]["lat"]
    lon = domain_data[domain]["lon"]

    emission = domain_data[domain]["emission"]

    wrf_attributes = domain_data[domain]["wrf_attributes"]

    current = START_DATE

    file_count = 0

    while current <= END_DATE:

        filename = (
            f"wrfchemi_{domain}_"
            f"{current.strftime('%Y-%m-%d_%H:%M:%S')}"
        )

        output_file = os.path.join(
            OUTPUT_DIR,
            filename
        )

        create_wrfchemi(
            output_file,
            current,
            lat,
            lon,
            emission,
            wrf_attributes,
        )

        file_count += 1

        current += timedelta(
            hours=1
        )

    print()
    print(
        f"{domain.upper()} files created:",
        file_count
    )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

expected_hours = int(
    (
        END_DATE - START_DATE
    ).total_seconds()
    / 3600.0
) + 1

print()
print()
print("=" * 80)
print("STEP 2 COMPLETED")
print("=" * 80)

print()
print("Simulation:")
print(
    " ",
    START_DATE.strftime("%Y-%m-%d %H:%M:%S"),
    "to",
    END_DATE.strftime("%Y-%m-%d %H:%M:%S")
)

print()
print(
    "Expected hourly files per domain:",
    expected_hours
)

print()
print("Output directory:")
print(
    " ",
    os.path.abspath(OUTPUT_DIR)
)

print()
print("Domains:")
print("  d01")
print("  d02")
print("  d03")

print()
print("WRF-Chem variable:")
print("  E_CH4")

print()
print("Units:")
print("  mol km^-2 hr^-1")

print()
print("=" * 80)
print("OC MINE CH4 PROCESSING COMPLETE")
print("=" * 80)

