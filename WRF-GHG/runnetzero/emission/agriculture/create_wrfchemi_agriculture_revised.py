#!/usr/bin/env python3
"""
======================================================================
CREATE EDGAR AGRICULTURE CH4 EMISSION FILES FOR WRF-CHEM
======================================================================

Input
-----
EDGAR 2023 Agriculture CH4:

    EDGAR_2024_GHG_CH4_2023_bkl_AGRICULTURE_flx.nc

EDGAR variable:

    fluxes(time, lat, lon)

EDGAR units:

    kg m-2 s-1

WRF input files:

    wrfinput_d01
    wrfinput_d02
    wrfinput_d03

WRF emission variable:

    E_CH4

WRF-Chem units:

    mol km-2 hr-1

Purpose
-------
Create AGRICULTURE-ONLY WRF-Chem emission files.

The files contain ONLY:

    EDGAR Agriculture CH4

They do NOT contain the 13 open-cut mine emissions.

This is intended for the AGRICULTURE-ONLY WRF-Chem
source attribution simulation.

The EDGAR 2023 monthly fields are mapped onto the
corresponding months of the 2026 WRF-Chem simulation.

For example:

    EDGAR January 2023
        ->
    all January 2026 hourly emission files

    EDGAR February 2023
        ->
    all February 2026 hourly emission files

etc.

======================================================================
"""

import os
import glob
import numpy as np

from netCDF4 import Dataset, num2date
from scipy.interpolate import RegularGridInterpolator


# ====================================================================
# USER SETTINGS
# ====================================================================

EDGAR_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/edgar2024/"
    "EDGAR_2024_GHG_CH4_2023_bkl_AGRICULTURE_flx.nc"
)

# WRF input files
WRFINPUT_FILES = {
    1: "wrfinput_d01",
    2: "wrfinput_d02",
    3: "wrfinput_d03",
}

# Output directory
OUTPUT_DIR = "edgar_agr"

# Simulation period
START_YEAR = 2026
START_MONTH = 1
START_DAY = 1

END_YEAR = 2026
END_MONTH = 2
END_DAY = 28

# ====================================================================
# WRF-CHEM CH4 UNIT CONVERSION
# ====================================================================

# EDGAR:
#
#       kg CH4 m-2 s-1
#
# WRF-Chem:
#
#       mol CH4 km-2 hr-1
#
# 1 km2 = 1,000,000 m2
# 1 hour = 3600 s
# CH4 molecular weight = 16.04 g/mol
#                         = 0.01604 kg/mol
#
# Therefore:
#
# kg m-2 s-1
#
# * 1e6 m2/km2
# * 3600 s/hr
# / 0.01604 kg/mol
#
# = mol km-2 hr-1

CH4_MW_KG_PER_MOL = 0.01604

EDGAR_TO_WRFCHEM = (
    1.0e6
    * 3600.0
    / CH4_MW_KG_PER_MOL
)

print()
print("=" * 75)
print("EDGAR -> WRF-CHEM CH4 UNIT CONVERSION")
print("=" * 75)

print()
print(
    "Conversion factor =",
    EDGAR_TO_WRFCHEM
)

print(
    "kg m-2 s-1 -> mol km-2 hr-1"
)


# ====================================================================
# CHECK FILES
# ====================================================================

if not os.path.exists(EDGAR_FILE):

    raise FileNotFoundError(
        "\nEDGAR file not found:\n"
        + EDGAR_FILE
    )

for dom, filename in WRFINPUT_FILES.items():

    if not os.path.exists(filename):

        raise FileNotFoundError(
            f"\nWRF input file for d{dom:02d} "
            f"not found:\n{filename}"
        )

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ====================================================================
# READ EDGAR
# ====================================================================

print()
print("=" * 75)
print("READING EDGAR AGRICULTURE")
print("=" * 75)

print()
print("File:")
print(" ", EDGAR_FILE)

edgar = Dataset(
    EDGAR_FILE,
    "r"
)

edgar_lat = np.asarray(
    edgar.variables["lat"][:],
    dtype=float
)

edgar_lon = np.asarray(
    edgar.variables["lon"][:],
    dtype=float
)

edgar_time = np.asarray(
    edgar.variables["time"][:],
    dtype=float
)

flux_var = edgar.variables["fluxes"]

print()
print("EDGAR dimensions:")

print(
    "  lat  =",
    len(edgar_lat)
)

print(
    "  lon  =",
    len(edgar_lon)
)

print(
    "  time =",
    len(edgar_time)
)

print()
print("EDGAR latitude:")
print(
    "  min =",
    np.nanmin(edgar_lat)
)
print(
    "  max =",
    np.nanmax(edgar_lat)
)

print()
print("EDGAR longitude:")
print(
    "  min =",
    np.nanmin(edgar_lon)
)
print(
    "  max =",
    np.nanmax(edgar_lon)
)

print()
print("EDGAR units:")
print(
    " ",
    flux_var.units
)


# ====================================================================
# MAKE EDGAR GRID MONOTONIC
# ====================================================================

# RegularGridInterpolator requires increasing coordinates.

if edgar_lat[0] > edgar_lat[-1]:

    print()
    print("Reversing EDGAR latitude order.")

    edgar_lat = edgar_lat[::-1]

    reverse_lat = True

else:

    reverse_lat = False


if edgar_lon[0] > edgar_lon[-1]:

    print()
    print("Reversing EDGAR longitude order.")

    edgar_lon = edgar_lon[::-1]

    reverse_lon = True

else:

    reverse_lon = False


# ====================================================================
# READ EDGAR MONTHLY FIELDS
# ====================================================================

print()
print("=" * 75)
print("READING EDGAR MONTHLY FIELDS")
print("=" * 75)

edgar_monthly = {}

for month_index in range(12):

    print()
    print(
        f"Reading EDGAR month {month_index + 1:02d}"
    )

    field = np.asarray(
        flux_var[
            month_index,
            :, :
        ],
        dtype=float
    )

    # Convert masked / invalid values to zero

    field = np.ma.filled(
        np.ma.asarray(field),
        0.0
    )

    field = np.asarray(
        field,
        dtype=float
    )

    field[
        ~np.isfinite(field)
    ] = 0.0

    # Reorder latitude if necessary

    if reverse_lat:

        field = field[::-1, :]

    # Reorder longitude if necessary

    if reverse_lon:

        field = field[:, ::-1]

    # Negative emissions are not physically meaningful

    field[
        field < 0.0
    ] = 0.0

    print(
        "  min =",
        np.min(field)
    )

    print(
        "  max =",
        np.max(field)
    )

    print(
        "  mean =",
        np.mean(field)
    )

    edgar_monthly[
        month_index + 1
    ] = field


edgar.close()


# ====================================================================
# FUNCTION: READ WRF GRID
# ====================================================================

def read_wrf_grid(filename):

    print()
    print("=" * 75)
    print(
        "READING WRF GRID:",
        filename
    )
    print("=" * 75)

    nc = Dataset(
        filename,
        "r"
    )

    lat = np.asarray(
        nc.variables["XLAT"][0, :, :],
        dtype=float
    )

    lon = np.asarray(
        nc.variables["XLONG"][0, :, :],
        dtype=float
    )

    e_we = (
        len(nc.dimensions["west_east"])
    )

    e_sn = (
        len(nc.dimensions["south_north"])
    )

    nc.close()

    print()
    print("Grid:")
    print(
        "  south_north =",
        e_sn
    )
    print(
        "  west_east   =",
        e_we
    )

    print()
    print("Latitude:")
    print(
        "  min =",
        np.nanmin(lat)
    )
    print(
        "  max =",
        np.nanmax(lat)
    )

    print()
    print("Longitude:")
    print(
        "  min =",
        np.nanmin(lon)
    )
    print(
        "  max =",
        np.nanmax(lon)
    )

    print()
    print("Grid centre:")
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
# FUNCTION: REGRID EDGAR TO WRF
# ====================================================================

def regrid_to_wrf(
    edgar_field,
    wrf_lat,
    wrf_lon
):

    print()
    print("Regridding EDGAR -> WRF")

    # ---------------------------------------------------------------
    # EDGAR values in kg m-2 s-1
    # ---------------------------------------------------------------

    interpolator = (
        RegularGridInterpolator(
            (
                edgar_lat,
                edgar_lon
            ),
            edgar_field,
            method="linear",
            bounds_error=False,
            fill_value=0.0
        )
    )

    points = np.column_stack(
        (
            wrf_lat.ravel(),
            wrf_lon.ravel()
        )
    )

    result = interpolator(
        points
    )

    result = result.reshape(
        wrf_lat.shape
    )

    result[
        ~np.isfinite(result)
    ] = 0.0

    result[
        result < 0.0
    ] = 0.0

    return result


# ====================================================================
# CREATE AGRICULTURE FIELD FOR EACH DOMAIN
# ====================================================================

print()
print("=" * 75)
print("CREATING WRF AGRICULTURE GRIDS")
print("=" * 75)

wrf_agri = {}

for dom in [1, 2, 3]:

    filename = WRFINPUT_FILES[
        dom
    ]

    wrf_lat, wrf_lon = (
        read_wrf_grid(
            filename
        )
    )

    wrf_agri[dom] = {}

    for month in range(1, 13):

        print()
        print(
            f"D{dom:02d} "
            f"month {month:02d}"
        )

        field = regrid_to_wrf(
            edgar_monthly[month],
            wrf_lat,
            wrf_lon
        )

        # -----------------------------------------------------------
        # Convert EDGAR units:
        #
        # kg m-2 s-1
        #
        # ->
        #
        # mol km-2 hr-1
        # -----------------------------------------------------------

        field_wrf = (
            field
            * EDGAR_TO_WRFCHEM
        )

        field_wrf[
            ~np.isfinite(field_wrf)
        ] = 0.0

        field_wrf[
            field_wrf < 0.0
        ] = 0.0

        wrf_agri[dom][month] = (
            field_wrf
        )

        print(
            "  WRF agriculture min =",
            np.min(field_wrf)
        )

        print(
            "  WRF agriculture max =",
            np.max(field_wrf)
        )

        print(
            "  WRF agriculture mean =",
            np.mean(field_wrf)
        )


# ====================================================================
# FUNCTION: CREATE ONE WRF-CHEM FILE
# ====================================================================

def create_wrfchemi_file(
    domain,
    date_string,
    wrf_lat,
    wrf_lon,
    agri_field,
    wrfinput_file
):

    filename = os.path.join(
        OUTPUT_DIR,
        f"wrfchemi_d{domain:02d}_{date_string}"
    )

    print()
    print(
        "Creating:",
        filename
    )

    # ---------------------------------------------------------------
    # Read WRF dimensions
    # ---------------------------------------------------------------

    wrf = Dataset(
        wrfinput_file,
        "r"
    )

    south_north = (
        len(
            wrf.dimensions[
                "south_north"
            ]
        )
    )

    west_east = (
        len(
            wrf.dimensions[
                "west_east"
            ]
        )
    )

    wrf.close()

    # ---------------------------------------------------------------
    # Create NetCDF
    # ---------------------------------------------------------------

    nc = Dataset(
        filename,
        "w",
        format="NETCDF4"
    )

    # ---------------------------------------------------------------
    # Dimensions
    # ---------------------------------------------------------------

    nc.createDimension(
        "Time",
        None
    )

    nc.createDimension(
        "emissions_zdim_stag",
        1
    )

    nc.createDimension(
        "south_north",
        south_north
    )

    nc.createDimension(
        "west_east",
        west_east
    )

    nc.createDimension(
        "DateStrLen",
        19
    )

    # ---------------------------------------------------------------
    # Times
    # ---------------------------------------------------------------

    times = nc.createVariable(
        "Times",
        "S1",
        (
            "Time",
            "DateStrLen"
        )
    )

    time_bytes = np.array(
        list(date_string),
        dtype="S1"
    )

    times[0, :] = (
        time_bytes
    )

    # ---------------------------------------------------------------
    # XLAT
    # ---------------------------------------------------------------

    xlat = nc.createVariable(
        "XLAT",
        "f4",
        (
            "Time",
            "south_north",
            "west_east"
        ),
        zlib=True
    )

    xlat.MemoryOrder = "XY "
    xlat.description = (
        "LATITUDE, SOUTH IS NEGATIVE"
    )
    xlat.units = "degree north"
    xlat.stagger = ""

    xlat[0, :, :] = (
        wrf_lat
    )

    # ---------------------------------------------------------------
    # XLONG
    # ---------------------------------------------------------------

    xlong = nc.createVariable(
        "XLONG",
        "f4",
        (
            "Time",
            "south_north",
            "west_east"
        ),
        zlib=True
    )

    xlong.MemoryOrder = "XY "
    xlong.description = (
        "LONGITUDE, WEST IS NEGATIVE"
    )
    xlong.units = "degree east"
    xlong.stagger = ""

    xlong[0, :, :] = (
        wrf_lon
    )

    # ---------------------------------------------------------------
    # E_CH4
    # ---------------------------------------------------------------

    e_ch4 = nc.createVariable(
        "E_CH4",
        "f4",
        (
            "Time",
            "emissions_zdim_stag",
            "south_north",
            "west_east"
        ),
        zlib=True,
        complevel=4,
        fill_value=0.0
    )

    e_ch4.MemoryOrder = "XYZ"
    e_ch4.description = "EMISSIONS"
    e_ch4.units = (
        "mol km^-2 hr^-1"
    )
    e_ch4.stagger = "Z"
    e_ch4.coordinates = "XLONG XLAT"
    e_ch4.FieldType = 104

    # Agriculture only

    e_ch4[0, 0, :, :] = (
        agri_field
    )

    # ---------------------------------------------------------------
    # GLOBAL ATTRIBUTES
    # ---------------------------------------------------------------

    nc.TITLE = (
        "Anthropogenic CH4 EMISSIONS - "
        "EDGAR Agriculture only"
    )

    nc.History = (
        "Created from EDGAR 2023 Agriculture CH4 "
        "for WRF-Chem agriculture-only simulation"
    )

    nc.Author = (
        "WRF-Chem EDGAR agriculture processor"
    )

    nc.close()

    print(
        "  Maximum E_CH4 =",
        np.max(agri_field),
        "mol km^-2 hr^-1"
    )

    print(
        "  Total grid sum =",
        np.sum(agri_field),
        "mol km^-2 hr^-1"
    )


# ====================================================================
# CREATE HOURLY FILES
# ====================================================================

print()
print("=" * 75)
print("CREATING HOURLY AGRICULTURE EMISSION FILES")
print("=" * 75)


# Cache WRF grids

wrf_grids = {}

for dom in [1, 2, 3]:

    wrf_grids[dom] = (
        read_wrf_grid(
            WRFINPUT_FILES[dom]
        )
    )


# ====================================================================
# GENERATE 2026-01-01 TO 2026-02-28
# ====================================================================

import datetime


current = datetime.datetime(
    START_YEAR,
    START_MONTH,
    START_DAY,
    0
)

end_datetime = (
    datetime.datetime(
        END_YEAR,
        END_MONTH,
        END_DAY,
        23
    )
)


count = 0


while current <= end_datetime:

    year = current.year
    month = current.month
    day = current.day
    hour = current.hour

    date_string = (
        current.strftime(
            "%Y-%m-%d_%H:%M:%S"
        )
    )

    print()
    print(
        "=" * 75
    )

    print(
        "Processing:",
        date_string
    )

    print(
        "=" * 75
    )

    for dom in [1, 2, 3]:

        wrf_lat, wrf_lon = (
            wrf_grids[dom]
        )

        agri_field = (
            wrf_agri[dom][month]
        )

        create_wrfchemi_file(
            domain=dom,
            date_string=date_string,
            wrf_lat=wrf_lat,
            wrf_lon=wrf_lon,
            agri_field=agri_field,
            wrfinput_file=WRFINPUT_FILES[dom]
        )

    count += 1

    current += datetime.timedelta(
        hours=1
    )


# ====================================================================
# FINISHED
# ====================================================================

print()
print("=" * 75)
print("SUCCESS")
print("=" * 75)

print()
print(
    "Agriculture WRF-Chem files created:",
    count
)

print()
print(
    "Output directory:"
)

print(
    " ",
    os.path.abspath(
        OUTPUT_DIR
    )
)

print()
print(
    "These files contain ONLY:"
)

print(
    "  EDGAR Agriculture CH4"
)

print()
print(
    "They do NOT contain:"
)

print(
    "  OC1-OC13 mine emissions"
)

print()
print("=" * 75)

