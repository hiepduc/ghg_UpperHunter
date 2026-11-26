#!/usr/bin/env python3

"""
======================================================================
CREATE WRF-CHEM CH4 EMISSIONS FROM EDGAR POWER INDUSTRY
======================================================================

Input:
    EDGAR_2024_GHG_CH4_2023_bkl_POWER_INDUSTRY_flx.nc

EDGAR variable:
    fluxes(time, lat, lon)

EDGAR units:
    kg m-2 s-1

Output:
    wrfchemi_d01_YYYY-MM-DD_HH:00:00
    wrfchemi_d02_YYYY-MM-DD_HH:00:00
    wrfchemi_d03_YYYY-MM-DD_HH:00:00

WRF-Chem E_CH4 units:
    mol km^-2 hr^-1

IMPORTANT:
    This is POWER INDUSTRY CH4 ONLY.

    It does NOT include:
        - Upper Hunter open-cut mines
        - Agriculture
        - Other EDGAR sectors

EDGAR year:
    2023

WRF simulation:
    2026-01-01 00:00
    to
    2026-02-28 23:00

The 2023 EDGAR January field is repeated for
January 2026, February for February 2026, etc.

======================================================================
"""

import os
import numpy as np
import xarray as xr

from netCDF4 import Dataset
from datetime import datetime, timedelta


# =====================================================================
# USER SETTINGS
# =====================================================================

EDGAR_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/data/edgar2024/"
    "EDGAR_2024_GHG_CH4_2023_bkl_POWER_INDUSTRY_flx.nc"
)

# Actual WRF input files
WRF_INPUT = {
    1: "wrfinput_d01",
    2: "wrfinput_d02",
    3: "wrfinput_d03",
}

# -------------------------------------------------------------
# OUTPUT DIRECTORY
# -------------------------------------------------------------

OUTPUT_DIR = "power"

# -------------------------------------------------------------
# SIMULATION PERIOD
# -------------------------------------------------------------

START_DATE = datetime(
    2026, 1, 1, 0, 0
)

END_DATE = datetime(
    2026, 2, 28, 23, 0
)

# -------------------------------------------------------------
# CH4 MOLECULAR WEIGHT
# -------------------------------------------------------------

CH4_MW = 16.04       # g mol-1


# =====================================================================
# UNIT CONVERSION
# =====================================================================

"""
EDGAR:

    kg m-2 s-1

Convert:

    kg m-2 s-1
        |
        | x 1,000,000
        v
    kg km-2 s-1
        |
        | x 3600
        v
    kg km-2 hr-1
        |
        | / 0.01604 kg mol-1
        v
    mol km-2 hr-1
"""

KG_M2_S_TO_MOL_KM2_HR = (
    1.0e6 *
    3600.0 /
    (CH4_MW / 1000.0)
)


# =====================================================================
# PRINT HEADER
# =====================================================================

print()
print("=" * 78)
print("EDGAR POWER INDUSTRY CH4 -> WRF-Chem")
print("=" * 78)

print()
print("EDGAR file:")
print(" ", EDGAR_FILE)

print()
print("Output directory:")
print(" ", OUTPUT_DIR)

print()
print("Conversion factor:")
print(
    "  kg m-2 s-1 -> mol km-2 hr-1 =",
    KG_M2_S_TO_MOL_KM2_HR
)

print()
print("Simulation:")
print(
    " ",
    START_DATE,
    "to",
    END_DATE
)


# =====================================================================
# CHECK FILES
# =====================================================================

if not os.path.exists(EDGAR_FILE):

    raise FileNotFoundError(
        f"\nEDGAR file not found:\n{EDGAR_FILE}"
    )


for domain, filename in WRF_INPUT.items():

    if not os.path.exists(filename):

        raise FileNotFoundError(
            f"\nWRF input file for d0{domain} "
            f"not found:\n{filename}"
        )


# =====================================================================
# CREATE OUTPUT DIRECTORY
# =====================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print()
print(
    "Output directory ready:",
    OUTPUT_DIR
)


# =====================================================================
# READ EDGAR
# =====================================================================

print()
print("=" * 78)
print("READING EDGAR")
print("=" * 78)

ds = xr.open_dataset(
    EDGAR_FILE
)

print()
print(ds)


# =====================================================================
# EDGAR COORDINATES
# =====================================================================

lat_edgar = ds["lat"].values
lon_edgar = ds["lon"].values

flux_edgar = ds["fluxes"]


print()
print("EDGAR dimensions:")

print(
    "  time =",
    len(ds.time)
)

print(
    "  lat  =",
    len(lat_edgar)
)

print(
    "  lon  =",
    len(lon_edgar)
)


print()
print("EDGAR latitude:")

print(
    " ",
    lat_edgar.min(),
    "to",
    lat_edgar.max()
)


print()
print("EDGAR longitude:")

print(
    " ",
    lon_edgar.min(),
    "to",
    lon_edgar.max()
)


print()
print("EDGAR flux:")

print(
    "  min =",
    float(
        flux_edgar.min(
            skipna=True
        )
    )
)

print(
    "  max =",
    float(
        flux_edgar.max(
            skipna=True
        )
    )
)


# =====================================================================
# ENSURE LATITUDE ASCENDING
# =====================================================================

if lat_edgar[0] > lat_edgar[-1]:

    print()
    print(
        "Sorting EDGAR latitude..."
    )

    ds = ds.sortby("lat")

    lat_edgar = ds["lat"].values
    lon_edgar = ds["lon"].values
    flux_edgar = ds["fluxes"]


# =====================================================================
# ENSURE LONGITUDE ASCENDING
# =====================================================================

if lon_edgar[0] > lon_edgar[-1]:

    print()
    print(
        "Sorting EDGAR longitude..."
    )

    ds = ds.sortby("lon")

    lat_edgar = ds["lat"].values
    lon_edgar = ds["lon"].values
    flux_edgar = ds["fluxes"]


# =====================================================================
# READ WRF GRID
# =====================================================================

def read_wrf_grid(
    filename
):

    nc = Dataset(
        filename,
        "r"
    )

    lat = np.asarray(
        nc.variables["XLAT"][0, :, :],
        dtype=np.float64
    )

    lon = np.asarray(
        nc.variables["XLONG"][0, :, :],
        dtype=np.float64
    )

    nc.close()

    return lat, lon


# =====================================================================
# INTERPOLATE EDGAR TO WRF GRID
# =====================================================================

def interpolate_edgar_to_wrf(
    edgar_field,
    wrf_lat,
    wrf_lon
):

    """
    Interpolate EDGAR field onto actual WRF grid.

    EDGAR:
        lat(lat)
        lon(lon)

    WRF:
        2-D XLAT
        2-D XLONG
    """

    result = edgar_field.interp(
        lat=xr.DataArray(
            wrf_lat,
            dims=(
                "south_north",
                "west_east"
            )
        ),
        lon=xr.DataArray(
            wrf_lon,
            dims=(
                "south_north",
                "west_east"
            )
        ),
        method="linear"
    )

    result = result.values

    result = np.asarray(
        result,
        dtype=np.float64
    )

    # -------------------------------------------------------------
    # NaN -> zero
    # -------------------------------------------------------------

    result[
        ~np.isfinite(result)
    ] = 0.0

    # -------------------------------------------------------------
    # Negative emissions -> zero
    # -------------------------------------------------------------

    result[
        result < 0.0
    ] = 0.0

    return result


# =====================================================================
# CREATE WRF-CHEM FILE
# =====================================================================

def create_wrfchemi(
    output_file,
    date_time,
    lat,
    lon,
    emission
):

    """
    Create one WRF-Chem hourly emission file.
    """

    ny, nx = lat.shape

    print()
    print(
        "Creating:"
    )

    print(
        " ",
        output_file
    )

    nc = Dataset(
        output_file,
        "w",
        format="NETCDF4"
    )

    # ================================================================
    # DIMENSIONS
    # ================================================================

    nc.createDimension(
        "Time",
        1
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

    # ================================================================
    # TIMES
    # ================================================================

    times_var = nc.createVariable(
        "Times",
        "S1",
        (
            "Time",
            "DateStrLen"
        )
    )

    times_string = (
        date_time.strftime(
            "%Y-%m-%d_%H:%M:%S"
        )
    )

    times_var[0, :] = np.array(
        list(times_string),
        dtype="S1"
    )

    # ================================================================
    # LATITUDE
    # ================================================================

    lat_var = nc.createVariable(
        "XLAT",
        "f4",
        (
            "Time",
            "south_north",
            "west_east"
        )
    )

    lat_var[0, :, :] = lat

    lat_var.MemoryOrder = "XY "
    lat_var.description = (
        "LATITUDE, SOUTH IS NEGATIVE"
    )
    lat_var.units = "degree north"
    lat_var.stagger = ""
    lat_var.FieldType = 104

    # ================================================================
    # LONGITUDE
    # ================================================================

    lon_var = nc.createVariable(
        "XLONG",
        "f4",
        (
            "Time",
            "south_north",
            "west_east"
        )
    )

    lon_var[0, :, :] = lon

    lon_var.MemoryOrder = "XY "
    lon_var.description = (
        "LONGITUDE, WEST IS NEGATIVE"
    )
    lon_var.units = "degree east"
    lon_var.stagger = ""
    lon_var.FieldType = 104

    # ================================================================
    # E_CH4
    # ================================================================

    evar = nc.createVariable(
        "E_CH4",
        "f4",
        (
            "Time",
            "emissions_zdim_stag",
            "south_north",
            "west_east"
        ),
        zlib=True
    )

    evar.MemoryOrder = "XYZ"

    evar.description = (
        "EDGAR Power Industry CH4 EMISSIONS"
    )

    evar.units = (
        "mol km^-2 hr^-1"
    )

    evar.stagger = "Z"

    evar.coordinates = "XLONG XLAT"

    evar.FieldType = 104

    evar[0, 0, :, :] = (
        emission.astype(
            np.float32
        )
    )

    # ================================================================
    # GLOBAL ATTRIBUTES
    # ================================================================

    nc.TITLE = (
        "EDGAR Power Industry CH4 EMISSIONS"
    )

    nc.History = (
        "Created from EDGAR 2024 GHG "
        "Power Industry CH4 dataset"
    )

    nc.Author = (
        "WRF-Chem CH4 emission converter"
    )

    nc.close()


# =====================================================================
# PROCESS DOMAINS
# =====================================================================

for domain in [1, 2, 3]:

    print()
    print("=" * 78)
    print(
        f"PROCESSING DOMAIN d0{domain}"
    )
    print("=" * 78)

    wrf_file = WRF_INPUT[domain]

    # ---------------------------------------------------------------
    # READ ACTUAL WRF GRID
    # ---------------------------------------------------------------

    wrf_lat, wrf_lon = read_wrf_grid(
        wrf_file
    )

    ny, nx = wrf_lat.shape

    print()
    print("WRF grid:")
    print(
        "  shape =",
        ny,
        nx
    )

    print(
        "  lat =",
        np.nanmin(wrf_lat),
        "to",
        np.nanmax(wrf_lat)
    )

    print(
        "  lon =",
        np.nanmin(wrf_lon),
        "to",
        np.nanmax(wrf_lon)
    )

    # ---------------------------------------------------------------
    # LOOP THROUGH HOURS
    # ---------------------------------------------------------------

    current = START_DATE

    while current <= END_DATE:

        print()
        print(
            "Processing:",
            current.strftime(
                "%Y-%m-%d %H:%M"
            )
        )

        # ============================================================
        # EDGAR MONTH
        # ============================================================

        month_index = (
            current.month - 1
        )

        edgar_month = (
            flux_edgar.isel(
                time=month_index
            )
        )

        # ============================================================
        # UNIT CONVERSION
        # ============================================================

        edgar_mol = (
            edgar_month *
            KG_M2_S_TO_MOL_KM2_HR
        )

        # ============================================================
        # INTERPOLATE TO WRF
        # ============================================================

        emission = (
            interpolate_edgar_to_wrf(
                edgar_mol,
                wrf_lat,
                wrf_lon
            )
        )

        # ============================================================
        # DIAGNOSTICS
        # ============================================================

        positive = emission[
            emission > 0
        ]

        print()
        print(
            "E_CH4 diagnostics:"
        )

        print(
            "  min =",
            np.min(emission)
        )

        print(
            "  max =",
            np.max(emission)
        )

        print(
            "  sum =",
            np.sum(emission)
        )

        print(
            "  non-zero cells =",
            positive.size
        )

        # ============================================================
        # OUTPUT FILE
        # ============================================================

        filename = (
            "wrfchemi_d0"
            f"{domain}_"
            f"{current.strftime('%Y-%m-%d_%H:%M:%S')}"
        )

        output_file = os.path.join(
            OUTPUT_DIR,
            filename
        )

        # ============================================================
        # WRITE
        # ============================================================

        create_wrfchemi(
            output_file,
            current,
            wrf_lat,
            wrf_lon,
            emission
        )

        current += timedelta(
            hours=1
        )


# =====================================================================
# CLOSE EDGAR
# =====================================================================

ds.close()


# =====================================================================
# FINISHED
# =====================================================================

print()
print("=" * 78)
print("POWER INDUSTRY PROCESSING COMPLETE")
print("=" * 78)

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
    "Files created:"
)

print(
    "  wrfchemi_d01_*"
)

print(
    "  wrfchemi_d02_*"
)

print(
    "  wrfchemi_d03_*"
)

print()
print(
    "Sector:"
)

print(
    "  EDGAR Power Industry CH4 ONLY"
)

print()
print("=" * 78)

