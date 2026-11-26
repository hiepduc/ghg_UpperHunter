#!/usr/bin/env python3
"""
===============================================================================
Build FLEXPART-WRF 13-mine daily H matrix
Upper Hunter CH4 inverse modelling

FLEXPART-WRF CONC:

    CONC(Time, ageclass, releases, bottom_top, south_north, west_east)

CONC units:

    s m3 kg-1

Vertical layers:

    bottom_top = 0 : 0-10 m
    bottom_top = 1 : 10-100 m
    bottom_top = 2 : 100-20000 m

This script uses ONLY:

    bottom_top = 0

i.e. the 0-10 m FLEXPART layer.

===============================================================================
H MATRIX
===============================================================================

Rows:

    FLEXPART receptor/time cases

Columns:

    13 mines x 31 daily scaling factors

    13 x 30 = 403 columns

Dates:

    2026-07-01 through 2026-07-31

Daily factor definition:

    E_posterior(m,d) = f(m,d) * E_prior(m)

Therefore H represents the predicted CH4 enhancement [ppmv]
for a unit daily scaling factor:

    f(m,d) = 1

===============================================================================
FLEXPART FOOTPRINT UNIT CONVERSION
===============================================================================

FLEXPART CONC:

    [s m3 kg-1]

Mine emission:

    [kg CH4 s-1]

Therefore:

    CONC * emission
        = m3

For the 0-10 m layer:

    volume = GRIDAREA * 10 m

where GRIDAREA comes directly from:

    header_d01.nc

GRIDAREA units:

    m2

Therefore:

    CONC * emission / volume
        = kg CH4 / kg air

This is a mass mixing ratio.

Convert to volume mixing ratio:

    ppmv =
        kg/kg
        * M_AIR / M_CH4
        * 1e6

Thus:

    H =
        sum[
            CONC * emission / (GRIDAREA * 10)
        ]
        * M_AIR / M_CH4
        * 1e6

The final H units are:

    ppmv per unit daily scaling factor

Since the source vector contains dimensionless daily scaling factors,
H itself has units of ppmv.

===============================================================================
GRID AREA
===============================================================================

GRIDAREA is read directly from:

    header_d01.nc

and is NOT approximated using DX * DY.

Each grid cell can therefore have its actual FLEXPART surface area:

    GRIDAREA[iy,ix] [m2]

The 0-10 m volume is:

    GRIDAREA[iy,ix] * 10 [m3]

===============================================================================
TIME MAPPING
===============================================================================

FLEXPART Times are:

    TIME OF OUTPUT (END OF AVERAGING INTERVAL)

For each hourly output:

    source_time_UTC = end_time_UTC - 1 hour

Then:

    source_time_AEST = source_time_UTC + 10 hours

Only source dates:

    2026-07-01 through 2026-07-31

are retained.

===============================================================================
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from shapely.geometry import box
from shapely.prepared import prep

from pyproj import CRS, Transformer


# =============================================================================
# PATHS
# =============================================================================

BASE = Path(
    "/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf/upperhunter_flexpart"
)

FLEX_ROOT = (
    BASE /
    "run" / 
    "2026-07" /
    "hourly_obs_3layer"
)

WRFOUT_DIR = (
    BASE /
    "WRFOUT" /
    "2026-07" 
)

MINE_DIR = Path(
    "/mnt/scratch_lustre/nzem_meim_scratch/data/emission_location"
)

PRIOR_FILE = (
    MINE_DIR /
    "UpperHuntercoalmines.csv"
)

OUT_DIR = (
    BASE /
    "H_matrix_flexpart" /
    "2026-07"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# FLEXPART HEADER
# =============================================================================

# The FLEXPART output is flxout_d01_*.nc, therefore use the
# matching header_d01.nc.

HEADER_FILE = (
    FLEX_ROOT /
    "muswellbrook" /
    "20260714_180000" /
    "header_d01.nc"
)


# =============================================================================
# RECEPTORS
# =============================================================================

RECEPTORS = {

    "singleton": {
        "lat": -32.57123,
        "lon": 151.16132,
    },

    "muswellbrook": {
        "lat": -32.26202,
        "lon": 150.89297,
    },

    "merriwa": {
        "lat": -32.14097,
        "lon": 150.35767,
    },

}


# =============================================================================
# MINE DEFINITIONS
# =============================================================================

MINE_NAMES = [
    f"OC{i}"
    for i in range(1, 14)
]

SHAPEFILES = {

    f"OC{i}":
    MINE_DIR / f"OC{i}.shp"

    for i in range(1, 14)

}


# =============================================================================
# DATES
# =============================================================================

START_DATE = pd.Timestamp(
    "2026-07-01"
)

END_DATE = pd.Timestamp(
    "2026-07-31"
)

DATES = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)

N_DAYS = len(DATES)

assert N_DAYS == 31


# =============================================================================
# H COLUMNS
# =============================================================================

H_COLUMNS = [

    f"{mine}_{date:%Y%m%d}"

    for date in DATES

    for mine in MINE_NAMES

]

assert len(H_COLUMNS) == 403


# =============================================================================
# FLEXPART FILES
# =============================================================================

FLEX_FILES = sorted(
    FLEX_ROOT.glob(
        "*/*/flxout_d01_*.nc"
    )
)


print("=" * 90)
print("FLEXPART-WRF 13-MINE H MATRIX")
print("=" * 90)

print(
    f"FLEXPART files : {len(FLEX_FILES)}"
)

print(
    f"Mine columns   : {len(MINE_NAMES)}"
)

print(
    f"Days           : {N_DAYS}"
)

print(
    f"H columns      : {len(H_COLUMNS)}"
)


if not FLEX_FILES:

    raise FileNotFoundError(
        f"No FLEXPART files found in {FLEX_ROOT}"
    )


if len(FLEX_FILES) != 1730:

    print(
        f"WARNING: expected 1730 FLEXPART files, "
        f"found {len(FLEX_FILES)}"
    )


# =============================================================================
# READ PRIOR MINE EMISSIONS
# =============================================================================

print(
    "\nReading prior mine emissions:"
)

print(
    PRIOR_FILE
)

prior = pd.read_csv(
    PRIOR_FILE
)

print(
    "\nPrior columns:"
)

print(
    prior.columns.tolist()
)


if "Facility_code_name" not in prior.columns:

    raise ValueError(
        "Prior file does not contain Facility_code_name"
    )


if "gCH4/s" not in prior.columns:

    raise ValueError(
        "Prior file does not contain gCH4/s"
    )


prior["Facility_code_name"] = (

    prior["Facility_code_name"]
    .astype(str)
    .str.strip()

)


prior_oc = prior[
    prior["Facility_code_name"].isin(
        MINE_NAMES
    )
].copy()


if len(prior_oc) != 13:

    raise ValueError(
        "Expected exactly 13 open-cut mines "
        f"OC1-OC13, found {len(prior_oc)}"
    )


prior_gps = dict(

    zip(

        prior_oc[
            "Facility_code_name"
        ],

        prior_oc[
            "gCH4/s"
        ].astype(float)

    )

)


missing_mines = [

    mine

    for mine in MINE_NAMES

    if mine not in prior_gps

]


if missing_mines:

    raise ValueError(
        f"Missing prior emissions for: "
        f"{missing_mines}"
    )


print(
    "\nCorrect OC1-OC13 prior emissions:"
)


for mine in MINE_NAMES:

    print(

        f"{mine:5s} "
        f"{prior_gps[mine]:12.6f} "
        f"g CH4/s"

    )


total_prior = sum(
    prior_gps.values()
)


print(
    f"\nTotal OC1-OC13 prior = "
    f"{total_prior:.6f} g CH4/s"
)


if not np.isclose(
    total_prior,
    2553.86,
    rtol=1e-5
):

    raise ValueError(
        f"Unexpected OC1-OC13 total: "
        f"{total_prior}"
    )


# =============================================================================
# READ FLEXPART HEADER
# =============================================================================

print(
    "\nReading FLEXPART grid header:"
)

print(
    HEADER_FILE
)


if not HEADER_FILE.exists():

    raise FileNotFoundError(
        f"FLEXPART header not found: "
        f"{HEADER_FILE}"
    )


with xr.open_dataset(
    HEADER_FILE,
    decode_times=False
) as header:

    # ---------------------------------------------------------
    # Grid dimensions
    # ---------------------------------------------------------

    fp_nx = header.sizes[
        "west_east"
    ]

    fp_ny = header.sizes[
        "south_north"
    ]


    # ---------------------------------------------------------
    # Coordinates
    # ---------------------------------------------------------

    XLONG_FP = np.asarray(
        header["XLONG"]
    )

    XLAT_FP = np.asarray(
        header["XLAT"]
    )


    # ---------------------------------------------------------
    # ACTUAL FLEXPART GRID AREA
    # ---------------------------------------------------------

    if "GRIDAREA" not in header:

        raise ValueError(
            "GRIDAREA is not present in "
            f"{HEADER_FILE}"
        )


    GRIDAREA = np.asarray(
        header["GRIDAREA"],
        dtype=np.float64
    )


    gridarea_units = (
        header["GRIDAREA"]
        .attrs
        .get("units", "")
        .strip()
    )


    # ---------------------------------------------------------
    # Projection
    # ---------------------------------------------------------

    cen_lat = float(
        header.attrs["CEN_LAT"]
    )

    cen_lon = float(
        header.attrs["CEN_LON"]
    )

    stand_lon = float(
        header.attrs["STAND_LON"]
    )

    truelat1 = float(
        header.attrs["TRUELAT1"]
    )

    truelat2 = float(
        header.attrs["TRUELAT2"]
    )

    dx = float(
        header.attrs["DX"]
    )

    dy = float(
        header.attrs["DY"]
    )


print(
    "\nFLEXPART header grid:"
)

print(
    f"  nx = {fp_nx}"
)

print(
    f"  ny = {fp_ny}"
)

print(
    f"  XLONG shape = {XLONG_FP.shape}"
)

print(
    f"  XLAT shape  = {XLAT_FP.shape}"
)

print(
    f"  GRIDAREA shape = {GRIDAREA.shape}"
)

print(
    f"  GRIDAREA units = {gridarea_units}"
)


# =============================================================================
# VERIFY GRID AREA
# =============================================================================

if gridarea_units != "m2":

    raise ValueError(
        "GRIDAREA has unexpected units: "
        f"'{gridarea_units}'. "
        "Expected 'm2'."
    )


if GRIDAREA.shape != (
    fp_ny,
    fp_nx
):

    raise ValueError(
        "GRIDAREA shape does not match "
        "FLEXPART grid: "
        f"{GRIDAREA.shape} vs "
        f"{(fp_ny, fp_nx)}"
    )


if not np.all(
    np.isfinite(GRIDAREA)
):

    raise ValueError(
        "GRIDAREA contains non-finite values."
    )


if np.any(
    GRIDAREA <= 0
):

    raise ValueError(
        "GRIDAREA contains zero or negative "
        "grid-cell areas."
    )


print(
    "\nActual FLEXPART grid-cell area:"
)

print(
    f"  Minimum = "
    f"{np.min(GRIDAREA):.6e} m2"
)

print(
    f"  Maximum = "
    f"{np.max(GRIDAREA):.6e} m2"
)

print(
    f"  Mean    = "
    f"{np.mean(GRIDAREA):.6e} m2"
)

print(
    f"  Median  = "
    f"{np.median(GRIDAREA):.6e} m2"
)


# =============================================================================
# 0-10 m LAYER VOLUME
# =============================================================================

LAYER_HEIGHT = 10.0

CELL_VOLUME = (
    GRIDAREA *
    LAYER_HEIGHT
)


print(
    "\n0-10 m layer:"
)

print(
    f"  Height = "
    f"{LAYER_HEIGHT:.3f} m"
)

print(
    "  Cell volume range:"
)

print(
    f"    Minimum = "
    f"{np.min(CELL_VOLUME):.6e} m3"
)

print(
    f"    Maximum = "
    f"{np.max(CELL_VOLUME):.6e} m3"
)

print(
    f"    Mean    = "
    f"{np.mean(CELL_VOLUME):.6e} m3"
)


# =============================================================================
# VERIFY AGAINST FLEXPART OUTPUT
# =============================================================================

fp_test = FLEX_FILES[0]

print(
    "\nChecking FLEXPART output grid:"
)

print(
    fp_test
)


with xr.open_dataset(
    fp_test,
    decode_times=False
) as ds:

    out_nx = ds.sizes[
        "west_east"
    ]

    out_ny = ds.sizes[
        "south_north"
    ]

    conc_units = (
        ds["CONC"]
        .attrs
        .get("units", "")
        .strip()
    )


if (
    out_nx != fp_nx
    or
    out_ny != fp_ny
):

    raise ValueError(
        "FLEXPART output/header grid mismatch: "
        f"output={(out_ny, out_nx)}, "
        f"header={(fp_ny, fp_nx)}"
    )


if conc_units != "s m3 kg-1":

    raise ValueError(
        "Unexpected CONC units: "
        f"'{conc_units}'. "
        "Expected 's m3 kg-1'."
    )


print(
    "FLEXPART output/header grid match confirmed."
)

print(
    f"CONC units = {conc_units}"
)


# =============================================================================
# READ WRF GRID
# =============================================================================

print(
    "\nSearching for WRF d03 file..."
)


wrf_files = sorted(
    WRFOUT_DIR.glob(
        "wrfout_met_d03_*"
    )
)


if not wrf_files:

    raise FileNotFoundError(
        f"No wrfout_met_d03_* found in "
        f"{WRFOUT_DIR}"
    )


wrf_file = wrf_files[0]


print(
    "Using WRF file:"
)

print(
    wrf_file
)


with xr.open_dataset(
    wrf_file,
    decode_times=False
) as wrf:

    XLONG_WRF = np.asarray(
        wrf["XLONG"].isel(Time=0)
    )

    XLAT_WRF = np.asarray(
        wrf["XLAT"].isel(Time=0)
    )


print(
    "\nWRF grid:"
)

print(
    "XLONG shape:",
    XLONG_WRF.shape
)

print(
    "XLAT shape :",
    XLAT_WRF.shape
)


# =============================================================================
# IMPORTANT GRID CHECK
# =============================================================================

if XLONG_WRF.shape != (
    fp_ny,
    fp_nx
):

    raise ValueError(
        "WRF d03/FLEXPART grid mismatch: "
        f"WRF={XLONG_WRF.shape}, "
        f"FLEXPART={(fp_ny, fp_nx)}"
    )


print(
    "\nWRF/FLEXPART grid dimensions "
    "match."
)


# =============================================================================
# CREATE FLEXPART PROJECTION
# =============================================================================

print(
    "\nCreating FLEXPART projection..."
)


fp_crs = CRS.from_proj4(

    f"+proj=lcc "
    f"+lat_1={truelat1} "
    f"+lat_2={truelat2} "
    f"+lat_0={cen_lat} "
    f"+lon_0={stand_lon} "
    f"+a=6370000 "
    f"+b=6370000"

)


to_xy = Transformer.from_crs(

    "EPSG:4326",

    fp_crs,

    always_xy=True

)


# =============================================================================
# GRID CENTRES
# =============================================================================

xgrid, ygrid = to_xy.transform(

    XLONG_FP,
    XLAT_FP

)


dx_grid = np.nanmedian(

    np.diff(
        xgrid,
        axis=1
    )

)


dy_grid = np.nanmedian(

    np.diff(
        ygrid,
        axis=0
    )

)


print(
    f"Projected grid spacing: "
    f"dx={dx_grid:.3f} m, "
    f"dy={dy_grid:.3f} m"
)


if not np.isclose(
    abs(dx_grid),
    abs(dx),
    atol=2
):

    warnings.warn(
        f"Unexpected X grid spacing: "
        f"{dx_grid:.3f} m"
    )


if not np.isclose(
    abs(dy_grid),
    abs(dy),
    atol=2
):

    warnings.warn(
        f"Unexpected Y grid spacing: "
        f"{dy_grid:.3f} m"
    )


# =============================================================================
# CREATE GRID POLYGONS
# =============================================================================

grid_polygons = []


half_dx = abs(dx) / 2.0

half_dy = abs(dy) / 2.0


for iy in range(fp_ny):

    for ix in range(fp_nx):

        xc = xgrid[
            iy,
            ix
        ]

        yc = ygrid[
            iy,
            ix
        ]

        cell = box(

            xc - half_dx,
            yc - half_dy,

            xc + half_dx,
            yc + half_dy

        )


        grid_polygons.append(

            (
                iy,
                ix,
                cell
            )

        )


print(
    f"Created "
    f"{len(grid_polygons):,} grid cells"
)


# =============================================================================
# READ MINE POLYGONS
# =============================================================================

print(
    "\nReading mine polygons..."
)


mine_gdfs = {}


for mine in MINE_NAMES:

    shp = SHAPEFILES[mine]


    if not shp.exists():

        raise FileNotFoundError(
            shp
        )


    gdf = gpd.read_file(
        shp
    )


    if gdf.empty:

        raise ValueError(
            f"{mine} shapefile is empty: "
            f"{shp}"
        )


    if gdf.crs is None:

        raise ValueError(
            f"{mine} shapefile has no CRS: "
            f"{shp}"
        )


    gdf = gdf.to_crs(
        fp_crs
    )


    geom = gdf.geometry.unary_union


    mine_gdfs[mine] = geom


    print(

        f"{mine:5s}: "
        f"area = "
        f"{geom.area / 1e6:.4f} km2"

    )


# =============================================================================
# CALCULATE MINE/GRID OVERLAP WEIGHTS
# =============================================================================

print(
    "\nCalculating mine/grid overlap weights..."
)


mine_weights = {}


for mine in MINE_NAMES:

    geom = mine_gdfs[mine]

    prepared = prep(
        geom
    )

    entries = []

    total_intersection_area = 0.0


    for iy, ix, cell in grid_polygons:

        if not prepared.intersects(
            cell
        ):

            continue


        inter = geom.intersection(
            cell
        )


        if inter.is_empty:

            continue


        area = inter.area


        if area <= 0:

            continue


        entries.append(

            (
                iy,
                ix,
                area
            )

        )


        total_intersection_area += area


    if total_intersection_area <= 0:

        raise ValueError(
            f"No grid overlap found "
            f"for {mine}"
        )


    entries_norm = [

        (
            iy,
            ix,
            area /
            total_intersection_area
        )

        for iy, ix, area
        in entries

    ]


    mine_weights[mine] = (
        entries_norm
    )


    print(

        f"{mine:5s}: "
        f"{len(entries_norm):5d} "
        f"grid cells"

    )


# =============================================================================
# CREATE PRIOR EMISSION FIELDS
# =============================================================================

print(
    "\nCreating mine emission fields..."
)

print(
    "Units: kg CH4/s per grid cell"
)


prior_field = {}


for mine in MINE_NAMES:

    field = np.zeros(

        (
            fp_ny,
            fp_nx
        ),

        dtype=np.float64

    )


    total_kg_s = (

        prior_gps[mine]
        /
        1000.0

    )


    for iy, ix, frac in (
        mine_weights[mine]
    ):

        field[
            iy,
            ix
        ] = (

            total_kg_s *
            frac

        )


    prior_field[mine] = (
        field
    )


    print(

        f"{mine:5s}: "
        f"{field.sum():.8f} kg/s"

    )


# =============================================================================
# CASE INFORMATION
# =============================================================================

CASE_RE = re.compile(

    r"/(singleton|muswellbrook|merriwa)/"
    r"(\d{8}_\d{6})/"

)


def case_info(path):

    m = CASE_RE.search(
        str(path)
    )


    if not m:

        raise ValueError(
            f"Cannot identify receptor/time "
            f"from {path}"
        )


    receptor = m.group(1)


    timestamp = pd.to_datetime(

        m.group(2),

        format="%Y%m%d_%H%M%S"

    )


    return (
        receptor,
        timestamp
    )


# =============================================================================
# FLEXPART TIME PARSER
# =============================================================================

def parse_flex_times(times):

    result = []


    for t in times:

        if isinstance(
            t,
            bytes
        ):

            t = t.decode()


        t = (

            str(t)
            .replace("*", "")
            .strip()

        )


        result.append(

            pd.to_datetime(

                t,

                format="%Y%m%d_%H%M%S"

            )

        )


    return result


# =============================================================================
# MOLECULAR WEIGHT CONVERSION
# =============================================================================

M_AIR = 28.965

M_CH4 = 16.043


MASS_TO_VOLUME_FACTOR = (

    M_AIR /
    M_CH4
    *
    1.0e6

)


print(
    "\nCH4 conversion:"
)


print(
    f"M_AIR = {M_AIR} g/mol"
)


print(
    f"M_CH4 = {M_CH4} g/mol"
)


print(
    f"kg/kg -> ppmv factor = "
    f"{MASS_TO_VOLUME_FACTOR:.6e}"
)


# =============================================================================
# PROCESS ONE FLEXPART FILE
# =============================================================================

def process_file(
    ncfile
):

    receptor, release_utc = (
        case_info(
            ncfile
        )
    )


    with xr.open_dataset(

        ncfile,

        decode_times=False

    ) as ds:

        # ---------------------------------------------------------
        # Dimensions
        # ---------------------------------------------------------

        if ds.sizes[
            "bottom_top"
        ] != 3:

            raise ValueError(

                f"{ncfile}: "
                "expected 3 vertical levels"

            )


        if (

            ds.sizes[
                "south_north"
            ] != fp_ny

            or

            ds.sizes[
                "west_east"
            ] != fp_nx

        ):

            raise ValueError(

                f"{ncfile}: "
                "unexpected grid dimensions"

            )


        # ---------------------------------------------------------
        # CONC units
        # ---------------------------------------------------------

        units = (

            ds["CONC"]
            .attrs
            .get("units", "")
            .strip()

        )


        if units != "s m3 kg-1":

            raise ValueError(

                f"{ncfile}: "
                f"unexpected CONC units "
                f"'{units}'"

            )


        # ---------------------------------------------------------
        # ONLY 0-10 m
        # ---------------------------------------------------------

        conc = np.asarray(

            ds["CONC"].isel(

                ageclass=0,

                releases=0,

                bottom_top=0

            ),

            dtype=np.float64

        )


        # ---------------------------------------------------------
        # Times
        # ---------------------------------------------------------

        times = parse_flex_times(

            ds["Times"].values

        )


    # =============================================================================
    # H FOR THIS RECEPTOR/TIME CASE
    # =============================================================================

    h = np.zeros(

        len(H_COLUMNS),

        dtype=np.float64

    )


    # =============================================================================
    # HOURLY FOOTPRINTS
    # =============================================================================

    for it, end_time_utc in enumerate(
        times
    ):

        # FLEXPART Times is the END of the
        # 1-hour averaging interval.

        source_time_utc = (

            end_time_utc
            -
            pd.Timedelta(
                hours=1
            )

        )


        # UTC -> AEST

        source_time_aest = (

            source_time_utc
            +
            pd.Timedelta(
                hours=10
            )

        )


        source_date = (

            source_time_aest
            .normalize()

        )


        # ---------------------------------------------------------
        # Keep only July 1-July 31
        # ---------------------------------------------------------

        if (

            source_date <
            START_DATE

            or

            source_date >
            END_DATE

        ):

            continue


        day_index = int(

            (
                source_date
                -
                START_DATE
            ).days

        )


        # ---------------------------------------------------------
        # Footprint
        # ---------------------------------------------------------

        R = np.asarray(

            conc[it],

            dtype=np.float64

        )


        R = np.nan_to_num(

            R,

            nan=0.0,

            posinf=0.0,

            neginf=0.0

        )


        if not np.any(R):

            continue


        # =============================================================================
        # MINE LOOP
        # =============================================================================

        for im, mine in enumerate(
            MINE_NAMES
        ):

            emission_field = (
                prior_field[mine]
            )


            # ---------------------------------------------------------
            # PHYSICAL CONVERSION
            #
            # R:
            #     s m3 kg-1
            #
            # E:
            #     kg CH4 s-1
            #
            # R * E:
            #     m3
            #
            # GRIDAREA * 10:
            #     m3
            #
            # R * E / V:
            #     kg/kg
            #
            # Then:
            #
            # kg/kg * M_AIR/M_CH4 * 1e6
            #     = ppmv
            # ---------------------------------------------------------

            contribution_mass_mixing_ratio = np.sum(

                R
                *
                emission_field
                /
                CELL_VOLUME

            )


            contribution_ppmv = (

                contribution_mass_mixing_ratio
                *
                MASS_TO_VOLUME_FACTOR

            )


            # ---------------------------------------------------------
            # Mine/day column
            # ---------------------------------------------------------

            col = (

                day_index * 13
                +
                im

            )


            h[col] += (
                contribution_ppmv
            )


    return (
        receptor,
        release_utc,
        h
    )


# =============================================================================
# PROCESS ALL FLEXPART FILES
# =============================================================================

rows = []

metadata = []


print(
    "\nProcessing FLEXPART files..."
)


for n, ncfile in enumerate(

    FLEX_FILES,

    start=1

):

    try:

        (
            receptor,
            release_utc,
            h
        ) = process_file(
            ncfile
        )


        obs_time_aest = (

            release_utc
            +
            pd.Timedelta(
                hours=10
            )

        )


        rows.append(
            h
        )


        metadata.append(

            {

                "receptor":
                receptor,

                "release_utc":
                release_utc,

                "obs_time_aest":
                obs_time_aest,

                "flexpart_file":
                str(ncfile),

            }

        )


    except Exception as e:

        print(

            "\nERROR processing:"
            f"\n{ncfile}"
            f"\n{e}"

        )

        raise


    if n % 50 == 0:

        print(

            f"Processed "
            f"{n}/{len(FLEX_FILES)}"

        )


# =============================================================================
# BUILD MATRIX
# =============================================================================

H = np.vstack(
    rows
)


metadata_df = pd.DataFrame(
    metadata
)


H_df = pd.DataFrame(

    H,

    columns=H_COLUMNS

)


H_output = pd.concat(

    [

        metadata_df.reset_index(
            drop=True
        ),

        H_df.reset_index(
            drop=True
        ),

    ],

    axis=1

)


# =============================================================================
# OUTPUT FILES
# =============================================================================

H_file = (

    OUT_DIR
    /
    "H_flexpart_0_10m_13mine_Jul2026.csv"

)


META_file = (

    OUT_DIR
    /
    "H_flexpart_metadata_Jul2026.csv"

)


H_output.to_csv(

    H_file,

    index=False,

    float_format="%.10e"

)


metadata_df.to_csv(

    META_file,

    index=False

)


# =============================================================================
# DIAGNOSTICS
# =============================================================================

print(
    "\n" + "=" * 90
)

print(
    "H MATRIX COMPLETE"
)

print(
    "=" * 90
)


print(

    f"Shape: "
    f"{H.shape[0]} rows x "
    f"{H.shape[1]} columns"

)


print(
    "Expected: 1798 x 403"
)


print(

    f"Non-zero elements: "
    f"{np.count_nonzero(H):,}"

)


print(

    f"Minimum: "
    f"{np.nanmin(H):.6e} ppmv"

)


print(

    f"Maximum: "
    f"{np.nanmax(H):.6e} ppmv"

)


nonzero = H[
    H > 0
]


if len(nonzero) > 0:

    print(

        f"Median non-zero: "
        f"{np.median(nonzero):.6e} ppmv"

    )


print(
    "\nH units:"
)

print(
    "  ppmv per unit daily scaling factor"
)


print(
    "\nGrid-area source:"
)

print(
    f"  {HEADER_FILE}"
)

print(
    "  GRIDAREA [m2]"
)


print(
    "\nLayer:"
)

print(
    "  0-10 m"
)

print(
    f"  Height = "
    f"{LAYER_HEIGHT:.2f} m"
)


print(
    "\nCell-volume range:"
)

print(

    f"  "
    f"{np.min(CELL_VOLUME):.6e}"
    " to "
    f"{np.max(CELL_VOLUME):.6e} m3"

)


print(
    "\nSaved:"
)

print(
    H_file
)

print(
    META_file
)


print(
    "\nFirst 13 H columns:"
)

print(
    H_df.iloc[
        :,
        :13
    ].head()
)


print(
    "\nFirst row non-zero H values:"
)


nz = np.where(
    H[0] > 0
)[0]


for j in nz:

    print(

        f"{H_COLUMNS[j]:20s} "
        f"{H[0,j]:.8e} ppmv"

    )


print(
    "\nDone."
)
