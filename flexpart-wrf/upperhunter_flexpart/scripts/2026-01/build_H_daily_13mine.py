#!/usr/bin/env python3
"""
===============================================================================
FLEXPART-WRF -> DAILY 13-MINE H MATRIX
Upper Hunter CH4 inverse modelling
===============================================================================

Purpose
-------
Build the source-receptor sensitivity matrix H for the Bayesian inversion.

The inversion parameters are DAILY mine scaling factors:

    13 mines x 27 days = 351 parameters

Source period:

    2026-01-05 through 2026-01-31

Observation vector:

    Valid Picarro observations from 2026-01-05 00:00
    through 2026-01-31 23:00 AEST

FLEXPART-WRF runs:

    One 24-hour backward run for every valid
    observation/receptor pair.

Final matrix:

    H shape = (N observations, 351)

Column ordering:

    OC1_20260105
    OC2_20260105
    ...
    OC13_20260105
    OC1_20260106
    ...
    OC13_20260131

Units
-----
CONC:
    s m3 kg-1

H_mass:
    mixing ratio response per kg/s

H_ppm:
    ppmv per kg/s

The conversion is:

    H_ppm = H_mass * 1e6 * MW_air / MW_CH4

with:

    MW_air = 28.97
    MW_CH4 = 16.043

Important
---------
Do NOT multiply CONC by air density again.

The FLEXPART-WRF backward CONC formulation already gives the
appropriate source sensitivity for the calculation:

    H_mass =
        sum(CONC * mine_weight / cell_volume)

where:

    mine_weight = overlap_area / mine_polygon_area

and:

    cell_volume = GRIDAREA * layer_thickness

Spin-up handling
----------------
The WRF-Chem spin-up period is Jan 1-4.

Only source contributions from:

    Jan 5 00:00 through Jan 31 23:00

are assigned to inversion parameters.

Any FLEXPART source-time contribution before Jan 5 is ignored.

TEST mode
---------
By default the script processes only:

    singleton / 20260104_140000

which corresponds to:

    Observation: 2026-01-05 00:00 AEST
    Release:     2026-01-04 14:00 UTC

After verifying TEST output, change:

    MODE = "FULL"

to process all valid observation cases.

===============================================================================
"""

from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from netCDF4 import Dataset


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE = Path("/mnt/scratch_lustre/nzem_meim_scratch") / "flexpart-wrf" / "upperhunter_flexpart"

RUN_ROOT = BASE / "run" / "2026-01" / "hourly_obs"

HMATRIX_ROOT = BASE / "hmatrix"

Picarro_CSV = (
    Path.home()
    / "flexpart-wrf"
    / "emission_UH"
    / "RH_upperHunter"
    / "tapm_output"
    / "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

# ---------------------------------------------------------------------------
# TEST or FULL
# ---------------------------------------------------------------------------

MODE = "TEST"

# TEST case:
TEST_RECEPTOR = "singleton"
TEST_RELEASE = "20260110_020000"

# ---------------------------------------------------------------------------
# Inversion source period
# ---------------------------------------------------------------------------

START_DAY = datetime(2026, 1, 5)
END_DAY   = datetime(2026, 1, 31)

# ---------------------------------------------------------------------------
# Receptors
# ---------------------------------------------------------------------------

RECEPTORS = {
    "singleton": {
        "lat": -32.57123,
        "lon": 151.16132,
        "obs_column": "Singleton cleaned",
    },

    "muswellbrook": {
        "lat": -32.26202,
        "lon": 150.89297,
        "obs_column": "Muswellbrook cleaned",
    },

    "merriwa": {
        "lat": -32.14097,
        "lon": 150.35767,
        "obs_column": "Merriwa cleaned",
    },
}

# ---------------------------------------------------------------------------
# Mines
# ---------------------------------------------------------------------------

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

NMINE = len(MINE_NAMES)

# ---------------------------------------------------------------------------
# Mine overlap weights
#
# The weights are spatial and therefore do not change from day to day.
# The existing Jan-05 diagnostic file is used because it was calculated
# on the same FLEXPART 1-km grid.
# ---------------------------------------------------------------------------

WEIGHT_FILE = (
    HMATRIX_ROOT
    / "test_20260105"
    / "mine_area_fraction_20260105.npy"
)

# ---------------------------------------------------------------------------
# Molecular weights
# ---------------------------------------------------------------------------

MW_AIR = 28.97
MW_CH4 = 16.043

PPMV_FACTOR = 1.0e6 * MW_AIR / MW_CH4


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_time(x):
    """
    Convert a netCDF Times value to a normal string.

    Handles:
        S arrays
        U arrays
        bytes
        normal strings

    Removes FLEXPART '*' padding.
    """

    arr = np.asarray(x)

    if arr.dtype.kind == "S":
        return b"".join(arr.tolist()).decode().replace("*", "")

    if arr.dtype.kind == "U":
        return "".join(arr.tolist()).replace("*", "")

    if isinstance(x, bytes):
        return x.decode().replace("*", "")

    return str(x).replace("*", "")


def make_source_dates():
    """
    Return the 27 inversion source days.
    """

    days = []

    current = START_DAY

    while current <= END_DAY:
        days.append(current)
        current += timedelta(days=1)

    return days


def make_column_names():
    """
    Create the 351 H-matrix column names.

    Ordering:

        OC1-Jan5 ... OC13-Jan5
        OC1-Jan6 ... OC13-Jan6
        ...
    """

    columns = []

    for day in make_source_dates():

        day_string = day.strftime("%Y%m%d")

        for mine in MINE_NAMES:
            columns.append(f"{mine}_{day_string}")

    return columns


def load_observations():
    """
    Read Picarro observations.

    IMPORTANT:
    The existing Datetime column is used directly because the CSV contains
    Time values such as 24:00.
    """

    print("=" * 80)
    print("READING PICARRO OBSERVATIONS")
    print("=" * 80)

    df = pd.read_csv(Picarro_CSV)

    if "Datetime" not in df.columns:
        raise RuntimeError(
            "CSV does not contain the expected 'Datetime' column."
        )

    df["Datetime"] = pd.to_datetime(
        df["Datetime"],
        dayfirst=True,
        errors="coerce",
    )

    # Observation period is AEST.
    mask = (
        (df["Datetime"] >= START_DAY)
        &
        (
            df["Datetime"]
            <= END_DAY + timedelta(hours=23)
        )
    )

    df = df.loc[mask].copy()

    df = df.sort_values("Datetime")

    print(f"Observation rows : {len(df)}")
    print(
        f"Period           : "
        f"{START_DAY:%Y-%m-%d %H:%M} "
        f"to "
        f"{END_DAY + timedelta(hours=23):%Y-%m-%d %H:%M} AEST"
    )

    print()

    return df


def build_observation_cases(df):
    """
    Construct the valid observation-receptor cases.

    Each case contains:

        receptor
        datetime_aest
        datetime_utc
        release_directory
        observed_ch4

    FLEXPART release time is:

        AEST - 10 hours
    """

    cases = []

    for receptor, info in RECEPTORS.items():

        column = info["obs_column"]

        if column not in df.columns:
            raise RuntimeError(
                f"Missing observation column: {column}"
            )

        for _, row in df.iterrows():

            value = pd.to_numeric(
                row[column],
                errors="coerce",
            )

            if not np.isfinite(value):
                continue

            datetime_aest = row["Datetime"]

            datetime_utc = (
                datetime_aest
                - timedelta(hours=10)
            )

            release_dir = datetime_utc.strftime(
                "%Y%m%d_%H%M%S"
            )

            cases.append(
                {
                    "receptor": receptor,
                    "datetime_aest": datetime_aest,
                    "datetime_utc": datetime_utc,
                    "release_dir": release_dir,
                    "observation": float(value),
                }
            )

    return cases


def find_conc_file(run_dir):
    """
    Find the FLEXPART-WRF concentration NetCDF file.

    Expected form:

        flxout_d01_*.nc
    """

    files = sorted(
        run_dir.glob("flxout_d01_*.nc")
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"No flxout_d01_*.nc found in {run_dir}"
        )

    if len(files) > 1:
        print(
            f"WARNING: multiple FLEXPART output files found in "
            f"{run_dir}; using {files[0].name}"
        )

    return files[0]


def load_weights():
    """
    Load mine-grid overlap fractions.

    Expected shape:

        (13, 147, 165)

    where:

        weights[m, y, x]

    represents the fraction of mine m's area in grid cell y,x.
    """

    print("=" * 80)
    print("READING MINE GRID WEIGHTS")
    print("=" * 80)

    if not WEIGHT_FILE.exists():
        raise FileNotFoundError(
            f"Mine weight file not found:\n{WEIGHT_FILE}"
        )

    weights = np.load(WEIGHT_FILE)

    print(f"Weight file : {WEIGHT_FILE}")
    print(f"Shape       : {weights.shape}")
    print(f"Dtype       : {weights.dtype}")

    expected_shape = (NMINE, 147, 165)

    if weights.shape != expected_shape:
        raise ValueError(
            f"Unexpected mine weight shape.\n"
            f"Expected: {expected_shape}\n"
            f"Found:    {weights.shape}"
        )

    # Diagnostics
    for m, mine in enumerate(MINE_NAMES):

        w = weights[m]

        nonzero = np.count_nonzero(w)

        print(
            f"{mine:5s} "
            f"nonzero={nonzero:4d} "
            f"sum={np.sum(w):.8f} "
            f"max={np.max(w):.8f}"
        )

    print()

    return weights.astype(np.float64)

def read_grid_geometry(run_dir):
    """
    Read FLEXPART-WRF grid geometry from header_d01.nc.

    GRIDAREA and ZTOP are stored in the FLEXPART header file,
    not necessarily in the flxout concentration NetCDF file.

    Returns
    -------
    gridarea :
        (south_north, west_east) grid-cell area [m2]

    ztop :
        vertical layer tops [m]

    thickness :
        vertical layer thicknesses [m]

    volume :
        (bottom_top, south_north, west_east) cell volumes [m3]
    """

    header_file = run_dir / "header_d01.nc"

    if not header_file.exists():
        raise FileNotFoundError(
            f"FLEXPART header file not found:\n{header_file}"
        )

    print(f"Grid geometry file: {header_file.name}")

    with Dataset(header_file, "r") as nc:

        print(
            f"Header dimensions: "
            f"west_east={len(nc.dimensions['west_east'])}, "
            f"south_north={len(nc.dimensions['south_north'])}, "
            f"bottom_top={len(nc.dimensions['bottom_top'])}"
        )

        if "GRIDAREA" not in nc.variables:
            raise RuntimeError(
                f"GRIDAREA variable is not present in {header_file}"
            )

        gridarea = np.asarray(
            nc.variables["GRIDAREA"][:],
            dtype=np.float64,
        )

        gridarea = np.squeeze(gridarea)

        if gridarea.shape != (147, 165):
            raise ValueError(
                f"Unexpected GRIDAREA shape: {gridarea.shape}"
            )

        if "ZTOP" not in nc.variables:
            raise RuntimeError(
                f"ZTOP variable is not present in {header_file}"
            )

        ztop = np.asarray(
            nc.variables["ZTOP"][:],
            dtype=np.float64,
        ).squeeze()


    # -------------------------------------------------------------------------
    # Vertical layer thickness
    # -------------------------------------------------------------------------

    thickness = np.empty(
        len(ztop),
        dtype=np.float64,
    )

    thickness[0] = ztop[0]

    for k in range(1, len(ztop)):
        thickness[k] = (
            ztop[k] - ztop[k - 1]
        )

    # -------------------------------------------------------------------------
    # Grid-cell volume
    # -------------------------------------------------------------------------

    volume = (
        gridarea[None, :, :]
        *
        thickness[:, None, None]
    )

    print(
        f"GRIDAREA min/max/mean : "
        f"{np.min(gridarea):.3f} / "
        f"{np.max(gridarea):.3f} / "
        f"{np.mean(gridarea):.3f} m2"
    )

    print(
        f"ZTOP                  : {ztop}"
    )

    print(
        f"Layer thickness       : {thickness} m"
    )

    print(
        f"Volume shape          : {volume.shape}"
    )

    return (
        gridarea,
        ztop,
        thickness,
        volume,
    )


def calculate_hourly_mine_sensitivity(
    conc,
    weights,
    volume,
):
    """
    Calculate mine sensitivity for every FLEXPART output time.

    Parameters
    ----------
    conc :
        Array with shape:

            (time, bottom_top, south_north, west_east)

    weights :
        Array:

            (13, south_north, west_east)

    volume :
        Array:

            (bottom_top, south_north, west_east)

    Returns
    -------
    H_hourly_mass :

        shape:

            (time, 13)

        units:

            mixing ratio response per kg/s

    H_hourly_ppm :

        same shape, ppmv per kg/s

    Formula
    -------

        H_mass[t,m] =
            sum(
                CONC[t,k,y,x]
                *
                weights[m,y,x]
                /
                volume[k,y,x]
            )

    This is the same formulation used in the previously validated
    FLEXPART-WRF H test calculation.
    """

    # Ensure double precision for the calculation.
    conc = np.asarray(
        conc,
        dtype=np.float64,
    )

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    volume = np.asarray(
        volume,
        dtype=np.float64,
    )

    if conc.ndim != 4:
        raise ValueError(
            f"Expected CONC with 4 dimensions after selecting "
            f"ageclass/release. Got {conc.shape}"
        )

    nt, nz, ny, nx = conc.shape

    if weights.shape != (NMINE, ny, nx):
        raise ValueError(
            f"Weight dimensions {weights.shape} do not match "
            f"CONC horizontal dimensions {(ny, nx)}."
        )

    if volume.shape != (nz, ny, nx):
        raise ValueError(
            f"Volume dimensions {volume.shape} do not match "
            f"CONC dimensions {(nz, ny, nx)}."
        )

    # -------------------------------------------------------------------------
    # Avoid division by zero.
    # -------------------------------------------------------------------------

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        # Shape:
        #
        # conc    = (time, layer, y, x)
        # weights = (mine, y, x)
        # volume  = (layer, y, x)
        #
        # Result:
        #
        # (time, mine, layer, y, x)

        kernel = (
            conc[:, None, :, :, :]
            *
            weights[None, :, None, :, :]
            /
            volume[None, None, :, :, :]
        )

    kernel = np.nan_to_num(
        kernel,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    H_hourly_mass = np.sum(
        kernel,
        axis=(2, 3, 4),
    )

    H_hourly_ppm = (
        H_hourly_mass
        * PPMV_FACTOR
    )

    return (
        H_hourly_mass,
        H_hourly_ppm,
    )


def get_conc_for_run(run_dir):
    """
    Read CONC and geometry from one FLEXPART run.

    Returns:

        conc
        times
        volume
    """

    conc_file = find_conc_file(run_dir)

    print(f"CONC file: {conc_file.name}")

    with Dataset(conc_file, "r") as nc:

        if "CONC" not in nc.variables:
            raise RuntimeError(
                f"CONC variable not found in {conc_file}"
            )

        conc_var = nc.variables["CONC"]

        print(f"CONC dimensions: {conc_var.dimensions}")
        print(f"CONC shape     : {conc_var.shape}")
        print(f"CONC units     : {getattr(conc_var, 'units', 'unknown')}")

        # Expected:
        #
        # Time, ageclass, releases, bottom_top, south_north, west_east
        #

        if len(conc_var.shape) != 6:
            raise ValueError(
                f"Unexpected CONC dimensions: {conc_var.shape}"
            )

        nt, nage, nrelease, nz, ny, nx = conc_var.shape

        if nage != 1:
            raise ValueError(
                f"Expected ageclass=1, found {nage}"
            )

        if nrelease != 1:
            raise ValueError(
                f"Expected releases=1, found {nrelease}"
            )

        # Read:
        #
        # (time, layer, y, x)

        conc = np.asarray(
            conc_var[:, 0, 0, :, :, :],
            dtype=np.float64,
        )

        times = [
            clean_time(x)
            for x in nc.variables["Times"][:]
        ]

        gridarea, ztop, thickness, volume = (
            read_grid_geometry(run_dir)
        )

    print(f"CONC selected shape: {conc.shape}")
    print(f"ZTOP               : {ztop}")
    print(f"Layer thickness     : {thickness}")
    print(
        f"CONC min/max        : "
        f"{np.nanmin(conc):.8e} / "
        f"{np.nanmax(conc):.8e}"
    )

    return (
        conc,
        times,
        volume,
    )


def calculate_source_time(release_utc, output_index):
    """
    Map FLEXPART output record to the source-time hour.

    For a 24-hour backward run:

        output 0  -> release - 24 h
        output 1  -> release - 23 h
        ...
        output 23 -> release - 1 h

    Example:

        release = 2026-01-04 14:00 UTC

        source hours:
            Jan 3 14:00
            ...
            Jan 4 13:00

    However, for the particular FLEXPART output used here, the previously
    validated hourly-H implementation associated the first record with:

        release - 24 hours

    and the last record with:

        release - 1 hour.

    This function therefore follows that established convention.
    """

    return (
        release_utc
        - timedelta(hours=24 - output_index)
    )

def parse_flexpart_time(time_record):
    """
    Convert one FLEXPART NetCDF Times record to a UTC datetime.
    """
    chars = []

    for value in time_record:
        if isinstance(value, bytes):
            chars.append(value.decode("ascii"))
        else:
            chars.append(str(value))

    time_string = "".join(chars).strip()

    return datetime.strptime(
        time_string,
        "%Y%m%d_%H%M%S",
    )


def build_one_H_row(
    case,
    weights,
    verbose=True,
):
    """
    Build the 351-element H row for one observation.

    A 24-hour FLEXPART footprint is integrated into the appropriate
    calendar source day.

    Source hours before START_DAY are discarded.
    """

    receptor = case["receptor"]
    release_utc = case["datetime_utc"]

    # ------------------------------------------------------------------
    # TEST OVERRIDE
    # ------------------------------------------------------------------
    # The successful Singleton test run currently lives directly in
    # run/singleton_10m rather than the production hourly_obs tree.
    #
    # This is only for MODE="TEST".
    # ------------------------------------------------------------------

    if MODE.upper() == "TEST":

        run_dir = (
            BASE
            / "run"
            / "singleton_2layer_clean"
        )

    else:

        run_dir = (
            RUN_ROOT
            / receptor
            / release_dir
        )

    if not run_dir.exists():
        raise FileNotFoundError(
            f"FLEXPART run directory does not exist:\n{run_dir}"
        )

    if verbose:
        print()
        print("=" * 80)
        print("PROCESSING FLEXPART RUN")
        print("=" * 80)
        print(f"Receptor       : {receptor}")
        print(f"Observation    : {case['datetime_aest']}")
        print(f"Observation CH4: {case['observation']:.8f} ppm")
        print(f"Release UTC    : {release_utc}")
        print(f"Run directory  : {run_dir}")

    conc, times, volume = get_conc_for_run(
        run_dir
    )

    H_hourly_mass, H_hourly_ppm = (
        calculate_hourly_mine_sensitivity(
            conc,
            weights,
            volume,
        )
    )

    nt = H_hourly_ppm.shape[0]

    if nt != 24:
        raise ValueError(
            f"Expected 24 FLEXPART output records; "
            f"found {nt}"
        )

    # -------------------------------------------------------------------------
    # Final 351-column row
    # -------------------------------------------------------------------------

    nday = (
        END_DAY - START_DAY
    ).days + 1

    H_row = np.zeros(
        nday * NMINE,
        dtype=np.float64,
    )

    source_records = []

    for it in range(nt):

        # FLEXPART NetCDF Times are UTC.
        source_time_utc = parse_flexpart_time(
            times[it]
        )

        # The inversion source-day columns are defined
        # using AEST calendar days.
        source_time = (
            source_time_utc
            + timedelta(hours=10)
        )

        # Ignore source hours before inversion start.
        if source_time < START_DAY:
            source_records.append(
                (
                    source_time,
                    "SPINUP_EXCLUDED",
                )
            )
            continue

        # Ignore source hours after inversion end.
        if source_time > (
            END_DAY + timedelta(hours=23)
        ):
            source_records.append(
                (
                    source_time,
                    "OUTSIDE_PERIOD",
                )
            )
            continue

        # Determine source day.
        source_day = source_time.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        day_index = (
            source_day - START_DAY
        ).days

        if day_index < 0 or day_index >= nday:
            source_records.append(
                (
                    source_time,
                    "OUTSIDE_PERIOD",
                )
            )
            continue

        start_col = (
            day_index * NMINE
        )

        end_col = (
            start_col + NMINE
        )

        # Add this source-hour footprint to the corresponding
        # daily source factor.
        H_row[start_col:end_col] += (
            H_hourly_ppm[it, :]
        )

        source_records.append(
            (
                source_time,
                f"DAY_{source_day:%Y%m%d}",
            )
        )

    if verbose:

        print()
        print("SOURCE-TIME MAPPING")
        print("-" * 80)

        for source_time, status in source_records:

            print(
                f"{source_time:%Y-%m-%d %H:%M} UTC  "
                f"{status}"
            )

        print()
        print("H ROW SUMMARY")
        print("-" * 80)

        print(
            f"Nonzero elements : "
            f"{np.count_nonzero(H_row)} / {len(H_row)}"
        )

        print(
            f"Maximum H        : "
            f"{np.max(H_row):.12e} ppmv/(kg/s)"
        )

        print(
            f"Sum H            : "
            f"{np.sum(H_row):.12e} ppmv/(kg/s)"
        )

    return H_row


def save_test_result(
    H_row,
    case,
    column_names,
):
    """
    Save the TEST result.
    """

    outdir = (
        HMATRIX_ROOT
        / "daily_test"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    receptor = case["receptor"]
    release = case["release_dir"]

    npy_file = (
        outdir
        / f"H_{receptor}_{release}_daily_13mine.npy"
    )

    csv_file = (
        outdir
        / f"H_{receptor}_{release}_daily_13mine.csv"
    )

    np.save(
        npy_file,
        H_row,
    )

    df = pd.DataFrame(
        {
            "source": column_names,
            "H_ppm_per_kg_s": H_row,
        }
    )

    df.to_csv(
        csv_file,
        index=False,
    )

    print()
    print("=" * 80)
    print("TEST FILES WRITTEN")
    print("=" * 80)
    print(f"NPY : {npy_file}")
    print(f"CSV : {csv_file}")

    # Print nonzero values.
    print()
    print("NONZERO DAILY H VALUES")
    print("-" * 80)

    for name, value in zip(
        column_names,
        H_row,
    ):

        if value != 0.0:
            print(
                f"{name:25s} "
                f"{value:.12e}"
            )


def run_test(
    df,
    weights,
    column_names,
):
    """
    Process the known successful Singleton test case.
    """

    target = None

    for receptor in [TEST_RECEPTOR]:

        info = RECEPTORS[receptor]

        obs_column = info["obs_column"]

        for _, row in df.iterrows():

            value = pd.to_numeric(
                row[obs_column],
                errors="coerce",
            )

            if not np.isfinite(value):
                continue

            datetime_aest = row["Datetime"]

            datetime_utc = (
                datetime_aest
                - timedelta(hours=10)
            )

            release_dir = datetime_utc.strftime(
                "%Y%m%d_%H%M%S"
            )

            if release_dir == TEST_RELEASE:

                target = {
                    "receptor": receptor,
                    "datetime_aest": datetime_aest,
                    "datetime_utc": datetime_utc,
                    "release_dir": release_dir,
                    "observation": float(value),
                }

                break

    if target is None:
        raise RuntimeError(
            f"Could not find TEST case:\n"
            f"{TEST_RECEPTOR}/{TEST_RELEASE}"
        )

    H_row = build_one_H_row(
        target,
        weights,
        verbose=True,
    )

    save_test_result(
        H_row,
        target,
        column_names,
    )

    return H_row


def run_full(
    df,
    weights,
    column_names,
):
    """
    Process all valid observation-receptor cases.
    """

    cases = build_observation_cases(
        df
    )

    print()
    print("=" * 80)
    print("FULL H-MATRIX GENERATION")
    print("=" * 80)

    print(
        f"Total observation cases : {len(cases)}"
    )

    print(
        f"Mines                   : {NMINE}"
    )

    print(
        f"Source days              : "
        f"{len(make_source_dates())}"
    )

    print(
        f"Unknowns                 : "
        f"{len(column_names)}"
    )

    # -------------------------------------------------------------------------
    # Expected number based on the known Picarro data.
    # -------------------------------------------------------------------------

    expected_cases = 1124

    if len(cases) != expected_cases:

        print()
        print(
            "WARNING:"
        )

        print(
            f"Expected {expected_cases} cases "
            f"from the current Picarro dataset, "
            f"but found {len(cases)}."
        )

        print(
            "The script will continue, but verify "
            "the observation selection."
        )

    nobs = len(cases)
    nparam = len(column_names)

    H = np.zeros(
        (nobs, nparam),
        dtype=np.float64,
    )

    metadata = []

    failed = []

    for i, case in enumerate(cases):

        print()
        print(
            "=" * 80
        )

        print(
            f"CASE {i + 1} / {nobs}"
        )

        print(
            f"{case['receptor']} "
            f"{case['datetime_aest']}"
        )

        try:

            H[i, :] = build_one_H_row(
                case,
                weights,
                verbose=False,
            )

            metadata.append(
                {
                    "row": i,
                    "receptor": case["receptor"],
                    "datetime_aest":
                        case["datetime_aest"],
                    "datetime_utc":
                        case["datetime_utc"],
                    "release_dir":
                        case["release_dir"],
                    "observation_ppm":
                        case["observation"],
                    "status": "OK",
                }
            )

            print(
                f"OK: nonzero H = "
                f"{np.count_nonzero(H[i, :])}"
            )

        except Exception as exc:

            print(
                f"FAILED: {exc}"
            )

            failed.append(
                {
                    "row": i,
                    "receptor": case["receptor"],
                    "datetime_aest":
                        case["datetime_aest"],
                    "release_dir":
                        case["release_dir"],
                    "error": str(exc),
                }
            )

            metadata.append(
                {
                    "row": i,
                    "receptor": case["receptor"],
                    "datetime_aest":
                        case["datetime_aest"],
                    "datetime_utc":
                        case["datetime_utc"],
                    "release_dir":
                        case["release_dir"],
                    "observation_ppm":
                        case["observation"],
                    "status": "FAILED",
                }
            )

    # -------------------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------------------

    outdir = (
        HMATRIX_ROOT
        / "daily_20260105_20260131"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    H_npy = (
        outdir
        / "H_daily_13mine_1124x351_ppm.npy"
    )

    H_csv = (
        outdir
        / "H_daily_13mine_1124x351_ppm.csv"
    )

    obs_csv = (
        outdir
        / "observation_vector.csv"
    )

    metadata_csv = (
        outdir
        / "H_metadata.csv"
    )

    failed_csv = (
        outdir
        / "failed_cases.csv"
    )

    np.save(
        H_npy,
        H,
    )

    # H matrix CSV
    H_df = pd.DataFrame(
        H,
        columns=column_names,
    )

    H_df.to_csv(
        H_csv,
        index=False,
    )

    # Observation vector
    obs_values = [
        c["observation"]
        for c in cases
    ]

    obs_df = pd.DataFrame(
        {
            "row": np.arange(nobs),
            "receptor": [
                c["receptor"]
                for c in cases
            ],
            "datetime_aest": [
                c["datetime_aest"]
                for c in cases
            ],
            "datetime_utc": [
                c["datetime_utc"]
                for c in cases
            ],
            "observation_ppm": obs_values,
        }
    )

    obs_df.to_csv(
        obs_csv,
        index=False,
    )

    # Metadata
    metadata_df = pd.DataFrame(
        metadata
    )

    metadata_df.to_csv(
        metadata_csv,
        index=False,
    )

    # Failed cases
    if failed:

        failed_df = pd.DataFrame(
            failed
        )

        failed_df.to_csv(
            failed_csv,
            index=False,
        )

    # -------------------------------------------------------------------------
    # Final diagnostics
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FULL H MATRIX COMPLETE")
    print("=" * 80)

    print(
        f"H shape             : {H.shape}"
    )

    print(
        f"Expected shape      : "
        f"({nobs}, {nparam})"
    )

    print(
        f"Successful cases    : "
        f"{nobs - len(failed)}"
    )

    print(
        f"Failed cases        : "
        f"{len(failed)}"
    )

    print(
        f"Nonzero H elements  : "
        f"{np.count_nonzero(H)}"
    )

    print(
        f"Maximum H           : "
        f"{np.max(H):.12e}"
    )

    print(
        f"Mean H              : "
        f"{np.mean(H):.12e}"
    )

    print()
    print("OUTPUT FILES")
    print("-" * 80)

    print(
        f"H NPY       : {H_npy}"
    )

    print(
        f"H CSV       : {H_csv}"
    )

    print(
        f"Observations : {obs_csv}"
    )

    print(
        f"Metadata     : {metadata_csv}"
    )

    if failed:
        print(
            f"Failed       : {failed_csv}"
        )

    return H


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 80)
    print("FLEXPART-WRF DAILY 13-MINE H MATRIX")
    print("=" * 80)

    print(
        f"MODE        : {MODE}"
    )

    print(
        f"Source      : "
        f"{START_DAY:%Y-%m-%d} "
        f"to "
        f"{END_DAY:%Y-%m-%d}"
    )

    print(
        f"Mines       : {NMINE}"
    )

    print(
        f"Parameters  : "
        f"{NMINE * len(make_source_dates())}"
    )

    print(
        f"PPMV factor : {PPMV_FACTOR:.12f}"
    )

    print()

    # -------------------------------------------------------------------------
    # Check files
    # -------------------------------------------------------------------------

    if not Picarro_CSV.exists():
        raise FileNotFoundError(
            f"Picarro CSV not found:\n{Picarro_CSV}"
        )

    # -------------------------------------------------------------------------
    # Read observations
    # -------------------------------------------------------------------------

    df = load_observations()

    # -------------------------------------------------------------------------
    # Read mine weights
    # -------------------------------------------------------------------------

    weights = load_weights()

    # -------------------------------------------------------------------------
    # Column names
    # -------------------------------------------------------------------------

    column_names = make_column_names()

    print("=" * 80)
    print("H MATRIX COLUMN STRUCTURE")
    print("=" * 80)

    print(
        f"Number of columns : {len(column_names)}"
    )

    print(
        f"First 13 columns  : "
        f"{column_names[:13]}"
    )

    print(
        f"Last 13 columns   : "
        f"{column_names[-13:]}"
    )

    print()

    # -------------------------------------------------------------------------
    # TEST
    # -------------------------------------------------------------------------

    if MODE.upper() == "TEST":

        run_test(
            df,
            weights,
            column_names,
        )

    # -------------------------------------------------------------------------
    # FULL
    # -------------------------------------------------------------------------

    elif MODE.upper() == "FULL":

        run_full(
            df,
            weights,
            column_names,
        )

    else:

        raise ValueError(
            "MODE must be either 'TEST' or 'FULL'."
        )


if __name__ == "__main__":
    main()

