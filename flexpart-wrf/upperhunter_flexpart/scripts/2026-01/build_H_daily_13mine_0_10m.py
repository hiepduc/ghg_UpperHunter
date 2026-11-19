#!/usr/bin/env python3
"""
===============================================================================
Build FLEXPART-WRF H matrix for Upper Hunter CH4 inversion
===============================================================================

Period:
    2026-01-05 through 2026-01-31 AEST

Receptors:
    Singleton
    Muswellbrook
    Merriwa

Sources:
    13 open-cut coal mines (OC1 ... OC13)

Unknowns:
    Daily scaling factors for each mine

Dimensions:
    13 mines x 27 days = 351 unknowns
    1124 observations x 351 H matrix

IMPORTANT:
    ONLY the FLEXPART 0-10 m vertical layer is used.

    The 10-100 m and 100-20,000 m layers are NOT included.

Time alignment:
    Picarro timestamps are AEST.

    For an observation at:
        2026-01-10 12:00 AEST

    convert to:
        2026-01-10 02:00 UTC

    FLEXPART production run:
        20260110_020000

    FLEXPART output selected:
        2026-01-10 01:00 UTC

    because FLEXPART output 0 corresponds to release - 1 hour.

The script does NOT assume a fixed FLEXPART time index. It reads the
NetCDF Times variable and requires an exact match to observation UTC - 1 hour.

Output:
    H.npy
    observation_vector.csv
    column_metadata.csv
    row_metadata.csv
    H_summary.txt

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

BASE = Path("/mnt/scratch_lustre/nzem_meim_scratch/flexpart-wrf") / "upperhunter_flexpart"

# -------------------------------------------------------------------------
# FLEXPART-WRF production runs
#
# IMPORTANT:
# This must contain the NEW 3-layer runs:
#
#     0-10 m
#     10-100 m
#     100-20000 m
#
# Do NOT point this to run/hourly_obs/, which is the existing 2-layer run.
# -------------------------------------------------------------------------
RUN_ROOT = BASE / "run" / "2026-01" / "hourly_obs_3layer"

# -------------------------------------------------------------------------
# Picarro observations
# -------------------------------------------------------------------------
OBS_FILE = (
    Path("/mnt/scratch_lustre/nzem_meim_scratch")
    / "data"
    / "piccaro_obs"
    / "Picarro_CH4_Upper_Hunter_Jan-July_2026.csv"
)

# -------------------------------------------------------------------------
# Mine overlap weights
#
# Shape:
#     (13, 147, 165)
#
# weight = overlap_area / mine_polygon_area
# -------------------------------------------------------------------------
MINE_WEIGHT_FILE = (
    BASE
    / "H_matrix_flexpart"
    / "test_20260105"
    / "mine_area_fraction_20260105.npy"
)

# -------------------------------------------------------------------------
# Output
# -------------------------------------------------------------------------
OUTPUT_DIR = (
    BASE
    / "H_matrix_flexpart"
    / "daily_20260105_20260131_0_10m"
)

# -------------------------------------------------------------------------
# Period
# -------------------------------------------------------------------------
START_DATE = pd.Timestamp("2026-01-05")
END_DATE = pd.Timestamp("2026-01-31")

# -------------------------------------------------------------------------
# Background / molecular conversion
#
# H is calculated as:
#
#     FLEXPART footprint
#       x mine area fraction
#       / 0-10 m grid-cell volume
#       x ppm conversion
#
# FLEXPART CONC units:
#     s m3 kg-1
#
# For a surface emission source in kg/s:
#
#     ppmv = mass sensitivity * (MW_air / MW_CH4) * 1e6
# -------------------------------------------------------------------------
MW_AIR = 28.97
MW_CH4 = 16.043

PPMV_FACTOR = 1.0e6 * MW_AIR / MW_CH4

# 0-10 m layer
LAYER_BOTTOM_M = 0.0
LAYER_TOP_M = 10.0
LAYER_THICKNESS_M = 10.0

# -------------------------------------------------------------------------
# Receptors
# -------------------------------------------------------------------------
RECEPTORS = {
    "singleton": {
        "lat": -32.57123,
        "lon": 151.16132,
        "zagl": 5.0,
    },

    "muswellbrook": {
        "lat": -32.26202,
        "lon": 150.89297,
        "zagl": 5.0,
    },

    "merriwa": {
        "lat": -32.14097,
        "lon": 150.35767,
        "zagl": 5.0,
    },
}

# -------------------------------------------------------------------------
# Mine names
# -------------------------------------------------------------------------
MINES = [
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


# =============================================================================
# HELPERS
# =============================================================================

def parse_flexpart_times(nc):
    """
    Read FLEXPART NetCDF Times variable and return UTC datetimes.

    FLEXPART-WRF Times may contain byte strings or character arrays.
    """

    if "Times" not in nc.variables:
        raise RuntimeError(
            "NetCDF file does not contain a 'Times' variable."
        )

    times_var = nc.variables["Times"][:]

    result = []

    # Case 1:
    # Times = array of strings / bytes
    if times_var.ndim == 1:

        for value in times_var:

            if isinstance(value, bytes):
                s = value.decode("ascii")

            elif isinstance(value, str):
                s = value

            else:
                s = str(value)

            s = s.replace("*", "").strip()

            # FLEXPART format
            # YYYYMMDD_HHMMSS
            dt = datetime.strptime(s, "%Y%m%d_%H%M%S")

            result.append(pd.Timestamp(dt))

        return result

    # Case 2:
    # Times = character array [time, strlen]
    if times_var.ndim == 2:

        for row in times_var:

            chars = []

            for value in row:

                if isinstance(value, bytes):
                    chars.append(value.decode("ascii"))

                else:
                    chars.append(chr(int(value)))

            s = "".join(chars)

            s = s.replace("*", "").strip()

            dt = datetime.strptime(s, "%Y%m%d_%H%M%S")

            result.append(pd.Timestamp(dt))

        return result

    raise RuntimeError(
        f"Unsupported Times dimensions: {times_var.shape}"
    )


def find_exact_time_index(nc, target_utc):
    """
    Find the exact FLEXPART output corresponding to target_utc.

    We intentionally require an exact match.

    No nearest-neighbour matching is allowed because a one-hour shift would
    change the source-day attribution.
    """

    flex_times = parse_flexpart_times(nc)

    target_utc = pd.Timestamp(target_utc).tz_localize(None)

    matches = [
        i
        for i, t in enumerate(flex_times)
        if pd.Timestamp(t).tz_localize(None) == target_utc
    ]

    if len(matches) == 0:

        available = "\n".join(
            f"    {i:02d}: {t}"
            for i, t in enumerate(flex_times)
        )

        raise RuntimeError(
            "\n"
            "FLEXPART Times mismatch\n"
            "----------------------\n"
            f"Requested UTC time: {target_utc}\n"
            f"Available times:\n{available}\n"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Duplicate FLEXPART Times found for {target_utc}: {matches}"
        )

    return matches[0]


def get_gridarea(nc):
    """
    Extract FLEXPART grid-cell area.

    Expected dimensions:
        (y, x)

    The production grid is approximately 1 km x 1 km.
    """

    possible_names = [
        "GRIDAREA",
        "gridarea",
        "AREA",
        "area",
    ]

    for name in possible_names:

        if name in nc.variables:

            arr = np.asarray(nc.variables[name][:], dtype=np.float64)

            # Remove singleton dimensions if present
            arr = np.squeeze(arr)

            if arr.ndim != 2:
                raise RuntimeError(
                    f"Grid-area variable '{name}' has unexpected shape "
                    f"{arr.shape}"
                )

            return arr

    raise RuntimeError(
        "Could not find grid-cell area variable. "
        "Expected one of: GRIDAREA, gridarea, AREA, area."
    )


def get_concentration(nc):
    """
    Read FLEXPART CONC variable.

    Expected production shape:

        (time, release_point, species, layer, y, x)

    which is:

        (24, 1, 1, 3, 147, 165)

    Some files may have singleton dimensions removed, so this function
    squeezes only dimensions of length 1.

    Returns:
        array with dimensions:

            (time, layer, y, x)
    """

    if "CONC" not in nc.variables:
        raise RuntimeError(
            "NetCDF file does not contain CONC variable."
        )

    conc = np.asarray(
        nc.variables["CONC"][:],
        dtype=np.float64,
    )

    original_shape = conc.shape

    # Remove singleton dimensions
    conc = np.squeeze(conc)

    # Expected after squeeze:
    #
    # (24, 3, 147, 165)
    #
    if conc.ndim != 4:

        raise RuntimeError(
            "\n"
            "Unexpected CONC dimensions\n"
            "---------------------------\n"
            f"Original shape: {original_shape}\n"
            f"After squeeze: {conc.shape}\n"
            "Expected four dimensions: "
            "(time, layer, y, x)\n"
        )

    return conc


def check_grid_dimensions(gridarea, mine_weights):
    """
    Verify that gridarea and mine weights use the same spatial grid.
    """

    if gridarea.shape != mine_weights.shape[1:]:

        raise RuntimeError(
            "\n"
            "Grid dimension mismatch\n"
            "-----------------------\n"
            f"GRIDAREA:       {gridarea.shape}\n"
            f"Mine weights:   {mine_weights.shape[1:]}\n"
        )


def calculate_H_row(
    nc,
    time_index,
    gridarea,
    mine_weights,
):
    """
    Calculate one H row using ONLY the 0-10 m layer.

    Parameters
    ----------
    nc:
        Open FLEXPART NetCDF file.

    time_index:
        Exact FLEXPART output index corresponding to observation UTC - 1 h.

    gridarea:
        2-D grid cell area, m2.

    mine_weights:
        Shape (13, ny, nx).

    Returns
    -------
    H_mine_ppm:
        Array shape (13,)

        Units:
            ppmv / (kg CH4 s-1)
    """

    conc = get_concentration(nc)

    # -----------------------------------------------------------------
    # Confirm that this is a 3-layer production file.
    # -----------------------------------------------------------------
    n_time, n_layer, ny, nx = conc.shape

    if n_layer != 3:

        raise RuntimeError(
            "\n"
            "This H builder requires the 3-layer FLEXPART run.\n"
            f"CONC has {n_layer} vertical layers.\n"
            "Expected:\n"
            "    layer 0 = 0-10 m\n"
            "    layer 1 = 10-100 m\n"
            "    layer 2 = 100-20000 m\n"
        )

    if gridarea.shape != (ny, nx):

        raise RuntimeError(
            f"Grid area shape {gridarea.shape} does not match "
            f"CONC spatial shape {(ny, nx)}"
        )

    # -----------------------------------------------------------------
    # ONLY layer 0 = 0-10 m
    # -----------------------------------------------------------------
    footprint_0_10 = conc[
        time_index,
        0,
        :, :,
    ]

    # -----------------------------------------------------------------
    # 0-10 m volume
    #
    # volume = grid cell area x layer thickness
    #
    # thickness = 10 m
    # -----------------------------------------------------------------
    volume_0_10 = gridarea * LAYER_THICKNESS_M

    # -----------------------------------------------------------------
    # For each mine:
    #
    # footprint * mine overlap fraction / cell volume
    #
    # Then integrate over x,y.
    # -----------------------------------------------------------------

    kernel = (
        footprint_0_10[None, :, :]
        * mine_weights
        / volume_0_10[None, :, :]
    )

    H_mine_mass = np.sum(
        kernel,
        axis=(1, 2),
    )

    # Convert to ppmv per kg/s
    H_mine_ppm = H_mine_mass * PPMV_FACTOR

    return H_mine_ppm


def validate_flexpart_run(
    run_dir,
    expected_release_utc,
    target_output_utc,
):
    """
    Validate that a production FLEXPART run has the expected release
    and contains the requested output time.
    """

    nc_files = sorted(run_dir.glob("flxout_d01_*.nc"))

    if len(nc_files) == 0:

        raise FileNotFoundError(
            f"No flxout_d01_*.nc found in {run_dir}"
        )

    # Normally there is one output file for a 24-hour run.
    # Search all files in case the run has been split.
    for nc_path in nc_files:

        with Dataset(nc_path, "r") as nc:

            times = parse_flexpart_times(nc)

            for i, t in enumerate(times):

                if pd.Timestamp(t) == pd.Timestamp(
                    target_output_utc
                ):

                    return nc_path, i

    raise RuntimeError(
        "\n"
        "Requested FLEXPART output time not found\n"
        "-----------------------------------------\n"
        f"Run directory:      {run_dir}\n"
        f"Expected release:   {expected_release_utc}\n"
        f"Requested output:   {target_output_utc}\n"
        f"NetCDF files:       {len(nc_files)}\n"
    )


# =============================================================================
# LOAD OBSERVATIONS
# =============================================================================

def load_observations():

    print("=" * 80)
    print("READING PICARRO OBSERVATIONS")
    print("=" * 80)

    if not OBS_FILE.exists():

        raise FileNotFoundError(
            f"Observation file does not exist:\n{OBS_FILE}"
        )

    df = pd.read_csv(OBS_FILE)

    print(f"Observation file:")
    print(f"  {OBS_FILE}")
    print()

    required_columns = [
        "Datetime",
        "Singleton cleaned",
        "Muswellbrook cleaned",
        "Merriwa cleaned",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing required observation columns: {missing}"
        )

    # -----------------------------------------------------------------
    # IMPORTANT:
    #
    # Raw CSV timestamps are DD/MM/YYYY.
    # -----------------------------------------------------------------
    df["Datetime"] = pd.to_datetime(
        df["Datetime"],
        dayfirst=True,
        errors="coerce",
    )

    rows = []

    receptor_columns = {
        "singleton": "Singleton cleaned",
        "muswellbrook": "Muswellbrook cleaned",
        "merriwa": "Merriwa cleaned",
    }

    for receptor, column in receptor_columns.items():

        temp = df[
            ["Datetime", column]
        ].copy()

        temp = temp.rename(
            columns={
                "Datetime": "datetime_aest",
                column: "observation_ppm",
            }
        )

        temp["receptor"] = receptor

        # Remove invalid values
        temp = temp[
            temp["datetime_aest"].notna()
            & temp["observation_ppm"].notna()
        ].copy()

        # Restrict to requested AEST dates
        temp = temp[
            (temp["datetime_aest"].dt.normalize() >= START_DATE)
            & (temp["datetime_aest"].dt.normalize() <= END_DATE)
        ].copy()

        rows.append(temp)

    obs = pd.concat(
        rows,
        ignore_index=True,
    )

    # Sort consistently by AEST datetime, then receptor
    receptor_order = {
        "singleton": 0,
        "muswellbrook": 1,
        "merriwa": 2,
    }

    obs["_receptor_order"] = obs["receptor"].map(
        receptor_order
    )

    obs = obs.sort_values(
        ["datetime_aest", "_receptor_order"]
    ).reset_index(drop=True)

    obs = obs.drop(
        columns="_receptor_order"
    )

    # -----------------------------------------------------------------
    # Convert AEST -> UTC
    #
    # AEST = UTC + 10 h
    # -----------------------------------------------------------------
    obs["datetime_utc"] = (
        obs["datetime_aest"]
        - pd.Timedelta(hours=10)
    )

    # Source day remains the AEST calendar day
    obs["source_day"] = (
        obs["datetime_aest"]
        .dt.normalize()
    )

    obs["row"] = np.arange(
        len(obs),
        dtype=int,
    )

    print(f"Valid observations:")
    print(f"  Total:          {len(obs)}")
    print()

    for receptor in receptor_columns:

        n = np.sum(
            obs["receptor"] == receptor
        )

        print(
            f"  {receptor:15s}: {n:4d}"
        )

    print()

    print(
        f"Date range: "
        f"{obs['datetime_aest'].min()} "
        f"to "
        f"{obs['datetime_aest'].max()}"
    )

    print()

    return obs


# =============================================================================
# COLUMN METADATA
# =============================================================================

def build_column_metadata():

    rows = []

    column = 0

    for day in pd.date_range(
        START_DATE,
        END_DATE,
        freq="D",
    ):

        for mine in MINES:

            rows.append(
                {
                    "column": column,
                    "mine": mine,
                    "source_day": day.strftime(
                        "%Y-%m-%d"
                    ),
                    "source_day_aest": day.strftime(
                        "%Y-%m-%d"
                    ),
                }
            )

            column += 1

    return pd.DataFrame(rows)


# =============================================================================
# MAIN H BUILDER
# =============================================================================

def build_H():

    print("=" * 80)
    print("FLEXPART-WRF H MATRIX BUILDER")
    print("0-10 m ONLY")
    print("=" * 80)
    print()

    # -----------------------------------------------------------------
    # Check paths
    # -----------------------------------------------------------------

    print("CONFIGURATION")
    print("-" * 80)

    print(f"RUN_ROOT:")
    print(f"  {RUN_ROOT}")
    print()

    print(f"OBS_FILE:")
    print(f"  {OBS_FILE}")
    print()

    print(f"MINE_WEIGHT_FILE:")
    print(f"  {MINE_WEIGHT_FILE}")
    print()

    print(f"OUTPUT_DIR:")
    print(f"  {OUTPUT_DIR}")
    print()

    if not RUN_ROOT.exists():

        raise FileNotFoundError(
            "\n"
            "3-layer FLEXPART production directory does not exist:\n"
            f"{RUN_ROOT}\n\n"
            "Create/run the 3-layer production FLEXPART simulations first.\n"
            "Do NOT change RUN_ROOT to run/hourly_obs because that is the "
            "existing 2-layer configuration.\n"
        )

    if not MINE_WEIGHT_FILE.exists():

        raise FileNotFoundError(
            f"Mine weight file does not exist:\n{MINE_WEIGHT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Load observations
    # -----------------------------------------------------------------

    obs = load_observations()

    # -----------------------------------------------------------------
    # Load mine weights
    # -----------------------------------------------------------------

    print("=" * 80)
    print("READING MINE AREA FRACTIONS")
    print("=" * 80)

    mine_weights = np.load(
        MINE_WEIGHT_FILE
    ).astype(np.float64)

    print(
        f"Mine weight shape: {mine_weights.shape}"
    )

    expected_shape = (
        len(MINES),
        147,
        165,
    )

    if mine_weights.shape != expected_shape:

        raise RuntimeError(
            f"Unexpected mine-weight shape:\n"
            f"  Got:      {mine_weights.shape}\n"
            f"  Expected: {expected_shape}"
        )

    if not np.all(
        np.isfinite(mine_weights)
    ):

        raise RuntimeError(
            "Mine weights contain NaN or infinite values."
        )

    print(
        f"Weight min: {mine_weights.min():.6e}"
    )

    print(
        f"Weight max: {mine_weights.max():.6e}"
    )

    print()

    # -----------------------------------------------------------------
    # Column metadata
    # -----------------------------------------------------------------

    column_metadata = build_column_metadata()

    n_obs = len(obs)
    n_columns = len(column_metadata)

    expected_columns = 13 * 27

    if n_columns != expected_columns:

        raise RuntimeError(
            f"Expected {expected_columns} columns, "
            f"got {n_columns}"
        )

    print("=" * 80)
    print("H MATRIX DIMENSIONS")
    print("=" * 80)

    print(
        f"Observations:  {n_obs}"
    )

    print(
        f"Mines:         {len(MINES)}"
    )

    print(
        f"Days:          {len(pd.date_range(START_DATE, END_DATE))}"
    )

    print(
        f"Columns:       {n_columns}"
    )

    print()

    # -----------------------------------------------------------------
    # Allocate H
    # -----------------------------------------------------------------

    H = np.zeros(
        (n_obs, n_columns),
        dtype=np.float64,
    )

    # -----------------------------------------------------------------
    # Process observations
    # -----------------------------------------------------------------

    successful = 0
    failed = 0

    failures = []

    print("=" * 80)
    print("BUILDING H MATRIX")
    print("=" * 80)
    print()

    for i, row in obs.iterrows():

        receptor = row["receptor"]

        datetime_aest = pd.Timestamp(
            row["datetime_aest"]
        )

        datetime_utc = pd.Timestamp(
            row["datetime_utc"]
        )

        source_day = pd.Timestamp(
            row["source_day"]
        )

        release_dir_name = (
            datetime_utc.strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        # -------------------------------------------------------------
        # Production FLEXPART run
        # -------------------------------------------------------------
        run_dir = (
            RUN_ROOT
            / receptor
            / release_dir_name
        )

        # -------------------------------------------------------------
        # Critical time mapping:
        #
        # Observation UTC:
        #     datetime_utc
        #
        # FLEXPART selected output:
        #     observation UTC - 1 hour
        # -------------------------------------------------------------
        target_output_utc = (
            datetime_utc
            - pd.Timedelta(hours=1)
        )

        try:

            nc_path, time_index = validate_flexpart_run(
                run_dir=run_dir,
                expected_release_utc=datetime_utc,
                target_output_utc=target_output_utc,
            )

            with Dataset(
                nc_path,
                "r",
            ) as nc:

                # -----------------------------------------------------
                # Verify exact Times again
                # -----------------------------------------------------
                flex_times = parse_flexpart_times(
                    nc
                )

                selected_time = (
                    pd.Timestamp(
                        flex_times[time_index]
                    )
                )

                if selected_time != target_output_utc:

                    raise RuntimeError(
                        "Internal FLEXPART time validation failed:\n"
                        f"Expected: {target_output_utc}\n"
                        f"Selected: {selected_time}"
                    )

                # -----------------------------------------------------
                # Read grid area
                # -----------------------------------------------------
                gridarea = get_gridarea(
                    nc
                )

                # -----------------------------------------------------
                # Verify 147 x 165 grid
                # -----------------------------------------------------
                check_grid_dimensions(
                    gridarea,
                    mine_weights,
                )

                # -----------------------------------------------------
                # Calculate ONLY 0-10 m H
                # -----------------------------------------------------
                H_mine = calculate_H_row(
                    nc=nc,
                    time_index=time_index,
                    gridarea=gridarea,
                    mine_weights=mine_weights,
                )

            # ---------------------------------------------------------
            # Determine source-day block
            #
            # Source day is the AEST observation date.
            # ---------------------------------------------------------
            day_index = (
                source_day - START_DATE
            ).days

            if day_index < 0 or day_index >= 27:

                raise RuntimeError(
                    f"Invalid source day index: {day_index}"
                )

            # ---------------------------------------------------------
            # Columns for this source day
            #
            # column = day_index * 13 + mine_index
            # ---------------------------------------------------------
            start_col = (
                day_index * len(MINES)
            )

            end_col = (
                start_col + len(MINES)
            )

            H[
                i,
                start_col:end_col,
            ] = H_mine

            successful += 1

            # ---------------------------------------------------------
            # Progress
            # ---------------------------------------------------------
            if (
                successful <= 10
                or successful % 50 == 0
                or successful == n_obs
            ):

                print(
                    f"[{successful:4d}/{n_obs}] "
                    f"{receptor:15s} "
                    f"{datetime_aest} AEST "
                    f"-> "
                    f"{datetime_utc} UTC "
                    f"-> "
                    f"{target_output_utc} UTC "
                    f"| "
                    f"idx={time_index:02d} "
                    f"| "
                    f"nonzero={np.count_nonzero(H_mine):2d} "
                    f"| "
                    f"max={H_mine.max():.6e}"
                )

        except Exception as exc:

            failed += 1

            failures.append(
                {
                    "row": int(row["row"]),
                    "receptor": receptor,
                    "datetime_aest": str(
                        datetime_aest
                    ),
                    "datetime_utc": str(
                        datetime_utc
                    ),
                    "source_day": str(
                        source_day.date()
                    ),
                    "run_dir": str(
                        run_dir
                    ),
                    "target_output_utc": str(
                        target_output_utc
                    ),
                    "error": str(exc),
                }
            )

            print()
            print("ERROR")
            print("-" * 80)
            print(
                f"Observation row: {i}"
            )
            print(
                f"Receptor:        {receptor}"
            )
            print(
                f"AEST:            {datetime_aest}"
            )
            print(
                f"UTC:             {datetime_utc}"
            )
            print(
                f"Source day:      {source_day.date()}"
            )
            print(
                f"Run directory:   {run_dir}"
            )
            print(
                f"Target FLEXPART: {target_output_utc}"
            )
            print(
                f"Error:           {exc}"
            )
            print()

            # ---------------------------------------------------------
            # IMPORTANT:
            #
            # Do not silently continue with an incomplete H matrix.
            # ---------------------------------------------------------
            raise

    # =========================================================================
    # FINAL VALIDATION
    # =========================================================================

    print()
    print("=" * 80)
    print("FINAL H MATRIX VALIDATION")
    print("=" * 80)

    print(
        f"Shape:              {H.shape}"
    )

    print(
        f"Expected shape:     ({n_obs}, 351)"
    )

    if H.shape != (
        n_obs,
        351,
    ):

        raise RuntimeError(
            f"Unexpected H shape: {H.shape}"
        )

    print(
        f"Successful rows:    {successful}"
    )

    print(
        f"Failed rows:        {failed}"
    )

    print(
        f"Total nonzero:      {np.count_nonzero(H)}"
    )

    print(
        f"Nonzero fraction:   "
        f"{np.count_nonzero(H) / H.size:.6%}"
    )

    print(
        f"Minimum H:          {H.min():.12e}"
    )

    print(
        f"Maximum H:          {H.max():.12e}"
    )

    print(
        f"Mean H:             {H.mean():.12e}"
    )

    print(
        f"Median H:           {np.median(H):.12e}"
    )

    # -------------------------------------------------------------------------
    # Check for invalid numerical values
    # -------------------------------------------------------------------------

    if not np.all(
        np.isfinite(H)
    ):

        raise RuntimeError(
            "H contains NaN or infinite values."
        )

    # -------------------------------------------------------------------------
    # Check that no H values are negative
    # -------------------------------------------------------------------------

    if np.any(H < 0):

        raise RuntimeError(
            "H contains negative values."
        )

    # =========================================================================
    # SAVE H
    # =========================================================================

    print()
    print("=" * 80)
    print("SAVING OUTPUT")
    print("=" * 80)

    H_file = (
        OUTPUT_DIR
        / "H.npy"
    )

    np.save(
        H_file,
        H,
    )

    print(
        f"H matrix:\n  {H_file}"
    )

    # -------------------------------------------------------------------------
    # Observation vector
    # -------------------------------------------------------------------------

    obs_output = obs[
        [
            "row",
            "receptor",
            "datetime_aest",
            "datetime_utc",
            "source_day",
            "observation_ppm",
        ]
    ].copy()

    obs_output["datetime_aest"] = (
        obs_output["datetime_aest"]
        .dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    obs_output["datetime_utc"] = (
        obs_output["datetime_utc"]
        .dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    obs_output["source_day"] = (
        obs_output["source_day"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    obs_file = (
        OUTPUT_DIR
        / "observation_vector.csv"
    )

    obs_output.to_csv(
        obs_file,
        index=False,
    )

    print(
        f"Observation vector:\n  {obs_file}"
    )

    # -------------------------------------------------------------------------
    # Column metadata
    # -------------------------------------------------------------------------

    column_file = (
        OUTPUT_DIR
        / "column_metadata.csv"
    )

    column_metadata.to_csv(
        column_file,
        index=False,
    )

    print(
        f"Column metadata:\n  {column_file}"
    )

    # -------------------------------------------------------------------------
    # Row metadata
    # -------------------------------------------------------------------------

    row_file = (
        OUTPUT_DIR
        / "row_metadata.csv"
    )

    obs_output.to_csv(
        row_file,
        index=False,
    )

    print(
        f"Row metadata:\n  {row_file}"
    )

    # -------------------------------------------------------------------------
    # Failure file
    # -------------------------------------------------------------------------

    if failures:

        failure_file = (
            OUTPUT_DIR
            / "failures.csv"
        )

        pd.DataFrame(
            failures
        ).to_csv(
            failure_file,
            index=False,
        )

        print(
            f"Failures:\n  {failure_file}"
        )

    # =========================================================================
    # H SUMMARY
    # =========================================================================

    summary_file = (
        OUTPUT_DIR
        / "H_summary.txt"
    )

    with open(
        summary_file,
        "w",
    ) as f:

        f.write(
            "FLEXPART-WRF Upper Hunter CH4 H matrix\n"
        )

        f.write(
            "========================================\n\n"
        )

        f.write(
            f"Period: {START_DATE.date()} to "
            f"{END_DATE.date()} AEST\n"
        )

        f.write(
            "Vertical layer: 0-10 m ONLY\n"
        )

        f.write(
            "Excluded layers: 10-100 m, 100-20000 m\n\n"
        )

        f.write(
            f"Receptors: {len(RECEPTORS)}\n"
        )

        f.write(
            f"Mines: {len(MINES)}\n"
        )

        f.write(
            f"Days: 27\n"
        )

        f.write(
            f"Observations: {n_obs}\n"
        )

        f.write(
            f"Unknowns: {n_columns}\n\n"
        )

        f.write(
            f"H shape: {H.shape}\n"
        )

        f.write(
            f"Successful rows: {successful}\n"
        )

        f.write(
            f"Failed rows: {failed}\n\n"
        )

        f.write(
            f"Nonzero elements: {np.count_nonzero(H)}\n"
        )

        f.write(
            f"Nonzero fraction: "
            f"{np.count_nonzero(H) / H.size:.12e}\n"
        )

        f.write(
            f"Min H: {H.min():.12e}\n"
        )

        f.write(
            f"Max H: {H.max():.12e}\n"
        )

        f.write(
            f"Mean H: {H.mean():.12e}\n"
        )

        f.write(
            f"Median H: {np.median(H):.12e}\n\n"
        )

        f.write(
            f"PPMV conversion factor: "
            f"{PPMV_FACTOR:.12e}\n"
        )

        f.write(
            f"0-10 m volume = GRIDAREA x "
            f"{LAYER_THICKNESS_M} m\n"
        )

    print(
        f"H summary:\n  {summary_file}"
    )

    # =========================================================================
    # MINE/DAY NONZERO SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print("NONZERO H BY MINE")
    print("=" * 80)

    for mine_index, mine in enumerate(MINES):

        cols = np.arange(
            mine_index,
            n_columns,
            len(MINES),
        )

        values = H[:, cols]

        print(
            f"{mine:5s}: "
            f"nonzero={np.count_nonzero(values):6d} "
            f"max={values.max():.6e} "
            f"sum={values.sum():.6e}"
        )

    # =========================================================================
    # DAY SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print("H SUMMARY BY SOURCE DAY")
    print("=" * 80)

    for day_index, day in enumerate(
        pd.date_range(
            START_DATE,
            END_DATE,
            freq="D",
        )
    ):

        start = (
            day_index
            * len(MINES)
        )

        end = (
            start
            + len(MINES)
        )

        block = H[
            :,
            start:end,
        ]

        print(
            f"{day.strftime('%Y-%m-%d')}: "
            f"nonzero={np.count_nonzero(block):6d} "
            f"max={block.max():.6e} "
            f"sum={block.sum():.6e}"
        )

    print()
    print("=" * 80)
    print("H MATRIX BUILD COMPLETE")
    print("=" * 80)
    print()

    print(
        f"H = {H_file}"
    )

    print(
        f"Shape = {H.shape}"
    )

    print(
        "Vertical sensitivity = 0-10 m ONLY"
    )

    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    build_H()

