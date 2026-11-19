#!/usr/bin/env python3

"""
===============================================================================
Create STILT H Matrix for 13 Upper Hunter Open-Cut Mines
===============================================================================

For every unique STILT footprint / observation:

    H[i,j] = integrated STILT response to mine j

The mine emission fields are static in:

    stilt_mine_emissions_d03.nc

with variables:

    OC1, OC2, ..., OC13

Emission units:

    mol m-2 s-1

STILT footprint units:

    ppm / (umol m-2 s-1)

Therefore:

    H[i,j] = sum(foot * emission * 1e6)

The resulting H values are in ppm.

IMPORTANT
---------
- No 24-hour footprint is required.
- Each observation can have a different number of backward time slices.
- All available footprint time slices are integrated.
- Mine emissions are constant in time.
- Background CH4 is NOT included.
- One STILT footprint gives the response of all 13 mines.
- No separate STILT run is needed for each mine.

DUPLICATE HANDLING
------------------
The STILT footprint directory may contain duplicate paths, symlinks,
or repeated copies of the same receptor footprint.

This script therefore:

1. Removes duplicate physical file paths using realpath().
2. Checks for duplicate receptor/time combinations.
3. If duplicate rows are numerically identical, keeps only one.
4. If duplicate rows differ, stops with an error.

This guarantees that the final H matrix contains at most one row for
each unique site/receptor/time.
===============================================================================
"""

import os
import re
import glob

import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# USER SETTINGS
# =============================================================================

STILT_FOOTPRINT_ROOT = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/arl_mine_2026/2026-01/footprints"
)

MINE_EMISSION_FILE = (
    "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter/"
    "stilt_mine_emissions_d03.nc"
)

OUTPUT_DIR = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/bayesian/2026-01/H-matrix"
)

SITES = [
    "Singleton",
    "Muswellbrook",
    "Merriwa",
]

MINES = [f"OC{i}" for i in range(1, 14)]

# January 2026:
# UTC + 10 hours = AEST
UTC_TO_AEST_HOURS = 10

# Numerical tolerance used when comparing duplicate H rows.
DUPLICATE_RTOL = 1.0e-10
DUPLICATE_ATOL = 1.0e-14


# =============================================================================
# FUNCTIONS
# =============================================================================

def parse_footprint_filename(filename):
    """
    Parse STILT footprint filename.

    Example:

        202601151200_150.89297_-32.26202_5_foot.nc

    Returns:
        datetime_utc
        longitude
        latitude
        receptor_height
    """

    base = os.path.basename(filename)

    pattern = (
        r"^(\d{12})_"
        r"([-+]?\d+(?:\.\d+)?)_"
        r"([-+]?\d+(?:\.\d+)?)_"
        r"([-+]?\d+(?:\.\d+)?)_"
        r"foot\.nc$"
    )

    match = re.match(pattern, base)

    if match is None:
        return None

    timestamp = match.group(1)

    lon = float(match.group(2))
    lat = float(match.group(3))
    height = float(match.group(4))

    datetime_utc = pd.to_datetime(
        timestamp,
        format="%Y%m%d%H%M",
        utc=True
    )

    return datetime_utc, lon, lat, height


def get_coordinate_name(ds, candidates):
    """
    Find latitude or longitude coordinate name.
    """

    for name in candidates:

        if name in ds.coords:
            return name

    for name in candidates:

        if name in ds.variables:
            return name

    return None


def load_mine_emissions(filename):

    print("=" * 80)
    print("Loading mine emission field")
    print("=" * 80)

    print(filename)
    print()

    ds = xr.open_dataset(filename)

    lat_name = get_coordinate_name(
        ds,
        ["lat", "latitude", "XLAT"]
    )

    lon_name = get_coordinate_name(
        ds,
        ["lon", "longitude", "XLONG"]
    )

    if lat_name is None:

        raise RuntimeError(
            "Could not find latitude coordinate in mine emission file."
        )

    if lon_name is None:

        raise RuntimeError(
            "Could not find longitude coordinate in mine emission file."
        )

    missing = [
        mine
        for mine in MINES
        if mine not in ds.variables
    ]

    if missing:

        raise RuntimeError(
            "Missing mine variables: "
            + ", ".join(missing)
        )

    print("Latitude coordinate :", lat_name)
    print("Longitude coordinate:", lon_name)

    print(
        "Grid:",
        ds.sizes[lat_name],
        "x",
        ds.sizes[lon_name]
    )

    print()

    return ds, lat_name, lon_name


def check_grid_match(
    footprint_ds,
    mine_ds,
    footprint_lat_name,
    footprint_lon_name,
    mine_lat_name,
    mine_lon_name
):

    if (
        footprint_ds.sizes[footprint_lat_name]
        != mine_ds.sizes[mine_lat_name]
    ):

        return False

    if (
        footprint_ds.sizes[footprint_lon_name]
        != mine_ds.sizes[mine_lon_name]
    ):

        return False

    footprint_lat = np.asarray(
        footprint_ds[footprint_lat_name].values
    )

    footprint_lon = np.asarray(
        footprint_ds[footprint_lon_name].values
    )

    mine_lat = np.asarray(
        mine_ds[mine_lat_name].values
    )

    mine_lon = np.asarray(
        mine_ds[mine_lon_name].values
    )

    if not np.allclose(
        footprint_lat,
        mine_lat,
        rtol=0,
        atol=1e-7
    ):

        return False

    if not np.allclose(
        footprint_lon,
        mine_lon,
        rtol=0,
        atol=1e-7
    ):

        return False

    return True


def calculate_contributions(
    footprint_ds,
    mine_ds
):

    """
    Calculate integrated STILT contribution from all 13 mines.

    Footprint:
        ppm / (umol m-2 s-1)

    Mine emissions:
        mol m-2 s-1

    Conversion:

        mol m-2 s-1 -> umol m-2 s-1

        1 mol = 1e6 umol
    """

    if "foot" not in footprint_ds.variables:

        raise RuntimeError(
            "Variable 'foot' not found in STILT footprint."
        )

    footprint_values = np.asarray(
        footprint_ds["foot"].values,
        dtype=np.float64
    )

    results = {}

    for mine in MINES:

        emission = mine_ds[mine].squeeze()

        emission_values = np.asarray(
            emission.values,
            dtype=np.float64
        )

        emission_umol = (
            emission_values * 1.0e6
        )

        contribution = (
            footprint_values
            * emission_umol[np.newaxis, :, :]
        )

        value = np.nansum(
            contribution
        )

        results[mine] = float(value)

    results["OC_TOTAL"] = sum(
        results[mine]
        for mine in MINES
    )

    return results


def duplicate_rows_identical(row1, row2):

    """
    Compare the H values and important metadata of two rows.
    """

    fields = [
        "receptor_lat",
        "receptor_lon",
        "receptor_height_m",
        "n_time_slices",
        "foot_max",
        "foot_sum",
    ] + MINES + ["OC_TOTAL"]

    for field in fields:

        a = float(row1[field])
        b = float(row2[field])

        if not np.isclose(
            a,
            b,
            rtol=DUPLICATE_RTOL,
            atol=DUPLICATE_ATOL
        ):

            return False

    return True


# =============================================================================
# MAIN
# =============================================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # =========================================================================
    # Load mine emissions
    # =========================================================================

    mine_ds, mine_lat_name, mine_lon_name = (
        load_mine_emissions(
            MINE_EMISSION_FILE
        )
    )

    # =========================================================================
    # Find STILT footprints
    # =========================================================================

    print("=" * 80)
    print("Searching for STILT footprints")
    print("=" * 80)
    print()

    footprint_files_raw = []

    for site in SITES:

        site_directory = os.path.join(
            STILT_FOOTPRINT_ROOT,
            site
        )

        files = sorted(
            glob.glob(
                os.path.join(
                    site_directory,
                    "**",
                    "*_foot.nc"
                ),
                recursive=True
            )
        )

        print(
            f"{site:15s}: {len(files):8d} footprint paths"
        )

        for filename in files:

            footprint_files_raw.append(
                (site, filename)
            )

    print()

    print(
        "Total footprint paths found:",
        len(footprint_files_raw)
    )

    print()

    # =========================================================================
    # Remove duplicate physical files
    # =========================================================================

    print("=" * 80)
    print("Removing duplicate physical footprint paths")
    print("=" * 80)
    print()

    unique_files = []
    seen_realpaths = set()

    duplicate_physical_paths = 0

    for site, filename in footprint_files_raw:

        realpath = os.path.realpath(
            filename
        )

        if realpath in seen_realpaths:

            duplicate_physical_paths += 1

            print(
                "Duplicate physical file skipped:"
            )

            print(
                "  path    :",
                filename
            )

            print(
                "  realpath:",
                realpath
            )

            continue

        seen_realpaths.add(
            realpath
        )

        unique_files.append(
            (site, filename)
        )

    print()

    print(
        "Unique physical footprints:",
        len(unique_files)
    )

    print(
        "Duplicate physical paths skipped:",
        duplicate_physical_paths
    )

    print()

    if len(unique_files) == 0:

        raise RuntimeError(
            "No unique STILT footprints found."
        )

    # =========================================================================
    # Process footprints
    # =========================================================================

    print("=" * 80)
    print("Processing unique STILT footprints")
    print("=" * 80)
    print()

    rows = []

    # Dictionary:
    #
    # key = site + receptor datetime + receptor location
    #
    # value = previously generated row
    #
    seen_receptors = {}

    n_processed = 0
    n_failed = 0

    n_duplicate_receptors = 0
    n_conflicting_duplicates = 0

    for number, (site, filename) in enumerate(
        unique_files,
        start=1
    ):

        print(
            f"[{number}/{len(unique_files)}] "
            f"{site}: "
            f"{os.path.basename(filename)}"
        )

        parsed = parse_footprint_filename(
            filename
        )

        if parsed is None:

            print(
                "  WARNING: filename format not recognised"
            )

            n_failed += 1

            continue

        (
            datetime_utc,
            filename_lon,
            filename_lat,
            receptor_height
        ) = parsed

        footprint_ds = None

        try:

            footprint_ds = xr.open_dataset(
                filename
            )

            # -----------------------------------------------------------------
            # Coordinates
            # -----------------------------------------------------------------

            footprint_lat_name = (
                get_coordinate_name(
                    footprint_ds,
                    ["lat", "latitude"]
                )
            )

            footprint_lon_name = (
                get_coordinate_name(
                    footprint_ds,
                    ["lon", "longitude"]
                )
            )

            if footprint_lat_name is None:

                raise RuntimeError(
                    "Could not identify footprint latitude."
                )

            if footprint_lon_name is None:

                raise RuntimeError(
                    "Could not identify footprint longitude."
                )

            # -----------------------------------------------------------------
            # Time dimension
            # -----------------------------------------------------------------

            if "time" not in footprint_ds.dims:

                raise RuntimeError(
                    "Footprint does not contain a time dimension."
                )

            n_time_slices = (
                footprint_ds.sizes["time"]
            )

            # -----------------------------------------------------------------
            # Grid check
            # -----------------------------------------------------------------

            grid_ok = check_grid_match(
                footprint_ds,
                mine_ds,
                footprint_lat_name,
                footprint_lon_name,
                mine_lat_name,
                mine_lon_name
            )

            if not grid_ok:

                raise RuntimeError(
                    "STILT footprint grid does not match "
                    "mine emission grid."
                )

            # -----------------------------------------------------------------
            # Footprint diagnostics
            # -----------------------------------------------------------------

            footprint_values = np.asarray(
                footprint_ds["foot"].values,
                dtype=np.float64
            )

            foot_max = np.nanmax(
                footprint_values
            )

            foot_sum = np.nansum(
                footprint_values
            )

            # -----------------------------------------------------------------
            # Mine contributions
            # -----------------------------------------------------------------

            contributions = calculate_contributions(
                footprint_ds,
                mine_ds
            )

            # -----------------------------------------------------------------
            # AEST
            # -----------------------------------------------------------------

            datetime_aest = (
                datetime_utc
                + pd.Timedelta(
                    hours=UTC_TO_AEST_HOURS
                )
            )

            # -----------------------------------------------------------------
            # Construct row
            # -----------------------------------------------------------------

            row = {

                "site": site,

                "datetime_utc":
                    datetime_utc.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "datetime_aest":
                    datetime_aest.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "receptor_lat":
                    filename_lat,

                "receptor_lon":
                    filename_lon,

                "receptor_height_m":
                    receptor_height,

                "n_time_slices":
                    n_time_slices,

                "foot_max":
                    foot_max,

                "foot_sum":
                    foot_sum,
            }

            for mine in MINES:

                row[mine] = (
                    contributions[mine]
                )

            row["OC_TOTAL"] = (
                contributions["OC_TOTAL"]
            )

            # -----------------------------------------------------------------
            # Duplicate receptor check
            # -----------------------------------------------------------------

            receptor_key = (
                site,
                datetime_utc,
                round(filename_lon, 8),
                round(filename_lat, 8),
                round(receptor_height, 6),
            )

            if receptor_key in seen_receptors:

                previous_row = seen_receptors[
                    receptor_key
                ]

                n_duplicate_receptors += 1

                if duplicate_rows_identical(
                    previous_row,
                    row
                ):

                    print(
                        "  DUPLICATE: identical receptor/H row "
                        "-> skipped"
                    )

                    continue

                else:

                    n_conflicting_duplicates += 1

                    print()
                    print(
                        "ERROR: conflicting duplicate receptor!"
                    )

                    print(
                        "  Site:",
                        site
                    )

                    print(
                        "  UTC:",
                        datetime_utc
                    )

                    print(
                        "  Current file:",
                        filename
                    )

                    raise RuntimeError(
                        "Two different STILT footprints exist "
                        "for the same site/receptor/time and "
                        "produce different H values. "
                        "The H matrix cannot safely choose between them."
                    )

            # -----------------------------------------------------------------
            # Store unique receptor
            # -----------------------------------------------------------------

            seen_receptors[
                receptor_key
            ] = row

            rows.append(
                row
            )

            n_processed += 1

            print(
                f"  Time slices : {n_time_slices}"
            )

            print(
                f"  OC total    : "
                f"{contributions['OC_TOTAL']:.8f} ppm"
            )

            dominant_mine = max(
                MINES,
                key=lambda m: contributions[m]
            )

            print(
                f"  Dominant    : "
                f"{dominant_mine} = "
                f"{contributions[dominant_mine]:.8f} ppm"
            )

        except Exception as exc:

            print(
                "  ERROR:",
                str(exc)
            )

            n_failed += 1

            # Re-raise conflicting duplicates.
            if "conflicting duplicate receptor" in str(
                exc
            ).lower():

                raise

        finally:

            if footprint_ds is not None:

                footprint_ds.close()

    # =========================================================================
    # Close mine dataset
    # =========================================================================

    mine_ds.close()

    # =========================================================================
    # Check results
    # =========================================================================

    if len(rows) == 0:

        raise RuntimeError(
            "No footprints were successfully processed."
        )

    # =========================================================================
    # DataFrame
    # =========================================================================

    df = pd.DataFrame(
        rows
    )

    df = df.sort_values(
        [
            "site",
            "datetime_utc"
        ]
    ).reset_index(
        drop=True
    )

    # =========================================================================
    # Final duplicate check
    # =========================================================================

    final_counts = (
        df.groupby(
            [
                "site",
                "datetime_utc"
            ]
        )
        .size()
    )

    remaining_duplicates = (
        final_counts[
            final_counts > 1
        ]
    )

    if len(remaining_duplicates) > 0:

        raise RuntimeError(
            "Final H matrix still contains duplicate "
            "site/datetime rows."
        )

    # =========================================================================
    # Full output
    # =========================================================================

    output_full = os.path.join(
        OUTPUT_DIR,
        "STILT_H_matrix_mines_Jan2026.csv"
    )

    df.to_csv(
        output_full,
        index=False,
        float_format="%.10e"
    )

    # =========================================================================
    # H matrix only
    # =========================================================================

    H = df[
        MINES
    ].to_numpy(
        dtype=np.float64
    )

    output_H = os.path.join(
        OUTPUT_DIR,
        "STILT_H_matrix_mines_only_Jan2026.csv"
    )

    H_df = df[
        [
            "site",
            "datetime_utc",
            "datetime_aest"
        ]
        + MINES
        + ["OC_TOTAL"]
    ]

    H_df.to_csv(
        output_H,
        index=False,
        float_format="%.10e"
    )

    # =========================================================================
    # NumPy H matrix
    # =========================================================================

    output_npy = os.path.join(
        OUTPUT_DIR,
        "STILT_H_matrix_mines_Jan2026.npy"
    )

    np.save(
        output_npy,
        H
    )

    # =========================================================================
    # Summary
    # =========================================================================

    print()
    print("=" * 80)
    print("H MATRIX COMPLETE")
    print("=" * 80)

    print()

    print(
        "Footprint paths found       :",
        len(footprint_files_raw)
    )

    print(
        "Unique physical footprints  :",
        len(unique_files)
    )

    print(
        "Duplicate physical paths    :",
        duplicate_physical_paths
    )

    print(
        "Unique H rows               :",
        len(df)
    )

    print(
        "Duplicate receptor rows     :",
        n_duplicate_receptors
    )

    print(
        "Conflicting duplicate rows  :",
        n_conflicting_duplicates
    )

    print(
        "Failed / skipped            :",
        n_failed
    )

    print()

    print(
        "H matrix shape              :",
        H.shape[0],
        "observations x",
        H.shape[1],
        "mines"
    )

    print()

    print("Rows by site:")

    print(
        df["site"].value_counts()
        .sort_index()
    )

    print()

    print("Output files:")

    print()
    print(
        "Full metadata:"
    )

    print(
        " ",
        output_full
    )

    print()
    print(
        "H matrix CSV:"
    )

    print(
        " ",
        output_H
    )

    print()
    print(
        "H matrix NumPy:"
    )

    print(
        " ",
        output_npy
    )

    # =========================================================================
    # H statistics
    # =========================================================================

    print()
    print("=" * 80)
    print("H MATRIX STATISTICS")
    print("=" * 80)

    print()

    print("Total mine enhancement:")

    print(
        f"  minimum = "
        f"{df['OC_TOTAL'].min():.8e} ppm"
    )

    print(
        f"  maximum = "
        f"{df['OC_TOTAL'].max():.8e} ppm"
    )

    print(
        f"  mean    = "
        f"{df['OC_TOTAL'].mean():.8e} ppm"
    )

    print(
        f"  median  = "
        f"{df['OC_TOTAL'].median():.8e} ppm"
    )

    # =========================================================================
    # Mine statistics
    # =========================================================================

    print()
    print("Individual mine H values:")

    for mine in MINES:

        values = df[mine].to_numpy(
            dtype=np.float64
        )

        print(
            f"  {mine:5s} "
            f"min={np.nanmin(values):.6e} "
            f"max={np.nanmax(values):.6e} "
            f"mean={np.nanmean(values):.6e}"
        )

    # =========================================================================
    # Dominant mine statistics
    # =========================================================================

    mine_array = df[
        MINES
    ].to_numpy(
        dtype=np.float64
    )

    dominant_indices = np.argmax(
        mine_array,
        axis=1
    )

    dominant_mines = np.asarray(
        MINES,
        dtype=object
    )[dominant_indices]

    dominant_counts = pd.Series(
        dominant_mines
    ).value_counts()

    print()
    print(
        "Dominant mine for each observation:"
    )

    for mine, count in dominant_counts.items():

        percentage = (
            100.0
            * count
            / len(df)
        )

        print(
            f"  {mine:5s}: "
            f"{count:6d} observations "
            f"({percentage:6.2f}%)"
        )

    # =========================================================================
    # Time-slice distribution
    # =========================================================================

    print()
    print(
        "Backward footprint time-slice distribution:"
    )

    print(
        df[
            "n_time_slices"
        ].value_counts().sort_index()
    )

    print()
    print("Done.")


if __name__ == "__main__":

    main()

