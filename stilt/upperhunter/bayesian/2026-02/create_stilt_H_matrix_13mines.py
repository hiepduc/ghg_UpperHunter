#!/usr/bin/env python3

"""
===============================================================================
Create STILT H Matrix for 13 Upper Hunter Open-Cut Mines
===============================================================================

For every STILT footprint / observation:

    H[i,j] = integrated STILT response to mine j

The mine emission fields are static in time and are supplied in:

    stilt_mine_emissions_d03.nc

with variables:

    OC1, OC2, ..., OC13

Emission units:

    mol m-2 s-1

STILT footprint units:

    ppm (umol-1 m2 s)

Therefore:

    H[i,j] = sum(foot * emission * 1e6)

The resulting H values are in ppm and represent the CH4 enhancement
produced by mine j when its emission scaling factor is 1.0.

IMPORTANT
---------
- No 24-hour footprint is required.
- Each observation can have a different number of backward time slices.
- All available footprint time slices are integrated.
- Mine emissions are constant in time.
- Background CH4 is NOT included.
- One STILT footprint gives the response of all 13 mines.
- No separate STILT run is needed for each mine.

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

# Root directory containing the STILT footprints
STILT_FOOTPRINT_ROOT = (
    "/mnt/scratch_lustre/duch/stilt/upperhunter/"
    "arl_mine_jan2026/footprints"
)

# Static STILT mine emission field
MINE_EMISSION_FILE = (
    "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter/"
    "stilt_mine_emissions_d03.nc"
)

# Output directory
OUTPUT_DIR = (
    "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter/stilt"
)

# Sites to process
SITES = [
    "Singleton",
    "Muswellbrook",
    "Merriwa",
]

# 13 mines
MINES = [f"OC{i}" for i in range(1, 14)]

# January 2026 uses AEST = UTC + 10
UTC_TO_AEST_HOURS = 10


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
    """
    Load the static mine emission fields.
    """

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
        mine for mine in MINES
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
    """
    Check that STILT footprint and mine emission grids match.
    """

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
    mine_ds,
    mine_lat_name,
    mine_lon_name
):
    """
    Calculate integrated STILT contribution from all 13 mines.

    Footprint:
        ppm / (umol m-2 s-1)

    Mine emissions:
        mol m-2 s-1

    Convert:
        mol m-2 s-1 -> umol m-2 s-1

    using:

        1 mol = 1e6 umol
    """

    if "foot" not in footprint_ds.variables:
        raise RuntimeError(
            "Variable 'foot' not found in STILT footprint."
        )

    footprint = footprint_ds["foot"]

    # Footprint should be:
    #
    #     time x lat x lon
    #
    # Number of time slices can be 4, 11, 24, etc.

    footprint_values = np.asarray(
        footprint.values,
        dtype=np.float64
    )

    results = {}

    for mine in MINES:

        emission = mine_ds[mine].squeeze()

        emission_values = np.asarray(
            emission.values,
            dtype=np.float64
        )

        # Convert:
        #
        # mol m-2 s-1
        #
        # to:
        #
        # umol m-2 s-1

        emission_umol = (
            emission_values * 1.0e6
        )

        # Broadcast static emission field over footprint time.
        #
        # footprint:
        #     time x lat x lon
        #
        # emission:
        #     lat x lon
        #
        # result:
        #     time x lat x lon

        contribution = (
            footprint_values
            * emission_umol[np.newaxis, :, :]
        )

        # Integrate over:
        #
        # time
        # latitude
        # longitude

        value = np.nansum(
            contribution
        )

        results[mine] = float(value)

    # Total contribution from all mines
    results["OC_TOTAL"] = sum(
        results[mine]
        for mine in MINES
    )

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # -------------------------------------------------------------------------
    # Load mine emission fields
    # -------------------------------------------------------------------------

    mine_ds, mine_lat_name, mine_lon_name = (
        load_mine_emissions(
            MINE_EMISSION_FILE
        )
    )

    # -------------------------------------------------------------------------
    # Find all STILT footprints
    # -------------------------------------------------------------------------

    print("=" * 80)
    print("Searching for STILT footprints")
    print("=" * 80)
    print()

    footprint_files = []

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
            f"{site:15s}: {len(files):8d} footprints"
        )

        for filename in files:

            footprint_files.append(
                (site, filename)
            )

    print()
    print(
        "Total footprints found:",
        len(footprint_files)
    )
    print()

    if len(footprint_files) == 0:

        raise RuntimeError(
            "No STILT *_foot.nc files found."
        )

    # -------------------------------------------------------------------------
    # Process footprints
    # -------------------------------------------------------------------------

    rows = []

    n_processed = 0
    n_failed = 0

    for number, (site, filename) in enumerate(
        footprint_files,
        start=1
    ):

        print(
            f"[{number}/{len(footprint_files)}] "
            f"{site}: "
            f"{os.path.basename(filename)}"
        )

        # ---------------------------------------------------------------------
        # Parse filename
        # ---------------------------------------------------------------------

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

            # -----------------------------------------------------------------
            # Open footprint
            # -----------------------------------------------------------------

            footprint_ds = xr.open_dataset(
                filename
            )

            # -----------------------------------------------------------------
            # Identify coordinates
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
            # Check time dimension
            # -----------------------------------------------------------------

            if "time" not in footprint_ds.dims:

                raise RuntimeError(
                    "Footprint does not contain a time dimension."
                )

            n_time_slices = footprint_ds.sizes["time"]

            # -----------------------------------------------------------------
            # Check spatial grid
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
            # Calculate mine responses
            # -----------------------------------------------------------------

            contributions = calculate_contributions(
                footprint_ds,
                mine_ds,
                mine_lat_name,
                mine_lon_name
            )

            # -----------------------------------------------------------------
            # Convert UTC -> AEST
            # -----------------------------------------------------------------

            datetime_aest = (
                datetime_utc
                + pd.Timedelta(
                    hours=UTC_TO_AEST_HOURS
                )
            )

            # -----------------------------------------------------------------
            # Construct H-matrix row
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

            # Add OC1 ... OC13
            for mine in MINES:

                row[mine] = (
                    contributions[mine]
                )

            # Add total
            row["OC_TOTAL"] = (
                contributions["OC_TOTAL"]
            )

            rows.append(row)

            n_processed += 1

            # -----------------------------------------------------------------
            # Print diagnostics
            # -----------------------------------------------------------------

            print(
                f"  Time slices : {n_time_slices}"
            )

            print(
                f"  OC total    : "
                f"{contributions['OC_TOTAL']:.8f} ppm"
            )

            # Find dominant mine
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

        finally:

            if footprint_ds is not None:

                footprint_ds.close()

    # -------------------------------------------------------------------------
    # Close mine dataset
    # -------------------------------------------------------------------------

    mine_ds.close()

    # -------------------------------------------------------------------------
    # Check results
    # -------------------------------------------------------------------------

    if len(rows) == 0:

        raise RuntimeError(
            "No footprints were successfully processed."
        )

    # -------------------------------------------------------------------------
    # Create dataframe
    # -------------------------------------------------------------------------

    df = pd.DataFrame(
        rows
    )

    # Sort chronologically
    df = df.sort_values(
        [
            "site",
            "datetime_utc"
        ]
    ).reset_index(
        drop=True
    )

    # -------------------------------------------------------------------------
    # Full output
    # -------------------------------------------------------------------------

    output_full = os.path.join(
        OUTPUT_DIR,
        "STILT_H_matrix_mines_Jan2026.csv"
    )

    df.to_csv(
        output_full,
        index=False,
        float_format="%.10e"
    )

    # -------------------------------------------------------------------------
    # H matrix only
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # NumPy H matrix
    # -------------------------------------------------------------------------

    output_npy = os.path.join(
        OUTPUT_DIR,
        "STILT_H_matrix_mines_Jan2026.npy"
    )

    np.save(
        output_npy,
        H
    )

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("H MATRIX COMPLETE")
    print("=" * 80)

    print()
    print(
        "Footprints found    :",
        len(footprint_files)
    )

    print(
        "Successfully used   :",
        n_processed
    )

    print(
        "Failed / skipped    :",
        n_failed
    )

    print()
    print(
        "H matrix shape      :",
        H.shape[0],
        "observations x",
        H.shape[1],
        "mines"
    )

    print()
    print("Output files:")
    print()
    print("Full metadata:")
    print(
        " ",
        output_full
    )

    print()
    print("H matrix CSV:")
    print(
        " ",
        output_H
    )

    print()
    print("H matrix NumPy:")
    print(
        " ",
        output_npy
    )

    # -------------------------------------------------------------------------
    # H statistics
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Mine statistics
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Dominant mine statistics
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Number of time slices
    # -------------------------------------------------------------------------

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

