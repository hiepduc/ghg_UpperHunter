#!/usr/bin/env python3
"""
Convert FLEXPART-WRF 3-layer footprints to the NetCDF format expected
by FLEXINVERT+.

Input:
    ~/flexpart-wrf/upperhunter_flexpart/run/hourly_obs_3layer/
        singleton/YYYYMMDD_HHMMSS/flxout_d01_*.nc
        muswellbrook/YYYYMMDD_HHMMSS/flxout_d01_*.nc
        merriwa/YYYYMMDD_HHMMSS/flxout_d01_*.nc

FLEXPART-WRF:
    CONC(Time, ageclass, releases, bottom_top, south_north, west_east)
    units = s m3 kg-1

Only FLEXPART-WRF layer 1 is used:
    CONC[:, 0, 0, 0, :, :]

Output:
    flexncdf/<receptor>/YYYYMM/
        grid_time_YYYYMMDDHHMMSS_001.nc

Output variable:
    srr(west_east, south_north, time)

IMPORTANT:
    The 24 backward hours are kept in the order given by FLEXPART-WRF.
    No summation over bottom_top is performed.
"""

from pathlib import Path
import re
import sys

import numpy as np
import xarray as xr


# ----------------------------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------------------------

INPUT_ROOT = Path(
    "/home/duch/flexpart-wrf/upperhunter_flexpart/run/hourly_obs_3layer"
)

OUTPUT_ROOT = Path(
    "/home/duch/flexpart-wrf/upperhunter_flexpart/run/flexncdf"
)

RECEPTORS = [
    "singleton",
    "muswellbrook",
    "merriwa",
]

# Set to None to process all release directories.
# Example: ("20260105_000000", "20260131_230000")
START_RELEASE = None
END_RELEASE = None
#START_RELEASE = "20260124_070000"
#END_RELEASE = "20260124_070000"

# FLEXPART-WRF layer index:
# 0 = 0-10 m
# 1 = 10-100 m
# 2 = 100-20000 m
LAYER_INDEX = 0

# FLEXPART-WRF CONC index
AGECLASS_INDEX = 0
RELEASE_INDEX = 0


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def release_datetime(dirname):
    """
    Convert release directory name YYYYMMDD_HHMMSS to string.
    """
    if not re.fullmatch(r"\d{8}_\d{6}", dirname):
        return None

    return dirname


def find_flxout(release_dir):
    """
    Find the FLEXPART-WRF output file associated with a release.
    """
    files = sorted(release_dir.glob("flxout_d01_*.nc"))

    if not files:
        return None

    if len(files) > 1:
        print(
            f"WARNING: {release_dir} contains {len(files)} "
            f"flxout files; using {files[0].name}"
        )

    return files[0]


def read_footprint(filename):
    """
    Read layer-1 FLEXPART-WRF footprint.

    Returns
    -------
    srr : ndarray
        Shape = (west_east, south_north, time)
    times : ndarray
        Character/string time values
    lon : ndarray
        XLONG grid
    lat : ndarray
        XLAT grid
    """

    print(f"    Reading: {filename.name}")

    ds = xr.open_dataset(
        filename,
        decode_times=False,
        mask_and_scale=False,
    )

    if "CONC" not in ds:
        ds.close()
        raise RuntimeError(f"CONC not found in {filename}")

    conc = ds["CONC"]

    print(f"      CONC dimensions: {conc.dims}")
    print(f"      CONC shape:      {conc.shape}")

    # --------------------------------------------------------------
    # Extract:
    #
    # CONC(Time, ageclass, releases, bottom_top,
    #      south_north, west_east)
    #
    # Only:
    #   ageclass    = 0
    #   releases    = 0
    #   bottom_top  = 0
    # --------------------------------------------------------------

    fp = conc.isel(
        ageclass=AGECLASS_INDEX,
        releases=RELEASE_INDEX,
        bottom_top=LAYER_INDEX,
    ).values

    # fp is now:
    #
    # (Time, south_north, west_east)

    if fp.ndim != 3:
        ds.close()
        raise RuntimeError(
            f"Unexpected extracted CONC shape: {fp.shape}"
        )

    # FLEXINVERT read_grid_ncdf expects:
    #
    # (west_east, south_north, time)
    #
    srr = np.transpose(fp, (2, 1, 0)).astype(np.float64)

    # --------------------------------------------------------------
    # Times
    # --------------------------------------------------------------

    if "Times" not in ds:
        ds.close()
        raise RuntimeError(f"Times variable not found in {filename}")

    times_raw = ds["Times"].values

    times = []

    for row in times_raw:
        if isinstance(row, bytes):
            t = row.decode("ascii")
        elif np.asarray(row).ndim == 0:
            t = str(row)
        else:
            t = "".join(
                x.decode("ascii") if isinstance(x, bytes) else str(x)
                for x in row
            )

        t = t.strip().replace("*", "")
        times.append(t)

    times = np.asarray(times)

    # --------------------------------------------------------------
    # Grid coordinates
    # --------------------------------------------------------------

    lon = None
    lat = None

    if "XLONG" in ds:
        lon = ds["XLONG"].values
        if lon.ndim == 3:
            lon = lon[0, :, :]

    if "XLAT" in ds:
        lat = ds["XLAT"].values
        if lat.ndim == 3:
            lat = lat[0, :, :]

    ds.close()

    return srr, times, lon, lat


def make_output(srr, times, lon, lat, output_file):
    """
    Write FLEXINVERT-compatible NetCDF.

    srr dimensions:
        west_east, south_north, time
    """

    nx = srr.shape[0]
    ny = srr.shape[1]
    nt = srr.shape[2]

    if len(times) != nt:
        raise RuntimeError(
            f"Number of times ({len(times)}) does not match "
            f"footprint time dimension ({nt})"
        )

    # FLEXINVERT's read_grid_ncdf looks specifically for:
    #
    #   time
    #   srr
    #
    # and expects srr(nx, ny, nt).

    data_vars = {
        "srr": (
            ("west_east", "south_north", "time"),
            srr,
        )
    }

    coords = {
        "west_east": np.arange(nx, dtype=np.int32),
        "south_north": np.arange(ny, dtype=np.int32),
        "time": np.arange(nt, dtype=np.float64),
    }

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
    )

    ds_out["srr"].attrs = {
        "long_name": "FLEXPART-WRF source-receptor footprint",
        "units": "s m3 kg-1",
    }

    ds_out["time"].attrs = {
        "long_name": "backward footprint time index",
        "units": "hours before release",
    }

    ds_out.attrs["source"] = "FLEXPART-WRF"
    ds_out.attrs["layer"] = "0-10 m"
    ds_out.attrs["layer_index"] = 0
    ds_out.attrs["original_variable"] = "CONC"
    ds_out.attrs["original_units"] = "s m3 kg-1"
    ds_out.attrs["footprint_times"] = ",".join(times.tolist())

    if lon is not None:
        ds_out["longitude"] = (
            ("south_north", "west_east"),
            lon,
        )
        ds_out["longitude"].attrs = {
            "long_name": "longitude",
            "units": "degrees_east",
        }

    if lat is not None:
        ds_out["latitude"] = (
            ("south_north", "west_east"),
            lat,
        )
        ds_out["latitude"].attrs = {
            "long_name": "latitude",
            "units": "degrees_north",
        }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    ds_out.to_netcdf(
        output_file,
        format="NETCDF4",
    )

    ds_out.close()


def process_receptor(receptor):
    """
    Process all release directories for one receptor.
    """

    input_dir = INPUT_ROOT / receptor

    if not input_dir.is_dir():
        print(f"WARNING: missing receptor directory: {input_dir}")
        return 0, 0

    release_dirs = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and release_datetime(d.name) is not None
    )

    if START_RELEASE is not None:
        release_dirs = [
            d for d in release_dirs
            if d.name >= START_RELEASE
        ]

    if END_RELEASE is not None:
        release_dirs = [
            d for d in release_dirs
            if d.name <= END_RELEASE
        ]

    print()
    print("=" * 80)
    print(f"RECEPTOR: {receptor}")
    print("=" * 80)
    print(f"Input : {input_dir}")
    print(f"Output: {OUTPUT_ROOT / receptor}")
    print(f"Release directories found: {len(release_dirs)}")

    if not release_dirs:
        return 0, 0

    n_ok = 0
    n_fail = 0

    for i, release_dir in enumerate(release_dirs, start=1):

        release = release_dir.name
        print()
        print(
            f"[{i}/{len(release_dirs)}] "
            f"{receptor}/{release}"
        )

        flxfile = find_flxout(release_dir)

        if flxfile is None:
            print("    WARNING: no flxout_d01_*.nc found")
            n_fail += 1
            continue

        try:
            srr, times, lon, lat = read_footprint(flxfile)

            print(f"      layer used:     0 (0-10 m)")
            print(f"      output shape:   {srr.shape}")
            print(f"      number times:   {len(times)}")

            if len(times) > 0:
                print(f"      first time:     {times[0]}")
                print(f"      last time:      {times[-1]}")

            # ------------------------------------------------------
            # Basic validation
            # ------------------------------------------------------

            if srr.shape[0] != 165 or srr.shape[1] != 147:
                print(
                    "    WARNING: unexpected grid size "
                    f"{srr.shape[0]} x {srr.shape[1]}"
                )

            if len(times) != 24:
                print(
                    f"    WARNING: expected 24 backward hours, "
                    f"found {len(times)}"
                )

            if not np.isfinite(srr).all():
                print(
                    "    WARNING: footprint contains NaN/Inf values"
                )

            # ------------------------------------------------------
            # Output filename
            # ------------------------------------------------------

            # release = YYYYMMDD_HHMMSS
            yyyymmdd = release[:8]
            hhmmss = release[9:]

            yyyymm = yyyymmdd[:6]

            output_file = (
                OUTPUT_ROOT
                / receptor
                / yyyymm
                / f"grid_time_{yyyymmdd}{hhmmss}_001.nc"
            )

            make_output(
                srr,
                times,
                lon,
                lat,
                output_file,
            )

            print(f"      wrote: {output_file}")

            n_ok += 1

        except Exception as exc:
            print(
                f"    ERROR processing {release}: {exc}"
            )
            n_fail += 1

    return n_ok, n_fail


def main():

    print()
    print("=" * 80)
    print("FLEXPART-WRF -> FLEXINVERT+ FOOTPRINT CONVERTER")
    print("=" * 80)
    print(f"Input root : {INPUT_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print()
    print("FLEXPART-WRF CONC:")
    print("  ageclass  = 0")
    print("  releases  = 0")
    print("  bottom_top= 0  [0-10 m]")
    print()
    print("Output:")
    print("  variable  = srr")
    print("  dimensions= west_east, south_north, time")
    print("  units     = s m3 kg-1")
    print()

    total_ok = 0
    total_fail = 0

    for receptor in RECEPTORS:
        ok, fail = process_receptor(receptor)
        total_ok += ok
        total_fail += fail

    print()
    print("=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)
    print(f"Successful: {total_ok}")
    print(f"Failed    : {total_fail}")
    print(f"Output    : {OUTPUT_ROOT}")
    print("=" * 80)


if __name__ == "__main__":
    main()

