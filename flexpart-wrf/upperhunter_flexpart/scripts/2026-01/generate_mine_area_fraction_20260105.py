#!/usr/bin/env python3
"""
Generate the missing 13-mine spatial weight array for the Upper Hunter
FLEXPART-WRF H-matrix workflow.

The calculation follows build_H_flexpart_13mine.py:
  - Same FLEXPART header and grid coordinates
  - Same Lambert conformal conic projection
  - Same DX/DY rectangular grid-cell geometry
  - Same OC1-OC13 shapefiles
  - Same intersection-area normalisation

Output:
  H_matrix_flexpart/test_20260105/mine_area_fraction_20260105.npy

Array shape:
  (13, south_north, west_east)

For every mine, the spatial weights sum to one.
"""

from pathlib import Path
import warnings

import numpy as np
import xarray as xr
import geopandas as gpd

from shapely.geometry import box
from shapely.prepared import prep
from pyproj import CRS, Transformer


# ============================================================
# PATHS — matched to build_H_flexpart_13mine.py
# ============================================================

BASE = Path(
    "/mnt/scratch_lustre/nzem_meim_scratch/"
    "flexpart-wrf/upperhunter_flexpart"
)

RUN_ROOT = (
    BASE / "run" / "2026-01" / "hourly_obs_3layer"
)

HEADER_FILE = (
    RUN_ROOT
    / "muswellbrook"
    / "20260114_180000"
    / "header_d01.nc"
)

MINE_DIR = Path(
    "/mnt/scratch_lustre/nzem_meim_scratch/"
    "data/emission_location"
)

OUTPUT_FILE = (
    BASE
    / "H_matrix_flexpart"
    / "test_20260105"
    / "mine_area_fraction_20260105.npy"
)

MINE_NAMES = [f"OC{i}" for i in range(1, 14)]

EXPECTED_SHAPE = (13, 147, 165)


def main():

    # --------------------------------------------------------
    # Check inputs
    # --------------------------------------------------------

    if not HEADER_FILE.is_file():
        raise FileNotFoundError(
            f"FLEXPART header not found:\n{HEADER_FILE}"
        )

    missing_shapefiles = [
        MINE_DIR / f"{mine}.shp"
        for mine in MINE_NAMES
        if not (MINE_DIR / f"{mine}.shp").is_file()
    ]

    if missing_shapefiles:
        raise FileNotFoundError(
            "The following mine shapefiles are missing:\n"
            + "\n".join(str(p) for p in missing_shapefiles)
        )

    print("=" * 78)
    print("GENERATING FLEXPART-WRF 13-MINE SPATIAL WEIGHTS")
    print("=" * 78)
    print("Header:", HEADER_FILE)
    print("Mine directory:", MINE_DIR)
    print("Output:", OUTPUT_FILE)

    # --------------------------------------------------------
    # Read the FLEXPART grid and its projection parameters
    # --------------------------------------------------------

    with xr.open_dataset(
        HEADER_FILE,
        decode_times=False
    ) as header:

        fp_nx = int(header.sizes["west_east"])
        fp_ny = int(header.sizes["south_north"])

        XLONG_FP = np.asarray(
            header["XLONG"],
            dtype=np.float64
        )

        XLAT_FP = np.asarray(
            header["XLAT"],
            dtype=np.float64
        )

        cen_lat = float(header.attrs["CEN_LAT"])
        stand_lon = float(header.attrs["STAND_LON"])
        truelat1 = float(header.attrs["TRUELAT1"])
        truelat2 = float(header.attrs["TRUELAT2"])

        dx = float(header.attrs["DX"])
        dy = float(header.attrs["DY"])

    if XLONG_FP.shape != (fp_ny, fp_nx):
        raise ValueError(
            f"Unexpected XLONG shape: {XLONG_FP.shape}; "
            f"expected {(fp_ny, fp_nx)}"
        )

    if XLAT_FP.shape != (fp_ny, fp_nx):
        raise ValueError(
            f"Unexpected XLAT shape: {XLAT_FP.shape}; "
            f"expected {(fp_ny, fp_nx)}"
        )

    if not np.all(np.isfinite(XLONG_FP)):
        raise ValueError("XLONG contains non-finite values.")

    if not np.all(np.isfinite(XLAT_FP)):
        raise ValueError("XLAT contains non-finite values.")

    if not np.isfinite(dx) or not np.isfinite(dy) or dx == 0 or dy == 0:
        raise ValueError(f"Invalid DX/DY values: DX={dx}, DY={dy}")

    print()
    print(f"FLEXPART grid: nx={fp_nx}, ny={fp_ny}")
    print(f"Header DX={dx:.3f} m, DY={dy:.3f} m")

    if (fp_ny, fp_nx) != EXPECTED_SHAPE[1:]:
        raise ValueError(
            f"Grid shape is {(fp_ny, fp_nx)}, but the H-matrix "
            f"workflow expects {EXPECTED_SHAPE[1:]}. "
            "No output file has been written."
        )

    # --------------------------------------------------------
    # Reproduce the original FLEXPART projection
    # --------------------------------------------------------

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

    xgrid, ygrid = to_xy.transform(
        XLONG_FP,
        XLAT_FP
    )

    dx_grid = np.nanmedian(np.diff(xgrid, axis=1))
    dy_grid = np.nanmedian(np.diff(ygrid, axis=0))

    print(
        f"Projected centre spacing: "
        f"dx={dx_grid:.3f} m, dy={dy_grid:.3f} m"
    )

    if not np.isclose(abs(dx_grid), abs(dx), atol=2):
        warnings.warn(
            f"Projected X spacing differs from header DX: "
            f"{dx_grid:.3f} versus {dx:.3f} m"
        )

    if not np.isclose(abs(dy_grid), abs(dy), atol=2):
        warnings.warn(
            f"Projected Y spacing differs from header DY: "
            f"{dy_grid:.3f} versus {dy:.3f} m"
        )

    # --------------------------------------------------------
    # Construct grid-cell polygons exactly as the original
    # --------------------------------------------------------

    half_dx = abs(dx) / 2.0
    half_dy = abs(dy) / 2.0

    grid_polygons = []

    for iy in range(fp_ny):
        for ix in range(fp_nx):

            xc = xgrid[iy, ix]
            yc = ygrid[iy, ix]

            cell = box(
                xc - half_dx,
                yc - half_dy,
                xc + half_dx,
                yc + half_dy
            )

            grid_polygons.append((iy, ix, cell))

    print(f"Constructed {len(grid_polygons):,} grid-cell polygons.")

    # --------------------------------------------------------
    # Read mine geometries in the FLEXPART projected CRS
    # --------------------------------------------------------

    mine_geometries = {}

    print()
    print("Reading mine shapefiles:")

    for mine in MINE_NAMES:

        shp = MINE_DIR / f"{mine}.shp"
        gdf = gpd.read_file(shp)

        if gdf.empty:
            raise ValueError(f"{mine} shapefile is empty: {shp}")

        if gdf.crs is None:
            raise ValueError(f"{mine} shapefile has no CRS: {shp}")

        gdf = gdf.to_crs(fp_crs)
        geom = gdf.geometry.unary_union

        if geom.is_empty:
            raise ValueError(f"{mine} geometry is empty after reprojection.")

        if not np.isfinite(geom.area) or geom.area <= 0:
            raise ValueError(f"{mine} has invalid or zero polygon area.")

        mine_geometries[mine] = geom

        print(f"  {mine:5s} shapefile area = {geom.area / 1e6:.4f} km2")

    # --------------------------------------------------------
    # Calculate normalised mine/grid intersection weights
    #
    # weight[m, iy, ix] =
    #   intersection_area(mine m, cell iy/ix)
    #   / total intersection area of mine m within this grid
    # --------------------------------------------------------

    weights = np.zeros(
        (len(MINE_NAMES), fp_ny, fp_nx),
        dtype=np.float64
    )

    print()
    print("Calculating mine/grid overlap weights:")

    for im, mine in enumerate(MINE_NAMES):

        geom = mine_geometries[mine]
        prepared = prep(geom)

        entries = []
        total_intersection_area = 0.0

        for iy, ix, cell in grid_polygons:

            if not prepared.intersects(cell):
                continue

            inter = geom.intersection(cell)

            if inter.is_empty:
                continue

            area = float(inter.area)

            if not np.isfinite(area) or area <= 0:
                continue

            entries.append((iy, ix, area))
            total_intersection_area += area

        if not np.isfinite(total_intersection_area):
            raise ValueError(
                f"Non-finite intersection area for {mine}"
            )

        if total_intersection_area <= 0:
            raise ValueError(
                f"No positive grid overlap found for {mine}"
            )

        for iy, ix, area in entries:
            weights[im, iy, ix] = area / total_intersection_area

        weight_sum = float(weights[im].sum())

        if not np.isclose(weight_sum, 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError(
                f"{mine} weights sum to {weight_sum}, not 1."
            )

        coverage = total_intersection_area / geom.area

        print(
            f"  {mine:5s} cells={len(entries):5d}, "
            f"grid-overlap area={total_intersection_area / 1e6:.4f} km2, "
            f"weight sum={weight_sum:.12f}, "
            f"grid coverage={100.0 * coverage:.2f}%"
        )

        if coverage < 0.99:
            warnings.warn(
                f"{mine}: less than 99% of shapefile area overlaps "
                "the FLEXPART grid. Weights are normalised over the "
                "portion of the mine that overlaps the grid."
            )

    # --------------------------------------------------------
    # Validate and save
    # --------------------------------------------------------

    if weights.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Incorrect output shape: {weights.shape}; "
            f"expected {EXPECTED_SHAPE}"
        )

    if not np.all(np.isfinite(weights)):
        raise ValueError("Output contains non-finite values.")

    if np.any(weights < 0):
        raise ValueError("Output contains negative weights.")

    row_sums = weights.sum(axis=(1, 2))

    if not np.allclose(row_sums, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError(
            f"One or more mine weights do not sum to one: {row_sums}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_FILE, weights)

    # Read it back to verify the saved file.
    check = np.load(OUTPUT_FILE)

    if check.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            f"Saved file has unexpected shape: {check.shape}"
        )

    print()
    print("=" * 78)
    print("SUCCESS")
    print("=" * 78)
    print("Saved:", OUTPUT_FILE)
    print("Shape:", check.shape)
    print("Data type:", check.dtype)
    print(f"Minimum weight: {check.min():.8e}")
    print(f"Maximum weight: {check.max():.8e}")
    print("Weight sums by mine:")

    for mine, total in zip(MINE_NAMES, check.sum(axis=(1, 2))):
        print(f"  {mine:5s} {total:.12f}")


if __name__ == "__main__":
    main()
