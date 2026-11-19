#!/usr/bin/env python3

import os
import numpy as np
import geopandas as gpd
from netCDF4 import Dataset

BASE_DIR = "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter"

FOOTPRINT = (
    "/mnt/scratch_lustre/duch/stilt/upperhunter/"
    "arl_mine_jan2026/footprints/Muswellbrook/footprints/"
    "202601151200_150.89297_-32.26202_5_foot.nc"
)

MINE_NAMES = [
    "OC1", "OC2", "OC3", "OC4", "OC5", "OC6",
    "OC7", "OC8", "OC9", "OC10", "OC11", "OC12", "OC13"
]

print("=" * 80)
print("STILT / MINE GEOMETRY DIAGNOSTIC")
print("=" * 80)

# ============================================================================
# READ STILT GRID
# ============================================================================

nc = Dataset(FOOTPRINT, "r")

lat = np.asarray(nc.variables["lat"][:], dtype=float)
lon = np.asarray(nc.variables["lon"][:], dtype=float)

print()
print("STILT FOOTPRINT")
print("-" * 80)

print("lat shape:", lat.shape)
print("lon shape:", lon.shape)

print("lat range:",
      np.nanmin(lat),
      "to",
      np.nanmax(lat))

print("lon range:",
      np.nanmin(lon),
      "to",
      np.nanmax(lon))

print("lat spacing:",
      np.diff(lat).min(),
      "to",
      np.diff(lat).max())

print("lon spacing:",
      np.diff(lon).min(),
      "to",
      np.diff(lon).max())

nc.close()

# ============================================================================
# CREATE STILT BOUNDING BOX
# ============================================================================

lat_spacing = np.median(np.abs(np.diff(lat)))
lon_spacing = np.median(np.abs(np.diff(lon)))

stilt_min_lat = np.min(lat) - lat_spacing / 2
stilt_max_lat = np.max(lat) + lat_spacing / 2

stilt_min_lon = np.min(lon) - lon_spacing / 2
stilt_max_lon = np.max(lon) + lon_spacing / 2

print()
print("STILT GRID BOUNDING BOX")
print("-" * 80)

print("longitude:",
      stilt_min_lon,
      "to",
      stilt_max_lon)

print("latitude :",
      stilt_min_lat,
      "to",
      stilt_max_lat)

# ============================================================================
# READ MINES
# ============================================================================

print()
print("=" * 80)
print("MINE SHAPEFILE DIAGNOSTICS")
print("=" * 80)

for mine in MINE_NAMES:

    shp = os.path.join(
        BASE_DIR,
        mine + ".shp"
    )

    print()
    print("-" * 80)
    print(mine)

    gdf = gpd.read_file(shp)

    print("File:", shp)
    print("CRS :", gdf.crs)
    print("Features:", len(gdf))

    if gdf.empty:
        print("WARNING: EMPTY SHAPEFILE")
        continue

    print("Original bounds:")
    print(" ", gdf.total_bounds)

    # ------------------------------------------------------------------------
    # Convert to WGS84
    # ------------------------------------------------------------------------

    gdf4326 = gdf.to_crs("EPSG:4326")

    print("WGS84 bounds:")
    print(" ", gdf4326.total_bounds)

    minx, miny, maxx, maxy = gdf4326.total_bounds

    # ------------------------------------------------------------------------
    # Test bounding-box overlap
    # ------------------------------------------------------------------------

    overlaps = not (
        maxx < stilt_min_lon
        or
        minx > stilt_max_lon
        or
        maxy < stilt_min_lat
        or
        miny > stilt_max_lat
    )

    print()
    print("STILT bounding-box overlap:",
          overlaps)

    if overlaps:

        print(
            "  >>> MINE IS INSIDE/OVERLAPS STILT DOMAIN"
        )

    else:

        print(
            "  >>> NO OVERLAP WITH STILT DOMAIN"
        )

# ============================================================================
# END
# ============================================================================

print()
print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

