#!/usr/bin/env python3

"""
Convert FLEXPART-WRF Lambert-grid footprints to a regular lat/lon grid
for the unmodified FLEXINVERT+ code.

Input:
    FLEXPART-WRF 3-layer footprints:
        hourly_obs_3layer/
            singleton/
            muswellbrook/
            merriwa/

Uses:
    CONC[:, 0, 0, 0, :, :]       -> lowest 0-10 m layer
    GRIDAREA                     -> actual WRF/FLEXPART cell area
    XLAT_CORNER / XLONG_CORNER   -> source cell geometry

Method:
    Source integrated quantity:
        Q_source = footprint * GRIDAREA

    Redistribute Q_source according to geometric overlap:
        overlap_area / source_polygon_area

    Then:
        target_footprint =
            target_integrated_quantity / target_cell_area

The target grid is a regular geographic lon/lat grid required by
the existing FLEXINVERT+ code.

No FLEXINVERT scaling is applied here.

Output:
    flexncdf/<receptor>/YYYYMM/grid_time_<release>_001.nc

Output variable:
    srr(west_east, south_north, time)

Units:
    s m3 kg-1
"""

import os
import glob
import time as timer

import numpy as np
from netCDF4 import Dataset

from pyproj import CRS, Transformer
from shapely.geometry import Polygon
from shapely.strtree import STRtree


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_ROOT = (
    "/home/duch/flexpart-wrf/"
    "upperhunter_flexpart/run/hourly_obs_3layer"
)

OUTPUT_ROOT = (
    "/home/duch/flexpart-wrf/"
    "upperhunter_flexpart/run/flexncdf"
)

RECEPTORS = [
    "singleton",
    "muswellbrook",
    "merriwa",
]

# ------------------------------------------------------------
# Target regular lat/lon grid
# ------------------------------------------------------------

LON_MIN = 150.22
LON_MAX = 152.02

LAT_MIN = -33.22
LAT_MAX = -31.88

DX_DEG = 0.01
DY_DEG = 0.01


# ============================================================
# TARGET GRID
# ============================================================

def get_target_grid():

    lon_edges = np.arange(
        LON_MIN,
        LON_MAX + DX_DEG * 0.5,
        DX_DEG
    )

    lat_edges = np.arange(
        LAT_MIN,
        LAT_MAX + DY_DEG * 0.5,
        DY_DEG
    )

    # Make sure final edge is included
    if lon_edges[-1] < LON_MAX - 1.e-10:
        lon_edges = np.append(
            lon_edges,
            LON_MAX
        )

    if lat_edges[-1] < LAT_MAX - 1.e-10:
        lat_edges = np.append(
            lat_edges,
            LAT_MAX
        )

    lon_centres = (
        0.5
        * (
            lon_edges[:-1]
            + lon_edges[1:]
        )
    )

    lat_centres = (
        0.5
        * (
            lat_edges[:-1]
            + lat_edges[1:]
        )
    )

    return (
        lon_edges,
        lat_edges,
        lon_centres,
        lat_centres
    )


# ============================================================
# PROJECTION
# ============================================================

def make_laea_transformer(
    lon0,
    lat0
):

    """
    Local Lambert Azimuthal Equal Area projection.

    Used only internally for polygon-area calculations.
    """

    laea = CRS.from_proj4(
        f"+proj=laea "
        f"+lat_0={lat0} "
        f"+lon_0={lon0} "
        f"+datum=WGS84 "
        f"+units=m "
        f"+no_defs"
    )

    wgs84 = CRS.from_epsg(4326)

    transformer = Transformer.from_crs(
        wgs84,
        laea,
        always_xy=True
    )

    return transformer


# ============================================================
# BUILD ONE POLYGON
# ============================================================

def polygon_from_lonlat(
    lon0,
    lat0,
    lon1,
    lat1,
    lon2,
    lat2,
    lon3,
    lat3,
    transformer
):

    lon = np.array(
        [
            lon0,
            lon1,
            lon2,
            lon3
        ],
        dtype=float
    )

    lat = np.array(
        [
            lat0,
            lat1,
            lat2,
            lat3
        ],
        dtype=float
    )

    x, y = transformer.transform(
        lon,
        lat
    )

    poly = Polygon(
        zip(x, y)
    )

    if not poly.is_valid:
        poly = poly.buffer(0)

    return poly


# ============================================================
# SOURCE WRF GEOMETRY
# ============================================================

def build_source_geometry(
    lat_corner,
    lon_corner,
    transformer
):

    """
    XLAT_CORNER / XLONG_CORNER are treated as the lower-left
    geographic corner of each WRF cell.

    A complete boundary grid is reconstructed with one extra
    row and column.
    """

    ny, nx = lat_corner.shape

    # --------------------------------------------------------
    # Complete cell-boundary arrays
    # --------------------------------------------------------

    lat_bnd = np.empty(
        (ny + 1, nx + 1),
        dtype=float
    )

    lon_bnd = np.empty(
        (ny + 1, nx + 1),
        dtype=float
    )

    # Existing corners
    lat_bnd[:-1, :-1] = lat_corner
    lon_bnd[:-1, :-1] = lon_corner

    # --------------------------------------------------------
    # Extrapolate final column
    # --------------------------------------------------------

    lat_dx = (
        lat_corner[:, -1]
        - lat_corner[:, -2]
    )

    lon_dx = (
        lon_corner[:, -1]
        - lon_corner[:, -2]
    )

    lat_bnd[:-1, -1] = (
        lat_corner[:, -1]
        + lat_dx
    )

    lon_bnd[:-1, -1] = (
        lon_corner[:, -1]
        + lon_dx
    )

    # --------------------------------------------------------
    # Extrapolate final row
    # --------------------------------------------------------

    lat_dy = (
        lat_corner[-1, :]
        - lat_corner[-2, :]
    )

    lon_dy = (
        lon_corner[-1, :]
        - lon_corner[-2, :]
    )

    lat_bnd[-1, :-1] = (
        lat_corner[-1, :]
        + lat_dy
    )

    lon_bnd[-1, :-1] = (
        lon_corner[-1, :]
        + lon_dy
    )

    # --------------------------------------------------------
    # Bottom-right corner
    # --------------------------------------------------------

    lat_bnd[-1, -1] = (
        lat_bnd[-1, -2]
        + lat_bnd[-2, -1]
        - lat_bnd[-2, -2]
    )

    lon_bnd[-1, -1] = (
        lon_bnd[-1, -2]
        + lon_bnd[-2, -1]
        - lon_bnd[-2, -2]
    )

    # --------------------------------------------------------
    # Build polygons
    # --------------------------------------------------------

    polygons = []

    valid = np.zeros(
        (ny, nx),
        dtype=bool
    )

    for iy in range(ny):

        for ix in range(nx):

            poly = polygon_from_lonlat(
                lon_bnd[iy, ix],
                lat_bnd[iy, ix],

                lon_bnd[iy, ix + 1],
                lat_bnd[iy, ix + 1],

                lon_bnd[iy + 1, ix + 1],
                lat_bnd[iy + 1, ix + 1],

                lon_bnd[iy + 1, ix],
                lat_bnd[iy + 1, ix],

                transformer
            )

            if (
                poly.is_empty
                or poly.area <= 0
            ):

                polygons.append(None)

            else:

                polygons.append(poly)

                valid[iy, ix] = True

    return (
        polygons,
        valid,
        lon_bnd,
        lat_bnd
    )


# ============================================================
# TARGET POLYGONS
# ============================================================

def build_target_polygons(
    lon_edges,
    lat_edges,
    transformer
):

    nx = len(lon_edges) - 1
    ny = len(lat_edges) - 1

    polygons = []

    areas = np.zeros(
        nx * ny,
        dtype=float
    )

    for iy in range(ny):

        for ix in range(nx):

            poly = polygon_from_lonlat(
                lon_edges[ix],
                lat_edges[iy],

                lon_edges[ix + 1],
                lat_edges[iy],

                lon_edges[ix + 1],
                lat_edges[iy + 1],

                lon_edges[ix],
                lat_edges[iy + 1],

                transformer
            )

            polygons.append(poly)

            areas[
                iy * nx + ix
            ] = poly.area

    return (
        polygons,
        areas
    )


# ============================================================
# SOURCE GEOMETRY DIAGNOSTIC
# ============================================================

def diagnostic_source_geometry(
    source_polygons,
    source_valid,
    gridarea
):

    geom_area = np.zeros_like(
        gridarea,
        dtype=float
    )

    valid_flat = source_valid.ravel()

    geom_flat = geom_area.ravel()

    for i, poly in enumerate(
        source_polygons
    ):

        if (
            poly is not None
            and valid_flat[i]
        ):

            geom_flat[i] = poly.area

    mask = (
        source_valid
        & np.isfinite(gridarea)
        & (gridarea > 0)
        & (geom_area > 0)
    )

    ratio = (
        geom_area[mask]
        / gridarea[mask]
    )

    print()
    print("=" * 70)
    print("SOURCE GEOMETRY vs GRIDAREA")
    print("=" * 70)

    print()
    print("GRIDAREA:")
    print(
        "  min  :",
        gridarea[mask].min()
    )

    print(
        "  max  :",
        gridarea[mask].max()
    )

    print(
        "  mean :",
        gridarea[mask].mean()
    )

    print(
        "  total:",
        gridarea[mask].sum()
    )

    print()
    print("Geometric polygon area:")

    print(
        "  min  :",
        geom_area[mask].min()
    )

    print(
        "  max  :",
        geom_area[mask].max()
    )

    print(
        "  mean :",
        geom_area[mask].mean()
    )

    print(
        "  total:",
        geom_area[mask].sum()
    )

    print()
    print("Geometric area / GRIDAREA:")

    print(
        "  min  :",
        ratio.min()
    )

    print(
        "  max  :",
        ratio.max()
    )

    print(
        "  mean :",
        ratio.mean()
    )

    print()
    print("Percent difference:")

    print(
        "  min  :",
        100.0 * (ratio.min() - 1.0)
    )

    print(
        "  max  :",
        100.0 * (ratio.max() - 1.0)
    )

    print(
        "  mean :",
        100.0 * (ratio.mean() - 1.0)
    )

    return geom_area


# ============================================================
# BUILD OVERLAP MATRIX
# ============================================================

def build_overlap_matrix(
    source_polygons,
    source_valid,
    target_polygons,
    source_shape
):

    """
    Build source-target overlap relationships.

    For each overlap:

        fraction =
            overlap_area / source_geometry_area
    """

    ny, nx = source_shape

    valid_indices = np.flatnonzero(
        source_valid.ravel()
    )

    source_valid_polys = [
        source_polygons[i]
        for i in valid_indices
    ]

    tree = STRtree(
        source_valid_polys
    )

    overlap_targets = []
    overlap_sources = []
    overlap_areas = []

    print()
    print("=" * 70)
    print("BUILDING SOURCE-TARGET OVERLAP MATRIX")
    print("=" * 70)

    print()
    print(
        "Source cells:",
        len(source_valid_polys)
    )

    print(
        "Target cells:",
        len(target_polygons)
    )

    for target_idx, target_poly in enumerate(
        target_polygons
    ):

        candidates = tree.query(
            target_poly
        )

        for candidate in candidates:

            # Shapely 2.x returns integer indices
            if isinstance(
                candidate,
                (int, np.integer)
            ):

                local_idx = int(
                    candidate
                )

                source_flat_idx = int(
                    valid_indices[local_idx]
                )

                source_poly = (
                    source_valid_polys[
                        local_idx
                    ]
                )

            # Compatibility fallback
            else:

                source_poly = candidate

                try:

                    local_idx = (
                        source_valid_polys.index(
                            source_poly
                        )
                    )

                except ValueError:

                    continue

                source_flat_idx = int(
                    valid_indices[
                        local_idx
                    ]
                )

            if not source_poly.intersects(
                target_poly
            ):
                continue

            intersection = (
                source_poly.intersection(
                    target_poly
                )
            )

            if intersection.is_empty:
                continue

            area = intersection.area

            if area <= 0:
                continue

            overlap_targets.append(
                target_idx
            )

            overlap_sources.append(
                source_flat_idx
            )

            overlap_areas.append(
                area
            )

    overlap_targets = np.asarray(
        overlap_targets,
        dtype=np.int64
    )

    overlap_sources = np.asarray(
        overlap_sources,
        dtype=np.int64
    )

    overlap_areas = np.asarray(
        overlap_areas,
        dtype=np.float64
    )

    print()
    print(
        "Overlap pairs:",
        len(overlap_areas)
    )

    return (
        overlap_targets,
        overlap_sources,
        overlap_areas
    )


# ============================================================
# BUILD COMPLETE REMAPPING
# ============================================================

def build_remapping(
    header_file,
    lon_edges,
    lat_edges
):

    with Dataset(
        header_file
    ) as nc:

        lat_corner = np.asarray(
            nc.variables[
                "XLAT_CORNER"
            ][:],
            dtype=float
        )

        lon_corner = np.asarray(
            nc.variables[
                "XLONG_CORNER"
            ][:],
            dtype=float
        )

        gridarea = np.asarray(
            nc.variables[
                "GRIDAREA"
            ][:],
            dtype=float
        )

        ny_source, nx_source = (
            gridarea.shape
        )

        cen_lat = float(
            nc.getncattr(
                "CEN_LAT"
            )
        )

        cen_lon = float(
            nc.getncattr(
                "CEN_LON"
            )
        )

    print()
    print("=" * 70)
    print("BUILDING CONSERVATIVE REMAPPING")
    print("=" * 70)

    print()
    print("Source grid:")
    print(
        "  nx =",
        nx_source
    )

    print(
        "  ny =",
        ny_source
    )

    print()
    print("Target grid:")
    print(
        "  nx =",
        len(lon_edges) - 1
    )

    print(
        "  ny =",
        len(lat_edges) - 1
    )

    print()
    print("Projection centre:")
    print(
        "  lon =",
        cen_lon
    )

    print(
        "  lat =",
        cen_lat
    )

    # --------------------------------------------------------
    # Equal-area projection
    # --------------------------------------------------------

    transformer = make_laea_transformer(
        cen_lon,
        cen_lat
    )

    # --------------------------------------------------------
    # Source geometry
    # --------------------------------------------------------

    (
        source_polygons,
        source_valid,
        lon_bnd,
        lat_bnd
    ) = build_source_geometry(
        lat_corner,
        lon_corner,
        transformer
    )

    # --------------------------------------------------------
    # Source area diagnostic
    # --------------------------------------------------------

    geom_area = diagnostic_source_geometry(
        source_polygons,
        source_valid,
        gridarea
    )

    # --------------------------------------------------------
    # Target geometry
    # --------------------------------------------------------

    (
        target_polygons,
        target_area
    ) = build_target_polygons(
        lon_edges,
        lat_edges,
        transformer
    )

    nx_target = (
        len(lon_edges) - 1
    )

    ny_target = (
        len(lat_edges) - 1
    )

    # --------------------------------------------------------
    # Overlap matrix
    # --------------------------------------------------------

    (
        target_idx,
        source_idx,
        overlap_area
    ) = build_overlap_matrix(
        source_polygons,
        source_valid,
        target_polygons,
        (ny_source, nx_source)
    )

    # --------------------------------------------------------
    # Convert overlap area to source fraction
    # --------------------------------------------------------

    source_geom_flat = (
        geom_area.ravel()
    )

    overlap_fraction = (
        overlap_area
        / source_geom_flat[
            source_idx
        ]
    )

    # --------------------------------------------------------
    # Source coverage
    # --------------------------------------------------------

    source_coverage = np.bincount(
        source_idx,
        weights=overlap_fraction,
        minlength=(
            nx_source
            * ny_source
        )
    )

    covered = (
        source_valid.ravel()
        & (source_geom_flat > 0)
    )

    cov = source_coverage[
        covered
    ]

    print()
    print("=" * 70)
    print("SOURCE COVERAGE DIAGNOSTIC")
    print("=" * 70)

    print()
    print(
        "Coverage fraction across target domain:"
    )

    print(
        "  min  :",
        cov.min()
    )

    print(
        "  max  :",
        cov.max()
    )

    print(
        "  mean :",
        cov.mean()
    )

    print()
    print("Number of source cells:")

    print(
        "  total:",
        covered.sum()
    )

    print(
        "  fully covered (>0.999999):",
        np.sum(
            cov > 0.999999
        )
    )

    print(
        "  partially covered:",
        np.sum(
            (cov > 0)
            & (cov < 0.999999)
        )
    )

    print(
        "  not covered:",
        np.sum(
            cov <= 0
        )
    )

    print()

    if (
        np.max(
            np.abs(
                cov - 1.0
            )
        )
        > 1.e-4
    ):

        print(
            "WARNING: target domain does not completely "
            "cover every source cell."
        )

        print(
            "Maximum source coverage error =",
            np.max(
                np.abs(
                    cov - 1.0
                )
            )
        )

        print(
            "Partial clipping is accounted for in the "
            "conservation diagnostic."
        )

    else:

        print(
            "SOURCE COVERAGE CHECK: PASSED"
        )

    return {
        "nx_source": nx_source,
        "ny_source": ny_source,

        "nx_target": nx_target,
        "ny_target": ny_target,

        "gridarea": gridarea,

        "target_area": target_area,

        "target_idx": target_idx,
        "source_idx": source_idx,

        "overlap_fraction":
            overlap_fraction,

        "source_coverage":
            source_coverage,

        "lon_edges": lon_edges,
        "lat_edges": lat_edges,
    }


# ============================================================
# REMAP ONE FOOTPRINT
# ============================================================

def remap_footprint(
    source_footprint,
    remap
):

    """
    Input:

        source_footprint:
            (ny_source, nx_source, time)

    Output:

        target:
            (nx_target, ny_target, time)
    """

    gridarea = remap[
        "gridarea"
    ]

    target_idx = remap[
        "target_idx"
    ]

    source_idx = remap[
        "source_idx"
    ]

    overlap_fraction = remap[
        "overlap_fraction"
    ]

    target_area = remap[
        "target_area"
    ]

    nx_target = remap[
        "nx_target"
    ]

    ny_target = remap[
        "ny_target"
    ]

    ny_source, nx_source, nt = (
        source_footprint.shape
    )

    # --------------------------------------------------------
    # Flatten source grid
    # --------------------------------------------------------

    source_flat = (
        source_footprint.reshape(
            ny_source * nx_source,
            nt
        )
    )

    # --------------------------------------------------------
    # Source integrated quantity
    #
    # s m3 kg-1 × m2
    # --------------------------------------------------------

    source_integrated = (
        source_flat
        * gridarea.ravel()[:, None]
    )

    # --------------------------------------------------------
    # Target integrated quantity
    # --------------------------------------------------------

    target_integrated = np.zeros(
        (
            nx_target * ny_target,
            nt
        ),
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Conservative redistribution
    # --------------------------------------------------------

    for k in range(
        len(overlap_fraction)
    ):

        ti = target_idx[k]
        si = source_idx[k]

        frac = overlap_fraction[k]

        target_integrated[
            ti,
            :
        ] += (
            source_integrated[
                si,
                :
            ]
            * frac
        )

    # --------------------------------------------------------
    # Convert integrated quantity to
    # target footprint
    # --------------------------------------------------------

    target_flat = (
        target_integrated
        / target_area[:, None]
    )

    # --------------------------------------------------------
    # Reshape:
    #
    # target_flat:
    #   target_flattened, time
    #
    # intermediate:
    #   south_north, west_east, time
    #
    # final:
    #   west_east, south_north, time
    # --------------------------------------------------------

    target = target_flat.reshape(
        ny_target,
        nx_target,
        nt
    )

    target = np.transpose(
        target,
        (1, 0, 2)
    )

    return target


# ============================================================
# WRITE OUTPUT
# ============================================================

def write_output(
    outfile,
    target,
    lon_centres,
    lat_centres
):

    """
    Write FLEXINVERT-compatible NetCDF.

    The previous version passed times=None here, which caused:

        TypeError:
        object of type 'NoneType' has no len()

    This version obtains nt directly from target.shape[2].
    """

    os.makedirs(
        os.path.dirname(outfile),
        exist_ok=True
    )

    nx = len(
        lon_centres
    )

    ny = len(
        lat_centres
    )

    nt = target.shape[2]

    with Dataset(
        outfile,
        "w",
        format="NETCDF4"
    ) as nc:

        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

        nc.createDimension(
            "west_east",
            nx
        )

        nc.createDimension(
            "south_north",
            ny
        )

        nc.createDimension(
            "time",
            nt
        )

        # ----------------------------------------------------
        # Coordinates
        # ----------------------------------------------------

        lonvar = nc.createVariable(
            "west_east",
            "f8",
            ("west_east",)
        )

        latvar = nc.createVariable(
            "south_north",
            "f8",
            ("south_north",)
        )

        timevar = nc.createVariable(
            "time",
            "f8",
            ("time",)
        )

        # ----------------------------------------------------
        # Footprint
        # ----------------------------------------------------

        srr = nc.createVariable(
            "srr",
            "f8",
            (
                "west_east",
                "south_north",
                "time"
            ),
            zlib=True,
            complevel=4,
            fill_value=0.0
        )

        lonvar[:] = lon_centres
        latvar[:] = lat_centres

        # ----------------------------------------------------
        # FLEXPART time convention
        #
        # t=0 corresponds to the first footprint time,
        # i.e. one hour before the release in the current
        # hourly FLEXPART-WRF output.
        #
        # FLEXINVERT uses this as relative time.
        # ----------------------------------------------------

        timevar[:] = np.arange(
            nt,
            dtype=float
        )

        timevar.units = (
            "hours before release"
        )

        srr.units = (
            "s m3 kg-1"
        )

        srr.long_name = (
            "FLEXPART-WRF source-receptor relationship"
        )

        # ----------------------------------------------------
        # Data
        # ----------------------------------------------------

        srr[:, :, :] = target

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        nc.setncattr(
            "description",
            "FLEXPART-WRF footprint conservatively "
            "remapped to regular latitude/longitude grid "
            "for FLEXINVERT+"
        )

        nc.setncattr(
            "source_grid",
            "FLEXPART-WRF Lambert conformal"
        )

        nc.setncattr(
            "remapping",
            "area-conservative polygon overlap"
        )

        nc.setncattr(
            "source_layer",
            "bottom_top=0, approximately 0-10 m"
        )


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(
    infile,
    outfile,
    remap,
    lon_centres,
    lat_centres
):

    with Dataset(
        infile
    ) as nc:

        conc = np.asarray(
            nc.variables[
                "CONC"
            ][:],
            dtype=np.float64
        )

        # ----------------------------------------------------
        # CONC dimensions:
        #
        # Time,
        # ageclass,
        # releases,
        # bottom_top,
        # south_north,
        # west_east
        #
        # Select:
        #   ageclass     = 0
        #   releases     = 0
        #   bottom_top   = 0
        #
        # This is the 0-10 m layer.
        # ----------------------------------------------------

        source = conc[
            :,
            0,
            0,
            0,
            :,
            :
        ]

        # ----------------------------------------------------
        # Current shape:
        #
        # time, south_north, west_east
        #
        # Convert to:
        #
        # south_north, west_east, time
        # ----------------------------------------------------

        source = np.transpose(
            source,
            (1, 2, 0)
        )

    # --------------------------------------------------------
    # Remap
    # --------------------------------------------------------

    target = remap_footprint(
        source,
        remap
    )

    # --------------------------------------------------------
    # Conservation diagnostic
    #
    # Source integral:
    #
    #     sum(
    #         footprint *
    #         GRIDAREA
    #     )
    #
    # Expected target:
    #
    #     sum(
    #         footprint *
    #         GRIDAREA *
    #         source_coverage
    #     )
    #
    # Target:
    #
    #     sum(
    #         target_footprint *
    #         target_area
    #     )
    # --------------------------------------------------------

    source_integrated = (
        source
        * remap["gridarea"][:, :, None]
    )

    source_total = (
        source_integrated.sum(
            axis=(0, 1)
        )
    )

    # Target area
    target_area = (
        remap["target_area"].reshape(
            remap["ny_target"],
            remap["nx_target"]
        )
    )

    # Convert target back to:
    #
    # south_north, west_east, time
    #
    target_geo = np.transpose(
        target,
        (1, 0, 2)
    )

    target_total = (
        target_geo
        * target_area[:, :, None]
    ).sum(
        axis=(0, 1)
    )

    # --------------------------------------------------------
    # Expected target after source-domain clipping
    # --------------------------------------------------------

    source_flat = source.reshape(
        remap["ny_source"]
        * remap["nx_source"],
        source.shape[2]
    )

    coverage = remap[
        "source_coverage"
    ]

    expected_target = np.zeros_like(
        source_total
    )

    for t in range(
        source.shape[2]
    ):

        expected_target[t] = np.sum(
            source_flat[:, t]
            * remap["gridarea"].ravel()
            * coverage
        )

    denominator = np.maximum(
        np.abs(expected_target),
        1.e-30
    )

    error = (
        target_total
        - expected_target
    ) / denominator

    maxerr = np.max(
        np.abs(error)
    )

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    write_output(
        outfile,
        target,
        lon_centres,
        lat_centres
    )

    return (
        source_total,
        target_total,
        expected_target,
        maxerr
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "FLEXPART-WRF -> FLEXINVERT+ "
        "CONSERVATIVE CONVERTER"
    )
    print("=" * 78)

    print()
    print(
        "Input :",
        INPUT_ROOT
    )

    print(
        "Output:",
        OUTPUT_ROOT
    )

    # --------------------------------------------------------
    # Target grid
    # --------------------------------------------------------

    (
        lon_edges,
        lat_edges,
        lon_centres,
        lat_centres
    ) = get_target_grid()

    print()
    print("TARGET GRID")

    print(
        "  longitude edges:",
        lon_edges[0],
        "to",
        lon_edges[-1]
    )

    print(
        "  latitude edges :",
        lat_edges[0],
        "to",
        lat_edges[-1]
    )

    print(
        "  resolution     :",
        DX_DEG,
        "x",
        DY_DEG,
        "degrees"
    )

    print(
        "  dimensions     :",
        len(lon_centres),
        "x",
        len(lat_centres)
    )

    # --------------------------------------------------------
    # Find first header
    # --------------------------------------------------------

    first_header = None

    for receptor in RECEPTORS:

        headers = sorted(
            glob.glob(
                os.path.join(
                    INPUT_ROOT,
                    receptor,
                    "*",
                    "header_d01.nc"
                )
            )
        )

        if headers:

            first_header = headers[0]

            break

    if first_header is None:

        raise RuntimeError(
            "No header_d01.nc found."
        )

    print()
    print(
        "Reference header:"
    )

    print(
        " ",
        first_header
    )

    # --------------------------------------------------------
    # Build remapping once
    # --------------------------------------------------------

    remap = build_remapping(
        first_header,
        lon_edges,
        lat_edges
    )

    # --------------------------------------------------------
    # Find all footprint files
    # --------------------------------------------------------

    files = []

    for receptor in RECEPTORS:

        pattern = os.path.join(
            INPUT_ROOT,
            receptor,
            "*",
            "flxout_d01_*.nc"
        )

        found = sorted(
            glob.glob(pattern)
        )

        files.extend(
            [
                (
                    receptor,
                    filename
                )
                for filename in found
            ]
        )

    files.sort()

    print()
    print("=" * 78)
    print("FILES")
    print("=" * 78)

    print(
        "Total files:",
        len(files)
    )

    successful = 0
    failed = 0

    global_max_error = 0.0

    # --------------------------------------------------------
    # Process all files
    # --------------------------------------------------------

    for count, (
        receptor,
        infile
    ) in enumerate(
        files,
        start=1
    ):

        t0 = timer.time()

        try:

            basename = os.path.basename(
                infile
            )

            # Example:
            #
            # flxout_d01_20260124_060000.nc
            #
            # -> 20260124060000

            release = basename[
                len("flxout_d01_"):
                -len(".nc")
            ]

            # ------------------------------------------------
            # Output:
            #
            # receptor/YYYYMM/
            # ------------------------------------------------

            year = release[:4]
            month = release[4:6]

            outdir = os.path.join(
                OUTPUT_ROOT,
                receptor,
                year + month
            )

            outfile = os.path.join(
                outdir,
                "grid_time_"
                + release
                + "_001.nc"
            )

            (
                source_total,
                target_total,
                expected_target,
                maxerr
            ) = process_file(
                infile,
                outfile,
                remap,
                lon_centres,
                lat_centres
            )

            global_max_error = max(
                global_max_error,
                maxerr
            )

            dt = (
                timer.time()
                - t0
            )

            successful += 1

            print()
            print(
                f"[{count}/{len(files)}] "
                f"OK {receptor}/{release} "
                f"({dt:.1f} s)"
            )

            print(
                "  time 0 source       =",
                source_total[0]
            )

            print(
                "  time 0 expected     =",
                expected_target[0]
            )

            print(
                "  time 0 target       =",
                target_total[0]
            )

            print(
                "  time 0 relative err =",
                (
                    target_total[0]
                    - expected_target[0]
                )
                / max(
                    abs(
                        expected_target[0]
                    ),
                    1.e-30
                )
            )

            print(
                "  maximum absolute "
                "relative error =",
                maxerr
            )

        except Exception as exc:

            failed += 1

            print()
            print(
                f"[{count}/{len(files)}] "
                f"FAILED {receptor}"
            )

            print(
                "  File:",
                infile
            )

            print(
                "  Error:",
                repr(exc)
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("CONVERSION COMPLETE")
    print("=" * 78)

    print(
        "Successful:",
        successful
    )

    print(
        "Failed    :",
        failed
    )

    print(
        "Total     :",
        len(files)
    )

    print()
    print(
        "Maximum remapping conservation error:",
        global_max_error
    )

    print()
    print(
        "Output:",
        OUTPUT_ROOT
    )

    if (
        failed == 0
        and global_max_error < 1.e-5
    ):

        print()
        print(
            "STATUS: "
            "CONSERVATION CHECK PASSED"
        )

    elif failed == 0:

        print()
        print(
            "STATUS: files converted, "
            "but inspect conservation diagnostic."
        )

    else:

        print()
        print(
            "STATUS: conversion had failures."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

