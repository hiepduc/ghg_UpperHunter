#!/usr/bin/env python3
"""
Compare FLEXPART-WRF spatial footprints for 2-layer and 3-layer
vertical configurations.

Case:
    Singleton receptor
    Release: 2026-01-10 02:00 UTC

Compare the H-relevant footprint density:

    K = CONC / VOLUME

Fields:
    1. 3-layer 0-10 m
    2. 3-layer 10-100 m
    3. 3-layer combined 0-100 m
    4. 2-layer 0-100 m

The 0-100 m 3-layer field is constructed as:

    K_3L_0_100 =
        CONC_3L[0] / V_3L[0]
        +
        CONC_3L[1] / V_3L[1]

This is the quantity corresponding directly to the
layer-by-layer summation used in the H calculation.

Output:
    hmatrix/daily_test/spatial_2L_vs_3L/

    *.npy
    *.csv
    PNG maps
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================================
# PATHS
# ============================================================================

BASE = Path.home() / "flexpart-wrf" / "upperhunter_flexpart"

RUN_3L = BASE / "run" / "singleton_10m"
RUN_2L = BASE / "run" / "singleton_2layer_clean"

OUTDIR = (
    BASE
    / "hmatrix"
    / "daily_test"
    / "spatial_2L_vs_3L"
)

OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# FILES
# ============================================================================

CONC_3L = RUN_3L /  "flxout_d01_20260110_010000.nc"
CONC_2L = RUN_2L /  "flxout_d01_20260110_010000.nc"

HEADER_3L = RUN_3L / "header_d01.nc"
HEADER_2L = RUN_2L / "header_d01.nc"


# ============================================================================
# FUNCTIONS
# ============================================================================

def read_grid_area(header_file):
    """
    Read GRIDAREA from FLEXPART header.

    Returns
    -------
    gridarea : ndarray
        Shape (ny, nx)
    """

    ds = xr.open_dataset(header_file)

    if "GRIDAREA" not in ds:
        raise RuntimeError(
            f"GRIDAREA not found in {header_file}"
        )

    gridarea = ds["GRIDAREA"].values.astype(np.float64)

    ds.close()

    return gridarea


def read_ztop(header_file):
    """
    Read ZTOP from FLEXPART header.
    """

    ds = xr.open_dataset(header_file)

    if "ZTOP" not in ds:
        raise RuntimeError(
            f"ZTOP not found in {header_file}"
        )

    ztop = np.asarray(
        ds["ZTOP"].values,
        dtype=np.float64
    ).ravel()

    ds.close()

    return ztop

def get_times(ds):
    """
    Read FLEXPART-WRF Times.

    FLEXPART stores Times here as a 1-D byte-string array:

        shape = (24,)
        dtype = |S15

    Example:
        b'20260110_010000'
        b'20260110_000000'
        b'20260109_230000'

    Returns UTC timestamps.
    """

    if "Times" not in ds.variables:
        raise RuntimeError(
            "Times variable not found. "
            f"Available variables: {list(ds.variables)}"
        )

    values = ds["Times"].values

    print("Times raw shape:", values.shape)
    print("Times dtype:", values.dtype)

    times = []

    for value in values:

        # --------------------------------------------------------
        # Decode the 15-character FLEXPART timestamp
        # --------------------------------------------------------

        if isinstance(value, (bytes, np.bytes_)):
            s = value.decode(
                "ascii",
                errors="ignore",
            )
        else:
            s = str(value)

        # Remove possible padding
        s = (
            s
            .replace("\x00", "")
            .replace("*", "")
            .strip()
        )

        # --------------------------------------------------------
        # Parse YYYYMMDD_HHMMSS
        # --------------------------------------------------------

        try:

            t = pd.to_datetime(
                s,
                format="%Y%m%d_%H%M%S",
                utc=True,
            )

        except Exception as exc:

            raise RuntimeError(
                f"Could not parse FLEXPART Times value: "
                f"'{s}'"
            ) from exc

        times.append(t)

    return pd.DatetimeIndex(times)


def calculate_volume(gridarea, ztop):
    """
    Calculate 3-D grid-cell volume.

    FLEXPART vertical boundaries are represented by ZTOP.

    For:
        ZTOP = [10, 100, 20000]

    layer thickness =
        [10, 90, 19900]
    """

    thickness = np.empty(
        len(ztop),
        dtype=np.float64
    )

    thickness[0] = ztop[0]

    for k in range(1, len(ztop)):
        thickness[k] = (
            ztop[k]
            - ztop[k - 1]
        )

    volume = (
        gridarea[None, :, :]
        * thickness[:, None, None]
    )

    return volume, thickness


def load_case(conc_file, header_file, name):

    print("\n" + "=" * 80)
    print(f"READING {name}")
    print("=" * 80)

    ds = xr.open_dataset(conc_file)

    print("CONC dimensions:")
    print(ds["CONC"].dims)

    conc = ds["CONC"].values.astype(
        np.float64
    )

    times = get_times(ds)

    ds.close()

    # ------------------------------------------------------------
    # Determine spatial dimensions
    # ------------------------------------------------------------

    # Expected:
    #
    # time, release, ageclass, bottom_top, south_north, west_east
    #
    # after selecting release/ageclass:
    #
    # time, bottom_top, south_north, west_east

    ds = xr.open_dataset(conc_file)

    da = ds["CONC"]

    # Select singleton release / ageclass if present
    if "release" in da.dims:
        da = da.isel(release=0)

    if "nageclass" in da.dims:
        da = da.isel(nageclass=0)

    if "ageclass" in da.dims:
        da = da.isel(ageclass=0)

    da = da.squeeze()

    conc = da.values.astype(np.float64)

    dims = da.dims

    print("\nSelected CONC dimensions:")
    print(dims)

    ds.close()

    # Find vertical dimension
    vertical_candidates = [
        "bottom_top",
        "bottom_top_stag",
        "height",
        "level",
    ]

    vertical_dim = None

    for d in vertical_candidates:
        if d in dims:
            vertical_dim = d
            break

    if vertical_dim is None:
        raise RuntimeError(
            f"Could not identify vertical dimension: {dims}"
        )

    k = dims.index(vertical_dim)

    if k != 1:
        # Expected order time, vertical, y, x.
        # Reorder explicitly.
        time_dim = [
            d for d in dims
            if d.lower() in ["time", "times"]
        ][0]

        spatial_dims = [
            d for d in dims
            if d not in [time_dim, vertical_dim]
        ]

        da = da.transpose(
            time_dim,
            vertical_dim,
            spatial_dims[0],
            spatial_dims[1],
        )

        conc = da.values.astype(np.float64)

    print("Final CONC shape:")
    print(conc.shape)

    # ------------------------------------------------------------
    # Grid geometry
    # ------------------------------------------------------------

    gridarea = read_grid_area(header_file)
    ztop = read_ztop(header_file)

    volume, thickness = calculate_volume(
        gridarea,
        ztop,
    )

    print("\nVertical geometry:")
    print("ZTOP:")
    print(ztop)

    print("Thickness:")
    print(thickness)

    print("Volume shape:")
    print(volume.shape)

    if conc.shape[1] != len(ztop):
        raise RuntimeError(
            f"CONC vertical dimension = {conc.shape[1]}, "
            f"but ZTOP has {len(ztop)} layers."
        )

    # ------------------------------------------------------------
    # H-relevant footprint density
    # ------------------------------------------------------------

    K = (
        conc
        /
        volume[None, :, :, :]
    )

    print("\nK = CONC / volume")
    print("K shape:", K.shape)

    print(
        "K min:",
        np.nanmin(K)
    )

    print(
        "K max:",
        np.nanmax(K)
    )

    return {
        "conc": conc,
        "K": K,
        "times": times,
        "gridarea": gridarea,
        "ztop": ztop,
        "thickness": thickness,
        "volume": volume,
    }


def save_field(name, field, times, suffix=""):

    np.save(
        OUTDIR / f"{name}{suffix}.npy",
        field,
    )

    rows = []

    for t, value in zip(times, field):

        rows.append({
            "datetime_utc": t.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "max": np.nanmax(value),
            "sum": np.nansum(value),
            "mean": np.nanmean(value),
            "nonzero": np.count_nonzero(
                np.isfinite(value)
                & (value != 0)
            ),
        })

    pd.DataFrame(rows).to_csv(
        OUTDIR / f"{name}{suffix}_summary.csv",
        index=False,
    )


def plot_field(
    field,
    title,
    filename,
    log=False,
):

    # Use output 0:
    # 2026-01-10 01:00 UTC

    data = field[0]

    finite = np.isfinite(data)

    if not np.any(finite):
        print(
            f"WARNING: no finite values for {title}"
        )
        return

    positive = data[
        finite & (data > 0)
    ]

    if len(positive) == 0:
        print(
            f"WARNING: no positive values for {title}"
        )
        return

    if log:

        plotdata = np.full_like(
            data,
            np.nan,
        )

        plotdata[finite & (data > 0)] = np.log10(
            data[finite & (data > 0)]
        )

        label = "log10(CONC / volume)"

    else:

        plotdata = data

        label = "CONC / volume"

    plt.figure(
        figsize=(10, 8)
    )

    im = plt.imshow(
        plotdata,
        origin="lower",
        aspect="equal",
    )

    plt.colorbar(
        im,
        label=label,
    )

    plt.xlabel("West-East grid index")
    plt.ylabel("South-North grid index")

    plt.title(
        title
        + "\n"
        + "FLEXPART output 0: 2026-01-10 01:00 UTC"
    )

    plt.tight_layout()

    plt.savefig(
        OUTDIR / filename,
        dpi=200,
    )

    plt.close()


# ============================================================================
# LOAD
# ============================================================================

case3 = load_case(
    CONC_3L,
    HEADER_3L,
    "3-LAYER",
)

case2 = load_case(
    CONC_2L,
    HEADER_2L,
    "2-LAYER",
)


# ============================================================================
# CHECK TIMES
# ============================================================================

print("\n" + "=" * 80)
print("TIME CHECK")
print("=" * 80)

print("3-layer:")
for i, t in enumerate(case3["times"]):
    print(i, t)

print("\n2-layer:")
for i, t in enumerate(case2["times"]):
    print(i, t)

if not np.array_equal(
    case3["times"].values,
    case2["times"].values,
):

    raise RuntimeError(
        "2-layer and 3-layer output times differ."
    )

print("\nTimes are identical.")


# ============================================================================
# EXTRACT FOOTPRINTS
# ============================================================================

K3 = case3["K"]
K2 = case2["K"]


# 3-layer:
#
# layer 0 = 0-10 m
# layer 1 = 10-100 m
# layer 2 = 100-20000 m

K3_0_10 = K3[:, 0, :, :]

K3_10_100 = K3[:, 1, :, :]

K3_100_20000 = K3[:, 2, :, :]

K3_0_100 = (
    K3_0_10
    +
    K3_10_100
)


# 2-layer:
#
# layer 0 = 0-100 m
# layer 1 = 100-20000 m

K2_0_100 = K2[:, 0, :, :]

K2_100_20000 = K2[:, 1, :, :]


# ============================================================================
# SAVE
# ============================================================================

print("\n" + "=" * 80)
print("SAVING FOOTPRINT FIELDS")
print("=" * 80)

save_field(
    "K_3L_0_10m",
    K3_0_10,
    case3["times"],
)

save_field(
    "K_3L_10_100m",
    K3_10_100,
    case3["times"],
)

save_field(
    "K_3L_0_100m",
    K3_0_100,
    case3["times"],
)

save_field(
    "K_3L_100_20000m",
    K3_100_20000,
    case3["times"],
)

save_field(
    "K_2L_0_100m",
    K2_0_100,
    case2["times"],
)

save_field(
    "K_2L_100_20000m",
    K2_100_20000,
    case2["times"],
)


# ============================================================================
# DIRECT DIFFERENCE
# ============================================================================

difference = (
    K2_0_100
    -
    K3_0_100
)

ratio = np.full_like(
    K3_0_100,
    np.nan,
)

mask = (
    np.isfinite(K3_0_100)
    &
    (K3_0_100 != 0)
)

ratio[mask] = (
    K2_0_100[mask]
    /
    K3_0_100[mask]
)


np.save(
    OUTDIR / "K_2L_0_100_minus_K_3L_0_100.npy",
    difference,
)

np.save(
    OUTDIR / "K_2L_to_K_3L_ratio.npy",
    ratio,
)


# ============================================================================
# NUMERICAL COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("SPATIAL COMPARISON")
print("=" * 80)

for i in [0, 1]:

    A = K3_0_100[i]
    B = K2_0_100[i]

    valid = (
        np.isfinite(A)
        &
        np.isfinite(B)
    )

    nonzero = (
        valid
        &
        ((A != 0) | (B != 0))
    )

    print("\nOutput", i)
    print("UTC:", case3["times"][i])

    print(
        "3L 0-100 sum:",
        np.nansum(A)
    )

    print(
        "2L 0-100 sum:",
        np.nansum(B)
    )

    print(
        "2L / 3L total:",
        np.nansum(B)
        /
        np.nansum(A)
        if np.nansum(A) != 0
        else np.nan
    )

    print(
        "Common valid cells:",
        np.count_nonzero(valid)
    )

    print(
        "Common nonzero cells:",
        np.count_nonzero(nonzero)
    )

    if np.count_nonzero(nonzero) > 1:

        x = A[nonzero]
        y = B[nonzero]

        corr = np.corrcoef(
            x,
            y,
        )[0, 1]

        print(
            "Raw spatial correlation:",
            corr
        )

        # Normalised spatial correlation
        xn = x / np.sum(x)
        yn = y / np.sum(y)

        corr_norm = np.corrcoef(
            xn,
            yn,
        )[0, 1]

        print(
            "Normalised spatial correlation:",
            corr_norm
        )

        # Overlap coefficient
        overlap = (
            np.sum(
                np.minimum(
                    xn,
                    yn,
                )
            )
        )

        print(
            "Overlap coefficient:",
            overlap
        )

        # Normalised RMS difference
        nrms = np.sqrt(
            np.mean(
                (xn - yn) ** 2
            )
        )

        print(
            "Normalised RMS difference:",
            nrms
        )


# ============================================================================
# VERTICAL COMPONENTS
# ============================================================================

print("\n" + "=" * 80)
print("VERTICAL COMPONENTS — OUTPUT 0")
print("=" * 80)

fields = {
    "3L 0-10m": K3_0_10[0],
    "3L 10-100m": K3_10_100[0],
    "3L 0-100m": K3_0_100[0],
    "3L 100-20000m": K3_100_20000[0],
    "2L 0-100m": K2_0_100[0],
    "2L 100-20000m": K2_100_20000[0],
}

for name, field in fields.items():

    print(
        f"{name:20s}"
        f" sum = {np.nansum(field):.12e}"
        f" max = {np.nanmax(field):.12e}"
        f" nonzero = {np.count_nonzero(field)}"
    )


# ============================================================================
# PLOTS
# ============================================================================

print("\n" + "=" * 80)
print("CREATING PNG MAPS")
print("=" * 80)

plot_field(
    K3_0_10,
    "FLEXPART 3-layer: 0–10 m",
    "K_3L_0_10m.png",
    log=True,
)

plot_field(
    K3_10_100,
    "FLEXPART 3-layer: 10–100 m",
    "K_3L_10_100m.png",
    log=True,
)

plot_field(
    K3_0_100,
    "FLEXPART 3-layer: H-relevant 0–100 m",
    "K_3L_0_100m.png",
    log=True,
)

plot_field(
    K2_0_100,
    "FLEXPART 2-layer: H-relevant 0–100 m",
    "K_2L_0_100m.png",
    log=True,
)

plot_field(
    difference,
    "2-layer minus 3-layer: 0–100 m",
    "K_difference_2L_minus_3L.png",
    log=False,
)


# ============================================================================
# FINISH
# ============================================================================

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(
    f"\nResults written to:\n{OUTDIR}"
)

print("\nKey files:")

for f in sorted(OUTDIR.glob("*")):
    print(" ", f.name)

