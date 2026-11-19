#!/usr/bin/env Rscript

library(ncdf4)
library(terra)

# ============================================================
# SETTINGS
# ============================================================

foot_file <- "/mnt/scratch_lustre/duch/stilt/upperhunter/test_UH_ARL_Jan10/out_oldgood/by-id/202601100200_151.16132_-32.57123_5_foot.nc"

foot_file <- "/mnt/scratch_lustre/duch/stilt/upperhunter/test_UH_ARL_Jan10/out_oldgood/by-id/202601100200_151.16132_-32.57123_5/202601100200_151.16132_-32.57123_5_foot.nc"

outdir <- "/mnt/scratch_lustre/duch/stilt/upperhunter/test_UH_ARL_Jan10/plots"

mine_dir <- "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter"

dir.create(
    outdir,
    recursive = TRUE,
    showWarnings = FALSE
)

# Singleton receptor
receptor_lon <- 151.16132
receptor_lat <- -32.57123


# ============================================================
# 1. READ STILT FOOTPRINT
# ============================================================

nc <- nc_open(foot_file)

lon  <- ncvar_get(nc, "lon")
lat  <- ncvar_get(nc, "lat")
time <- ncvar_get(nc, "time")
foot <- ncvar_get(nc, "foot")

nc_close(nc)

time_utc <- as.POSIXct(
    time,
    origin = "1970-01-01",
    tz = "UTC"
)

cat("Footprint dimensions:\n")
print(dim(foot))

cat("\nLongitude range:\n")
print(range(lon))

cat("\nLatitude range:\n")
print(range(lat))

cat("\nNumber of time layers:", length(time_utc), "\n")

cat("\nFootprint times:\n")
print(time_utc)


# ============================================================
# 2. READ 13 MINE POLYGONS
# ============================================================

mines <- lapply(
    1:13,
    function(i) {

        f <- file.path(
            mine_dir,
            paste0("OC", i, ".shp")
        )

        vect(f)
    }
)

names(mines) <- paste0("OC", 1:13)

# Convert all mines to WGS84
mines <- lapply(
    mines,
    function(x) project(x, "EPSG:4326")
)


# ============================================================
# 3. CONVERT STILT FOOTPRINT TO TERRA RASTER
# ============================================================

make_foot_raster <- function(k) {

    # STILT NetCDF as returned by ncvar_get:
    #
    # [longitude, latitude, time]

    z <- foot[, , k]

    # longitude x latitude
    # ->
    # latitude x longitude

    z <- t(z)

    # STILT latitude runs south -> north.
    # terra raster rows run north -> south.

    z <- z[nrow(z):1, ]

    # --------------------------------------------------------
    # Coordinate spacing
    # --------------------------------------------------------

    dx <- median(diff(lon))
    dy <- median(diff(lat))

    # lon/lat are cell centres.
    # Therefore extend the raster by half a grid cell.

    xmin <- min(lon) - dx / 2
    xmax <- max(lon) + dx / 2

    ymin <- min(lat) - dy / 2
    ymax <- max(lat) + dy / 2

    r <- rast(
        nrows = length(lat),
        ncols = length(lon),
        xmin = xmin,
        xmax = xmax,
        ymin = ymin,
        ymax = ymax,
        crs = "EPSG:4326"
    )

    # R matrix is column-major.
    # terra raster values are row-wise.

    values(r) <- as.vector(t(z))

    return(r)
}


# ============================================================
# 4. CREATE ALL FOOTPRINT RASTERS
# ============================================================

foot_rasters <- lapply(
    seq_len(dim(foot)[3]),
    make_foot_raster
)

names(foot_rasters) <- format(
    time_utc,
    "%Y%m%d_%H%MUTC"
)


# ============================================================
# 5. DISPLAY RASTER GEOMETRY
# ============================================================

cat("\nRaster geometry:\n")
print(foot_rasters[[1]])

cat("\nRaster extent:\n")
print(ext(foot_rasters[[1]]))

cat("\nRaster resolution:\n")
print(res(foot_rasters[[1]]))


# ============================================================
# 6. PLOT FUNCTION
# ============================================================

draw_footprint <- function(r, title_text) {

    # Log10 footprint
    r_plot <- ifel(
        r > 0,
        log10(r),
        NA
    )

    plot(
        r_plot,
        main = title_text,
        xlab = "Longitude",
        ylab = "Latitude",
        col = hcl.colors(
            100,
            "YlOrRd",
            rev = TRUE
        )
    )

    # --------------------------------------------------------
    # Mine boundaries
    # --------------------------------------------------------

    for (i in seq_along(mines)) {

        lines(
            mines[[i]],
            lwd = 2
        )
    }

    # --------------------------------------------------------
    # Mine labels
    # --------------------------------------------------------

    for (i in seq_along(mines)) {

        xy <- crds(
            centroids(mines[[i]])
        )

        text(
            x = xy[1, 1],
            y = xy[1, 2],
            labels = names(mines)[i],
            cex = 0.75,
            font = 2
        )
    }

    # --------------------------------------------------------
    # Singleton receptor
    # --------------------------------------------------------

    points(
        receptor_lon,
        receptor_lat,
        pch = 19,
        cex = 1.3
    )

    text(
        receptor_lon,
        receptor_lat,
        labels = "Singleton",
        pos = 4,
        cex = 0.8,
        font = 2
    )
}


# ============================================================
# 7. OPEN X11
# ============================================================

if (!capabilities("X11")) {
    stop("X11 graphics capability is not available.")
}

X11(
    width = 10,
    height = 8
)


# ============================================================
# 8. DISPLAY EACH FOOTPRINT ONE AT A TIME
# ============================================================

for (k in seq_along(foot_rasters)) {

    timestamp <- format(
        time_utc[k],
        "%Y-%m-%d %H:%M UTC"
    )

    outfile <- file.path(
        outdir,
        paste0(
            "footprint_",
            format(
                time_utc[k],
                "%Y%m%d_%H%M"
            ),
            ".png"
        )
    )

    title_text <- paste0(
        "STILT footprint\n",
        timestamp,
        "\nlog10(foot)"
    )

    # --------------------------------------------------------
    # DISPLAY ON X11
    # --------------------------------------------------------

    draw_footprint(
        foot_rasters[[k]],
        title_text
    )

    cat("\n")
    cat("============================================\n")
    cat(
        "Footprint ",
        k,
        " of ",
        length(foot_rasters),
        "\n",
        sep = ""
    )
    cat(
        "Time: ",
        timestamp,
        "\n",
        sep = ""
    )
    cat("============================================\n")
    cat("\n")
    cat("The footprint is displayed on the X11 window.\n")
    cat("Click ONCE anywhere inside the plot to continue.\n")

    # --------------------------------------------------------
    # WAIT FOR USER CLICK
    # --------------------------------------------------------

    locator(1)

    # --------------------------------------------------------
    # AFTER CLICK, SAVE SAME PLOT TO PNG
    # --------------------------------------------------------

    png(
        filename = outfile,
        width = 1400,
        height = 1100,
        res = 150
    )

    draw_footprint(
        foot_rasters[[k]],
        title_text
    )

    dev.off()

    cat(
        "Saved PNG: ",
        outfile,
        "\n",
        sep = ""
    )
}


# ============================================================
# 9. COMBINED FOUR-HOUR FOOTPRINT
# ============================================================

combined <- sum(
    rast(foot_rasters),
    na.rm = TRUE
)


# ------------------------------------------------------------
# Display combined footprint
# ------------------------------------------------------------

draw_footprint(
    combined,
    paste0(
        "STILT combined footprint\n",
        format(time_utc[1], "%Y-%m-%d %H:%M"),
        " to ",
        format(
            time_utc[length(time_utc)],
            "%Y-%m-%d %H:%M"
        ),
        " UTC\n",
        "sum of 4 hourly footprints\n",
        "log10(sum foot)"
    )
)

cat("\n")
cat("============================================\n")
cat("COMBINED FOUR-HOUR FOOTPRINT\n")
cat("============================================\n")
cat("\n")
cat("Click the plot when you have finished inspecting it.\n")

locator(1)


# ============================================================
# 10. SAVE COMBINED PNG
# ============================================================

combined_file <- file.path(
    outdir,
    "footprint_combined_4hours.png"
)

png(
    filename = combined_file,
    width = 1400,
    height = 1100,
    res = 150
)

draw_footprint(
    combined,
    paste0(
        "STILT combined footprint\n",
        format(time_utc[1], "%Y-%m-%d %H:%M"),
        " to ",
        format(
            time_utc[length(time_utc)],
            "%Y-%m-%d %H:%M"
        ),
        " UTC\n",
        "sum of 4 hourly footprints\n",
        "log10(sum foot)"
    )
)

dev.off()


# ============================================================
# 11. SAVE COMBINED GEOTIFF
# ============================================================

combined_tif <- file.path(
    outdir,
    "footprint_combined_4hours.tif"
)

writeRaster(
    combined,
    combined_tif,
    overwrite = TRUE
)


# ============================================================
# 12. FINISHED
# ============================================================

cat("\n")
cat("============================================\n")
cat("STILT FOOTPRINT PLOTTING COMPLETED\n")
cat("============================================\n")
cat("PNG directory:\n")
cat(outdir, "\n\n")

cat("Combined PNG:\n")
cat(combined_file, "\n\n")

cat("Combined GeoTIFF:\n")
cat(combined_tif, "\n")

