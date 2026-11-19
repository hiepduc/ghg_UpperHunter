#!/usr/bin/env Rscript

library(ncdf4)
library(terra)

# ============================================================
# STILT footprint + Upper Hunter mine polygons
# Singleton receptor
# ============================================================

# ------------------------------------------------------------
# 1. STILT footprint
# ------------------------------------------------------------

#footfile <- paste0(
#    "/home/duch/software/hysplit_data2arl/",
#    "test_upperhunter/stilt_singleton_20260110/",
#    "out_clean/by-id/",
#    "202601100200_151.16132_-32.57123_5/",
#    "202601100200_151.16132_-32.57123_5_foot.nc"
#)

footfile <- "/mnt/scratch_lustre/duch/stilt/upperhunter/test_UH_ARL_Jan10/out_oldgood/by-id/202601100200_151.16132_-32.57123_5/202601100200_151.16132_-32.57123_5_foot.nc"

# ------------------------------------------------------------
# 2. Mine shapefiles
# ------------------------------------------------------------

mine_dir <- "/home/duch/flexpart-wrf/emission_UH/RH_upperHunter"

# ------------------------------------------------------------
# 3. Output directory
# ------------------------------------------------------------

outdir <- "footprint_plots"
dir.create(outdir, showWarnings = FALSE)

# ------------------------------------------------------------
# 4. Read footprint NetCDF
# ------------------------------------------------------------

cat("Reading STILT footprint:\n")
cat(footfile, "\n\n")

nc <- nc_open(footfile)

lon  <- ncvar_get(nc, "lon")
lat  <- ncvar_get(nc, "lat")
foot <- ncvar_get(nc, "foot")
time <- ncvar_get(nc, "time")

nc_close(nc)

dates <- as.POSIXct(
    time,
    origin = "1970-01-01",
    tz = "UTC"
)

cat("Footprint dimensions:\n")
print(dim(foot))

cat("\nFootprint times:\n")
print(dates)

# ------------------------------------------------------------
# 5. Read mine polygons
# ------------------------------------------------------------

cat("\nReading mine shapefiles...\n")

mines <- vector("list", 13)

for (i in 1:13) {

    shp <- file.path(
        mine_dir,
        paste0("OC", i, ".shp")
    )

    if (!file.exists(shp)) {
        warning("Missing shapefile: ", shp)
        next
    }

    cat("  Reading OC", i, "\n", sep="")

    mines[[i]] <- vect(shp)

    # Make sure polygon CRS is WGS84
    if (!is.lonlat(mines[[i]])) {
        mines[[i]] <- project(
            mines[[i]],
            "EPSG:4326"
        )
    } else {
        mines[[i]] <- project(
            mines[[i]],
            "EPSG:4326"
        )
    }
}

names(mines) <- paste0("OC", 1:13)

# ------------------------------------------------------------
# 6. Footprint raster
# ------------------------------------------------------------

# IMPORTANT:
# NetCDF footprint has dimensions:
# lon x lat x time
#
# terra raster uses rows = latitude and columns = longitude.
# The transpose below makes the orientation correct.

make_foot_raster <- function(k) {

    r <- rast(
        nrows = length(lat),
        ncols = length(lon),
        xmin = min(lon),
        xmax = max(lon),
        ymin = min(lat),
        ymax = max(lat),
        crs = "EPSG:4326"
    )

    # NetCDF array is lon x lat.
    # Convert to lat x lon for raster rows/columns.
    vals <- t(foot[,,k])

    # Raster latitude rows run from ymax to ymin.
    # Reverse the rows so spatial orientation is correct.
    vals <- vals[nrow(vals):1, ]

    values(r) <- as.vector(vals)

    return(r)
}

# ------------------------------------------------------------
# 7. Plot individual footprint layers
# ------------------------------------------------------------

for (k in seq_along(dates)) {

    cat(
        "Plotting footprint ",
        k,
        ": ",
        format(dates[k], "%Y-%m-%d %H:%M"),
        " UTC\n",
        sep=""
    )

    r <- make_foot_raster(k)

    outfile <- file.path(
        outdir,
        sprintf(
            "footprint_%02d_%s_mines.png",
            k,
            format(dates[k], "%Y%m%d_%H%M")
        )
    )

    png(
        outfile,
        width = 1300,
        height = 1000,
        res = 140
    )

    # Footprint
    plot(
        r,
        main = paste0(
            "STILT footprint + Upper Hunter mines\n",
            "Singleton — ",
            format(dates[k], "%Y-%m-%d %H:%M"),
            " UTC"
        ),
        xlab = "Longitude",
        ylab = "Latitude"
    )

    # --------------------------------------------------------
    # Mine polygons
    # --------------------------------------------------------

    for (i in 1:13) {

        if (is.null(mines[[i]]))
            next

        lines(
            mines[[i]],
            lwd = 2
        )

        # Label using polygon centroid
        ctr <- centroids(mines[[i]])

        xy <- crds(ctr)

        text(
            xy[,1],
            xy[,2],
            labels = paste0("OC", i),
            cex = 0.75,
            pos = 3
        )
    }

    # --------------------------------------------------------
    # Singleton receptor
    # --------------------------------------------------------

    points(
        151.16132,
        -32.57123,
        pch = 8,
        cex = 2.5,
        lwd = 3
    )

    text(
        151.16132,
        -32.57123,
        labels = " Singleton",
        pos = 4,
        font = 2
    )

    dev.off()

    cat("  Created: ", outfile, "\n")
}

###########################
# Plot one by one k=1,2,3,4
###########################

k <- 1
r <- make_foot_raster(k)

X11(width = 10, height = 8)

plot(
    r,
    main = paste0(
        "STILT footprint + Upper Hunter mines\n",
        "Singleton — ",
        format(dates[k], "%Y-%m-%d %H:%M"),
        " UTC"
    ),
    xlab = "Longitude",
    ylab = "Latitude"
)

for (i in 1:13) {

    if (is.null(mines[[i]]))
        next

    lines(mines[[i]], lwd = 2)

    ctr <- centroids(mines[[i]])
    xy <- crds(ctr)

    text(
        xy[,1],
        xy[,2],
        labels = paste0("OC", i),
        cex = 0.75,
        pos = 3
    )
}

points(
    151.16132,
    -32.57123,
    pch = 8,
    cex = 2.5,
    lwd = 3
)

text(
    151.16132,
    -32.57123,
    labels = " Singleton",
    pos = 4,
    font = 2
)

# Better with log scale

# ============================================================
# Plot one STILT footprint with logarithmic colour scaling
# ============================================================

k <- 1
r <- make_foot_raster(k)

# ------------------------------------------------------------
# Get footprint values
# ------------------------------------------------------------

v <- values(r)

v <- v[
    is.finite(v) &
    v > 0
]

cat("\nFootprint statistics:\n")
cat("Minimum :", min(v), "\n")
cat("Median  :", median(v), "\n")
cat("95%     :", quantile(v, 0.95), "\n")
cat("99%     :", quantile(v, 0.99), "\n")
cat("99.5%   :", quantile(v, 0.995), "\n")
cat("Maximum :", max(v), "\n")

# ------------------------------------------------------------
# Log10 transformation
# ------------------------------------------------------------

r_log <- log10(r + 1e-12)

# Limit the colour scale to the 99.5 percentile
vlog <- values(r_log)

vlog <- vlog[
    is.finite(vlog)
]

zmax <- as.numeric(
    quantile(
        vlog,
        0.995,
        na.rm = TRUE
    )
)

zmin <- min(vlog, na.rm = TRUE)

cat("\nLog plot range:\n")
cat("zmin =", zmin, "\n")
cat("zmax =", zmax, "\n")

# ------------------------------------------------------------
# Colour palette
# ------------------------------------------------------------

foot_cols <- colorRampPalette(
    c(
        "darkblue",
        "blue",
        "cyan",
        "green",
        "yellow",
        "orange",
        "red"
    )
)(100)

# ------------------------------------------------------------
# X11
# ------------------------------------------------------------

X11(
    width = 10,
    height = 8
)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plot(
    r_log,
    col = foot_cols,
    zlim = c(zmin, zmax),
    main = paste0(
        "STILT footprint + Upper Hunter mines\n",
        "Singleton — ",
        format(dates[k], "%Y-%m-%d %H:%M"),
        " UTC"
    ),
    xlab = "Longitude",
    ylab = "Latitude"
)

# ============================================================
# Mine polygons
# ============================================================

for (i in 1:13) {

    if (is.null(mines[[i]]))
        next

    lines(
        mines[[i]],
        lwd = 2
    )

    ctr <- centroids(mines[[i]])

    xy <- crds(ctr)

    text(
        xy[,1],
        xy[,2],
        labels = paste0("OC", i),
        cex = 0.75,
        pos = 3,
        font = 2
    )
}

# ============================================================
# Singleton receptor
# ============================================================

points(
    151.16132,
    -32.57123,
    pch = 8,
    cex = 2.5,
    lwd = 3
)

text(
    151.16132,
    -32.57123,
    labels = " Singleton",
    pos = 4,
    font = 2
)

# ============================================================
# Plot one STILT footprint: ON SCREEN + PNG
# Log-scaled footprint for better visualization
# ============================================================

k <- 1

r <- make_foot_raster(k)

# ============================================================
# 1. Log-transform for visualization
# ============================================================

# Small value only to avoid log10(0)
r_log <- log10(r + 1e-12)

# ------------------------------------------------------------
# Determine colour scale
# ------------------------------------------------------------

v <- values(r)

v <- v[
    is.finite(v) &
    v > 0
]

# Work in log space
v_log <- log10(v)

# Ignore the extreme upper tail for visualization
zmin <- quantile(
    v_log,
    0.01,
    na.rm = TRUE
)

zmax <- quantile(
    v_log,
    0.995,
    na.rm = TRUE
)

cat("\nFootprint plotting range:\n")
cat("1%    =", zmin, "\n")
cat("99.5% =", zmax, "\n")

# ============================================================
# 2. Colour palette
# ============================================================

foot_cols <- colorRampPalette(
    c(
        "darkblue",
        "blue",
        "cyan",
        "green",
        "yellow",
        "orange",
        "red"
    )
)(100)

# ============================================================
# 3. Function to draw complete map
# ============================================================

draw_map <- function() {

    plot(
        r_log,
        col = foot_cols,
        zlim = c(zmin, zmax),
        main = paste0(
            "STILT footprint + Upper Hunter mines\n",
            "Singleton — ",
            format(dates[k], "%Y-%m-%d %H:%M"),
            " UTC"
        ),
        xlab = "Longitude",
        ylab = "Latitude"
    )

    # --------------------------------------------------------
    # Mine polygons
    # --------------------------------------------------------

    for (i in 1:13) {

        if (is.null(mines[[i]]))
            next

        lines(
            mines[[i]],
            lwd = 2
        )

        ctr <- centroids(mines[[i]])

        xy <- crds(ctr)

        text(
            xy[, 1],
            xy[, 2],
            labels = paste0("OC", i),
            cex = 0.75,
            pos = 3,
            font = 2
        )
    }

    # --------------------------------------------------------
    # Singleton receptor
    # --------------------------------------------------------

    points(
        151.16132,
        -32.57123,
        pch = 8,
        cex = 2.5,
        lwd = 3
    )

    text(
        151.16132,
        -32.57123,
        labels = " Singleton",
        pos = 4,
        font = 2
    )
}

# ============================================================
# 4. OPEN X11 AND DISPLAY ON SCREEN
# ============================================================

X11(
    width = 10,
    height = 8
)

draw_map()

cat("\nMap displayed on X11.\n")

# ============================================================
# 5. SAVE EXACTLY THE SAME MAP TO PNG
# ============================================================

outfile <- file.path(
    outdir,
    sprintf(
        "footprint_%02d_%s_mines_log.png",
        k,
        format(dates[k], "%Y%m%d_%H%M")
    )
)

# Copy current X11 plot to PNG
dev.copy(
    png,
    filename = outfile,
    width = 1300,
    height = 1000,
    res = 140
)

# Close PNG device
dev.off()

cat(
    "Created PNG: ",
    outfile,
    "\n",
    sep = ""
)

# ============================================================
# 6. Make sure X11 is active again
# ============================================================

cat("\nCurrent graphics device:\n")
print(dev.cur())

# ============================================================
# 8. Combined footprint: sum of all available time layers
# ============================================================

cat("\nCreating combined footprint...\n")

combined <- apply(
    foot,
    c(1, 2),
    sum,
    na.rm = TRUE
)

r_combined <- rast(
    nrows = length(lat),
    ncols = length(lon),
    xmin = min(lon),
    xmax = max(lon),
    ymin = min(lat),
    ymax = max(lat),
    crs = "EPSG:4326"
)

combined <- t(combined)

combined <- combined[
    nrow(combined):1,
]

values(r_combined) <- as.vector(combined)

outfile <- file.path(
    outdir,
    "footprint_combined_4layers_mines.png"
)

png(
    outfile,
    width = 1300,
    height = 1000,
    res = 140
)

plot(
    r_combined,
    main = paste0(
        "STILT footprint + Upper Hunter mines\n",
        "Singleton — sum of 4 footprint layers"
    ),
    xlab = "Longitude",
    ylab = "Latitude"
)

# Mine polygons
for (i in 1:13) {

    if (is.null(mines[[i]]))
        next

    lines(
        mines[[i]],
        lwd = 2
    )

    ctr <- centroids(mines[[i]])
    xy <- crds(ctr)

    text(
        xy[,1],
        xy[,2],
        labels = paste0("OC", i),
        cex = 0.75,
        pos = 3
    )
}

# Singleton
points(
    151.16132,
    -32.57123,
    pch = 8,
    cex = 2.5,
    lwd = 3
)

text(
    151.16132,
    -32.57123,
    labels = " Singleton",
    pos = 4,
    font = 2
)

dev.off()

cat("Created: ", outfile, "\n")

#########################
# Better plot of footprints over mines
#####################################

# ============================================================
# Combined footprint + mines
# Plot ON SCREEN (X11) and SAVE PNG
# ============================================================

library(terra)

cat("\nCreating combined footprint...\n")

# ------------------------------------------------------------
# 1. Sum all available footprint layers
# ------------------------------------------------------------

combined <- apply(
    foot,
    c(1, 2),
    sum,
    na.rm = TRUE
)

# ------------------------------------------------------------
# 2. Create raster
# ------------------------------------------------------------

r_combined <- rast(
    nrows = length(lat),
    ncols = length(lon),
    xmin = min(lon),
    xmax = max(lon),
    ymin = min(lat),
    ymax = max(lat),
    crs = "EPSG:4326"
)

# Correct orientation
combined <- t(combined)

combined <- combined[
    nrow(combined):1,
]

values(r_combined) <- as.vector(combined)

# ============================================================
# 3. Plotting range
# ============================================================

vals <- values(r_combined)

vals <- vals[
    is.finite(vals) &
    vals >= 0
]

upper <- as.numeric(
    quantile(
        vals,
        probs = 0.995,
        na.rm = TRUE
    )
)

cat("\nFootprint statistics:\n")
cat("Minimum :", min(vals), "\n")
cat("Median  :", median(vals), "\n")
cat("95%     :", quantile(vals, 0.95), "\n")
cat("99%     :", quantile(vals, 0.99), "\n")
cat("99.5%   :", upper, "\n")
cat("Maximum :", max(vals), "\n")

# Visualization-only raster
r_plot <- clamp(
    r_combined,
    lower = 0,
    upper = upper,
    values = TRUE
)

# ============================================================
# 4. Colours
# ============================================================

foot_cols <- colorRampPalette(
    c(
        "darkblue",
        "blue",
        "cyan",
        "green",
        "yellow",
        "orange",
        "red"
    )
)(100)

# ============================================================
# 5. Function containing the complete plot
# ============================================================

draw_footprint_plot <- function() {

    plot(
        r_plot,
        col = foot_cols,
        zlim = c(0, upper),
        main = paste0(
            "STILT footprint + Upper Hunter mines\n",
            "Singleton — sum of 4 footprint layers"
        ),
        xlab = "Longitude",
        ylab = "Latitude",
        axes = TRUE
    )

    # --------------------------------------------------------
    # Mine polygons
    # --------------------------------------------------------

    for (i in 1:13) {

        if (is.null(mines[[i]]))
            next

        lines(
            mines[[i]],
            lwd = 2
        )

        ctr <- centroids(mines[[i]])

        xy <- crds(ctr)

        text(
            xy[, 1],
            xy[, 2],
            labels = paste0("OC", i),
            cex = 0.75,
            pos = 3,
            font = 2
        )
    }

    # --------------------------------------------------------
    # Singleton receptor
    # --------------------------------------------------------

    points(
        151.16132,
        -32.57123,
        pch = 8,
        cex = 2.5,
        lwd = 3
    )

    text(
        151.16132,
        -32.57123,
        labels = " Singleton",
        pos = 4,
        font = 2
    )
}

# ============================================================
# 6. Open X11 device
# ============================================================

X11(
    width = 10,
    height = 8
)

# Draw on screen
draw_footprint_plot()

# ============================================================
# 7. Save exactly the same plot to PNG
# ============================================================

outfile <- file.path(
    outdir,
    "footprint_combined_4layers_mines.png"
)

png(
    outfile,
    width = 1300,
    height = 1000,
    res = 140
)

draw_footprint_plot()

dev.off()

cat(
    "\nCreated PNG: ",
    outfile,
    "\n",
    sep = ""
)

# ============================================================
# 8. Return to X11 device
# ============================================================

# Check devices
dev.list()

# Make X11 the active device
# Usually it will be device 2 after PNG is closed,
# but select it explicitly if necessary.

if ("X11" %in% names(dev.list())) {
    dev.set(
        which = which(names(dev.list()) == "X11")[1]
    )
}

cat("\nActive device:\n")
print(dev.cur())


# ============================================================
# 9. Print footprint values over each mine
# ============================================================

cat("\n============================================================\n")
cat("FOOTPRINT OVER MINE POLYGONS\n")
cat("============================================================\n")

for (i in 1:13) {

    if (is.null(mines[[i]]))
        next

    # Extract combined footprint inside polygon
    x <- extract(
        r_combined,
        mines[[i]],
        fun = sum,
        na.rm = TRUE
    )

    val <- if (nrow(x) > 0) {
        x[,2]
    } else {
        NA
    }

    cat(
        sprintf(
            "OC%-2d  footprint sum = %.8e\n",
            i,
            val
        )
    )
}

cat("\nDone.\n")

