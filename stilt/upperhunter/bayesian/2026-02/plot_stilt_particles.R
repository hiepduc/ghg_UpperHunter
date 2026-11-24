
#!/usr/bin/env Rscript

# ============================================================
# Plot STILT backward particle trajectories and distributions
# Upper Hunter - Merriwa receptor
# ============================================================

suppressPackageStartupMessages({
    library(ggplot2)
    library(dplyr)
    library(sf)
})

# ============================================================
# USER SETTINGS
# ============================================================

INPUT_FILE <- paste0(
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/",
    "upperhunter/arl_mine_2026/2026-02/",
    "footprints/Merriwa/particles/",
    "202602010100_150.35767_-32.14097_5_traj.rds"
)

MINE_SHP_DIR <- paste0(
    "/mnt/scratch_lustre/nzem_meim_scratch/data/",
    "emission_location"
)

COASTLINE_FILE <- paste0(
    "/mnt/scratch_lustre/nzem_meim_scratch/data/",
    "coastline/ne_50m_coastline.shp"
)

OUTPUT_DIR <- paste0(
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/",
    "upperhunter/bayesian/2026-02/",
    "STILT_particle_plots"
)

RECEPTOR_NAME <- "Merriwa"
RECEPTOR_LON <- 150.35767
RECEPTOR_LAT <- -32.14097

# Desired backward age for particle distribution plot
PLOT_AGE_HOURS <- 12

# Figure dimensions
FIG_WIDTH <- 11
FIG_HEIGHT <- 9
FIG_DPI <- 300

# Display plots interactively through X11 (MobaXterm)
SHOW_PLOTS <- TRUE

# Check whether X11 forwarding is available
X11_AVAILABLE <- nzchar(Sys.getenv("DISPLAY"))

if (SHOW_PLOTS && !X11_AVAILABLE) {
    warning(
        "DISPLAY is not set. Plots will be saved as PNG only. ",
        "Reconnect using MobaXterm with X11 forwarding enabled."
    )
}

# Trajectory appearance
TRAJECTORY_ALPHA <- 0.30
TRAJECTORY_WIDTH <- 0.25

# Particle appearance
PARTICLE_SIZE <- 1.5

dir.create(
    OUTPUT_DIR,
    recursive = TRUE,
    showWarnings = FALSE
)

# ============================================================
# READ RDS FILE
# ============================================================

cat("\n")
cat("====================================================\n")
cat("READING STILT PARTICLE TRAJECTORIES\n")
cat("====================================================\n")

traj <- readRDS(INPUT_FILE)

if (!is.list(traj) || is.null(traj$particle)) {
    stop("The RDS object does not contain a particle data frame.")
}

particle <- as.data.frame(traj$particle)

required_columns <- c(
    "time", "indx", "long", "lati", "zagl",
    "foot", "mlht"
)

missing_columns <- setdiff(
    required_columns,
    names(particle)
)

if (length(missing_columns) > 0) {
    stop(
        "Missing particle columns: ",
        paste(missing_columns, collapse = ", ")
    )
}

# Remove invalid coordinates
particle <- particle %>%
    filter(
        is.finite(long),
        is.finite(lati),
        is.finite(time),
        is.finite(indx)
    ) %>%
    mutate(
        age_hours = abs(time)
    )

cat("Receptor:", RECEPTOR_NAME, "\n")
cat("Number of particle records:", nrow(particle), "\n")
cat("Number of particles:", n_distinct(particle$indx), "\n")

cat(
    "Backward time range:",
    min(particle$time),
    "to",
    max(particle$time),
    "\n"
)

cat(
    "Backward age range:",
    min(particle$age_hours),
    "to",
    max(particle$age_hours),
    "hours\n"
)

cat(
    "Particle height range:",
    round(min(particle$zagl), 2),
    "to",
    round(max(particle$zagl), 2),
    "m AGL\n"
)

cat("\nAvailable backward ages:\n")
print(sort(unique(particle$age_hours)))

# ============================================================
# READ MINE SHAPEFILES
# ============================================================

cat("\n")
cat("====================================================\n")
cat("READING MINE SHAPEFILES\n")
cat("====================================================\n")

mine_list <- list()

for (i in 1:13) {

    mine_name <- paste0("OC", i)

    shp_file <- file.path(
        MINE_SHP_DIR,
        paste0(mine_name, ".shp")
    )

    if (!file.exists(shp_file)) {
        warning("Missing shapefile: ", shp_file)
        next
    }

    mine <- st_read(
        shp_file,
        quiet = TRUE
    )

    if (is.na(st_crs(mine))) {
        stop("Missing CRS for ", mine_name)
    }

    mine <- st_transform(
        mine,
        crs = 4326
    )

    mine$mine_name <- mine_name

    mine_list[[mine_name]] <- mine

    cat("Loaded", mine_name, "\n")
}

if (length(mine_list) == 0) {
    stop("No mine shapefiles were loaded.")
}

mines <- do.call(
    rbind,
    mine_list
)

# Mine label positions
mine_label_geometry <- st_point_on_surface(
    mines
)

mine_label_coordinates <- st_coordinates(
    mine_label_geometry
)

mine_labels <- data.frame(
    longitude = mine_label_coordinates[, 1],
    latitude = mine_label_coordinates[, 2],
    mine_name = mines$mine_name
)

# ============================================================
# READ COASTLINE
# ============================================================

cat("\nReading coastline...\n")

coastline <- st_read(
    COASTLINE_FILE,
    quiet = TRUE
)

if (is.na(st_crs(coastline))) {
    stop("Coastline shapefile has no CRS.")
}

coastline <- st_transform(
    coastline,
    crs = 4326
)

cat("Coastline loaded.\n")

# ============================================================
# COMMON MAP EXTENT
# ============================================================

lon_range <- range(
    particle$long,
    na.rm = TRUE
)

lat_range <- range(
    particle$lati,
    na.rm = TRUE
)

lon_padding <- max(
    0.05,
    diff(lon_range) * 0.05
)

lat_padding <- max(
    0.05,
    diff(lat_range) * 0.05
)

map_lon_min <- lon_range[1] - lon_padding
map_lon_max <- lon_range[2] + lon_padding

map_lat_min <- lat_range[1] - lat_padding
map_lat_max <- lat_range[2] + lat_padding

# ============================================================
# COMMON MAP LAYERS
# ============================================================

map_layers <- list(

    geom_sf(
        data = coastline,
        colour = "darkblue",
        linewidth = 0.35,
        inherit.aes = FALSE
    ),

    geom_sf(
        data = mines,
        fill = NA,
        colour = "black",
        linewidth = 0.55,
        inherit.aes = FALSE
    ),

    geom_text(
        data = mine_labels,
        aes(
            x = longitude,
            y = latitude,
            label = mine_name
        ),
        size = 2.8,
        fontface = "bold",
        colour = "black",
        inherit.aes = FALSE,
        check_overlap = TRUE
    ),

    geom_point(
        aes(
            x = RECEPTOR_LON,
            y = RECEPTOR_LAT
        ),
        shape = 8,
        size = 5,
        colour = "red",
        stroke = 1.1,
        inherit.aes = FALSE
    ),

    coord_sf(
        xlim = c(map_lon_min, map_lon_max),
        ylim = c(map_lat_min, map_lat_max),
        expand = FALSE
    ),

    labs(
        x = "Longitude (degrees east)",
        y = "Latitude (degrees north)"
    ),

    theme_bw(),

    theme(
        plot.title = element_text(
            size = 14,
            face = "bold",
            hjust = 0.5
        ),
        plot.subtitle = element_text(
            size = 10,
            hjust = 0.5
        ),
        axis.title = element_text(size = 11),
        axis.text = element_text(size = 9),
        panel.grid.major = element_line(
            colour = "grey85",
            linewidth = 0.3
        ),
        legend.position = "right"
    )
)

# ============================================================
# PLOT 1: INDIVIDUAL PARTICLE TRAJECTORIES
# ============================================================

cat("\n")
cat("====================================================\n")
cat("PLOTTING BACKWARD PARTICLE TRAJECTORIES\n")
cat("====================================================\n")

# Arrange each particle from release towards the past.
# This assumes time is negative for backward transport.
trajectory_data <- particle %>%
    arrange(indx, desc(time))

p1 <- ggplot() +

    geom_path(
        data = trajectory_data,
        aes(
            x = long,
            y = lati,
            group = indx,
            colour = age_hours
        ),
        alpha = TRAJECTORY_ALPHA,
        linewidth = TRAJECTORY_WIDTH,
        lineend = "round"
    ) +

    map_layers +

    scale_colour_viridis_c(
        name = "Backward age\n(hours)",
        option = "turbo"
    ) +

    labs(
        title = paste(
            "STILT Backward Particle Trajectories -",
            RECEPTOR_NAME
        ),
        subtitle = paste(
            "Release:",
            format(traj$receptor$run_time, "%Y-%m-%d %H:%M UTC"),
            "|",
            n_distinct(particle$indx),
            "particles"
        )
    )

trajectory_file <- file.path(
    OUTPUT_DIR,
    paste0(
        RECEPTOR_NAME,
        "_202602010100_particle_trajectories.png"
    )
)

ggsave(
    filename = trajectory_file,
    plot = p1,
    width = FIG_WIDTH,
    height = FIG_HEIGHT,
    dpi = FIG_DPI
)

# Display trajectory plot in a separate X11 window
if (SHOW_PLOTS && X11_AVAILABLE) {

    grDevices::x11(
        width = FIG_WIDTH,
        height = FIG_HEIGHT,
        title = "STILT Backward Particle Trajectories"
    )

    print(p1)
    grDevices::dev.flush()

    cat("Trajectory plot displayed on screen.\n")
}

cat("Saved:", trajectory_file, "\n")

# ============================================================
# PLOT 2: PARTICLE DISTRIBUTION AT SELECTED BACKWARD AGE
# ============================================================

cat("\n")
cat("====================================================\n")
cat("PLOTTING PARTICLE DISTRIBUTION\n")
cat("====================================================\n")

available_ages <- sort(
    unique(particle$age_hours)
)

# Find the available time closest to requested age
selected_age <- available_ages[
    which.min(
        abs(available_ages - PLOT_AGE_HOURS)
    )
]

cat(
    "Requested backward age:",
    PLOT_AGE_HOURS,
    "hours\n"
)

cat(
    "Actual available backward age:",
    selected_age,
    "hours\n"
)

particle_snapshot <- particle %>%
    filter(age_hours == selected_age)

cat(
    "Number of particles in snapshot:",
    nrow(particle_snapshot),
    "\n"
)

p2 <- ggplot() +

    geom_point(
        data = particle_snapshot,
        aes(
            x = long,
            y = lati,
            colour = zagl
        ),
        size = PARTICLE_SIZE,
        alpha = 0.75
    ) +

    map_layers +

    scale_colour_viridis_c(
        name = "Particle height\n(m AGL)",
        option = "turbo"
    ) +

    labs(
        title = paste(
            "STILT Particle Distribution -",
            RECEPTOR_NAME
        ),
        subtitle = paste(
            "Backward age:",
            selected_age,
            "hours |",
            nrow(particle_snapshot),
            "particles"
        )
    )

distribution_file <- file.path(
    OUTPUT_DIR,
    paste0(
        RECEPTOR_NAME,
        "_202602010100_particle_distribution_",
        selected_age,
        "h.png"
    )
)

ggsave(
    filename = distribution_file,
    plot = p2,
    width = FIG_WIDTH,
    height = FIG_HEIGHT,
    dpi = FIG_DPI
)

# Display trajectory plot in a separate X11 window
if (SHOW_PLOTS && X11_AVAILABLE) {

    grDevices::x11(
        width = FIG_WIDTH,
        height = FIG_HEIGHT,
        title = "STILT Backward Particle Trajectories"
    )

    print(p2)
    grDevices::dev.flush()

    cat("Trajectory plot displayed on screen.\n")
}

cat("Saved:", trajectory_file, "\n")

# ============================================================
# FINISHED
# ============================================================

cat("\n")
cat("====================================================\n")
cat("PARTICLE PLOTTING COMPLETED\n")
cat("====================================================\n")

cat("Trajectory plot:\n", trajectory_file, "\n\n")
cat("Particle distribution plot:\n", distribution_file, "\n")
cat("\n")

# ============================================================
# KEEP GRAPHICAL WINDOWS OPEN
# ============================================================

if (SHOW_PLOTS && X11_AVAILABLE) {

    cat("\n")
    cat("====================================================\n")
    cat("INTERACTIVE PLOT DISPLAY\n")
    cat("====================================================\n")

    cat("Two graphical windows have been opened.\n")
    cat("Close the plot windows when finished.\n")
    cat("Press ENTER in the terminal to finish the R script.\n")

    readline()
}

