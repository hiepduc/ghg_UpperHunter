
#!/usr/bin/env Rscript

# Inspect the structure of a STILT particle trajectory RDS file

input_file <- paste0(
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/upperhunter/arl_mine_2026/2026-03/footprints/Merriwa/particles/202603010100_150.35767_-32.14097_5_traj.rds"
)

cat("Reading:", input_file, "\n\n")

traj <- readRDS(input_file)

cat("========================================\n")
cat("OBJECT CLASS\n")
cat("========================================\n")
print(class(traj))

cat("\n========================================\n")
cat("OBJECT STRUCTURE\n")
cat("========================================\n")
str(traj, max.level = 3)

cat("\n========================================\n")
cat("OBJECT NAMES\n")
cat("========================================\n")
print(names(traj))

if (is.data.frame(traj)) {

    cat("\n========================================\n")
    cat("DATA FRAME DIMENSIONS\n")
    cat("========================================\n")
    print(dim(traj))

    cat("\n========================================\n")
    cat("COLUMN NAMES\n")
    cat("========================================\n")
    print(names(traj))

    cat("\n========================================\n")
    cat("FIRST 10 ROWS\n")
    cat("========================================\n")
    print(head(traj, 10))

    cat("\n========================================\n")
    cat("SUMMARY\n")
    cat("========================================\n")
    print(summary(traj))
}

