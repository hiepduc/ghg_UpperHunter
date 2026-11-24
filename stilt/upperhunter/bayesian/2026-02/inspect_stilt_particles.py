
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspect STILT particle trajectory data stored in an RDS file.

Uses rdata rather than pyreadr because the STILT RDS file
contains a nested R list.
"""

import rdata
import numpy as np
import pandas as pd


RDS_FILE = (
    "/mnt/scratch_lustre/nzem_meim_scratch/stilt/"
    "upperhunter/arl_mine_2026/2026-03/footprints/"
    "Merriwa/particles/"
    "202603010100_150.35767_-32.14097_5_traj.rds"
)


def print_structure(obj, name="root", indent=0):

    prefix = " " * indent

    print(f"{prefix}{name}: {type(obj)}")

    if isinstance(obj, dict):

        for key, value in obj.items():
            print_structure(value, str(key), indent + 4)

    elif isinstance(obj, (list, tuple)):

        for i, value in enumerate(obj):
            print_structure(value, f"[{i}]", indent + 4)

    elif isinstance(obj, pd.DataFrame):

        print(f"{prefix}  Shape: {obj.shape}")
        print(f"{prefix}  Columns: {list(obj.columns)}")


print("=" * 70)
print("READING STILT PARTICLE RDS FILE")
print("=" * 70)

# Read the RDS file
data = rdata.read_rds(RDS_FILE)

print("\nRDS object structure:")
print_structure(data)

# Extract particle data
if isinstance(data, dict) and "particle" in data:

    particle = data["particle"]

elif hasattr(data, "particle"):

    particle = data.particle

else:

    raise ValueError(
        "Could not locate the particle data frame. "
        "Please check the object structure printed above."
    )

# Convert to pandas DataFrame if necessary
if not isinstance(particle, pd.DataFrame):

    particle = pd.DataFrame(particle)

print("\n" + "=" * 70)
print("PARTICLE DATA INFORMATION")
print("=" * 70)

print("\nDataFrame information:")
print(particle.info())

print("\nFirst 10 records:")
print(particle.head(10).to_string())

print("\nColumn names:")
print(particle.columns.tolist())

# Check time
print("\n" + "=" * 70)
print("PARTICLE TIME INFORMATION")
print("=" * 70)

print("\nTime statistics:")
print(particle["time"].describe())

unique_times = np.sort(particle["time"].dropna().unique())

print("\nUnique time values:")
print(unique_times)

print("\nNumber of unique time values:")
print(len(unique_times))

# Particle IDs
print("\n" + "=" * 70)
print("PARTICLE ID INFORMATION")
print("=" * 70)

print("\nNumber of particles:")
print(particle["indx"].nunique())

print("\nRecords per particle:")
print(particle.groupby("indx").size().describe())

# Spatial coordinates
print("\n" + "=" * 70)
print("PARTICLE LOCATION INFORMATION")
print("=" * 70)

for column in ["long", "lati", "zagl", "foot"]:

    print(f"\n{column}:")
    print(particle[column].describe())

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)


