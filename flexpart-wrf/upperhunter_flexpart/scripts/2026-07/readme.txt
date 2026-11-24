 - First run create_hourly_obs_3layer.py  to create the directory structure and input files 
 - Then run generate_hourly_obs_inputs.py to create input files from observation data (missing data will not be used) for input to flexpart
 - Then run run_flexpart_3layer_array.slurm to run flexpart-wrf for backward trajectory to find the footprints for all observation in January 2026
 - Then build H matrix  build_H_flexpart_13mine.py
 - Finally run  bayesian_inverse_flexpart_13mine.py

(1) for slurm job to run flexpart-wrf to do backward trajectory to calculate the foot prints for the whole jan 2026.

run_flexpart_3layer_array.slurm is used to run for each of hour of observation.
Because there are some missing gap in the observation manifest.txt is used to tell slurm to run on day that has observation

SDC has  Slurm configuration has a maximum array index of 1000, so:


Since we  have 1,124 cases, the clean solution is to split the production into two arrays:

Array 1: tasks 1–1000
Array 2: tasks 1001–1124

We don't need to change the manifest or regenerate anything.

1. Submit the first 1,000
sbatch --array=1-1000%48 run_flexpart_3layer_array.slurm
2. Submit the remaining 124
MANIFEST_OFFSET=1000 sbatch --array=1-124%48 run_flexpart_3layer_array.slurm

production_manifest.txt
       │
       │ tells Slurm which case to run
       ▼
20260104_140000/
       ├── flexwrf.input
       └── AVAILABLE
              │
              ▼
       flexwrf33_omp
              │
              ▼
       flxout_d01_20260104_130000.nc

A safer approach is:

sbatch --array=1-1000%48 run_flexpart_3layer_array.slurm

Wait until that finishes, then:

MANIFEST_OFFSET=1000 sbatch --array=1-124%48 run_flexpart_3layer_array.slurm

(2) 
The manifest was created by the 3-layer case-generation script:

~/flexpart-wrf/upperhunter_flexpart/scripts/create_hourly_obs_3layer.py

That is the script we used to generate the 1,124 production case directories.

The manifest is:

~/flexpart-wrf/upperhunter_flexpart/run/hourly_obs_3layer/production_manifest.txt

create_hourly_obs_3layer.py
          │
          ├── creates 1,124 case directories
          │
          ├── creates flexwrf.input
          │
          ├── creates AVAILABLE
          │
          └── creates production_manifest.txt
                         │
                         ▼
              run_flexpart_3layer_array.slurm
                         │
                         ▼
                  flexwrf33_omp

(3)  Build H matrix

python3 build_H_flexpart_13mine.py

creating the H matrix file H_flexpart_0_10m_13mine_Jan2026.csv

(4) Finally, run bayesian_inverse_flexpart_13mine.py


Note: paths, options/ and output/ are the old flexpart-wrf, they are included here for compatibility
`
