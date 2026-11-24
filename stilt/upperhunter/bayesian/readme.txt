The work flow as below

(1) Create the footprint from STILT program

use the script

run_stilt_jan2026.sh
run_stilt_feb2026.sh
...

(2) Create the H matrix

use the script

create_stilt_H_matrix_13mines_revised.py

(3) Check and diagnose the H matrix

use the script

diagnostic_H_matrix.py

(4) Run inverse method using Bayesian

use the script

posterior_emission_daily_scaling_factors.py

(5) Plotting the results

plot_posterior_daily_scaling_factors_Jan2026.py
plot_prior_posterior_receptor_comparison_Jan2026_new.py
plot_prior_posterior_receptor_comparison_Jan2026.py

To run/plot R script use

R_LIBS_USER="$CONDA_PREFIX/lib/R/library" $CONDA_PREFIX/bin/Rscript --vanilla plot_stilt_particles.R

