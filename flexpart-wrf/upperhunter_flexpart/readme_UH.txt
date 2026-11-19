In this directory, all the work on flexpart-wrf inverse modelling is carried out
We have the following subdirectories 

- scripts: All python scripts for each of the months to run flexpart_wrf backward trajectory to determine the footprints, scripts to build the H matrix
and scripts to run inverse modelling for analytical linear inverse solution (MAP), MCMC (Gibbs sampler), MCMC (Metropolis-Hastings). Also scripts to plotthe footprints, to check the H-matrix, to plot daily scaling factors, prediction

- run: The output result of flexpart-wrf backward trajectories for each of the receptors and contains all the footprints. Slurm job logs are in this run directory as well
 
- H_matrix_flexpart : The H-matrix as created by scripts

- inverse_results : All results of inverse modelling using different methods-   

- WRFOUT : directory to link (hook) to WRF meteorological files

- options, output, pathnames are legacy flexpart-wrf files (only for compatibilty)
