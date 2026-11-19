 
(1) First enter the spack environment for WRF-Chem 4.4.1
To load the model:
source /mnt/aqf2-scratch1/forecast/wrf-cmaq/inputs/builds/aqf-med-s15sp5/spack/share/spack/setup-env.sh

(2) These packages have to be loaded in spack (david)

spack load intel-oneapi-compilers-classic@2021.10.0
spack load netcdf-fortran@4.6.1
spack load netcdf-c@4.9.2
spack load hdf5@1.14.3

spack find --loaded

(3) The makefile needs some modification

For omp linking 

ifort *.o -o flexwrf33_omp \
>   -L/mnt/aqf2-scratch1/forecast/wrf-cmaq/inputs/builds/aqf-med-s15sp5/spack/opt/spack/linux-sles15-x86_64_v3/intel-2021.10.0/netcdf-fortran-4.6.1-jzmova4nferkvwoszojfsqtvt7mprcyz/lib \
>   -lnetcdff \
>   -qopenmp -O2 -mcmodel=medium -shared-intel

Or full pathname of ifort

/mnt/aqf2-scratch1/forecast/wrf-cmaq/inputs/builds/aqf-med-s15sp5/spack/opt/spack/linux-sles15-x86_64_v3/gcc-12.4.0/intel-oneapi-compilers-2023.2.4-akc7voxnaxs43c4uycch5sioifpq3sjf/compiler/2023.2.4/linux/bin/intel64/ifort *.o -o flexwrf33_omp   -L/mnt/aqf2-scratch1/forecast/wrf-cmaq/inputs/builds/aqf-med-s15sp5/spack/opt/spack/linux-sles15-x86_64_v3/intel-2021.10.0/netcdf-fortran-4.6.1-jzmova4nferkvwoszojfsqtvt7mprcyz/lib   -lnetcdff   -qopenmp -O2 -mcmodel=medium -shared-intel

(b) Make sure to load netcdff (netcdf-fortran) in LD_LIBRARY_PATH

spack location -i netcdf-fortran@4.6.1
(/mnt/aqf2-scratch1/forecast/wrf-cmaq/inputs/builds/aqf-med-s15sp5/spack/opt/spack/linux-sles15-x86_64_v3/intel-2021.10.0/netcdf-fortran-4.6.1-jzmova4nferkvwoszojfsqtvt7mprcyz)

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/aqf2-scratch1/forecast/wrf-cmaq/inputs/builds/aqf-med-s15sp5/spack/opt/spack/linux-sles15-x86_64_v3/intel-2021.10.0/netcdf-fortran-4.6.1-jzmova4nferkvwoszojfsqtvt7mprcyz/lib

(3) Running flexpart

flexwrf33_omp flexwrf_example.input
