 
(1) First enter the spack environment for WRF-Chem
source /mnt/scratch_lustre/ar_biogen_scratch/stack/aqf-med-s15/spack/share/spack/setup-env.sh
spack load intel-oneapi-compilers-classic@2021.10.0
spack find --loaded
spack load netcdf-fortran
spack load hdf5
spack location -i netcdf-fortran
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/scratch_lustre/ar_biogen_scratch/stack/aqf-med-s15/spack/opt/spack/linux-x86_64_v3/netcdf-fortran-4.6.2-6xxxvdhat3tdlcydf56dey23pmxod4sx/lib
ldd /mnt/scratch_lustre/duch/flexpart-wrf/flexpart-wrf/flexwrf33_omp
ldd /mnt/scratch_lustre/duch/flexpart-wrf/flexpart-wrf/flexwrf33_omp | grep "not found"
spack find | grep -Ei 'intel|oneapi|intel-oneapi'
spack load intel-oneapi-compilers-classic@2021.10.0
echo $LD_LIBRARY_PATH
spack location -i intel-oneapi-compilers-classic@2021.10.0
PREFIX=$(spack location -i intel-oneapi-compilers-classic@2021.10.0)
echo "$PREFIX"
find "$PREFIX" -type f   \( -name 'libifport.so*'   -o -name 'libifcoremt.so*'   -o -name 'libimf.so*'   -o -name 'libsvml.so*'   -o -name 'libiomp5.so*'   -o -name 'libintlc.so*' \)   2>/dev/null
export INTEL_LIB=/mnt/scratch_lustre/ar_biogen_scratch/stack/aqf-med-s15/spack/opt/spack/linux-x86_64_v3/intel-oneapi-compilers-2023.2.4-phxvcdfl6i64vlmkipsngnsfpqnrxosy/compiler/latest/linux/compiler/lib/intel64_lin
export LD_LIBRARY_PATH=$INTEL_LIB:$LD_LIBRARY_PATH


spack find --loaded

(2) The makefile needs some modification

For omp linking 

ifort *.o -o flexwrf33_omp \
>   -L/mnt/aqf2-scratch1/forecast/wrf-cmaq/inputs/builds/aqf-med-s15sp5/spack/opt/spack/linux-sles15-x86_64_v3/intel-2021.10.0/netcdf-fortran-4.6.1-jzmova4nferkvwoszojfsqtvt7mprcyz/lib \
>   -lnetcdff \
>   -qopenmp -O2 -mcmodel=medium -shared-intel

Or full pathname of ifort

/mnt/aqf2-scratch1/forecast/wrf-cmaq/inputs/builds/aqf-med-s15sp5/spack/opt/spack/linux-sles15-x86_64_v3/gcc-12.4.0/intel-oneapi-compilers-2023.2.4-akc7voxnaxs43c4uycch5sioifpq3sjf/compiler/2023.2.4/linux/bin/intel64/ifort *.o -o flexwrf33_omp   -L/mnt/aqf2-scratch1/forecast/wrf-cmaq/inputs/builds/aqf-med-s15sp5/spack/opt/spack/linux-sles15-x86_64_v3/intel-2021.10.0/netcdf-fortran-4.6.1-jzmova4nferkvwoszojfsqtvt7mprcyz/lib   -lnetcdff   -qopenmp -O2 -mcmodel=medium -shared-intel

(3) Running flexpart

flexwrf33_omp flexwrf_example.input


                         DAILY SOURCE PARAMETERS
                  ┌──── Day 1 ────┐ ┌──── Day 2 ────┐
                  OC1 ... OC13     OC1 ... OC13
Observation 1     H11 ... H1,13    H1,14 ... H1,26
Observation 2     H21 ... H2,13    H2,14 ... H2,26
...
Observation N
