#!/bin/bash
#SBATCH --job-name=UH_MCMC
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=UH_MCMC_%j.out
#SBATCH --error=UH_MCMC_%j.err

set -e

cd ~/flexpart-wrf/upperhunter_flexpart

echo "=========================================="
echo "Upper Hunter blocked MH MCMC"
echo "=========================================="

date
hostname

which python3
python3 --version

python3 scripts/bayesian_inverse_mcmc.py

date
