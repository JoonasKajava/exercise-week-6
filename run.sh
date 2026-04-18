#!/bin/bash
#SBATCH --account=project_2018026
#SBATCH --job-name=analysis
#SBATCH --time=00:05:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=small
module load python-data
module load gcc/11.3.0 openmpi/4.1.4

srun --mpi=pmix_v3 python analysis-with-mpi.py /projappl/project_2018026/super_data
