#!/bin/bash
#SBATCH --account=project_2018026
#SBATCH --job-name=analysis
#SBATCH --time=00:05:00
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=small
module load python-data
module load openmpi

srun python analysis-with-mpi.py /projappl/project_2018026/super_data
