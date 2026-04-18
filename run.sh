#!/bin/bash
#SBATCH --account=project_2018026
#SBATCH --job-name=analysis
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=small
module load python-data
srun python analysis.py /projappl/project_2018026/super_data
