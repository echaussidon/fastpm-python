#!/bin/bash

#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=68
#SBATCH --image=edmondchau/fastpm-python:latest
#SBATCH --time=00:01:00 --qos=debug --constraint=knl --account=desi
#SBATCH --job-name=test --output="%x-%j.out"

#SBATCH hetjob
#SBATCH --nodes=1 --ntasks=2 --cpus-per-task=1
#SBATCH --image=edmondchau/fastpm-python:latest
#SBATCH --time=00:01:00 --qos=debug --constraint=knl --account=desi

module load python

srun python test.py : python test.py
