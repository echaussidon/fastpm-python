#!/bin/bash


## Heteregenous job --> does not work due to mpi communication problem between the two componants...

#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=68
##SBATCH --image=edmondchau/fastpm-python:latest
#SBATCH --time=00:01:00 --qos=debug --constraint=knl --account=desi
#SBATCH --job-name=test --output="%x-%j.out"

#SBATCH hetjob
#SBATCH --nodes=1 --ntasks=2 --cpus-per-task=1
##SBATCH --image=edmondchau/fastpm-python:latest
#SBATCH --time=00:01:00 --qos=debug --constraint=knl --account=desi

module load python

srun --mpi=pmi2 --resv-ports=0 --het-group=0,1 python test.py

## Test with map-cpu

#
#    # #SBATCH --image=edmondchau/fastpm-python:latest
#    # SBATCH --time=00:02:00 --qos=debug --constraint=knl --account=desi
#    # SBATCH --job-name=test --output="%x-%j.out"
#
# srun --cpu-bind=map_cpu:0,300,305,309 python test.py
