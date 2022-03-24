#!/bin/bash
#SBATCH --image=edmondchau/fastpm-python:latest
#SBATCH --nodes=2
#SBATCH --qos=debug
#SBATCH --constraint=knl

# Problem with shifter at NERSC, it does not clear the PYTHONPATH... (exactly what we do not want to do...)
export PYTHONPATH=''

srun -n 64 shifter python -m fastpm.main /global/cscratch1/sd/edmondc/Mocks/debug/
