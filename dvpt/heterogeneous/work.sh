#!/bin/bash

#SBATCH --nodes=2
#SBATCH --image=edmondchau/fastpm-python:latest
#SBATCH --time=00:02:00 --qos=debug --constraint=knl --account=desi
#SBATCH --job-name=test --output="%x-%j.out"

# collect the list of node names
scontrol show hostname $SLURM_NODELIST > nodelist_$SLURM_JOB_ID.txt
# create hostfile with rank 0 isolated in the first node
python build_hostfile.py -i nodelist_$SLURM_JOB_ID.txt -o hostfile_$SLURM_JOB_ID.txt --ntask_per_nodes 10
# some cleaning
rm nodelist_$SLURM_JOB_ID.txt
# Initialize SLURM_HOSTFILE for --distribution=arbitrary option in srun
export SLURM_HOSTFILE=hostfile_$SLURM_JOB_ID.txt

## OU alors ca :
# # Initialize SLURM_HOSTFILE and built it for --distribution=arbitrary option in srun
# export SLURM_HOSTFILE=hostfile_$SLURM_JOB_ID.txt
# end=1 # rank 0 isolated in node 0
# for node in `scontrol show hostname $SLURM_NODELIST`; do
#     for ((i=1; i<=$end; i++)); do echo $node >> $SLURM_HOSTFILE; done
#     end=68 # 68 ranks per nodes
# done

srun -n 11 --distribution=arbitrary python test.py
