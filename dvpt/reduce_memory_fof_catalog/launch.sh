#!/bin/bash
# sbatch launch.sh

#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --time=00:10:00
#SBATCH --constraint=haswell
#SBTACH --account=desi

#SBATCH --job-name=test-memory-fof
#SBATCH --output="%x-%j.out"

export OMP_NUM_THREADS=1

# Parameters:
nproc=64

echo
echo "Start test 1 --> implementation in fof_catalog"
srun -n $nproc python test1.py
echo

echo
echo "Start test 3 --> New implementation (slower but use less memory)"
srun -n $nproc python test3.py
echo

echo
echo "Compare the two output --> Should be True"
python compare_test.py
echo
