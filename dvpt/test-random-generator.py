import mpytools as mpy
from mpi4py import MPI
mpicomm = MPI.COMM_WORLD
rank = mpicomm.Get_rank()
start_ini = MPI.Wtime()

rng = mpy.random.MPIRandomState(int(10 / mpicomm.size), seed=42)  # invariant under number of processes

a = rng.uniform(low=0., high=10., dtype=int)

print('Rank: ', rank, ', ', a)
