""" Test memory implication of what is coded in fof_catalog compare to what we can do with sparse idea (ie) conserving the information only where the value is not zero"""

from mpi4py import MPI
import numpy as np
import time

from fastpm.memory_monitor import MemoryMonitor, plot_memory

# import scipy.sparse as ss

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

mem = MemoryMonitor(log_file=f'memory-monitor/test1-memory_monitor_rank_{rank}.txt')
mem()

# generate int label as halos id
N_max = 1000000
n_part = 20000
np.random.seed(2021 + rank * 100)
label = np.random.randint(1, N_max + 1, n_part)  # exclude right
mem()
# print(f"rank={rank}: label={label}")
time.sleep(0.5)
mem()

# count the number of particle inside the same halos --> How it is done in fof
N = np.bincount(label, minlength=N_max)
mem()
# print(f"rank={rank}: N size={N.size}")
for i in range(N.size):
    _ = N[i]
time.sleep(0.5)
mem()

comm.Barrier()
# Collect the information in all the processor
comm.Allreduce(MPI.IN_PLACE, N, op=MPI.SUM)
mem()
# print(f"rank={rank}: N size={N.size}")
time.sleep(0.5)
mem()

mem.stop_monitoring()
comm.Barrier()

if rank == 0:
    plot_memory('.', prefix='test1-')

if rank == 0:
    print(N.shape)
    print(np.argwhere(N > 0))
    print(N[N > 0])
    np.save('test1.npy', N)
    print("\n \n")

# do the same thing to compare the execution time
tic = MPI.Wtime()
# count the number of particle inside the same halos --> How it is done in fof
N = np.bincount(label, minlength=N_max)
# Collect the information in all the processor
comm.Allreduce(MPI.IN_PLACE, N, op=MPI.SUM)
toc = MPI.Wtime()
if rank == 0:
    print(f"Done in {toc - tic} s.")
