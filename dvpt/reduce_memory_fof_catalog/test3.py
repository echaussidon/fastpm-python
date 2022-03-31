""" Test memory implication of what is coded in fof_catalog compare to what we can do with sparse idea (ie) conserving the information only where the value is not zero"""

from mpi4py import MPI
import numpy as np
import time

from fastpm.memory_monitor import MemoryMonitor, plot_memory
from fastpm.utils import GatherArray

# import scipy.sparse as ss

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
tic = MPI.Wtime()

mem = MemoryMonitor(log_file=f'memory-monitor/test3-memory_monitor_rank_{rank}.txt')
mem()

# generate int label as halos id
N_max = 1000000
n_part = 20000
np.random.seed(2021 + rank * 100)
label = np.random.randint(1, N_max + 1, n_part)  # exclude right
mem()
print(f"rank={rank}: label={label}")
time.sleep(0.5)
mem()

# count the number of particle inside the same halos --> How it is done in fof
unique_label, counts = np.unique(label, return_counts=True)
mem()
print(f"rank={rank}: unique_label={unique_label.size} -- nbr_counts={counts.sum()}")
time.sleep(0.5)
mem()

comm.Barrier()

# Collect the information in all the processor
nbr_unique_label_per_rank = GatherArray(np.array([unique_label.size]), comm=comm, root=0)
unique_label = GatherArray(unique_label, comm=comm, root=0)
counts = GatherArray(counts, comm=comm, root=0)


mem()
print(f"rank={rank}: unique_label={unique_label} counts={counts}")
time.sleep(0.5)
mem()

if rank == 0:
    print(nbr_unique_label_per_rank)
    N = np.zeros(N_max)
    start, stop = 0, 0
    for nbr_unique_label in nbr_unique_label_per_rank:
        stop += nbr_unique_label
        N[unique_label[start:stop]] += counts[start:stop]
        start += nbr_unique_label

mem()
time.sleep(0.5)
mem()

mem.stop_monitoring()
comm.Barrier()

if rank == 0:
    plot_memory('.', prefix='test3-')

if rank == 0:
    print(N.shape)
    print(np.argwhere(N > 0))
    print(N[N > 0])
    np.save('test3.npy', N)
    print("\n \n")


# do the same thing to compare the execution time
tic = MPI.Wtime()
# count the number of particle inside the same halos --> How it is done in fof
unique_label, counts = np.unique(label, return_counts=True)
# Collect the information in all the processor
comm.Barrier()
nbr_unique_label_per_rank = GatherArray(np.array([unique_label.size]), comm=comm, root=0)
unique_label = GatherArray(unique_label, comm=comm, root=0)
counts = GatherArray(counts, comm=comm, root=0)
if rank == 0:
    print(nbr_unique_label_per_rank)
    N = np.zeros(N_max)
    start, stop = 0, 0
    for nbr_unique_label in nbr_unique_label_per_rank:
        stop += nbr_unique_label
        N[unique_label[start:stop]] += counts[start:stop]
        start += nbr_unique_label

toc = MPI.Wtime()
if rank == 0:
    print(f"Done in {toc - tic} s.")
