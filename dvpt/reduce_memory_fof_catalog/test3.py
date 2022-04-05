"""
Test memory implication of what is coded in fof_catalog compare to what we can do with sparse idea (ie) conserving the information only where the value is not zero

see: https://stackoverflow.com/questions/21088420/mpi4py-send-recv-with-tag
"""

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
N_max = 100000000
n_part = 200000

if rank == 0:
    print(N_max, n_part)
np.random.seed(2021 + rank * 100)
label = np.random.randint(1, N_max + 1, n_part)  # exclude right
mem()
# print(f"rank={rank}: label={label}")
time.sleep(0.5)
mem()

# count the number of particle inside the same halos
unique_label, counts = np.unique(label, return_counts=True)
mem()
# print(f"rank={rank}: unique_label={unique_label.size} -- nbr_counts={counts.sum()}")
time.sleep(0.5)
mem()

comm.Barrier()

# Send information to the root rank
nbr_unique_label = comm.gather(unique_label.size, root=0)

if rank != 0:  # .Send is a blocking communication
    comm.Send(unique_label, dest=0, tag=1)
    comm.Send(counts, dest=0, tag=2)

mem()
# print(f"rank={rank}: unique_label={unique_label} counts={counts}")
time.sleep(0.5)
mem()

if rank == 0:
    # Define used quantity to collect the output from all the ranks
    N = np.zeros(N_max, dtype='i8')

    # add rank 0
    N[unique_label] += counts

    # collect other rank
    for send_rank in range(1, comm.size):
        unique_label, counts = np.zeros(nbr_unique_label[send_rank], dtype='i8'), np.zeros(nbr_unique_label[send_rank], dtype='i8')
        comm.Recv(unique_label, source=send_rank, tag=1)
        comm.Recv(counts, source=send_rank, tag=2)
        N[unique_label] += counts

# pour que le dessin soit lisible
comm.Barrier()

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
    print(" ")

# do the same thing to compare the execution time
tic = MPI.Wtime()
# count the number of particle inside the same halos
unique_label, counts = np.unique(label, return_counts=True)

# Send information to the root rank
nbr_unique_label = comm.gather(unique_label.size, root=0)

if rank != 0:  # .Send is a blocking communication
    comm.Send(unique_label, dest=0, tag=1)
    comm.Send(counts, dest=0, tag=2)

if rank == 0:
    # Define used quantity to collect the output from all the ranks
    N = np.zeros(N_max, dtype='i8')

    # add rank 0
    N[unique_label] += counts

    # collect other rank
    for send_rank in range(1, comm.size):
        # TAKE CARE to the type used here, 'i8' can be not enough for large int value --> use np.int32 / np.int64
        unique_label, counts = np.zeros(nbr_unique_label[send_rank], dtype='i8'), np.zeros(nbr_unique_label[send_rank], dtype='i8')
        comm.Recv(unique_label, source=send_rank, tag=1)
        comm.Recv(counts, source=send_rank, tag=2)
        N[unique_label] += counts

toc = MPI.Wtime()
if rank == 0:
    np.save('test3.npy', N)
    print(f"\n Done in {toc - tic} s.\n")


# do the same thing to compare the execution time
tic = MPI.Wtime()
# count the number of particle inside the same halos
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
    np.save('test3.npy', N)
    print(f"\n Done in {toc - tic} s.\n")
