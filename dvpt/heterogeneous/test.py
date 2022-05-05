from mpi4py import MPI
import numpy as np
import socket


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print(f"Hello, I'm rank {rank} / {comm.size} runnning on {socket.gethostname()}", flush=True)

tt = np.zeros(3, dtype='i8')
to_send = np.array([1, 2, 3], dtype='i8')
print("INIT ", rank, tt, flush=True)

if rank == 1: comm.Send(to_send, dest=2, tag=1)
if rank == 2: comm.Recv(tt, source=1, tag=1)
print("A (rank 1 --> rank 2) ", rank, tt, flush=True)

if rank == 2: comm.Send(to_send, dest=0, tag=2)
if rank == 0: comm.Recv(tt, source=2, tag=2)
print("B (rank 2 --> rank 0)", rank, tt, flush=True)
