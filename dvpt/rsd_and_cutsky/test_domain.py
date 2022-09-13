import os

import numpy as np
import matplotlib.pyplot as plt

from fastpm.io import BigFile
from mockfactory import BoxCatalog

from pmesh.domain import GridND
from fastpm.utils import split_size_2d

from mpi4py import MPI
mpicomm = MPI.COMM_WORLD
rank = mpicomm.Get_rank()
start_ini = MPI.Wtime()


# LOAD DATA:
sim = os.path.join('/global/u2/e/edmondc/Scratch/Mocks/', 'test-memory')
start = MPI.Wtime()
halos = BigFile(os.path.join(sim, 'halos-0.5000'), dataset='1/', mode='r', mpicomm=mpicomm)
halos.Position = halos.read('Position')
box = BoxCatalog(data=halos, columns=['Position'], boxsize=halos.attrs['boxsize'][0], boxcenter=halos.attrs['boxsize'][0] // 2, mpicomm=mpicomm)
# recenter the box to make rotation easier
box.recenter()
mpicomm.Barrier()
print(" ")
mpicomm.Barrier()

# TEST DATA DISTRIBUTION
if True:
    print(f'rank={rank}: Stage1: x -> min, max', np.min(box['Position'][:, 0]), np.max(box['Position'][:, 0]))
    print(f'rank={rank}: Stage1: y -> min, max', np.min(box['Position'][:, 1]), np.max(box['Position'][:, 1]))

    plt.figure()
    plt.scatter(box['Position'][:, 0], box['Position'][:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'test-rank-1-{rank}.png')
    plt.close()

# GROUP THE DATA

periodic = True
nd = split_size_2d(mpicomm.size)  # np.prod(nd) = mpicomm.size

position = box['Position'][:, :2]

if periodic:
    BoxSize = box.boxsize
    if np.isscalar(BoxSize):
        BoxSize = [BoxSize, BoxSize]
    left = [0, 0, 0]
    right = BoxSize
else:
    BoxSize = None
    left = np.min(mpicomm.allgather(position.min(axis=0)), axis=0)
    right = np.max(mpicomm.allgather(position.max(axis=0)), axis=0)

grid = [np.linspace(left[0], right[0], nd[0] + 1, endpoint=True),
        np.linspace(left[1], right[1], nd[1] + 1, endpoint=True)]
domain = GridND(grid, comm=mpicomm, periodic=periodic)

domain.loadbalance(domain.load(position))  # balance the load
layout = domain.decompose(position, smoothing=0)

position = layout.exchange(position, pack=False)  # exchange first particles


# TEST THE NEW DISTRIBUTION:
if True:
    print(f'rank={rank}: Stage2: x -> min, max', np.min(position[:, 0]), np.max(position[:, 0]))
    print(f'rank={rank}: Stage2: y -> min, max', np.min(position[:, 1]), np.max(position[:, 1]))

    plt.figure()
    plt.scatter(position[:, 0], position[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'test-rank-2-{rank}.png')
    plt.close()


# GO BACK INITIAL ORDER:
if True:
    position = layout.gather(position, mode='all', out=None)

    print(f'rank={rank}: Stage3: x -> min, max', np.min(position[:, 0]), np.max(position[:, 0]))
    print(f'rank={rank}: Stage3: y -> min, max', np.min(position[:, 1]), np.max(position[:, 1]))

    plt.figure()
    plt.scatter(position[:, 0], position[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'test-rank-3-{rank}.png')
    plt.close()
