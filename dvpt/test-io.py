from fastpm.io import BigFile

from nbodykit.source.catalog.file import BigFileCatalog

import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


start = MPI.Wtime()
f = BigFile('/global/homes/e/edmondc/Software/fastpm-python/examples/run/fpm-1.0000', dataset='1/', mode='r', mpicomm=comm)
if rank == 0:
    print(f.attrs)
    print(f.size)
    print(f.csize)
position = f.read('Position')
stop = MPI.Wtime()
if rank == 0:
    print(stop - start, "s to read ", position.shape, 'particles position -- ', position[0])
print(position.shape)


start = MPI.Wtime()
position = BigFileCatalog('/Users/ec263193/Desktop/test_cfastpm/fastpm-nersc/fpm-1.0000', dataset='1/', comm=comm)['Position'].compute()
stop = MPI.Wtime()
if rank == 0:
    print(stop - start, "s to read ", position.shape, 'particles position -- ', position[0])


write = BigFile('/global/homes/e/edmondc/Software/fastpm-python/examples/run/test', dataset='1/', mode='w', mpicomm=comm)
write.attrs = f.attrs
write.write({'position': rank * np.ones(100)})


f = BigFile('/global/homes/e/edmondc/Software/fastpm-python/examples/run/test', dataset='1/', mode='r', mpicomm=comm)
position = f.read('position')
if rank == 3:
    print(position)
    print(f.attrs)
