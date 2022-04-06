from fastpm.io import BigFile

import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# start = MPI.Wtime()
#
f = BigFile('/Users/ec263193/Desktop/test_cfastpm/fastpm-nersc/fpm-1.0000', dataset='1/', mode='r', mpicomm=comm)

if rank == 0:
    print(f.attrs)
    print(f.size)
    print(f.csize)

# position = f.read('Position')
#
# stop = MPI.Wtime()
#
# if rank == 0:
#     print(stop - start, "s to read ", position.shape, 'particles position -- ', position[0])
#
# print(position.shape)

# from nbodykit.source.catalog.file import BigFileCatalog
#
# start = MPI.Wtime()
#
# position = BigFileCatalog('/Users/ec263193/Desktop/test_cfastpm/fastpm-nersc/fpm-1.0000', dataset='1/', comm=comm)['Position'].compute()
#
# stop = MPI.Wtime()
#
# if rank == 0:
#     print(stop - start, "s to read ", position.shape, 'particles position -- ', position[0])

write = BigFile('/Users/ec263193/Desktop/test_cfastpm/fastpm-nersc/test', dataset='1/', mode='w', mpicomm=comm)
write.attrs = f.attrs

# il faut sauvegarder les attributs avec f.attrs =

write.write({'position': np.ones(100)})
