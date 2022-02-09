from fastpm.state import StateVector, Matter, Baryon, CDM, NCDM
from runtests.mpi import MPITest

from cosmoprimo.fiducial import DESI
cosmo_fid = DESI('class')

import numpy

BoxSize = 100.
Q = numpy.zeros((100, 3))

@MPITest([1, 4])
def test_create(comm):

    matter = Matter(cosmo_fid, BoxSize, Q, comm)

    cdm = CDM(cosmo_fid, BoxSize, Q, comm)
    cdm.a['S'] = 1.0
    cdm.a['P'] = 1.0
    baryon = Baryon(cosmo_fid, BoxSize, Q, comm)
    baryon.a['S'] = 1.0
    baryon.a['P'] = 1.0

    state = StateVector(cosmo_fid, {'0': baryon, '1' : cdm}, comm)
    state.a['S'] = 1.0
    state.a['P'] = 1.0
    state.save("state")
