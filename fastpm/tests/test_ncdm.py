# TO BE UPDATED WITH COSMOPRIMO

from runtests.mpi import MPITest
from fastpm.core import leapfrog, autostages
from fastpm.background import PerturbationGrowth

from pmesh.pm import ParticleMesh
import numpy
from numpy.testing import assert_allclose

from cosmoprimo.fiducial import DESI
cosmo_fid = DESI('class')

from fastpm.ncdm import Solver


@MPITest([1, 4])
def test_ncdm(comm):
    pm = ParticleMesh(BoxSize=512., Nmesh=[8, 8, 8], comm=comm)
    Plin = cosmo_fid.get_fourier().pk_interpolator().to_1d(z=0.)
    solver = Solver(pm, cosmo_fid, B=1)
    Q = pm.generate_uniform_particle_grid(shift=0)

    wn = solver.whitenoise(1234)
    dlin = solver.linear(wn, lambda k: numpy.where(k >0, Plin(k, bounds_error=False), 0))
    state = solver.lpt(dlin, Q, a=1.0, order=2)

    dnonlin = solver.nbody(state, leapfrog([0.1, 1.0]))

    dnonlin.save('nonlin')
