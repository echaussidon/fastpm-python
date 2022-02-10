from fastpm.core import leapfrog, autostages
from fastpm.hold import Solver

from pmesh.pm import ParticleMesh
import numpy

from cosmoprimo.fiducial import DESI


cosmo_fid = DESI('class')

pm = ParticleMesh(BoxSize=32., Nmesh=[16, 16, 16])

def test_solver():
    Plin = cosmo_fid.get_fourier().pk_interpolator().to_1d(z=0.)
    solver = Solver(pm, cosmo_fid, B=2)
    Q = pm.generate_uniform_particle_grid(shift=0)

    wn = solver.whitenoise(1234)
    dlin = solver.linear(wn, lambda k: numpy.where(k >0, Plin(k, bounds_error=False), 0))

    state = solver.lpt(dlin, Q, a=0.3, order=2)

    dnonlin = solver.nbody(state, leapfrog([0.3, 0.35]))

    dnonlin.save('nonlin')
