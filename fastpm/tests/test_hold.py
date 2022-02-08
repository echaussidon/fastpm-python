from fastpm.core import leapfrog, autostages
from fastpm.hold import Solver

from pmesh.pm import ParticleMesh
import numpy

from cosmoprimo.fiducial import DESI
# faudra le mettre ailleur mais pour l'instant tant que cosmo_fid.Omega0_m n'est pas corrigé on attend.
cosmo_fid = DESI('class')
cosmo_fid.Omega0_m = cosmo_fid.Omega_m(0.)
# on definit le spectre de puissance --> attention to_1D() raise une erreur mega relou...
linear_power_spectrum_interp = cosmo_fid.get_fourier().pk_interpolator(extrap_kmin=1e-8, extrap_kmax=1e3)
def linear_power_spectrum(k):
    return linear_power_spectrum_interp(k, z=0.)

pm = ParticleMesh(BoxSize=32., Nmesh=[16, 16, 16])

def test_solver():
    Plin = linear_power_spectrum
    solver = Solver(pm, cosmo_fid, B=2)
    Q = pm.generate_uniform_particle_grid(shift=0)

    wn = solver.whitenoise(1234)
    dlin = solver.linear(wn, lambda k: Plin(k))

    state = solver.lpt(dlin, Q, a=0.3, order=2)

    dnonlin = solver.nbody(state, leapfrog([0.3, 0.35]))

    dnonlin.save('nonlin')
