from fastpm.core import leapfrog, Solver, autostages
from fastpm.background import MatterDominated

from pmesh.pm import ParticleMesh
import numpy
from numpy.testing import assert_allclose

from cosmoprimo.fiducial import DESI
# faudra le mettre ailleur mais pour l'instant tant que cosmo_fid.Omega0_m n'est pas corrigé on attend.
cosmo_fid = DESI('class')
cosmo_fid.Omega0_m = cosmo_fid.Omega_m(0.)
# on definit le spectre de puissance --> attention to_1D() raise une erreur mega relou...
linear_power_spectrum_interp = cosmo_fid.get_fourier().pk_interpolator(extrap_kmin=1e-8, extrap_kmax=1e3)
def linear_power_spectrum(k):
    return linear_power_spectrum_interp(k, z=0.)

pm = ParticleMesh(BoxSize=128., Nmesh=[8, 8, 8])

def test_autostages():
    for knots in [
        [0.1, 1.0],
        [1.0],
        [0.5],
        [0.1, 0.2, 1.0],
        ]:

        l = autostages(knots, astart=0.1, N=12)

        assert(len(l) == 12)
        for k in knots:
            assert k in l

def test_leapfrog():
    l = list(leapfrog(numpy.linspace(0.1, 1.0, 2, endpoint=True)))
    assert len(l) == 1 + 4 * (2 - 1)

    l = list(leapfrog([1.0]))
    assert len(l) == 1

def test_solver():
    Plin = linear_power_spectrum
    solver = Solver(pm, cosmo_fid, B=1)
    Q = pm.generate_uniform_particle_grid(shift=0)

    wn = solver.whitenoise(1234)
    dlin = solver.linear(wn, lambda k: Plin(k))

    state = solver.lpt(dlin, Q, a=1.0, order=2)

    dnonlin = solver.nbody(state, leapfrog([1.0]))

    dnonlin.save('nonlin')

def test_lpt():
    Plin = linear_power_spectrum
    solver = Solver(pm, cosmo_fid, B=1)
    Q = pm.generate_uniform_particle_grid(shift=0)

    wn = solver.whitenoise(1234)
    dlin = solver.linear(wn, lambda k: Plin(k))

    state1 = solver.lpt(dlin, Q, a=0.01, order=1)
    state2 = solver.lpt(dlin, Q, a=1.0, order=1)

    pt = MatterDominated(cosmo_fid.Omega0_m, a=[0.01, 1.0], a_normalize=1.0)
#    print((state2.P[...] / state1.P[...]))
    print((state2.P[...] - state1.P[...]) / state1.F[...])

    fac = 1 / (0.01 ** 2 * pt.E(0.01)) * (pt.Gf(1.0) - pt.Gf(0.01)) / pt.gf(0.01)
    assert_allclose(state2.P[...], state1.P[...] + fac * state1.F[...])

from runtests.mpi import MPITest
@MPITest([1, 2, 3, 4])
def test_solver_convergence(comm):
    from mpi4py import MPI
    pm = ParticleMesh(BoxSize=128., Nmesh=[8, 8, 8], comm=comm)
    pm_ref = ParticleMesh(BoxSize=128., Nmesh=[8, 8, 8], comm=MPI.COMM_SELF)

    Plin = linear_power_spectrum

    Q = pm_ref.generate_uniform_particle_grid(shift=0)

    def solve(pm):
        solver = Solver(pm, cosmo_fid, B=1)
        q = numpy.array_split(Q, pm.comm.size)[pm.comm.rank]
        wn = solver.whitenoise(1234)
        dlin = solver.linear(wn, lambda k: Plin(k))

        state = solver.lpt(dlin, q, a=0.1, order=2)
        state = solver.nbody(state, leapfrog([0.1, 0.5, 1.0]))

        d = {}
        for key in 'X', 'P', 'F':
            d[key] = numpy.concatenate(pm.comm.allgather(getattr(state, key)), axis=0)
        return d

    s = solve(pm)
    r = solve(pm_ref)
    assert_allclose(s['X'], r['X'])
    assert_allclose(s['P'], r['P'])
    assert_allclose(s['F'], r['F'])
