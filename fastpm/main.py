from argparse import ArgumentParser
from pmesh.pm import ParticleMesh
ap = ArgumentParser()
ap.add_argument("config")

from .core import Solver
from .core import leapfrog
from .core import autostages
from .background import PerturbationGrowth

import numpy as np

from cosmoprimo.fiducial import DESI
from cosmoprimo import constants

from pypower import MeshFFTPower


class Config(dict):
    def __init__(self, path):
        self.prefix = '%s' % path
        filename = self.makepath('config.py')

        self['boxsize'] = 1380.0     # size of the box for the simulation.
        self['shift'] = 0.0          # shifting the grid by this much relative to the size of each grid cell.
        self['nc'] = 64              # number of mesh points along one direction for
        self['ndim'] = 3             # number of dimensions.
        self['seed'] = 1985          # fix the random seed for reproductibility.
        self['pm_nc_factor'] = 2     # force resolution parameter (B parameter). The size per side of the mesh(Nm) used for force calculation is B times the number of particles per side (Ng) (B=2 seems to be a good choice).
        self['resampler'] = 'tsc'    # type of window (ie) how the particle will be painted in a Mesh ('tsc' is order 3)
        self['cosmology'] = DESI('class')  # cosmology in which the simulation is done
        self['powerspectrum'] = self['cosmology'].get_fourier().pk_interpolator().to_1d(z=0.) # power spectrum with which the initialisation is done
        self['unitary'] = False      # True for a unitary gaussian field (amplitude is fixed to 1) False for a true gaussian field
        self['stages'] = np.linspace(0.1, 1.0, 5, endpoint=True) # a_start -- a_end -- N = # time step --> use autostages instead of linspace
        self['aout'] = [1.0]         # in which a the particles are saved

        # add initial non-gaussianity:
        self['use_non_gaussianity'] = False
        self['kmax_primordial_over_knyquist'] = 0.5
        self['fnl'] = 0.
        self['gnl'] = 0.

        # default param for power spectrum computation
        self['power_kedges'] = np.geomspace(1e-3, 5e-1, 80)

        local = {} # these names will be usable in the config file, can add cosmo to use specific cosmology.
        local['linspace'] = np.linspace
        local['autostages'] = autostages

        names = set(self.__dict__.keys())

        exec(open(filename).read(), local, self)

        unknown = set(self.__dict__.keys()) - names
        assert len(unknown) == 0

        self.finalize()
        global _config
        _config = self

    def finalize(self):
        self['aout'] = np.array(self['aout'])

        self.pm = ParticleMesh(BoxSize=self['boxsize'], Nmesh= [self['nc']] * self['ndim'], resampler=self['resampler'])
        mask = np.array([a not in self['stages'] for a in self['aout']], dtype='?')
        missing_stages = self['aout'][mask]
        if len(missing_stages):
            raise ValueError('Some stages are requested for output but missing: %s' % str(missing_stages))

    def makepath(self, filename):
        import os.path
        return os.path.join(self.prefix, filename)

def main(args=None):
    # Supress the logger from pypower
    import logging
    logging.getLogger("MeshFFTPower").setLevel(logging.ERROR)

    def write_power(d, path, a):
        """
        Compute and save the powerspectrum for a given complex mesh d.
        """
        poles = MeshFFTPower(d.c2r(), edges=config['power_kedges'], ells=(0), wnorm=d.pm.Nmesh.prod()**2/d.pm.BoxSize.prod(), shotnoise=0.).poles
        if config.pm.comm.rank == 0:
            print('Writing matter power spectrum at %s' % path)
            # only root rank saves
            poles.save(path)

    def my_where(k, mask, fun, default=0):
        """
        Implementation of np.where to avoid to call the function in np.where and raise warning for division error for instance.
        """
        toret = np.empty_like(k)
        toret[mask] = fun(k[mask])
        toret[~mask] = default
        return toret

    def monitor(action, ai, ac, af, state, event):
        """
        Monitor function. Action to do for different steps of the computation.
        """
        if config.pm.comm.rank == 0:
            print('Step %s %06.4f - (%06.4f) -> %06.4f' %( action, ai, ac, af),
                  'S %(S)06.4f P %(P)06.4f F %(F)06.4f' % (state.a))

        if action == 'F':
            a = state.a['F']
            path = config.makepath('power-%06.4f.npy' % a)
            write_power(event['delta_k'], path, a)

        if state.synchronized:
            a = state.a['S']
            if a in config['aout']:
                path = config.makepath('fpm-%06.4f' % a) % a
                if config.pm.comm.rank == 0:
                    print('Writing a snapshot at %s' % path)
                # collective save
                state.save(path, attrs=config)

    # load configuration:
    ns = ap.parse_args(args)
    config = Config(ns.config)

    # Create the mesh on which we will run the nbody simulation. Solver class contain all the function that we need.
    solver = Solver(config.pm, cosmology=config['cosmology'], B=config['pm_nc_factor'])

    # generate random vector of norm 1 in complex plane in each point of the grid --> we generate delta_k
    # remind: the fourier transform of real function is complex and even !
    # remind: f real --> f(k) complex & even | f real et even --> f(k) real et even
    whitenoise = solver.whitenoise(seed=config['seed'], unitary=config['unitary'])
    # whitenoise is a symetric matrix by construction (to have a real field after fourier transform) -> One of the dimension is two time smaller. -->  cf section 6 de Large-scale dark matter simulations

    # Match delta_k with the desired power spectrum --> cf section6 of Large-scale dark matter simulations
    # Here dlin is delta(k, z=0) following the linear power spectrum at z=0. (set correct redshift thanks to the growth_rate at displacement stage.)
    dlin = solver.linear(whitenoise, Pk=lambda k : my_where(k, k>0, config['powerspectrum'], 0))

    if config['use_non_gaussianity']:
        dlin = solver.add_local_non_gaussianity(dlin, fnl=config['fnl'], kmax_primordial_over_knyquist=config['kmax_primordial_over_knyquist'])

    # generate particles in grid with uniform law:
    Q = config.pm.generate_uniform_particle_grid(shift=config['shift'])

    # compute the displacement for each particle in the grid either with Zeldovich approximation (order=1) or 2LPT (order=2)
    # Warning: We normalize here with the growth factor to have the power spectrum at the correct redhift !
    # then apply the displacement on each particle:
    state = solver.lpt(dlin, Q=Q, a=config['stages'][0], order=2)

    # Save input (matter power spectrum at redshift=0.):
    write_power(dlin, config.makepath('power-linear-1.0.npy'), a=1.0)

    # Evolution of particles via fastpm computation:
    solver.nbody(state, stepping=leapfrog(config['stages']), monitor=monitor)

if __name__ == '__main__':
    # to remove this warning from pmesh:
    # VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences
    # (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated.
    # If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
    import warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    main()
