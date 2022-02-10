from argparse import ArgumentParser
from pmesh.pm import ParticleMesh
ap = ArgumentParser()
ap.add_argument("config")

from .core import Solver
from .core import leapfrog
from .core import autostages
from .background import PerturbationGrowth

from cosmoprimo.fiducial import DESI
from pypower import MeshFFTPower
import numpy

class Config(dict):
    def __init__(self, path):
        self.prefix = '%s' % path
        filename = self.makepath('config.py')

        self['boxsize'] = 1380.0
        self['shift'] = 0.0
        self['nc'] = 64
        self['ndim'] = 3
        self['seed'] = 1985
        self['pm_nc_factor'] = 2
        self['resampler'] = 'tsc'
        self['cosmology'] = DESI('class')
        self['powerspectrum'] = self['cosmology'].get_fourier().pk_interpolator().to_1d(z=0.)
        self['unitary'] = False
        self['stages'] = numpy.linspace(0.1, 1.0, 5, endpoint=True)
        self['aout'] = [1.0]

        local = {} # these names will be usable in the config file
        local['linspace'] = numpy.linspace
        local['autostages'] = autostages

        names = set(self.__dict__.keys())

        exec(open(filename).read(), local, self)

        unknown = set(self.__dict__.keys()) - names
        assert len(unknown) == 0

        self.finalize()
        global _config
        _config = self

    def finalize(self):
        self['aout'] = numpy.array(self['aout'])

        self.pm = ParticleMesh(BoxSize=self['boxsize'], Nmesh= [self['nc']] * self['ndim'], resampler=self['resampler'])
        mask = numpy.array([ a not in self['stages'] for a in self['aout']], dtype='?')
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
        poles = MeshFFTPower(d.c2r(), edges=numpy.geomspace(1e-3, 5e-1, 80), ells=(0), wnorm=d.pm.Nmesh.prod()**2/d.pm.BoxSize.prod(), shotnoise=0.).poles
        if config.pm.comm.rank == 0:
            print('Writing matter power spectrum at %s' % path)
            # only root rank saves
            poles.save(path)

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
    dlin = solver.linear(whitenoise, Pk=lambda k : numpy.where(k >0, config['powerspectrum'](k, bounds_error=False), 0))

    # generate particles in grid with uniform law:
    Q = config.pm.generate_uniform_particle_grid(shift=config['shift'])

    # compute the displacement for each particle in the grid either with Zeldovich approximation (order=1) or 2LPT (order=2)
    # then apply the displacement on each particle:
    state = solver.lpt(dlin, Q=Q, a=config['stages'][0], order=2)

    # Save input (Linear power spectrum at redshift=0.):
    write_power(dlin, config.makepath('power-linear-1.0.npy'), a=1.0)

    # Evolution of particles via fastpm computation:
    solver.nbody(state, stepping=leapfrog(config['stages']), monitor=monitor)

if __name__ == '__main__':
    main()
