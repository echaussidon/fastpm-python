import os
import logging
import argparse

import numpy as np

from .fof import FOF
from .memory_monitor import MemoryMonitor, plot_memory

from pypower import CatalogFFTPower, setup_logging


logger = logging.getLogger('Post processing')

# To avoid error from NUMEXPR Package
os.environ.setdefault('NUMEXPR_MAX_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))
os.environ.setdefault('NUMEXPR_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))


def logger_info(logger, msg, rank, mpiroot=0):
    """Print something with the logger only for the rank == mpiroot to avoid duplication of message."""
    if rank == mpiroot:
        logger.info(msg)


def load_bigfile(path, dataset='1/', comm=None):
    """
    Load BigFile with BigFileCatalog removing FuturWarning. Just pass the commutator to spread the file in all the processes.

    Parameters
    ----------
    path : str
        Path where the BigFile is.
    dataset: str
        Which sub-dataset do you want to read from the BigFile ?
    comm : MPI commutator
        Pass the current commutator if you want to use MPI.

    Return
    ------
    cat : BigFileCatalog
        BigFileCatalog object from nbodykit.

    """
    from nbodykit.source.catalog.file import BigFileCatalog

    # to remove the following warning:
    # FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        logger_info(logger, f'Read {path}', comm.Get_rank())
        # read simulation output
        cat = BigFileCatalog(path, dataset=dataset, comm=comm)
        # add BoxSize attributs (mandatory for fof)
        cat.attrs['BoxSize'] = np.array([cat.attrs['boxsize'][0], cat.attrs['boxsize'][0], cat.attrs['boxsize'][0]])

    return cat


def load_fiducial_cosmo():
    """ Load fiducial DESI cosmology."""
    from cosmoprimo.fiducial import DESI
    # Load fiducial cosmology
    cosmo = DESI(engine='class')
    # precompute the bakcground
    _ = cosmo.get_background()
    return cosmo


def build_halos_catalog(particles, linking_length=0.2, nmin=8, particle_mass=1e12, rank=None, memory_monitor=None):
    """
    Determine the halos of Dark Matter with the FOF algorithm of nbody kit. The computation is parallelized with MPI if the file is open with a commutator.

    Parameters
    ----------
    particles : nbodykit BigFileCatalog
        Catalog containing the position of all the particle of the simulation. (Not necessary a BigFileCatalog)
    linking_lenght : float
        The linking length, either in absolute units, or relative to the mean particle separation.
    nmin = int
        Minimal number of particles to determine a halos.
    particle_mass : float
        Mass of each DM particle in Solar Mass unit.

    Return
    ------
    halos :  numpy dtype array

    attrs : attributs which will be saved in the BigFile.
    """
    # Run the fof algorithm
    fof = FOF(particles, linking_length, nmin, memory_monitor=memory_monitor)
    if memory_monitor is not None:
        memory_monitor()

    # build halos catalog:
    halos, attrs = fof.find_features()

    # remove halos with lenght == 0
    if memory_monitor is not None:
        memory_monitor()
    halos = halos[halos['Length'] > 0]
    if memory_monitor is not None:
        memory_monitor()

    # meta-data
    attrs = particles.attrs.copy()
    attrs['linking_length'] = linking_length
    attrs['nmin'] = nmin

    return halos, attrs


def collect_argparser():
    parser = argparse.ArgumentParser(description="Post processing of fastpm-python simulation. It run FOF halo finder, save the halo catalog into a .fits format. \
                                                  It computes also the power spectrum for the particle and the power spectrum for the halos with a given mass selection.")

    parser.add_argument("--path_to_sim", type=str, required=False, default='/global/u2/e/edmondc/Scratch/Mocks/',
                        help="Path to the Scratch where the simulations are saved")
    parser.add_argument("--sim", type=str, required=False, default='test',
                        help="Simulation name (e.g) fastpm-fnl-0")
    parser.add_argument("--aout", type=str, required=False, default='1.0000',
                        help="scale factor at which the particles are saved (e.g) '0.3300' or '1.0000'")

    parser.add_argument("--nmesh", type=int, required=False, default=1024,
                        help="nmesh used for the power spectrum computation")

    parser.add_argument("--k_min", type=float, required=False, default=5e-3,
                        help="to build np.geomspace(k_min, k_max, k_nbins)")
    parser.add_argument("--k_max", type=float, required=False, default=3e0,
                        help="to build np.geomspace(k_min, k_max, k_nbins)")
    parser.add_argument("--k_nbins", type=float, required=False, default=80,
                        help="to build np.geomspace(k_min, k_max, k_nbins)")

    parser.add_argument("--min_mass_halos", type=float, required=False, default=1e13,
                        help="minimal mass of the halos to be kept")
    parser.add_argument("--nmin", type=int, required=False, default=8,
                        help="minimal number of particle to form a halos")

    return parser.parse_args()


if __name__ == '__main__':

    setup_logging()

    # to remove the following warning from pmesh (arnaud l'a corrigé sur son github mais ok)
    # VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or
    # shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray. self.edges = numpy.asarray(edges)
    import warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    start_ini = MPI.Wtime()

    args = collect_argparser()
    sim = os.path.join(args.path_to_sim, args.sim)
    aout = args.aout

    mem_monitor = MemoryMonitor(log_file=os.path.join(sim, 'memory-monitor', f'halos-{aout}-memory_monitor_rank_{rank}.txt'))
    mem_monitor()

    start = MPI.Wtime()
    particles = load_bigfile(os.path.join(sim, f'fpm-{aout}'), comm=comm)
    mem_monitor()
    logger_info(logger, f"Number of DM particles: {particles.csize} read in {MPI.Wtime() - start:2.2f} s.", rank)

    start = MPI.Wtime()
    # need to use wrap = True since some particles are outside the box
    # no neeed to select rank == 0 it is automatic in .save method
   CatalogFFTPower(data_positions1=particles['Position'].compute(), wrap=True, edges=np.geomspace(args.k_min, args.k_max, args.k_nbins), ells=(0), nmesh=args.nmesh,
                   boxsize=particles.attrs['boxsize'][0], boxcenter=particles.attrs['boxsize'][0] // 2, resampler='tsc', interlacing=2, los='x', position_type='pos',
                   mpicomm=comm).poles.save(os.path.join(sim, f'particle-power-{aout}.npy'))
   mem_monitor()
   logger_info(logger, f'CatalogFFTPower with particles done in {MPI.Wtime() - start:2.2f} s.', rank)

    # take care if -N != 1 --> particles will be spread in the different nodes --> csize instead .size to get the full lenght
    start = MPI.Wtime()
    cosmo = load_fiducial_cosmo()
    particle_mass = (cosmo.get_background().rho_cdm(1 / float(aout) - 1) + cosmo.get_background().rho_b(1 / float(aout) - 1)) / cosmo.h * 1e10 * particles.attrs['boxsize'][0]**3 / particles.csize  # units: Solar Mass
    mem_monitor()
    halos, attrs = build_halos_catalog(particles, nmin=args.nmin, rank=rank, memory_monitor=mem_monitor)
    attrs['particle_mass'] = particle_mass
    attrs['min_mass_halos'] = args.min_mass_halos
    mem_monitor()
    logger_info(logger, f"Find halos (with nmin = {args.nmin}) done in {MPI.Wtime() - start:.2f} s.", rank)

    start = MPI.Wtime()
    from bigfile import FileMPI
    with FileMPI(comm, os.path.join(sim, f'halos-{aout}'), create=True) as ff:
        with ff.create('Header') as bb:
            keylist = ['N_eff', 'Omega0_Lambda', 'Omega0_b', 'Omega0_m', 'RSDFactor', 'T0_cmb', 'Time', 'boxsize', 'fnl', 'gnl', 'h', 'kmax_primordial_over_knyquist',
                       'nc', 'ndim', 'pm_nc_factor', 'resampler', 'seed', 'shift', 'unitary', 'use_non_gaussianity', 'particle_mass', 'nmin', 'min_mass_halos']
            for key in keylist:
                try:
                    bb.attrs[key] = attrs[key]
                except KeyError:
                    pass
        # work with center of mass
        ff.create_from_array('1/Position', halos['CMPosition'])
        ff.create_from_array('1/Velocity', halos['CMVelocity'])
        ff.create_from_array('1/Mass', attrs['particle_mass'] * halos['Length'])
    mem_monitor()

    nbr_halos = comm.reduce(halos['Length'].size, op=MPI.SUM, root=0)
    mem_monitor()
    logger_info(logger, f"Save {nbr_halos} halos done in {MPI.Wtime() - start:2.2f} s.", rank)

    start = MPI.Wtime()
    position = halos['CMPosition'][(halos['Length'] * attrs['particle_mass']) >= args.min_mass_halos]
    mem_monitor()
    CatalogFFTPower(data_positions1=position, edges=np.geomspace(args.k_min, args.k_max, args.k_nbins), ells=(0), nmesh=args.nmesh,
                    boxsize=attrs['boxsize'][0], boxcenter=attrs['boxsize'][0] // 2, resampler='tsc', interlacing=2, los='x', position_type='pos',
                    mpicomm=comm).poles.save(os.path.join(sim, f'halos-power-{aout}.npy'))
    mem_monitor()
    logger_info(logger, f'CatalogFFTPower with halos done in {MPI.Wtime() - start:2.2f} s.', rank)

    logger_info(logger, f"Post processing took {MPI.Wtime() - start_ini:2.2f} s.", rank)

    mem_monitor.stop_monitoring()
    comm.Barrier()

    if rank == 0:
        plot_memory(sim, prefix=f'halos-{aout}-')
