import os
import logging
import argparse

import numpy as np

from .io import BigFile
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
    particles : BigFileCatalog
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
    if memory_monitor is not None: memory_monitor()

    # build halos catalog:
    halos, attrs = fof.find_features()
    if memory_monitor is not None: memory_monitor()

    # remove halos with lenght == 0
    halos = halos[halos['Length'] > 0]
    if memory_monitor is not None: memory_monitor()

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

    parser.add_argument("--compute_power", type=str, required=False, default='True',
                        help="If True, then compute the particle power spectrum.")
    parser.add_argument("--compute_halos", type=str, required=False, default='True',
                        help="If True, then search halos with FOF algorithm. Take care, need to reduce the number of proc but not the number of nodes ...")

    parser.add_argument("--subsampling", type=str, required=False, default='False',
                        help="If True subsample particles, only --subsampling_ratio % of particles are kept")
    parser.add_argument("--subsampling_nbr", type=float, required=False, default=3 * 34,
                        help="Keep one particle on subsampling_nbr. WARNING: this number should be divided by 3 (e.g.) 3*34")
    parser.add_argument("--delet_original", type=str, required=False, default='False',
                        help="If True supress the original particle file to save memory")

    parser.add_argument("--nmesh", type=int, required=False, default=1024,
                        help="nmesh used for the power spectrum computation")
    parser.add_argument("--k_min", type=float, required=False, default=1e-3,
                        help="to build np.arange(k_min, k_max, kbin)")
    parser.add_argument("--k_max", type=float, required=False, default=2e0,
                        help="to build np.arange(k_min, k_max, kbin)")
    parser.add_argument("--kbin", type=float, required=False, default=1e-3,
                        help="to build np.arange(k_min, k_max, kbin)")

    parser.add_argument("--min_mass_halos", type=float, required=False, default=2.25e12,
                        help="minimal mass of the halos kept during the halos power spectrum computation")
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
    mpicomm = MPI.COMM_WORLD
    rank = mpicomm.Get_rank()
    start_ini = MPI.Wtime()

    args = collect_argparser()
    sim = os.path.join(args.path_to_sim, args.sim)
    aout = args.aout

    mem_monitor_prefix = 'halos' if args.compute_halos else 'subsampling'
    mem_monitor = MemoryMonitor(log_file=os.path.join(sim, 'memory-monitor', f'{mem_monitor_prefix}-{aout}-memory_monitor_rank_{rank}.txt'))
    mem_monitor()

    start = MPI.Wtime()
    particles = BigFile(os.path.join(sim, f'fpm-{aout}'), dataset='1/', mode='r', mpicomm=mpicomm)
    particles.attrs['BoxSize'] = np.array([particles.attrs['boxsize'][0], particles.attrs['boxsize'][0], particles.attrs['boxsize'][0]])
    particles.Position = particles.read('Position')
    particles.Velocity = particles.read('Velocity')
    mem_monitor()
    logger_info(logger, f"Number of DM particles: {particles.csize} read in {MPI.Wtime() - start:2.2f} s.", rank)

    if args.compute_power == 'True':
        start = MPI.Wtime()
        # need to use wrap = True since some particles are outside the box
        # no neeed to select rank == 0 it is automatic in .save method
        CatalogFFTPower(data_positions1=particles['Position'], wrap=True,
                        edges=np.arange(args.k_min, args.k_max, args.kbin), ells=(0), nmesh=args.nmesh,
                        boxsize=particles.attrs['boxsize'][0], boxcenter=particles.attrs['boxsize'][0] // 2, resampler='tsc', interlacing=2, los='x',
                        position_type='pos', mpicomm=mpicomm).poles.save(os.path.join(sim, f'particle-power-{aout}.npy'))
        mem_monitor()
        logger_info(logger, f'CatalogFFTPower with particles done in {MPI.Wtime() - start:2.2f} s.', rank)

    if args.compute_halos == 'True':
        # take care if -N != 1 --> particles will be spread in the different nodes --> csize instead .size to get the full lenght
        start = MPI.Wtime()
        cosmo = load_fiducial_cosmo()
        # for the units https://cosmoprimo.readthedocs.io/en/latest/api/api.html?highlight=rho_crit#cosmoprimo.camb.Background.rho_crit
        # here particle mass is in Solar Mass.
        particle_mass = (cosmo.get_background().rho_cdm(1 / float(aout) - 1) + cosmo.get_background().rho_b(1 / float(aout) - 1)) / cosmo.h * 1e10 * particles.attrs['boxsize'][0]**3 / particles.csize  # units: Solar Mass
        mem_monitor()
        halos, attrs = build_halos_catalog(particles, nmin=args.nmin, rank=rank, memory_monitor=mem_monitor)
        attrs['particle_mass'] = particle_mass
        attrs['min_mass_halos'] = args.nmin * particle_mass
        mem_monitor()
        logger_info(logger, f"Find halos (with nmin = {args.nmin} -- particle mass = {particle_mass:2.2e}) done in {MPI.Wtime() - start:.2f} s.", rank)

        start = MPI.Wtime()
        halos_file = BigFile(os.path.join(sim, f'halos-{aout}'), dataset='1/', mode='w', mpicomm=mpicomm)
        halos_file.attrs = attrs
        halos_file.write({'Position': halos['CMPosition'], 'Velocity': halos['CMVelocity'], 'Mass': attrs['particle_mass'] * halos['Length']})
        mem_monitor()

        # Collect the number of halos --> just to print the information
        nbr_halos = mpicomm.reduce(halos['Length'].size, op=MPI.SUM, root=0)
        mem_monitor()
        logger_info(logger, f"Save {nbr_halos} halos done in {MPI.Wtime() - start:2.2f} s.", rank)

        start = MPI.Wtime()
        mem_monitor()
        CatalogFFTPower(data_positions1=halos['CMPosition'][(halos['Length'] * attrs['particle_mass']) >= args.min_mass_halos],
                        edges=np.arange(args.k_min, args.k_max, args.kbin), ells=(0), nmesh=args.nmesh,
                        boxsize=attrs['boxsize'][0], boxcenter=attrs['boxsize'][0] // 2, resampler='tsc', interlacing=2, los='x',
                        position_type='pos', mpicomm=mpicomm).poles.save(os.path.join(sim, f'halos-power-{aout}.npy'))
        mem_monitor()
        logger_info(logger, f'CatalogFFTPower with halos done in {MPI.Wtime() - start:2.2f} s.', rank)

    if args.subsampling == 'True':
        start = MPI.Wtime()
        logger_info(logger, f'Start subsampling with subsampling_nbr={args.subsampling_nbr}', rank)

        nbr_particles = mpicomm.allgather(particles.size)
        kept = np.arange(0, particles.size, args.subsampling_nbr) + int(np.sum(nbr_particles[:rank]) % args.subsampling_nbr)
        kept = kept[kept < particles.size].astype(int)  # remove index out of range

        # save subsampled particles
        sub_particles = BigFile(os.path.join(sim, f'fpm-subsamp-{aout}'), dataset='1/', mode='w', mpicomm=mpicomm)
        sub_particles.attrs = particles.attrs
        sub_particles.attrs['subsampling_ratio'] = 1 / args.subsampling_nbr
        sub_particles.write({'Position': particles['Position'][kept], 'Velocity': particles['Velocity'][kept]})

        # compute power spectrum of subsampled particles:
        CatalogFFTPower(data_positions1=particles['Position'][kept], wrap=True,
                        edges=np.arange(args.k_min, args.k_max, args.kbin), ells=(0), nmesh=args.nmesh,
                        boxsize=attrs['boxsize'][0], boxcenter=attrs['boxsize'][0] // 2, resampler='tsc', interlacing=2, los='x',
                        position_type='pos', mpicomm=mpicomm).poles.save(os.path.join(sim, f'particle-subsamp-power-{aout}.npy'))
        mem_monitor()
        logger_info(logger, f"Subsampling (from {particles.csize:2.2e} to approx. {particles.csize / args.subsampling_nbr:2.2e}) done in {MPI.Wtime() - start:2.2f} s.", rank)

    mem_monitor.stop_monitoring()
    mpicomm.Barrier()

    if rank == 0:
        plot_memory(sim, prefix=f'{mem_monitor_prefix}-{aout}-')

    if args.delet_original == 'True':
        if rank == 0:
            import shutil
            shutil.rmtree(os.path.join(sim, f'fpm-{aout}'), ignore_errors=True)

    logger_info(logger, f"Post processing took {MPI.Wtime() - start_ini:2.2f} s.", rank)
