""" Stand alone to add rsd et compute multipole. Now it is in post_processing.py """


import os
import logging
import argparse

import numpy as np

from .io import BigFile

from pypower import CatalogFFTPower, setup_logging


logger = logging.getLogger('Apply RSD')

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


def collect_argparser():
    parser = argparse.ArgumentParser(description="Transform position in real space to redshift space and compute the multipoles.")

    parser.add_argument("--path_to_sim", type=str, required=False, default='/global/u2/e/edmondc/Scratch/Mocks/',
                        help="Path to the Scratch where the simulations are saved")
    parser.add_argument("--sim", type=str, required=False, default='test',
                        help="Simulation name (e.g) fastpm-fnl-0")
    parser.add_argument("--aout", type=str, required=False, default='1.0000',
                        help="scale factor at which the halos are saved (e.g) '0.3300' or '1.0000'")

    parser.add_argument("--min_mass_halos", type=float, required=False, default=2.25e12,
                        help="minimal mass of the halos kept during the halos power spectrum computation")

    parser.add_argument("--nmesh", type=int, required=False, default=1024,
                        help="nmesh used for the power spectrum computation")
    parser.add_argument("--k_min", type=float, required=False, default=1e-3,
                        help="to build np.arange(k_min, k_max, kbin)")
    parser.add_argument("--k_max", type=float, required=False, default=1.5e0,
                        help="to build np.arange(k_min, k_max, kbin)")
    parser.add_argument("--kbin", type=float, required=False, default=1e-3,
                        help="to build np.arange(k_min, k_max, kbin)")

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

    start = MPI.Wtime()
    halos = BigFile(os.path.join(sim, f'halos-{args.aout}'), dataset='1/', mode='r', mpicomm=mpicomm)
    halos.Position = halos.read('Position')
    halos.Velocity = halos.read('Velocity')
    halos.Mass = halos.read('Mass')
    logger_info(logger, f"Number of halos: {halos.csize} read in {MPI.Wtime() - start:2.2f} s.", rank)

    start = MPI.Wtime()
    # add RSD along the z axis (unitary vector)
    line_of_sight = [0, 0, 1]
    # the RSD normalization factor
    # the RSD factor is already saved in attrs: halos.attrs['RSDFactor'][0]
    # Warning: H(z) = 100 * h * E(z) in km.s^-1.Mpc^-1
    cosmo = load_fiducial_cosmo()
    rsd_factor = 1 / (float(args.aout) * 100 * cosmo.get_background().efunc(1 / float(args.aout) - 1))
    # compute position in redshift space
    position_rsd = halos['Position'] + rsd_factor * halos['Velocity'] * line_of_sight
    logger_info(logger, f"Compute positions in redshift space in {MPI.Wtime() - start:2.2f} s.", rank)

    # save the position in the same dataset
    start = MPI.Wtime()
    halos.write({'Position_rsd': position_rsd})
    logger_info(logger, f"Write positions in redshift space in {MPI.Wtime() - start:2.2f} s.", rank)

    start = MPI.Wtime()
    # need to use wrap = True since some halos are outside the box
    # due to pmesh auto wrapping.
    # no need to select rank == 0 it is automatic in .save method
    CatalogFFTPower(data_positions1=position_rsd[halos['Mass'] >= args.min_mass_halos], wrap=True,
                    edges=np.arange(args.k_min, args.k_max, args.kbin), ells=(0, 2, 4), nmesh=args.nmesh,
                    boxsize=halos.attrs['boxsize'][0], boxcenter=halos.attrs['boxsize'][0] // 2, resampler='tsc', interlacing=2, los=line_of_sight,
                    position_type='pos', mpicomm=mpicomm).poles.save(os.path.join(sim, f'halos-power-{args.aout}-rsd.npy'))
    logger_info(logger, f'CatalogFFTPower with halos for ell=(0, 2, 4) done in {MPI.Wtime() - start:2.2f} s.', rank)

    # wait every one (no need since rank 0 will be the last one and we write on rank 0)
    mpicomm.Barrier()
    logger_info(logger, f"Translate positions into redshift space and compute multipole took {MPI.Wtime() - start_ini:2.2f} s.", rank)
