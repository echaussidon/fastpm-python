import os
import logging
import argparse

import numpy as np


logger = logging.getLogger('Power Spectrum')


# To avoid error from NUMEXPR Package
os.environ.setdefault('NUMEXPR_MAX_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))
os.environ.setdefault('NUMEXPR_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))


def logger_info(logger, msg, rank, mpiroot=0):
    """Print something with the logger only for the rank == mpiroot to avoid duplication of message."""
    if rank == mpiroot:
        logger.info(msg)


def collect_argparser():
    parser = argparse.ArgumentParser(description="Transform position in real space to redshift space and compute the multipoles.")

    parser.add_argument("--path_to_sim", type=str, required=False, default='/global/u2/e/edmondc/Scratch/Mocks/',
                        help="Path to the Scratch where the simulations are saved")
    parser.add_argument("--sim", type=str, required=False, default='test',
                        help="Simulation name (e.g) fastpm-fnl-0")
    parser.add_argument("--aout", type=str, required=False, default='1.0000',
                        help="scale factor at which the halos are saved (e.g) '0.3300' or '1.0000'")

    parser.add_argument("--release", type=str, required=False, default='Y5',
                        help="match Y1 / Y5 footprint")
    parser.add_argument("--regions", nargs='+', type=str, required=False, default=['N', 'SNGC', 'SSGC'],
                        help="photometric regions")

    parser.add_argument("--name_randoms", type=str, required=False, default='fastpm-randoms',
                        help='Each region is be saved in one dataset.')

    return parser.parse_args()


if __name__ == '__main__':
    from fastpm.io import BigFile
    from pypower import CatalogFFTPower, setup_logging
    from cosmoprimo.fiducial import DESI

    # to remove pmesh warning
    import warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    setup_logging()

    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    rank = mpicomm.Get_rank()
    start_ini = MPI.Wtime()

    args = collect_argparser()
    sim = os.path.join(args.path_to_sim, args.sim)

    # parametre -> je ferai un rebin si necessaire
    kedges = np.arange(1e-3, 6e-1, 1e-3)
    logger_info(logger, "Default arange: np.arange(1e-3, 6e-1, 1e-3)", rank)

    # Load DESI fiducial cosmo to convert redshift to distance (ok car univers plat):
    distance = DESI(engine='class').comoving_radial_distance

    for region in args.regions:
        start = MPI.Wtime()
        randoms = BigFile(os.path.join(args.path_to_sim, args.name_randoms), dataset=f'{args.release}-{region}/', mode='r', mpicomm=mpicomm)
        randoms.RA, randoms.DEC, randoms.DISTANCE = randoms.read('RA'), randoms.read('DEC'), distance(randoms.read('Z'))
        logger_info(logger, f"Load Randoms: {randoms.csize} randoms read in {MPI.Wtime() - start:2.2f} s.", rank)

        start = MPI.Wtime()
        cutsky = BigFile(os.path.join(sim, f'desi-cutsky-{args.aout}'), dataset=f'{args.release}-{region}/', mode='r', mpicomm=mpicomm)
        cutsky.RA, cutsky.DEC, cutsky.DISTANCE = cutsky.read('RA'), cutsky.read('DEC'), distance(cutsky.read('Z'))
        cutsky.NMOCK, cutsky.WSYS, is_wsys_cont = cutsky.read('NMOCK'), cutsky.read('WSYS'), cutsky.read('IS_WSYS_CONT')
        logger_info(logger, f"Number of galaxies: {cutsky.csize} read in {MPI.Wtime() - start:2.2f} s.", rank)

        for num in range(np.max(cutsky['NMOCK'])):
            sel = cutsky['NMOCK'] == num

            # Compute the power spectrum
            # To fix the size of the box, we take the same number than those for the Ezmocks 6pc computation
            start = MPI.Wtime()
            CatalogFFTPower(data_positions1=[cutsky['RA'][sel], cutsky['DEC'][sel], cutsky['DISTANCE'][sel]],
                            randoms_positions1=[randoms['RA'], randoms['DEC'], randoms['DISTANCE']],
                            position_type='rdd',
                            edges=kedges, ells=(0), cellsize=[6, 6, 6], boxsize=[8000, 16000, 8000],
                            resampler='tsc', interlacing=3, los='firstpoint',
                            mpicomm=mpicomm).poles.save(os.path.join(sim, f'power-spectrum/desi-cutsky-{args.release}-{region}-{num}.npy'))
            logger_info(logger, f'CatalogFFTPower done with uncont in {MPI.Wtime() - start:2.2f} s.', rank)

            start = MPI.Wtime()
            CatalogFFTPower(data_positions1=[cutsky['RA'][sel & is_wsys_cont], cutsky['DEC'][sel & is_wsys_cont], cutsky['DISTANCE'][sel & is_wsys_cont]],
                            randoms_positions1=[randoms['RA'], randoms['DEC'], randoms['DISTANCE']],
                            position_type='rdd',
                            edges=kedges, ells=(0), cellsize=[6, 6, 6], boxsize=[8000, 16000, 8000],
                            resampler='tsc', interlacing=3, los='firstpoint',
                            mpicomm=mpicomm).poles.save(os.path.join(sim, f'power-spectrum/desi-cutsky-{args.release}-{region}-{num}-cont.npy'))
            logger_info(logger, f'CatalogFFTPower done with cont in {MPI.Wtime() - start:2.2f} s.', rank)

            start = MPI.Wtime()
            CatalogFFTPower(data_positions1=[cutsky['RA'][sel & is_wsys_cont], cutsky['DEC'][sel & is_wsys_cont], cutsky['DISTANCE'][sel & is_wsys_cont]], data_weights1=cutsky['WSYS'][sel & is_wsys_cont],
                            randoms_positions1=[randoms['RA'], randoms['DEC'], randoms['DISTANCE']],
                            position_type='rdd',
                            edges=kedges, ells=(0), cellsize=[6, 6, 6], boxsize=[8000, 16000, 8000],
                            resampler='tsc', interlacing=3, los='firstpoint',
                            mpicomm=mpicomm).poles.save(os.path.join(sim, f'power-spectrum/desi-cutsky-{args.release}-{region}-{num}-corr.npy'))
            logger_info(logger, f'CatalogFFTPower done with cont in {MPI.Wtime() - start:2.2f} s.', rank)
