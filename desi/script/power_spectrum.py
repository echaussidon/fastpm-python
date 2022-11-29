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
                        help="match y1 / y5 footprint")
    # parser.add_argument("--program", type=str, required=False, default='dark',
    #                     help="match bright / dark footprint")
    parser.add_argument("--npasses", type=int, required=False, default=None,
                        help="match footprint with more than npasses observation")
    parser.add_argument("--regions", nargs='+', type=str, required=False, default=['N', 'SNGC', 'SSGC'],
                        help="photometric regions")

    parser.add_argument("--maskbits", nargs='+', type=int, required=False, default=[1, 8, 9, 11, 12, 13],
                        help="DR9 maskbits used to cut the data and the randoms, default=[1, 8, 9, 11, 12, 13]\
                              The default one mimicking the clustering catalog.")

    parser.add_argument("--name_randoms", type=str, required=False, default='fastpm-randoms',
                        help='Each region is be saved in one dataset.')

    parser.add_argument("--compute_all", type=str, required=False, default='False', help='to compute cutsky power spectrum with the full sample. Take care to the number of randoms')
    parser.add_argument("--compute_ini", type=str, required=False, default='True')
    parser.add_argument("--compute_cont", type=str, required=False, default='True')
    parser.add_argument("--compute_corr", type=str, required=False, default='True')

    parser.add_argument("--nmock_ini", type=int, required=False, default=0, help='compute tu')

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
    # display args in logger to keep tracer of inout;
    logger_info(logger, args, rank)
    sim = os.path.join(args.path_to_sim, args.sim)

    # parametre -> je ferai un rebin si necessaire
    kedges = np.arange(1e-3, 6e-1, 1e-3)
    logger_info(logger, "Default arange: np.arange(1e-3, 6e-1, 1e-3)", rank)

    # Load DESI fiducial cosmo to convert redshift to distance (ok car univers plat):
    distance = DESI(engine='class').comoving_radial_distance

    for region in args.regions:
        # dataset in which the data will be writen
        dataset = f'{args.release}-{region}/'
        if (args.npasses is not None) and (args.npasses > 1): dataset = f'{args.release}-{region}-{args.npasses}pass/'

        start = MPI.Wtime()
        randoms = BigFile(os.path.join(args.path_to_sim, args.name_randoms), dataset=dataset, mode='r', mpicomm=mpicomm)
        maskbits = randoms.read('MASKBITS')
        # keep only objects without maskbits
        sel = np.ones(maskbits.size, dtype=bool)
        for mskb in args.maskbits:
            sel &= (maskbits & 2**mskb) == 0
        randoms.RA, randoms.DEC, randoms.DISTANCE = randoms.read('RA')[sel], randoms.read('DEC')[sel], distance(randoms.read('Z')[sel])
        logger_info(logger, f"Load Randoms and apply maskbits {args.maskbits}: {randoms.csize} randoms read in {MPI.Wtime() - start:2.2f} s.", rank)

        start = MPI.Wtime()
        cutsky = BigFile(os.path.join(sim, f'desi-cutsky-{args.aout}'), dataset=dataset, mode='r', mpicomm=mpicomm)
        maskbits = cutsky.read('MASKBITS')
        # keep only objects without maskbits
        sel = np.ones(maskbits.size, dtype=bool)
        for mskb in args.maskbits:
            sel &= (maskbits & 2**mskb) == 0
        cutsky.RA, cutsky.DEC, cutsky.DISTANCE = cutsky.read('RA')[sel], cutsky.read('DEC')[sel], distance(cutsky.read('Z')[sel])
        cutsky.NMOCK, is_for_uncont = cutsky.read('NMOCK')[sel], cutsky.read('IS_FOR_UNCONT')[sel]
        if args.compute_cont == 'True' or args.compute_corr == 'True':
            cutsky.NMOCK, cutsky.WSYS, is_for_uncont, is_wsys_cont = cutsky.read('NMOCK')[sel], cutsky.read('WSYS')[sel], cutsky.read('IS_FOR_UNCONT')[sel], cutsky.read('IS_WSYS_CONT')[sel]
        logger_info(logger, f"Load Data and apply maskbits {args.maskbits}: {cutsky.csize} data read in {MPI.Wtime() - start:2.2f} s.", rank)

        if args.compute_all == 'True':
            start = MPI.Wtime()
            CatalogFFTPower(data_positions1=[cutsky['RA'], cutsky['DEC'], cutsky['DISTANCE']],
                            randoms_positions1=[randoms['RA'], randoms['DEC'], randoms['DISTANCE']],
                            position_type='rdd',
                            edges=kedges, ells=(0, 2, 4), cellsize=[6, 6, 6], boxsize=[8000, 16000, 8000],
                            resampler='tsc', interlacing=3, los='firstpoint',
                            mpicomm=mpicomm).poles.save(os.path.join(sim, f'power-spectrum/desi-cutsky-{dataset[:-1]}-full.npy'))
            logger_info(logger, f'CatalogFFTPower done with the full sample in {MPI.Wtime() - start:2.2f} s.', rank)

        # Collect the number of subsample available in cutsky, take care all ranks do not have all the subsample, need to collect it across all the ranks!
        max_nmock = np.max(mpicomm.gather(np.max(cutsky['NMOCK'])))
        max_nmock = mpicomm.bcast(max_nmock, root=0)
        for num in range(args.nmock_ini, max_nmock):
            sel = cutsky['NMOCK'] == num
            # Compute the power spectrum
            # To fix the size of the box, we take the same number than those for the Ezmocks 6pc computation

            if args.compute_ini == 'True':
                start = MPI.Wtime()
                CatalogFFTPower(data_positions1=[cutsky['RA'][sel & is_for_uncont], cutsky['DEC'][sel & is_for_uncont], cutsky['DISTANCE'][sel & is_for_uncont]],
                                randoms_positions1=[randoms['RA'], randoms['DEC'], randoms['DISTANCE']],
                                position_type='rdd',
                                edges=kedges, ells=(0, 2, 4), cellsize=[6, 6, 6], boxsize=[8000, 16000, 8000],
                                resampler='tsc', interlacing=3, los='firstpoint',
                                mpicomm=mpicomm).poles.save(os.path.join(sim, f'power-spectrum/desi-cutsky-{dataset[:-1]}-{num}.npy'))
                logger_info(logger, f'CatalogFFTPower done with uncont in {MPI.Wtime() - start:2.2f} s.', rank)

            if args.compute_cont == 'True':
                start = MPI.Wtime()
                CatalogFFTPower(data_positions1=[cutsky['RA'][sel & is_wsys_cont], cutsky['DEC'][sel & is_wsys_cont], cutsky['DISTANCE'][sel & is_wsys_cont]],
                                randoms_positions1=[randoms['RA'], randoms['DEC'], randoms['DISTANCE']],
                                position_type='rdd',
                                edges=kedges, ells=(0, 2, 4), cellsize=[6, 6, 6], boxsize=[8000, 16000, 8000],
                                resampler='tsc', interlacing=3, los='firstpoint',
                                mpicomm=mpicomm).poles.save(os.path.join(sim, f'power-spectrum/desi-cutsky-{dataset[:-1]}-{num}-cont.npy'))
                logger_info(logger, f'CatalogFFTPower done with cont in {MPI.Wtime() - start:2.2f} s.', rank)

            if args.compute_corr == 'True':
                start = MPI.Wtime()
                CatalogFFTPower(data_positions1=[cutsky['RA'][sel & is_wsys_cont], cutsky['DEC'][sel & is_wsys_cont], cutsky['DISTANCE'][sel & is_wsys_cont]], data_weights1=cutsky['WSYS'][sel & is_wsys_cont],
                                randoms_positions1=[randoms['RA'], randoms['DEC'], randoms['DISTANCE']],
                                position_type='rdd',
                                edges=kedges, ells=(0, 2, 4), cellsize=[6, 6, 6], boxsize=[8000, 16000, 8000],
                                resampler='tsc', interlacing=3, los='firstpoint',
                                mpicomm=mpicomm).poles.save(os.path.join(sim, f'power-spectrum/desi-cutsky-{dataset[:-1]}-{num}-corr.npy'))
                logger_info(logger, f'CatalogFFTPower done with cont in {MPI.Wtime() - start:2.2f} s.', rank)
