import os
import logging
import argparse

import numpy as np


logger = logging.getLogger('Compute Window matrix')


# disable jax warning:
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)


# To avoid error from NUMEXPR Package
os.environ.setdefault('NUMEXPR_MAX_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))
os.environ.setdefault('NUMEXPR_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))


def logger_info(logger, msg, rank, mpiroot=0):
    """Print something with the logger only for the rank == mpiroot to avoid duplication of message."""
    if rank == mpiroot:
        logger.info(msg)


def collect_argparser():
    parser = argparse.ArgumentParser(description="Compute the window matrix with the randoms")

    parser.add_argument("--path_to_sim", type=str, required=False, default='/pscratch/sd/e/edmondc/Mocks/',
                        help="Path to the Scratch where the simulations are saved")
    parser.add_argument("--sim", type=str, required=False, default='test',
                        help="Simulation name (e.g) fastpm-fnl-0")
    parser.add_argument("--aout", type=str, required=False, default='1.0000',
                        help="scale factor at which the halos are saved (e.g) '0.3300' or '1.0000'")

    parser.add_argument("--release", type=str, required=False, default='Y5',
                        help="match Y1 / Y5 footprint")
    parser.add_argument("--npasses", type=int, required=False, default=None,
                        help="match footprint with more than npasses observation")
    parser.add_argument("--regions", nargs='+', type=str, required=False, default=['N', 'SNGC', 'SSGC'],
                        help="photometric regions")

    parser.add_argument("--maskbits", nargs='+', type=int, required=False, default=[1, 8, 9, 11, 12, 13],
                        help="DR9 maskbits used to cut the data and the randoms, default=[1, 8, 9, 11, 12, 13]\
                              The default one mimicking the clustering catalog.")

    parser.add_argument("--name_randoms", type=str, required=False, default='fastpm-randoms',
                        help='Each region is be saved in one dataset.')

    parser.add_argument("--boxsizes", nargs='+', type=int, required=False, default=[200000, 50000, 20000],
                        help="Compute the window matrix with different boxsizes to concatenate it ! (Increase a lot the accuracy)")

    return parser.parse_args()


if __name__ == '__main__':
    from fastpm.io import BigFile
    from cosmoprimo.fiducial import DESI
    from mockfactory import setup_logging
    from pypower import PowerSpectrumStatistics, CatalogSmoothWindow

    # to remove pmesh warning
    import warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    os.environ['DESI_LOGLEVEL'] = 'ERROR'
    setup_logging()

    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    rank = mpicomm.Get_rank()
    start_ini = MPI.Wtime()
    logger_info(logger, 'Run window function computation', rank)

    args = collect_argparser()
    sim = os.path.join(args.path_to_sim, args.sim)

    # Load DESI fiducial cosmo to convert redshift to distance (ok car univers plat):
    distance = DESI(engine='class').comoving_radial_distance

    # parameters to compute power spectrum
    kedges = np.arange(1e-3, 6e-1, 1e-3)
    logger_info(logger, "Power spectrum default arange: np.arange(1e-3, 6e-1, 1e-3)", rank)

    # Let's go:
    for region in args.regions:
        # dataset in which the data will be writen
        dataset = f'{args.release}-{region}'
        if (args.npasses is not None) and (args.npasses > 1): dataset = f'{args.release}-{region}-{args.npasses}pass'

        start = MPI.Wtime()
        randoms = BigFile(os.path.join(args.path_to_sim, args.name_randoms), dataset=dataset + '/', mode='r', mpicomm=mpicomm)
        maskbits = randoms.read('MASKBITS')
        # keep only objects without maskbits
        sel = np.ones(maskbits.size, dtype=bool)
        for mskb in args.maskbits:
            sel &= (maskbits & 2**mskb) == 0
        randoms.RA, randoms.DEC, randoms.DISTANCE = randoms.read('RA')[sel], randoms.read('DEC')[sel], distance(randoms.read('Z')[sel])
        logger_info(logger, f"Load Randoms and apply maskbits {args.maskbits}: {randoms.csize} randoms read in {MPI.Wtime() - start:2.2f} s.", rank)

        # Load reference poles:
        # la window doit être normalisée par le même facteur que le pk
        power = PowerSpectrumStatistics.load(os.path.join(sim, f'power-spectrum/desi-cutsky-{dataset}-0.npy')) if mpicomm.rank == 0 else 0
        power = mpicomm.bcast(power, root=0)

        start = MPI.Wtime()
        for boxsize in args.boxsizes:
            window = CatalogSmoothWindow(randoms_positions1=[randoms['RA'], randoms['DEC'], randoms['DISTANCE']], position_type='rdd',
                                         power_ref=power, edges={'step': 1e-4}, boxsize=boxsize,
                                         mpicomm=mpicomm).poles.save(os.path.join(args.path_to_sim, args.name_randoms, f'window-matrix/window_matrix_boxisze-{boxsize}_{dataset}.npy'))
        logger_info(logger, f'Window matrix computation for {len(args.boxsizes)} boxsize done with cont in {MPI.Wtime() - start:2.2f} s.', rank)
