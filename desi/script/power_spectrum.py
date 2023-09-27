import os
import logging
import argparse

import numpy as np


logger = logging.getLogger('Power Spectrum')


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
    parser = argparse.ArgumentParser(description="Transform position in real space to redshift space and compute the multipoles.")

    parser.add_argument("--path_to_sim", type=str, required=False, default='/pscratch/sd/e/edmondc/Mocks/',
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

    parser.add_argument("--maskbits", nargs='+', type=int, required=False, default=[1, 7, 8, 11, 12, 13],
                        help="DR9 maskbits used to cut the data and the randoms, default=[1, 7, 8, 11, 12, 13]\
                              The default one mimicking the clustering catalog.")

    parser.add_argument("--name_randoms", type=str, required=False, default='fastpm-randoms',
                        help='Each region is be saved in one dataset.')

    parser.add_argument("--compute_all", type=str, required=False, default='False', help='to compute cutsky power spectrum with the full sample. Take care to the number of randoms')
    
    parser.add_argument("--compute_ini", type=str, required=False, default='False', help='Warning (from 09/15/23) only one type of pk can be computed, launch several times the script --> not a problem with bash :)')
    parser.add_argument("--compute_inicorr", type=str, required=False, default='False')
    parser.add_argument("--compute_cont", type=str, required=False, default='False')
    parser.add_argument("--compute_corr", type=str, required=False, default='False')

    parser.add_argument("--nmock_ini", type=int, required=False, default=0)
    
    parser.add_argument("--use_fkp", type=str, required=False, default='False')
    parser.add_argument("--use_oqe", type=str, required=False, default='False')
    parser.add_argument("--pop", type=float, required=False, default=1.6)

    return parser.parse_args()


def b_qso(z):
    """
    QSO bias model taken from Laurent et al. 2016 (1705.04718)
    """
    alpha = 0.278
    beta = 2.393
    return alpha * ( (1+z)**2 - 6.565 ) + beta



if __name__ == '__main__':
    from fastpm.io import BigFile
    from pypower import CatalogFFTPower, PowerSpectrumMultipoles, setup_logging
    from cosmoprimo.fiducial import DESI

    # to remove pmesh warning
    import warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    from scipy.interpolate import interp1d
    
    setup_logging()

    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    rank = mpicomm.Get_rank()
    start_ini = MPI.Wtime()

    args = collect_argparser()
    # display args in logger to keep tracer of inout;
    logger_info(logger, args, rank)
    sim = os.path.join(args.path_to_sim, args.sim)
    
    suff_power = '_w_fkp' if (args.use_fkp == 'True') else ''
    suff_power += f'_w_oqe_p_{args.pop}' if (args.use_oqe == 'True') else ''
    logger_info(logger, f'Suffixe power: {suff_power}', rank)

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
        randoms.RA, randoms.DEC, randoms.Z, randoms.DISTANCE = randoms.read('RA')[sel], randoms.read('DEC')[sel], randoms.read('Z')[sel], distance(randoms.read('Z')[sel])
        logger_info(logger, f"Load Randoms and apply maskbits {args.maskbits}: {randoms.csize} randoms read in {MPI.Wtime() - start:2.2f} s.", rank)

        start = MPI.Wtime()
        cutsky = BigFile(os.path.join(sim, f'desi-cutsky-{args.aout}'), dataset=dataset, mode='r', mpicomm=mpicomm)
        maskbits = cutsky.read('MASKBITS')
        # keep only objects without maskbits
        sel = np.ones(maskbits.size, dtype=bool)
        for mskb in args.maskbits:
            sel &= (maskbits & 2**mskb) == 0
        cutsky.RA, cutsky.DEC, cutsky.Z, cutsky.DISTANCE = cutsky.read('RA')[sel], cutsky.read('DEC')[sel], cutsky.read('Z')[sel], distance(cutsky.read('Z')[sel])
        cutsky.NMOCK, cutsky.WSYS, cutsky.WSYS_INI, is_for_uncont, is_wsys_cont = cutsky.read('NMOCK')[sel], cutsky.read('WSYS')[sel], cutsky.read('WSYS_INI')[sel], cutsky.read('IS_FOR_UNCONT')[sel], cutsky.read('IS_WSYS_CONT')[sel]
        logger_info(logger, f"Load Data and apply maskbits {args.maskbits}: {cutsky.csize} data read in {MPI.Wtime() - start:2.2f} s.", rank)

        # Compute relevant weights for randoms:
        randoms.WEIGHT = np.ones(randoms['RA'].size)
        cutsky.WEIGHT = np.ones(cutsky['RA'].size)
        if (args.use_fkp == 'True') or (args.use_oqe == 'True'):
            P0 = 3e4
            zbin_min, zbin_max, n_z = np.load(f'/global/homes/e/edmondc/Software/fastpm-python/desi/script/data/nz_qso_Y1_{region}.npy') # load le nz du fichier clustering catalog ! en Mpc^-3h^3
            nz = interp1d((zbin_min + zbin_max) / 2, n_z, kind='quadratic', bounds_error=False, fill_value=(0, 0))
            randoms.WEIGHT_FKP = 1/(1 + nz(randoms['Z']) * P0)
            randoms.WEIGHT = randoms['WEIGHT'] * randoms['WEIGHT_FKP']
            cutsky.WEIGHT_FKP = 1/(1 + nz(cutsky['Z']) * P0)
            cutsky.WEIGHT  = cutsky['WEIGHT'] * cutsky['WEIGHT_FKP']
        if args.use_oqe == 'True':
            cosmo = DESI(engine='class')
            randoms.W_TILDE = b_qso(randoms['Z']) - args.pop
            randoms.W0 = cosmo.growth_factor(randoms['Z']) * (b_qso(randoms['Z']) + cosmo.growth_rate(randoms['Z']) / 3)
            randoms.W2 = 2 / 3 * cosmo.growth_factor(randoms['Z']) * cosmo.growth_rate(randoms['Z'])
            cutsky.W_TILDE = b_qso(cutsky['Z']) - args.pop
            cutsky.W0 = cosmo.growth_factor(cutsky['Z']) * (b_qso(cutsky['Z']) + cosmo.growth_rate(cutsky['Z']) / 3)
            cutsky.W2 = 2 / 3 * cosmo.growth_factor(cutsky['Z']) * cosmo.growth_rate(cutsky['Z'])

        if args.compute_all == 'True':
            start = MPI.Wtime()
            CatalogFFTPower(data_positions1=[cutsky['RA'], cutsky['DEC'], cutsky['DISTANCE']],
                            randoms_positions1=[randoms['RA'], randoms['DEC'], randoms['DISTANCE']],
                            position_type='rdd',
                            edges=kedges, ells=(0, 2, 4), cellsize=[6, 6, 6], boxsize=[8000, 16000, 8000],
                            resampler='tsc', interlacing=3, los='firstpoint',
                            mpicomm=mpicomm).poles.save(os.path.join(sim, f'power-spectrum/desi-cutsky-{dataset[:-1]}-full.npy'))
            logger_info(logger, f'CatalogFFTPower done with the full sample in {MPI.Wtime() - start:2.2f} s.', rank)

        
        if args.compute_ini == 'True':
            suff_pk = ''
            to_use = is_for_uncont
            data_weights = np.ones(is_for_uncont.size)
                
        if args.compute_inicorr == 'True':
            suff_pk = '-inicorr'
            to_use = is_for_uncont
            data_weights = cutsky['WSYS_INI']
            
        if args.compute_cont == 'True':
            suff_pk = '-cont'
            to_use = is_wsys_cont
            data_weights = np.ones(is_for_uncont.size)
            
        if args.compute_corr == 'True':
            suff_pk = '-corr'
            to_use = is_wsys_cont
            data_weights = cutsky['WSYS']

        # Collect the number of subsample available in cutsky, take care all ranks do not have all the subsample, need to collect it across all the ranks!
        max_nmock = np.max(mpicomm.gather(np.max(cutsky['NMOCK'])))
        max_nmock = mpicomm.bcast(max_nmock, root=0)
        for num in range(args.nmock_ini, max_nmock):
            sel = cutsky['NMOCK'] == num
            # Compute the power spectrum: To fix the size of the box, we take the same number than those for the Ezmocks 6pc computation

            if args.use_oqe == 'False':
                CatalogFFTPower(data_positions1=[cutsky['RA'][sel & to_use], cutsky['DEC'][sel & to_use], cutsky['DISTANCE'][sel & to_use]],
                                data_weights1=cutsky['WEIGHT'][sel & to_use] * data_weights[sel & to_use],
                                randoms_positions1=[randoms['RA'], randoms['DEC'], randoms['DISTANCE']],
                                randoms_weights1=randoms['WEIGHT'],
                                position_type='rdd',
                                edges=kedges, ells=(0, 2, 4), cellsize=[6, 6, 6], boxsize=[8000, 16000, 8000],
                                resampler='tsc', interlacing=3, los='firstpoint',
                                mpicomm=mpicomm).poles.save(os.path.join(sim, f'power-spectrum/desi-cutsky-{dataset[:-1]}-{num}{suff_pk}{suff_power}.npy'))                    
            else:
                # monopole
                poles_0 = CatalogFFTPower(data_positions1=[cutsky['RA'][sel & to_use], cutsky['DEC'][sel & to_use], cutsky['DISTANCE'][sel & to_use]], 
                                          data_weights1=cutsky['WEIGHT_FKP'][sel & to_use] * cutsky['W_TILDE'][sel & to_use] * data_weights[sel & to_use], 
                                          data_weights2=cutsky['WEIGHT_FKP'][sel & to_use] * cutsky['W0'][sel & to_use] * data_weights[sel & to_use],
                                          randoms_positions1=[randoms['RA'], randoms['DEC'], randoms['DISTANCE']], 
                                          randoms_weights1=randoms['WEIGHT_FKP'] * randoms['W_TILDE'], 
                                          randoms_weights2=randoms['WEIGHT_FKP'] * randoms['W0'],
                                          position_type='rdd', edges=kedges, ells=(0), cellsize=[6, 6, 6], boxsize=[8000, 16000, 8000],
                                          resampler='tsc', interlacing=3, los='firstpoint',
                                          mpicomm=mpicomm).poles
                # quadrupole
                poles_2 = CatalogFFTPower(data_positions1=[cutsky['RA'][sel & to_use], cutsky['DEC'][sel & to_use], cutsky['DISTANCE'][sel & to_use]], 
                                          data_weights1=cutsky['WEIGHT_FKP'][sel & to_use] * cutsky['W_TILDE'][sel & to_use] * data_weights[sel & to_use], 
                                          data_weights2=cutsky['WEIGHT_FKP'][sel & to_use] * cutsky['W2'][sel & to_use] * data_weights[sel & to_use],
                                          randoms_positions1=[randoms['RA'], randoms['DEC'], randoms['DISTANCE']], 
                                          randoms_weights1=randoms['WEIGHT_FKP'] * randoms['W_TILDE'], 
                                          randoms_weights2=randoms['WEIGHT_FKP'] * randoms['W2'],
                                          position_type='rdd', edges=kedges, ells=(2), cellsize=[6, 6, 6], boxsize=[8000, 16000, 8000],
                                          resampler='tsc', interlacing=3, los='firstpoint',
                                          mpicomm=mpicomm).poles

                # Then you can combine the two measurements together
                poles_02 = PowerSpectrumMultipoles.concatenate_proj(poles_0, poles_2).save(os.path.join(sim, f'power-spectrum/desi-cutsky-{dataset[:-1]}-{num}{suff_pk}{suff_power}.npy'))