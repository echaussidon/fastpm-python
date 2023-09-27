import os
import sys
import logging
import argparse

import numpy as np
import healpy as hp

from regressis import PhotometricDataFrame, DR9Footprint, Regression, PhotoWeight, build_healpix_map
from regressis.mocks import create_flag_imaging_systematic


logger = logging.getLogger('Imaging Systematics')


# disable jax warning:
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)


# To avoid error from NUMEXPR Package
os.environ.setdefault('NUMEXPR_MAX_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))
os.environ.setdefault('NUMEXPR_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))


def collect_argparser():
    parser = argparse.ArgumentParser(description="Apply imaging systematics and uncontaminate the mocks.")

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

    parser.add_argument("--seed", type=int, required=False, default=123,
                        help="for reproductibility")

    parser.add_argument("--which_contamination", type=str, required=False, default='TS',
                        help='Choose which type of contamination do you want, for now: TS or Y1')

    parser.add_argument("--name_randoms", type=str, required=False, default='fastpm-randoms',
                        help='Each region is saved in one dataset.')

    return parser.parse_args()


def apply_imaging_systematics(cutsky, sel, fracarea, nside=256, seed=946, fix_density=True,
                              wsys_path='/global/cfs/cdirs/desi/users/edmondc/sys_weight/photo/MAIN_QSO_imaging_weight_256.npy'):
    """ Warning: cannot be called with mpicomm.size > 1 !
        cutsky.size == contain nmock subsamples. Need to return is_wsys_cont[sel] to return the flag only for the object from the sel subsample.

        cutsky is alread full of masks .. --> need fracarea (from randoms to estimate) the mean density of the mocks !
        """

    # Load photometric weight
    wsys = PhotoWeight.load(wsys_path)

    # take care: create_flag_imaging_systematic use wsys.mask_region to compute the mean / ratio / fraction of objects to remove
    # we need to restricte wsys.mask_region only where the mocks have data
    mock_footprint = np.zeros(hp.nside2npix(nside))
    mock_footprint[cutsky['HPX'][sel]] += 1
    mock_footprint = mock_footprint > 0

    if mock_footprint[np.isnan(fracarea)].sum() > 0:
        logger.warning(f'There is {mock_footprint[np.isnan(fracarea)].sum()} pixels with data but no-randoms ... Remove the data --> No probleme if it is typically one')
        mock_footprint &= ~np.isnan(fracarea)

    for region in wsys.regions:
        wsys.mask_region[region] &= mock_footprint

    if fix_density:
        # ATTENTION pour l'instant on utilise wsys des targets donc la densite est bien plus elevee...
        # on va donc faire en sorte que ca soit 200 par degrée carré (ie) environ 10.5
        logger.warning("ATTENTION ON FIXE A LA MAIN LA DENSITE car on a que le wsys des targets")
        wsys.mean_density_region = {region: 10.5 for region in wsys.regions}

    _, is_wsys_cont, _ = create_flag_imaging_systematic(cutsky, sel, wsys, fracarea, pix_number=cutsky['HPX'], use_real_density=True, seed=seed)

    # is_wsys_cont is size of cutsky.size --> return only is_wsys_cont[sel] to work with one sub-sample
    return is_wsys_cont[sel]


def compute_imaging_systematic_weights(cutsky, sel, region, fracarea, nside=256, seed=123, n_jobs=6):
    """ Use regressis to compute imaging systematic weights
        Return the weight for each objects in the subsample contaminated mocks"""

    """ IMPORTANT: pour gnager du temps et ne pas m'mebeter je ne fais pas de seleciton sur les maskbits--> ok au preimer ordre tout cela est statistique """

    data_map = build_healpix_map(nside, cutsky['RA'][sel], cutsky['DEC'][sel], precomputed_pix=cutsky['HPX'][sel], in_deg2=False)

    # Set parameters for the dataframe:
    params = dict()
    params['data_dir'] = '/global/homes/e/edmondc/Software/regressis/data'
    params['output_dir'] = None
    params['use_median'] = False
    params['use_new_norm'] = False
    params['regions'] = {'N': ['North'], 'SNGC': ['South_mid_ngc'], 'SSGC': ['South_mid_sgc']}[region]  # attention ca doit etre une liste

    # Build PhotometricDataFrame class:
    dataframe = PhotometricDataFrame('MOCK', '', DR9Footprint(nside), **params)
    dataframe.set_features()
    dataframe.set_targets(targets=data_map, fracarea=fracarea)
    dataframe.build(cut_fracarea=False)

    # Which features will be used during the regression.
    feature_names = ['STARDENS', 'EBV', 'STREAM',
                     'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2',
                     'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']

    regressor_params = {region: {'random_state': seed + i} for i, region in enumerate(['North', 'South_mid_ngc', 'South_mid_sgc'])}
    regression = Regression(dataframe, feature_names=feature_names,
                            regressor='RF', suffix_regressor='', use_kfold=True,
                            n_jobs=n_jobs, regressor_params=regressor_params)

    return regression.get_weight(save=False)(cutsky['RA'][sel], cutsky['DEC'][sel])


if __name__ == '__main__':
    """ regressis does not work will MPI... --> launch this only with -n 1 """
    from fastpm.io import BigFile
    from pypower import setup_logging
    setup_logging()

    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD

    args = collect_argparser()
    # display args in logger to keep tracer of inout;
    if mpicomm.rank == 0: logger.info(args)
    sim = os.path.join(args.path_to_sim, args.sim)

    if mpicomm.size != 1:
        logger.error('Regressis is not design to work with MPI ! Please run this script with -n 1')
        sys.exit(1)

    # take care we generate the healpix number in make_desi_survey.py at nside=256
    nside = 256

    if args.which_contamination == 'TS':
        wsys_path_cont = f'/global/homes/e/edmondc/Software/fastpm-python/desi/script/data/MAIN_QSO_imaging_weight_{nside}.npy'
        fix_density = True
    elif args.which_contamination == 'Y1':
        wsys_path_cont = f'/global/homes/e/edmondc/Software/fastpm-python/desi/script/data/Y1_QSO_ssgc_imaging_weight_{nside}.npy'
        fix_density = False
    else:
        logger.error('Please choose correct flag for which contamination')
        sys.exit(1)

    seeds = {region: args.seed + 100 * i for i, region in enumerate(['N', 'SNGC', 'SSGC'])}
    for region in args.regions:
        # dataset in which the data will be writen
        dataset = f'{args.release}-{region}/'
        if (args.npasses is not None) and (args.npasses > 1): dataset = f'{args.release}-{region}-{args.npasses}pass/'

        # Load randoms:
        start = MPI.Wtime()
        randoms = BigFile(os.path.join(args.path_to_sim, args.name_randoms), dataset=dataset, mode='r', mpicomm=mpicomm)
        randoms_density = randoms.attrs['randoms_density']
        randoms.RA, randoms.DEC, randoms.HPX = randoms.read('RA'), randoms.read('DEC'), randoms.read('HPX')
        # compute randoms healpix map to build fracarea
        randoms_map = build_healpix_map(nside, randoms['RA'], randoms['DEC'], precomputed_pix=randoms['HPX'], in_deg2=True)
        fracarea = randoms_map / randoms_density
        fracarea[~(randoms_map > 0)] = np.nan
        logger.info(f"Load randoms and compute fracarea done in {MPI.Wtime() - start:2.2f} s.")

        # Load data:
        start = MPI.Wtime()
        cutsky = BigFile(os.path.join(sim, f'desi-cutsky-{args.aout}'), dataset=dataset, mode='rw', mpicomm=mpicomm)
        cutsky.RA, cutsky.DEC, cutsky.Z = cutsky.read('RA'), cutsky.read('DEC'), cutsky.read('Z')
        cutsky.NMOCK, cutsky.HPX = cutsky.read('NMOCK'), cutsky.read('HPX')
        is_for_uncont = cutsky.read('IS_FOR_UNCONT')
        logger.info(f"Number of galaxies: {cutsky.csize} read in {MPI.Wtime() - start:2.2f} s.")

        # Apply systematics contamination and apply the correction
        start = MPI.Wtime()
        sys_weight_ini = np.nan * np.ones(cutsky.size)
        is_wsys_cont, sys_weight = np.ones(cutsky.size, dtype='bool'), np.nan * np.ones(cutsky.size)

        # Collect the number of subsample available in cutsky, take care all ranks do not have all the subsample, need to collect it across all the ranks!
        max_nmock = np.max(mpicomm.gather(np.max(cutsky['NMOCK'])))
        max_nmock = mpicomm.bcast(max_nmock, root=0)
        for num in range(max_nmock):
            sel = cutsky['NMOCK'] == num
            
            # compute correction on uncontaminted mocks:
            sys_weight_ini[sel & is_for_uncont] = compute_imaging_systematic_weights(cutsky, sel & is_for_uncont, region, fracarea, nside=256, seed=(seeds[region] + 10) * num, n_jobs=64)
            
            # contaminate mocks:
            is_wsys_cont[sel] = apply_imaging_systematics(cutsky, sel, fracarea, seed=seeds[region] + num, fix_density=fix_density, wsys_path=wsys_path_cont)
            
            # compute correction on contaminated mocks:
            sys_weight[sel & is_wsys_cont] = compute_imaging_systematic_weights(cutsky, sel & is_wsys_cont, region, fracarea, nside=256, seed=(seeds[region] + 10) * num, n_jobs=64)
        
        cutsky.write({'WSYS_INI': sys_weight_ini})
        cutsky.write({'IS_WSYS_CONT': is_wsys_cont, 'WSYS': sys_weight})

        logger.info(f"Contaminate all subsamples in regions: {args.regions} done in {MPI.Wtime() - start:2.2f} s.")
