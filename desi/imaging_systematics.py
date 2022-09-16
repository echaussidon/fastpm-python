import os
import sys
import logging
import argparse

import numpy as np
import healpy as hp

from regressis import PhotometricDataFrame, DR9Footprint, Regression, PhotoWeight, build_healpix_map
from regressis.mocks import create_flag_imaging_systematic


logger = logging.getLogger('Imaging Systematics')


# To avoid error from NUMEXPR Package
os.environ.setdefault('NUMEXPR_MAX_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))
os.environ.setdefault('NUMEXPR_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))


def collect_argparser():
    parser = argparse.ArgumentParser(description="Apply imaging systematics and uncontaminate the mocks.")

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

    parser.add_argument("--seed", type=int, required=False, default=123,
                        help="for reproductibility")

    parser.add_argument("--name_randoms", type=str, required=False, default='fastpm-randoms',
                        help='Each region is saved in one dataset.')

    return parser.parse_args()


def apply_imaging_systematics(cutsky, sel, fracarea, nside=256, seed=946,
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

    for region in wsys.regions:
        wsys.mask_region[region] &= mock_footprint

    # ATTENTION pour l'instant on utilise wsys des targets donc la densite est bien plus elevee...
    # on va donc faire en sorte que ca soit 200 par degrée carré (ie) environ 10.5
    logger.warning("ATTENTION ON FIXE A LA MAIN LA DENSITE car on a que le wsys des targets")
    wsys.mean_density_region = {region: 10.5 for region in wsys.regions}

    _, is_wsys_cont, _ = create_flag_imaging_systematic(cutsky, sel, wsys, fracarea, use_real_density=True, seed=seed)

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
    sim = os.path.join(args.path_to_sim, args.sim)

    if mpicomm.size != 1:
        logger.error('Regressis is not design to work with MPI ! Please run this script with -n 1')
        sys.exit(1)

    seeds = {region: args.seed + 100 * i for i, region in enumerate(['N', 'SNGC', 'SSGC'])}
    for region in args.regions:
        # Load randoms:
        start = MPI.Wtime()
        randoms = BigFile(os.path.join(args.path_to_sim, args.name_randoms), dataset=f'{args.release}-{region}/', mode='r', mpicomm=mpicomm)
        randoms.RA, randoms.DEC, randoms.HPX = randoms.read('RA'), randoms.read('DEC'), randoms.read('HPX')
        # compute randoms healpix map to build fracarea
        nside = 256
        randoms_map = build_healpix_map(nside, randoms['RA'], randoms['DEC'], precomputed_pix=randoms['HPX'], in_deg2=True)
        sel = randoms_map > 0
        randoms_density = np.mean(randoms_map[sel])
        fracarea = randoms_map / randoms_density
        fracarea[~sel] = np.nan
        logger.info(f"Load randoms and compute fracarea done in {MPI.Wtime() - start:2.2f} s.")

        # Load data:
        start = MPI.Wtime()
        cutsky = BigFile(os.path.join(sim, f'desi-cutsky-{args.aout}'), dataset=f'{args.release}-{region}/', mode='rw', mpicomm=mpicomm)
        cutsky.RA, cutsky.DEC, cutsky.Z = cutsky.read('RA'), cutsky.read('DEC'), cutsky.read('Z')
        cutsky.NMOCK, cutsky.HPX = cutsky.read('NMOCK'), cutsky.read('HPX')
        logger.info(f"Number of galaxies: {cutsky.csize} read in {MPI.Wtime() - start:2.2f} s.")

        # Apply systematics contamination and apply the correction
        start = MPI.Wtime()
        is_wsys_cont, sys_weight = np.ones(cutsky.size, dtype='bool'), np.nan * np.ones(cutsky.size)
        for num in range(np.max(cutsky['NMOCK'])):
            sel = cutsky['NMOCK'] == num
            is_wsys_cont[sel] = apply_imaging_systematics(cutsky, sel, fracarea, seed=seeds[region] + num,
                                                          wsys_path=f'/global/cfs/cdirs/desi/users/edmondc/sys_weight/photo/MAIN_QSO_imaging_weight_{nside}.npy')

            sys_weight[sel & is_wsys_cont] = compute_imaging_systematic_weights(cutsky, sel & is_wsys_cont, region, fracarea, nside=256, seed=(seeds[region] + 10) * num, n_jobs=64)
        cutsky.write({'IS_WSYS_CONT': is_wsys_cont, 'WSYS': sys_weight})

        logger.info(f"Contaminate all subsamples in regions: {args.regions} done in {MPI.Wtime() - start:2.2f} s.")
