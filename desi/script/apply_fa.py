import os
import sys
import logging
import argparse

import numpy as np


logger = logging.getLogger('Apply F.A.')


# To avoid error from NUMEXPR Package
os.environ.setdefault('NUMEXPR_MAX_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))
os.environ.setdefault('NUMEXPR_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))


def logger_info(logger, msg, rank, mpiroot=0):
    """Print something with the logger only for the rank == mpiroot to avoid duplication of message."""
    if rank == mpiroot:
        logger.info(msg)


def add_info_for_fa(catalog, desi_target=2**2, obscondition=3, numobs=1, offset=0, seed=123):
    """
    We follow the MTL procedure described in Schalfy et al. 2023: https://www.overleaf.com/project/61424b13a719290754560897

    Note: for the QSO clustering. When a qso is observed to be at z>2.1, it is reobserved but with a lower probability than the unobserved QSO.
    Therefore, we do not care about the reobservation at all when we apply F.A. for QSO clustering. This is not the case for ELG/LRG case, in which we need to set up at 3 the NUMOBS_MORE when z>2.1.

    """
    # Add requiered columns for F.A.
    catalog['DESI_TARGET'] = desi_target * np.ones(catalog.size, dtype='i8')
    # Warning: the reproducibility (ie) the choice of target when multi-targets are available is done via SUBPRIORITY. Need random generator invariant under MPI scaling !
    catalog['SUBPRIORITY'] = catalog.rng(seed=seed).uniform(low=0, high=1, dtype='f8')
    catalog['OBSCONDITIONS'] = obscondition * np.ones(catalog.size, dtype='i8')
    catalog['NUMOBS_MORE'] = numobs * np.ones(catalog.size, dtype='i8')

    # take care with MPI ! TARGETID has to be unique !
    # use offset if you want to concatenate several catalogs :)
    cumsize = np.cumsum([0] + mpicomm.allgather(catalog.size))[mpicomm.rank]
    catalog['TARGETID'] = offset + cumsize + np.arange(catalog.size)

    return catalog


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
    parser.add_argument("--npasses", type=int, required=False, default=None,
                        help="match footprint with more than npasses observation")
    parser.add_argument("--regions", nargs='+', type=str, required=False, default=['N', 'SNGC', 'SSGC'],
                        help="photometric regions")

    parser.add_argument("--name_randoms", type=str, required=False, default='fastpm-randoms',
                        help='Each region is be saved in one dataset.')
    parser.add_argument("--name_contamination", type=str, required=False, default='fastpm-contamination',
                        help='Each region is be saved in one dataset.')

    parser.add_argument("--add_fake_stars", type=str, required=False, default='True',
                        help='If true add fake stars to mimick the DESI Target density')

    parser.add_argument("--seed", type=int, required=False, default=1325,
                        help='to fix reproducibility for the F.A.')

    return parser.parse_args()


if __name__ == '__main__':
    from fastpm.io import BigFile
    import fitsio
    from mpytools import Catalog
    from cosmoprimo.fiducial import DESI
    from mockfactory import setup_logging
    from mockfactory.desi import build_tiles_for_fa, read_sky_targets, apply_fiber_assignment, compute_completeness_weight
    from pypower import CatalogFFTPower

    # to remove pmesh warning
    import warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    os.environ['DESI_LOGLEVEL'] = 'ERROR'
    setup_logging()

    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    rank = mpicomm.Get_rank()
    start_ini = MPI.Wtime()
    logger_info(logger, 'Run F.A. script', rank)

    args = collect_argparser()
    sim = os.path.join(args.path_to_sim, args.sim)

    # Some parameter, for quick changes --> when the scprit is ready add it in the argparse !
    compute_power_wo_comp, compute_power, compute_wp, compute_wtheta = False, False, True, False
    use_sky_targets, preload_sky_targets = True, True
    add_fake_stars = args.add_fake_stars == 'True'
    nmock_ini, max_nmock = 0, 16
    fa_random_already_computed, fa_data_already_computed = True, False
    plot = False

    # Load DESI fiducial cosmo to convert redshift to distance (ok car univers plat):
    distance = DESI(engine='class').comoving_radial_distance

    # parameters to compute power spectrum
    kedges = np.arange(1e-3, 6e-1, 1e-3)
    logger_info(logger, "Power spectrum default arange: np.arange(1e-3, 6e-1, 1e-3)", rank)
    # LSS clustering catalog use these masbkits -> do the same
    maskbits = [1, 7, 8, 11, 12, 13]
    logger_info(logger, f"DR9 maskbits apply before power spectrum computation: {maskbits}", rank)

    # F.A. info:
    npasses = 7
    # Collect tiles from surveyops directory on which the fiber assignment will be applied
    tiles = build_tiles_for_fa(release_tile_path=f'/global/cfs/cdirs/desi/survey/catalogs/{args.release}/LSS/tiles-DARK.fits', program='dark', npasses=npasses)
    # Get info from origin fiberassign file and setup options for F.A. (see fiberassign.scripts.assign.parse_assign to modify margins, number of sky fibers for each petal ect...)
    ts = str(tiles['TILEID'][0]).zfill(6)
    fht = fitsio.read_header(f'/global/cfs/cdirs/desi/target/fiberassign/tiles/trunk/{ts[:3]}/fiberassign-{ts}.fits.gz')
    opts_for_fa = ["--target", " ", "--rundate", fht['RUNDATE'], "--mask_column", "DESI_TARGET"]
    # columns needed to run the F.A. and collect the info (They will be exchange between processes during the F.A.)
    columns_for_fa = ['RA', 'DEC', 'TARGETID', 'DESI_TARGET', 'SUBPRIORITY', 'OBSCONDITIONS', 'NUMOBS_MORE']
    # to speed up the process, we preload sky targets
    sky_targets = None
    if use_sky_targets and preload_sky_targets:
        # tiles is not restricted here, we will load sky_targets for all the Y1 footprint
        sky_targets = read_sky_targets(dirname='/global/cfs/cdirs/desi/users/edmondc/desi_targets/sky_targets_tmp/', filetype='bigfile', tiles=tiles, program='dark', mpicomm=mpicomm)

    # Let's go:
    for region in args.regions:
        # dataset in which the data will be writen
        dataset = f'{args.release}-{region}/'
        if (args.npasses is not None) and (args.npasses > 1): dataset = f'{args.release}-{region}-{args.npasses}pass/'

        start = MPI.Wtime()
        randoms = BigFile(os.path.join(args.path_to_sim, args.name_randoms), dataset=dataset, mode='r', mpicomm=mpicomm)
        randoms = Catalog(data={'RA': randoms.read('RA'), 'DEC': randoms.read('DEC'), 'DISTANCE': distance(randoms.read('Z')), 'MASKBITS': randoms.read('MASKBITS')})
        csize = randoms.csize
        logger_info(logger, f"Load Randoms: {csize} randoms read in {MPI.Wtime() - start:2.2f} s.", rank)

        start = MPI.Wtime()
        cutsky = BigFile(os.path.join(sim, f'desi-cutsky-{args.aout}'), dataset=dataset, mode='r', mpicomm=mpicomm)
        is_for_uncont = cutsky.read('IS_FOR_UNCONT')
        cutsky = Catalog(data={'RA': cutsky.read('RA'), 'DEC': cutsky.read('DEC'), 'DISTANCE': distance(cutsky.read('Z')), 'Z': cutsky.read('Z'),
                               'MASKBITS': cutsky.read('MASKBITS'), 'NMOCK': cutsky.read('NMOCK')})
        csize = cutsky.csize
        logger_info(logger, f"Number of galaxies: {csize} read in {MPI.Wtime() - start:2.2f} s.", rank)

        # Add F.A. columns: (reproductibility is fixed here)
        logger_info(logger, 'Add columns for F.A. to randoms and cutsky', rank)
        randoms = add_info_for_fa(randoms, seed=args.seed)
        cutsky = add_info_for_fa(cutsky, seed=args.seed + 34)

        # Apply F.A. for randoms
        if fa_random_already_computed:
            randoms['AVAILABLE'] = BigFile(os.path.join(args.path_to_sim, args.name_randoms), dataset=dataset, mode='r', mpicomm=mpicomm).read('AVAILABLE')
        else:
            # To speed up the process, we can reduce the randoms size (here randoms x 100 higher one subsample) (from small test, gain less than 1s. for exchange and 20 sec. for each pass)
            # randoms_tmp = {name: randoms[name][::10] for name in columns_for_fa + ['OBS_PASS']}
            apply_fiber_assignment(randoms, tiles, npasses, opts_for_fa, columns_for_fa, mpicomm, use_sky_targets=use_sky_targets, sky_targets=sky_targets)
            # save which randoms are available
            to_save = BigFile(os.path.join(args.path_to_sim, args.name_randoms), dataset=dataset, mode='rw', mpicomm=mpicomm)
            to_save.write({'AVAILABLE': randoms['AVAILABLE']})
        # keep all available randoms !
        sel_randoms = randoms['AVAILABLE']
        # remove also maskbits used in clustering catalog:
        for mskb in maskbits:
            sel_randoms &= (randoms['MASKBITS'] & 2**mskb) == 0

        # create array to save F.A. result:
        # numobs, comp_weight = np.zeros(cutsky.size), np.zeros(cutsky.size)

        if max_nmock is None:
            # Collect the number of subsample available in cutsky, take care all ranks do not have all the subsample, need to collect it across all the ranks!
            max_nmock = np.max(mpicomm.gather(np.max(cutsky['NMOCK'])))
            max_nmock = mpicomm.bcast(max_nmock, root=0)
        for num in range(nmock_ini, max_nmock):
            # extract info only for the submock with correct density (200 per deg^2)
            cutsky_tmp = cutsky[(cutsky['NMOCK'] == num) & is_for_uncont]

            if add_fake_stars:
                # Load contamination (fake stars) defining as: target density = 200 + stars. Contamination is the same for each subsample --> OK
                """ Here we neglect the imaging systeamtics:  Otherwise, when we emulate the systematic effect --> compute also the stellar contamnation ect ...---> NEXT STEP """
                suffix = '-with-stars'

                start = MPI.Wtime()
                stars = BigFile(os.path.join(args.path_to_sim, args.name_contamination), dataset=dataset, mode='r', mpicomm=mpicomm)
                stars = Catalog(data={'RA': stars.read('RA'), 'DEC': stars.read('DEC'), 'MASKBITS': stars.read('MASKBITS')})
                csize = stars.csize
                logger_info(logger, f"Number of contaminants: {csize} read in {MPI.Wtime() - start:2.2f} s.", rank)

                stars = add_info_for_fa(stars, offset=cutsky.csize, numobs=1, seed=args.seed + 2)

                # To concatenate cutsky_tmp and stars, stars need to have the same columns than cutsky_tmp
                # Add empty columns
                # Note: we are able to separe fake stars to cutsky with the TARGETID (and the offset used in add_info_for_fa)
                for name in cutsky_tmp.columns():
                    if not (name in stars.columns()):
                        stars[name] = stars.empty()
                targets = Catalog.concatenate(cutsky_tmp, stars)

                # Let's do the F.A.:
                apply_fiber_assignment(targets, tiles, npasses, opts_for_fa, columns_for_fa, mpicomm, use_sky_targets=use_sky_targets, sky_targets=sky_targets)

                # keep only good objects (ie) no stellar contaminant or objects unobserved (to compute correctly the comp_weight !)
                cutsky_tmp = targets[~((targets['TARGETID'] >= cutsky.csize) & (targets['NUMOBS'] > 0))]

            else:
                suffix = ''
                # Let's do the F.A.:
                apply_fiber_assignment(cutsky_tmp, tiles, npasses, opts_for_fa, columns_for_fa, mpicomm, use_sky_targets=use_sky_targets, sky_targets=sky_targets)

            # Now remove maskbits used in clustering catalog
            # Apply it before the completness weight computation ! (to avoid weighting targets with bad maskbits..)
            sel_good_maskbits = np.ones(cutsky_tmp.size, dtype='?')
            for mskb in maskbits:
                sel_good_maskbits &= (cutsky_tmp['MASKBITS'] & 2**mskb) == 0
            cutsky_tmp = cutsky_tmp[sel_good_maskbits]

            # Compute the completness weight:
            # Remark: the completness weight will not be perfect. Some good targets will not be observed and will be attached to stars or targets with bad maskbits ...
            # To minimize this effec consider only good targets when computing the completeness weight
            compute_completeness_weight(cutsky_tmp, tiles, npasses, mpicomm)

            # keep only targets observed
            sel_data = cutsky_tmp['NUMOBS'] > 0

            if compute_power_wo_comp:
                CatalogFFTPower(data_positions1=[cutsky_tmp['RA'][sel_data], cutsky_tmp['DEC'][sel_data], cutsky_tmp['DISTANCE'][sel_data]],
                                randoms_positions1=[randoms['RA'][sel_randoms], randoms['DEC'][sel_randoms], randoms['DISTANCE'][sel_randoms]],
                                position_type='rdd',
                                edges=kedges, ells=(0, 2, 4), cellsize=[6, 6, 6], boxsize=[8000, 16000, 8000],
                                resampler='tsc', interlacing=3, los='firstpoint',
                                mpicomm=mpicomm).poles.save(os.path.join(sim, f'power-spectrum/desi-cutsky-{dataset[:-1]}-{num}-with-fa-without-comp{suffix}.npy'))

            if compute_power:
                CatalogFFTPower(data_positions1=[cutsky_tmp['RA'][sel_data], cutsky_tmp['DEC'][sel_data], cutsky_tmp['DISTANCE'][sel_data]], data_weights1=cutsky_tmp['COMP_WEIGHT'][sel_data],
                                randoms_positions1=[randoms['RA'][sel_randoms], randoms['DEC'][sel_randoms], randoms['DISTANCE'][sel_randoms]],
                                position_type='rdd',
                                edges=kedges, ells=(0, 2, 4), cellsize=[6, 6, 6], boxsize=[8000, 16000, 8000],
                                resampler='tsc', interlacing=3, los='firstpoint',
                                mpicomm=mpicomm).poles.save(os.path.join(sim, f'power-spectrum/desi-cutsky-{dataset[:-1]}-{num}-with-fa{suffix}.npy'))

            if compute_wp:
                # Super useful to derive fs and Dfc parameters in the Hahn et. 2017 correction
                from pycorr import TwoPointCorrelationFunction
                edges = np.linspace(0., 2.5, 41)
                nsub = 5  # subsample randoms !

                # without F.A. : (need to remove maskbits)
                sel_randoms_tmp = np.ones(randoms.size, dtype='?')
                for mskb in maskbits:
                    sel_randoms_tmp &= (randoms['MASKBITS'] & 2**mskb) == 0
                # need to remove stellar contaminant (want to compute wp without f.a. but only for true objects):
                sel_data_tmp = cutsky_tmp['TARGETID'] < cutsky.csize
                TwoPointCorrelationFunction('rp', edges, position_type='rdd',
                                            data_positions1=[cutsky_tmp['RA'][sel_data_tmp], cutsky_tmp['DEC'][sel_data_tmp], cutsky_tmp['DISTANCE'][sel_data_tmp]],
                                            randoms_positions1=[randoms['RA'][sel_randoms_tmp][::nsub], randoms['DEC'][sel_randoms_tmp][::nsub], randoms['DISTANCE'][sel_randoms_tmp][::nsub]],
                                            engine='corrfunc', mpicomm=mpicomm).save(os.path.join(sim, f'correlation-function/rp-desi-cutsky-{dataset[:-1]}-{num}-without-fa{suffix}.npy'))

                TwoPointCorrelationFunction('rp', edges, position_type='rdd',
                                            data_positions1=[cutsky_tmp['RA'][sel_data], cutsky_tmp['DEC'][sel_data], cutsky_tmp['DISTANCE'][sel_data]], data_weights1=cutsky_tmp['COMP_WEIGHT'][sel_data],
                                            randoms_positions1=[randoms['RA'][sel_randoms][::nsub], randoms['DEC'][sel_randoms][::nsub], randoms['DISTANCE'][sel_randoms][::nsub]],
                                            engine='corrfunc', mpicomm=mpicomm).save(os.path.join(sim, f'correlation-function/rp-desi-cutsky-{dataset[:-1]}-{num}-with-fa{suffix}.npy'))

            if compute_wtheta:
                # Super useful to derive fs and Dfc parameters in the Hahn et. 2017 correction
                from pycorr import TwoPointCorrelationFunction
                # theta = rp / Xi(z=1.7) --> ca c'est en radians -> il faut donc multiplier par 180 / pi ... --> NE PAS L'OUBLIER WESH
                edges = np.linspace(5e-3, 1e-1, 101)
                nsub = 5  # subsample randoms !

                # without F.A. : (need to remove maskbits)
                sel_randoms_tmp = np.ones(randoms.size, dtype='?')
                for mskb in maskbits:
                    sel_randoms_tmp &= (randoms['MASKBITS'] & 2**mskb) == 0
                # need to remove stellar contaminant (want to compute wp without f.a. but only for true objects):
                sel_data_tmp = cutsky_tmp['TARGETID'] < cutsky.csize
                TwoPointCorrelationFunction('theta', edges, position_type='rdd',
                                            data_positions1=[cutsky_tmp['RA'][sel_data_tmp], cutsky_tmp['DEC'][sel_data_tmp], cutsky_tmp['DISTANCE'][sel_data_tmp]],
                                            randoms_positions1=[randoms['RA'][sel_randoms_tmp][::nsub], randoms['DEC'][sel_randoms_tmp][::nsub], randoms['DISTANCE'][sel_randoms_tmp][::nsub]],
                                            engine='corrfunc', mpicomm=mpicomm).save(os.path.join(sim, f'correlation-function/wtheta-desi-cutsky-{dataset[:-1]}-{num}-without-fa{suffix}.npy'))

                TwoPointCorrelationFunction('theta', edges, position_type='rdd',
                                            data_positions1=[cutsky_tmp['RA'][sel_data], cutsky_tmp['DEC'][sel_data], cutsky_tmp['DISTANCE'][sel_data]], data_weights1=cutsky_tmp['COMP_WEIGHT'][sel_data],
                                            randoms_positions1=[randoms['RA'][sel_randoms][::nsub], randoms['DEC'][sel_randoms][::nsub], randoms['DISTANCE'][sel_randoms][::nsub]],
                                            engine='corrfunc', mpicomm=mpicomm).save(os.path.join(sim, f'correlation-function/wtheta-desi-cutsky-{dataset[:-1]}-{num}-with-fa{suffix}.npy'))
            # Plot:
            if plot:
                ra, dec = cutsky_tmp.cget('RA', mpiroot=0), cutsky_tmp.cget('DEC', mpiroot=0)
                numobs, available, obs_pass = cutsky_tmp.cget('NUMOBS', mpiroot=0), cutsky_tmp.cget('AVAILABLE', mpiroot=0), cutsky_tmp.cget('OBS_PASS', mpiroot=0)

                if mpicomm.rank == 0:
                    import matplotlib.pyplot as plt
                    import desimodel.footprint

                    ra = ra - 120
                    ra[ra > 180] -= 360    # scale conversion to [-180, 180]
                    ra = -ra               # reverse the scale: East to the left

                    logger.info(f"Nbr of targets observed: {(numobs >= 1).sum()} -- per pass: {obs_pass.sum(axis=0)} -- Nbr of targets available: {available.sum()} -- Nbr of targets: {ra.size}")
                    logger.info(f"In percentage: Observed: {(numobs >= 1).sum()/ra.size:2.2%} -- Available: {available.sum()/ra.size:2.2%}")

                    tiles = tiles[tiles['PASS'] < npasses]
                    tile_id = np.unique(np.concatenate([tiles['TILEID'].values[np.array(idx, dtype='int64')] for idx in desimodel.footprint.find_tiles_over_point(tiles, ra, dec)]))

                    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

                    for id in tile_id:
                        tile = tiles[tiles['TILEID'] == id]
                        c = plt.Circle((tile['RA'].values[0], tile['DEC'].values[0]), np.sqrt(8 / 3.14), color='lightgrey', alpha=1)
                        ax.add_patch(c)

                    ax.scatter(ra, dec, c='red', s=0.3, label='targets')
                    ax.scatter(ra[available], dec[available], c='orange', s=0.3, label=f'available: {available.sum() / ra.size:2.2%}')

                    from matplotlib.axes._axes import _log as matplotlib_axes_logger
                    matplotlib_axes_logger.setLevel('ERROR')
                    colors = plt.cm.BuGn(np.linspace(0.6, 1, npasses))
                    for i in range(npasses):
                        ax.scatter(ra[obs_pass[:, i]], dec[obs_pass[:, i]], c=colors[i], s=0.3, label=f'Pass {i}: {obs_pass[:, i].sum() / ra.size:2.2%}')

                    ax.legend(loc='upper left')
                    ax.set_xlabel('R.A. [deg]')
                    ax.set_ylabel('Dec. [deg]')
                    ax.set_title(f'Fiber assignment for {args.release} release - {npasses} passes')
                    plt.tight_layout()
                    plt.savefig(f'fiberasignment_{npasses}npasses-{num}.png')
                    plt.close()
                    logger.info(f'Plot save in fiberasignment_{npasses}npasses-{num}.png')
