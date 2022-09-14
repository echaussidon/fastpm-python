import os
import sys
import logging
import argparse
import numpy as np
import mockfactory


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(mockfactory.__file__)), 'desi'))


logger = logging.getLogger('Make Survey')


# To avoid error from NUMEXPR Package
os.environ.setdefault('NUMEXPR_MAX_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))
os.environ.setdefault('NUMEXPR_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))


def collect_argparser():
    parser = argparse.ArgumentParser(description="Transform position in real space to redshift space and compute the multipoles.")

    parser.add_argument("--path_to_sim", type=str, required=False, default='/global/u2/e/edmondc/Scratch/Mocks/',
                        help="Path to the Scratch where the simulations are saved")
    parser.add_argument("--sim", type=str, required=False, default='test',
                        help="Simulation name (e.g) fastpm-fnl-0")
    parser.add_argument("--aout", type=str, required=False, default='1.0000',
                        help="scale factor at which the halos are saved (e.g) '0.3300' or '1.0000'")

    parser.add_argument("--zmin", type=float, required=False, default=0.8,
                        help="minimal redshift cut")
    parser.add_argument("--zmax", type=float, required=False, default=2.65,
                        help="maximum redshift cut. Reduce zmax to increase the sky coverage of the cutsky. 2.65 recovers all SNGC and SSGC and a large fraction of N. 3.5 recovers almost only SSGC.")
    parser.add_argument("--nz_filename", type=str, required=False, default='/global/homes/e/edmondc/Software/desi_ec/Data/nz_qso_final.dat',
                        help="where is saved the nz.dat file")
    parser.add_argument("--seed_data", type=int, required=False, default=28,
                        help="Set the seed for reproductibility when we match the n(z). \
                              seeds_data = {'N': args.seed_data, 'SNGC': args.seed_data + 1, 'SSGC': args.seed_data + 2}")

    parser.add_argument("--release", type=str, required=False, default='Y5',
                        help="match y1 / y5 footprint")
    parser.add_argument("--program", type=str, required=False, default='dark',
                        help="match bright / dark footprint")
    parser.add_argument("--npasses", type=int, required=False, default=None,
                        help="match footprint with more than npasses observation")
    parser.add_argument("--regions", nargs='+', type=str, required=False, default=['N', 'SNGC', 'SSGC'],
                        help="photometric regions")

    parser.add_argument("--maskbits", nargs='+', type=int, required=False, default=[1, 8, 9, 11, 12, 13],
                        help="DR9 maskbits used to cut the data and the randoms, default=[1, 8, 9, 11, 12, 13]")

    parser.add_argument("--expected_density", type=float, required=False, default=280,
                        help='Expected mock density in deg^-2 before systematic contamination. Typically, 1.4*n_obs is enough.')

    parser.add_argument("--generate_randoms", type=str, required=False, default='False',
                        help="if 'True' generate associated randoms for the expected density cutsky.")
    parser.add_argument("--seed_randoms", type=int, required=False, default=36,
                        help="Set the seed for reproductibility when we generate associated randoms \
                              seeds_randoms = {'N': args.seed_randoms, 'SNGC': args.seed_random + 1, 'SSGC': args.seed_random + 2}")
    parser.add_argument("--name_randoms", type=str, required=False, default='fastpm-randoms',
                        help='the randoms will be saved in os.path.join(args.path_to_sim, args.name_randoms). Each region will be saved in one dataset.')

    return parser.parse_args()


if __name__ == '__main__':
    """ This file is design to be launched with survey.sh"""
    from fastpm.io import BigFile

    from mockfactory import BoxCatalog, setup_logging
    from mockfactory.desi import get_brick_pixel_quantities

    from from_box_to_desi_cutsky import remap_the_box, apply_rsd_and_cutsky, \
        apply_radial_mask, generate_redshifts, photometric_region_center, \
        match_photo_desi_footprint

    from utils import logger_info, load_fiducial_cosmo, z_to_chi, chi_to_z, \
        apply_hod, split_the_mock

    setup_logging()

    # to remove the following warning from pmesh (arnaud l'a corrigé sur son github mais ok)
    import warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    rank = mpicomm.Get_rank()
    start_ini = MPI.Wtime()

    # collect args
    args = collect_argparser()
    sim = os.path.join(args.path_to_sim, args.sim)

    # load DESI fiducial cosmology and redshift-distance translator
    cosmo = load_fiducial_cosmo()
    z2chi, chi2z = z_to_chi(cosmo), chi_to_z(cosmo)

    # Fix seed on each photometric region for reproductibility
    seeds_data = {region: args.seed_data + i for i, region in enumerate(['N', 'SNGC', 'SSGC'])}
    seeds_randoms = {region: args.seed_randoms + i for i, region in enumerate(['N', 'SNGC', 'SSGC'])}

    # Load halos catalog and build Boxcatalog:
    start = MPI.Wtime()
    halos = BigFile(os.path.join(sim, f'halos-{args.aout}'), dataset='1/', mode='r', mpicomm=mpicomm)
    halos.Position = halos.read('Position')
    halos.Velocity = halos.read('Velocity')
    halos.Mass = halos.read('Mass')
    rsd_factor = halos.attrs['RSDFactor'][0]
    # build boxcatalog
    box = BoxCatalog(data=halos, columns=['Position', 'Velocity', 'Mass'], boxsize=halos.attrs['boxsize'][0], boxcenter=halos.attrs['boxsize'][0] // 2, mpicomm=mpicomm)
    # recenter the box to make rotation easier
    box.recenter()
    logger_info(logger, f"Number of halos: {halos.csize} read in {MPI.Wtime() - start:2.2f} s.", rank)

    # Apply HOD:
    start = MPI.Wtime()
    box = apply_hod(box)
    logger_info(logger, f"Apply HOD done in {MPI.Wtime() - start:2.2f} s.", rank)

    # Remap + cutsky + RSD + radial mask
    start = MPI.Wtime()
    box = remap_the_box(box)  # to increase the sky area
    logger_info(logger, f"Remapping done in {MPI.Wtime() - start:2.2f} s.", rank)

    # With the SAME realisation match North, Decalz_north and Decalz_south photometric region
    # The three regions are not idependent! Useful to test the geometrical / imaging systematic effects on each region.
    for region in args.regions:
        start = MPI.Wtime()

        # rotation of the box to match as best as possible each region
        add_ra, add_dec = photometric_region_center(region)
        logger_info(logger, f'Rotation to match region: {region} in release: {args.release} with add_ra: {add_ra} and add_dec: {add_dec}', rank)

        # create the cutsky
        cutsky = apply_rsd_and_cutsky(box, z2chi(args.zmin), z2chi(args.zmax), rsd_factor, add_ra=add_ra, add_dec=add_dec)
        # convert distance to redshift
        cutsky['Z'] = chi2z(cutsky['DISTANCE'])
        # match the nz distribution
        cutsky = apply_radial_mask(cutsky, args.zmin, args.zmax, nz_filename=args.nz_filename, cosmo=cosmo, seed=seeds_data[region])
        logger_info(logger, f"Remap + cutsky + RSD + radial mask done in {MPI.Wtime() - start:2.2f} s.", rank)

        # match the desi footprint and apply the DR9 mask:
        start = MPI.Wtime()
        desi_cutsky = match_photo_desi_footprint(cutsky, region, args.release, args.program, npasses=args.npasses, rank=rank)
        # add DR9 maskbits
        add_brick_quantities = {'maskbits': {'fn': '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/{region}/coadd/{brickname:.3s}/{brickname}/legacysurvey-{brickname}-maskbits.fits.fz', 'dtype': 'i2', 'default': 1}}
        desi_cutsky['MASKBITS'] = get_brick_pixel_quantities(desi_cutsky['RA'], desi_cutsky['DEC'], add_brick_quantities, mpicomm=mpicomm)['maskbits']
        # keep only objects without maskbits
        sel = np.ones(desi_cutsky.size, dtype=bool)
        for mskb in args.maskbits:
            sel &= (desi_cutsky['MASKBITS'] & 2**mskb) == 0
        desi_cutsky = desi_cutsky[sel]
        logger_info(logger, f"Match region: {region} and release footprint: {args.release}-{args.program}-{args.npasses} + apply DR9 maskbits: {args.maskbits} done in {MPI.Wtime() - start:2.2f} s.", rank)

        # split the mock into 'small mock' and add flags for imaging systematics
        start = MPI.Wtime()
        desi_cutsky['NMOCK'], nmocks = split_the_mock(desi_cutsky, mpicomm, expected_density=args.expected_density, seed=seeds_data[region] * 3)
        logger_info(logger, f"Split the simulation in {MPI.Wtime() - start:2.2f} s.", rank)

        # save desi_cutsky into the same bigfile --> N / SNGC / SSGC
        start = MPI.Wtime()
        mock = BigFile(os.path.join(sim, f'desi-cutsky-{args.aout}'), dataset=args.release + '-' + region + '/', mode='w', mpicomm=mpicomm)
        mock.attrs = halos.attrs
        mock.write({'RA': desi_cutsky['RA'], 'DEC': desi_cutsky['DEC'], 'Z': desi_cutsky['Z'],
                    'MASKBITS': desi_cutsky['MASKBITS'], 'NMOCK': desi_cutsky['NMOCK'],
                    'DISTANCE': desi_cutsky['DISTANCE'], 'HPX': desi_cutsky['HPX']})
        logger_info(logger, f"Save done in {MPI.Wtime() - start:2.2f} s.\n", rank)

        if args.generate_randoms == 'True':
            # generate associated randoms:
            from mockfactory import RandomCutskyCatalog, box_to_cutsky

            # We want 10 times more than the cutsky mock
            nrand_over_data = 10
            # Since random are generated not directly on DESI footprint, we take the size of cutsky and not desi_cutsky
            nbr_randoms = int(cutsky.csize * nrand_over_data + 0.5)
            # collect limit for the cone
            _, rarange, decrange = box_to_cutsky(box.boxsize, z2chi(args.zmax), dmin=z2chi(args.zmin))
            logger_info(logger, f'Generate randoms for region={region} with seed={seeds_randoms[region]}', rank)

            # Generate randoms with in the cutsky
            start = MPI.Wtime()
            randoms = RandomCutskyCatalog(rarange=add_ra + np.array(rarange), decrange=add_dec + np.array(decrange), csize=nbr_randoms, seed=seeds_randoms[region], mpicomm=mpicomm)
            logger_info(logger, f"RandomCutsky done in {MPI.Wtime() - start:2.2f} s.", rank)

            # match the desi footprint and apply the DR9 mask
            start = MPI.Wtime()
            randoms = match_photo_desi_footprint(randoms, region, args.release, args.program, npasses=args.npasses, rank=rank)
            randoms['MASKBITS'] = get_brick_pixel_quantities(randoms['RA'], randoms['DEC'], add_brick_quantities, mpicomm=mpicomm)['maskbits']
            # keep only objects without maskbits
            sel = np.ones(randoms.size, dtype=bool)
            for mskb in args.maskbits:
                sel &= (randoms['MASKBITS'] & 2**mskb) == 0
            randoms = randoms[sel]
            logger_info(logger, f"Match region: {region} and release footprint: {args.release}-{args.program}-{args.npasses} + apply DR9 maskbits: {args.maskbits} done in {MPI.Wtime() - start:2.2f} s.", rank)

            # use the naive implementation of mockfactory/make_survey/BaseRadialMask
            # draw numbers according to a uniform law until to find enough correct numbers
            # basically, this is the so-called 'methode du rejet'
            randoms['Z'] = generate_redshifts(randoms.size, args.zmin, args.zmax, nz_filename=args.nz_filename, cosmo=cosmo, seed=seeds_randoms[region] * 276)

            # save randoms
            start = MPI.Wtime()
            generated_randoms = BigFile(os.path.join(args.path_to_sim, args.name_randoms), dataset=args.release + '-' + region + '/', mode='w', mpicomm=mpicomm)
            generated_randoms.write({'RA': randoms['RA'], 'DEC': randoms['DEC'], 'Z': randoms['Z'],
                                    'MASKBITS': randoms['MASKBITS'], 'HPX': randoms['HPX']})
            logger_info(logger, f"Save done in {MPI.Wtime() - start:2.2f} s.\n", rank)

    logger_info(logger, f"Make survey took {MPI.Wtime() - start_ini:2.2f} s.", rank)
