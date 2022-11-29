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
    from mpytools import Catalog
    from cosmoprimo.fiducial import DESI
    from mockfactory import setup_logging
    from pypower import CatalogSmoothWindow

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
    # LSS clustering catalog use these masbkits -> do the same
    maskbits = [1, 7, 8, 11, 12, 13]
    logger_info(logger, f"DR9 maskbits apply before power spectrum computation: {maskbits}", rank)

    # Let's go:
    for region in args.regions:
        # dataset in which the data will be writen
        dataset = f'{args.release}-{region}/'
        if (args.npasses is not None) and (args.npasses > 1): dataset = f'{args.release}-{region}-{args.npasses}pass/'

        start = MPI.Wtime()
        randoms = BigFile(os.path.join(args.path_to_sim, args.name_randoms), dataset=dataset, mode='r', mpicomm=mpicomm)
        randoms = Catalog(data={'RA': randoms.read('RA'), 'DEC': randoms.read('DEC'), 'DISTANCE': distance(randoms.read('Z')), 'MASKBITS': randoms.read('MASKBITS'), 'AVAILBLE': randoms.read('AVAILABLE')})
        csize = randoms.csize
        logger_info(logger, f"Load Randoms: {csize} randoms read in {MPI.Wtime() - start:2.2f} s.", rank)

        # keep all available randoms !
        sel_randoms_available = randoms['AVAILABLE']
        # remove also maskbits used in clustering catalog:
        sel_randoms = np.ones(randoms.size, dtype='?')
        for mskb in maskbits:
            sel_randoms &= (randoms['MASKBITS'] & 2**mskb) == 0





# Let us compute the window function multipoles in k-space
boxsize = 10000.
edges = {'step': 2. * np.pi / boxsize}
window_large = CatalogSmoothWindow(randoms_positions1=randoms_positions, randoms_weights1=randoms_weights,
                                   power_ref=poles, edges=edges, boxsize=boxsize, position_type='pos', dtype='f4').poles
# You can save the window function
with tempfile.TemporaryDirectory() as tmp_dir:
    fn = os.path.join(tmp_dir, 'tmp.npy')
    window_large.save(fn)
    window = PowerSpectrumSmoothWindow.load(fn)
    print(window.projs)

ax = plt.gca()
for iproj, proj in enumerate(window.projs):
    ax.plot(window.k, window(proj=proj, complex=False), label=proj.latex(inline=True))
ax.set_xscale('log')
ax.legend(loc=1)
ax.grid(True)
ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
ax.set_ylabel(r'$P(k)$ [$(\mathrm{Mpc}/h)^{3}$]')
plt.show()


# Let us just take a look at the window function in configuration space
# Normalization is not quite 1 as s -> 0
# Small s is sensitive to Nyquist frequency,
# while convergence en large s is obtained with very large boxes
sep = np.geomspace(1e-2, 4e3, 2048)
window_real = window.to_real(sep=sep)
window_real_large = window_large.to_real(sep=sep)
window_real_small = window_small.to_real(sep=sep)
ax = plt.gca()
plt.gcf().set_size_inches(8, 5)
ax.plot([], [], linestyle='--', color='k', label='small box')
ax.plot([], [], linestyle=':', color='k', label='large box')
ax.plot([], [], linestyle='-', color='k', label='combined box')
for iproj, proj in enumerate(window_real.projs):
    ax.plot(window_real_small.sep, window_real_small(proj=proj), color='C{:d}'.format(iproj), linestyle='--', label=None)
    ax.plot(window_real_large.sep, window_real_large(proj=proj), color='C{:d}'.format(iproj), linestyle=':', label=None)
    ax.plot(window_real.sep, window_real(proj=proj), linestyle='-', color='C{:d}'.format(iproj), label=proj.latex(inline=True))
ax.set_xlim(sep[0], 1e4)
ax.legend(loc=1)
ax.grid(True)
ax.set_xscale('log')
ax.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
ax.set_ylabel(r'$W_{\ell}^{(n)}(s)$')
plt.show()
