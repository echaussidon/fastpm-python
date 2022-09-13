""" Ceci est la premiere version de mockfactor/desi/imaging_maskbits que arnaud a modifer.

    A conserver car toute la partie sur l'échange des particules est super interesante et ultra utile ! --> mpsort est aussi super malin

"""

import os
import logging

import numpy as np

from mpytools import CurrentMPIComm

from desiutil import brick
import fitsio
from astropy import wcs


logger = logging.getLogger('Maskbits')


def split_size_2d(s):
    """
    Split `s` into two integers, a, b such
    that a * b == s and a <= b
    Parameters
    -----------
    s : int
        integer to split
    Returns
    -------
    a, b: int
        integers such that a * b == s and a <= b
    """
    a = int(s ** 0.5) + 1
    while a > 1:
        if s % a == 0:
            s = s // a
            break
        a = a - 1
    b = s
    return a, b


def bitmask_radec(bitmask_fn, ra, dec):
    """ Extract bitmask associated to a (Ra, Dec) position from a legacy imaging brick"""
    if os.path.isfile(bitmask_fn):
        # read data and header:
        bitmask_img, header = fitsio.read(bitmask_fn, header=True)

        # convert ra, dec coordinates to brick coordinates:
        coadd_x, coadd_y = wcs.WCS(header).wcs_world2pix(ra, dec, 0)
        coadd_x, coadd_y = np.round(coadd_x).astype(int), np.round(coadd_y).astype(int)

        # extract mask information:
        bitmask = bitmask_img[coadd_y, coadd_x]

    else:
        # Sometimes we can have objects outside DR9 footprint:
        # Either because the footprint in regressis.footprint.DR9Footpint has a resolution of nside=256
        # Or because we do cut outside the footprint simply
        # Remove these objects setting bitmask 0 (NPRIMARY pixel)
        bitmask = 2**0 * np.ones(ra.size, dtype=np.int16)

    return bitmask


@CurrentMPIComm.enable
def add_dr9_maskbits(ra, dec, north_or_south='south', mpicomm=None):
    """ Starting from Rongpu Zhou code:
        https://github.com/rongpu/desi-examples/blob/master/bright_star_mask/read_pixel_bitmask.py
        This code used python multiprocessing and cannot achieve the job in large set of simulation used for DESI in small amount of times.
        Improve the code, increasing the I/O speed and modify it for MPI. """

    # if not empty, collect the information from brick
    if ra.size != 0:
        # collect birckid
        brickid = brick.Bricks(bricksize=0.25).brickid(ra, dec)

        # load bricks info
        bricks = fitsio.read('/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/survey-bricks.fits.gz', columns=['BRICKID', 'BRICKNAME'])

        # Just some tricks to speed up things up
        bid_unique, bidcnts = np.unique(brickid, return_counts=True)
        bidcnts = np.insert(bidcnts, 0, 0)
        bidcnts = np.cumsum(bidcnts)
        bidorder = np.argsort(brickid)

        nbr_bricks = len(bid_unique)

        idx, bitmask = np.array([]), np.array([])
        for bid_index in np.arange(nbr_bricks):
            # build brickname file in the correct field (north or south)
            brick_index = np.where(bricks['BRICKID'] == bid_unique[bid_index])[0][0]
            brickname = str(bricks['BRICKNAME'][brick_index])
            bitmask_fn = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/{}/coadd/{}/{}/legacysurvey-{}-maskbits.fits.fz'.format(north_or_south, brickname[:3], brickname, brickname)

            idx_tmp = bidorder[bidcnts[bid_index]: bidcnts[bid_index + 1]]

            bitmask_tmp = bitmask_radec(bitmask_fn, ra[idx_tmp], dec[idx_tmp])
            idx, bitmask = np.concatenate([idx, idx_tmp]), np.concatenate([bitmask, bitmask_tmp])
        maskbits = np.array(bitmask[np.argsort(idx)])

    # if empty, create empty array
    else:
        maskbits = np.array([])
        nbr_bricks = 0

    nbr_bricks = mpicomm.allgather(nbr_bricks)
    if mpicomm.Get_rank() == 0:
        logger.info(f'Number of bricks to open per rank: min={np.min(nbr_bricks)}, max={np.max(nbr_bricks)}, mean={np.mean(nbr_bricks)}')

    return maskbits


@CurrentMPIComm.enable
def reorganise_and_add_maskbits(source, periodic=False, use_ra_dec=False, wrap_ra=False, north_or_south='south', mpicomm=None):
    """ To speed the process (reading bricks is quite slow), we reorganise the objects in (RA, Dec) or in cartesian coordinates
        with GridND from pmesh.domain, instead of having them split across all the rank. Before reorganisation,
        We need to process the same number of bricks on each rank, now the number of bricks are split across the rank :)
        Parameters
        ----------
        source: Catalog containing 'RA', 'DEC' columns if you use use_ra_dec=True or 'Position' if you use use_ra_dec=False
    """
    from pmesh.domain import GridND

    nd = split_size_2d(mpicomm.size)  # we should have: np.prod(nd) = mpicomm.size

    if use_ra_dec:
        # to avoid problem with ra ~ 0 / ra ~ 360
        # otherwise it create a large zone without data ...
        ra, dec = source['RA'], source['DEC']
        if wrap_ra:
            ra[ra > 180] -= 360
        position = np.array((ra, dec)).T  # split the box into the sky
    else:
        position = source['Position'][:, :2]

    if periodic:
        boxsize = source.boxsize
        if np.isscalar(boxsize):
            boxsize = [boxsize, boxsize]
        left, right = [0, 0, 0], boxsize
    else:
        boxsize = None
        if position.size == 0:
            local_left, local_right = (np.nan, np.nan), (np.nan, -np.nan)
        else:
            local_left, local_right = position.min(axis=0), position.max(axis=0)
        left = np.nanmin(mpicomm.allgather(local_left), axis=0)
        right = np.nanmax(mpicomm.allgather(local_right), axis=0)

    grid = [np.linspace(lft, rht, n + 1, endpoint=True) for lft, rht, n in zip(left, right, nd)]
    domain = GridND(grid, comm=mpicomm, periodic=periodic)
    domain.loadbalance(domain.load(position))  # balance the load
    layout = domain.decompose(position, smoothing=0)

    # reorganise the data:
    ra = layout.exchange(source['RA'], pack=False)
    dec = layout.exchange(source['DEC'], pack=False)

    # collect the maskbits:
    maskbits = add_dr9_maskbits(ra, dec, north_or_south)

    # send to the initial rank and save it:
    maskbits = layout.gather(maskbits, mode='all', out=None)

    return maskbits


if __name__ == '__main__':
    """ This example can be run with srun -n 16 python imaging_maskbits.py. """

    from mockfactory import RandomCutskyCatalog, setup_logging

    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    rank = mpicomm.Get_rank()

    # to remove the following warning from pmesh (should be corrected in Arnaud branch)
    import warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    # Set up logging
    setup_logging()

    if rank == 0:
        logger.info('Run simple example to illustrate how to apply DR9 maskbits')

    cutsky = RandomCutskyCatalog(rarange=(20., 30.), decrange=(-0.5, 2.), size=10000, seed=44, mpicomm=mpicomm)
    start = MPI.Wtime()

    cutsky['MASKBITS'] = reorganise_and_add_maskbits(cutsky, periodic=False, use_ra_dec=True,
                                                     wrap_ra=False, north_or_south='south',
                                                     mpicomm=mpicomm)

    if mpicomm.rank == 0:
        logger.info(f'Apply DR9 maskbits done in {MPI.Wtime() - start:2.2f} s.')
