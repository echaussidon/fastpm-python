import logging
import numpy as np


logger = logging.getLogger('Utils')


def logger_info(logger, msg, rank, mpiroot=0):
    """ Print something with the logger only for the rank == mpiroot to avoid duplication of message. """
    if rank == mpiroot:
        logger.info(msg)


def load_fiducial_cosmo():
    """ Load fiducial DESI cosmology."""
    from cosmoprimo.fiducial import DESI
    # Load fiducial cosmology
    cosmo = DESI(engine='class')
    # precompute the bakcground
    _ = cosmo.get_background()
    return cosmo


def z_to_chi(cosmo):
    """ This is a callable which convert redshit into comoving distance at a given cosmology.
        The comoving distance is the radial comoving distance since we work in a flat Universe. """
    return cosmo.comoving_radial_distance


def chi_to_z(cosmo):
    """ This a callable which convert comoving distance into redshift at a given cosmology. """
    from mockfactory import DistanceToRedshift
    return DistanceToRedshift(cosmo.comoving_radial_distance)


def apply_hod(BoxCatalog, m_min=2.25e12):
    """ Put galaxies into the halos following the Halo Occupation Disitribution.
        For the moment we only use a simple mass step function, leading to p=1.
        See. Eftekharzadeh (2019) for more realitisic model (leading also to p=1
        since only depends on M) and its impact on the clustering at small scales. """

    def qso_hod_to_use(m, m_min=2.25e12, fn=1, Delta_m=0.75):
        """ Realistic model from Eftekharzadeh (2019). """
        return 1 / np.sqrt(2 * np.pi) / Delta_m * np.exp(-np.log(m / m_min)**2 / 2 / Delta_m**2)

    def qso_hod(m, m_min=2.25e12):
        """ Simple step function """
        return m > m_min

    return BoxCatalog[qso_hod(BoxCatalog['Mass'], m_min=m_min)]


def compute_sky_density(pixels, mpicomm, nside=256, in_deg2=True):
    """ todo. devrait aller dans mpytools ou mockfactory """
    import healpy as hp

    # collective unique and counts in order to compute the sky mean density of the simulation
    local_pixels, local_counts = np.unique(pixels, return_counts=True)

    # Send number of objects which will be send to the root rank
    nbr_obj_per_rank = mpicomm.gather(local_pixels.size, root=0)

    if mpicomm.Get_rank() != 0:  # .Send is a blocking communication
        mpicomm.Send(local_pixels, dest=0, tag=1)
        mpicomm.Send(local_counts, dest=0, tag=2)

        mean_sky_density = None

    if mpicomm.Get_rank() == 0:
        # collect and sum all the counts from each rank
        global_counts = np.zeros(hp.nside2npix(nside))

        # add rank 0
        global_counts[local_pixels] += local_counts

        # collect other rank
        for send_rank in range(1, mpicomm.size):
            local_pixels, local_counts = np.zeros(nbr_obj_per_rank[send_rank], dtype=np.int64), np.zeros(nbr_obj_per_rank[send_rank], dtype=np.int64)
            mpicomm.Recv(local_pixels, source=send_rank, tag=1)
            mpicomm.Recv(local_counts, source=send_rank, tag=2)
            global_counts[local_pixels] += local_counts

        if in_deg2:
            global_counts /= hp.nside2pixarea(nside, degrees=True)

        mean_sky_density = np.mean(global_counts[global_counts > 0])

    mean_sky_density = mpicomm.bcast(mean_sky_density)

    return mean_sky_density


def split_the_mock(pixels, mpicomm, expected_density=300, nside=256, seed=5162):
    """ Extract several mocks of exepected density from the same realisation (supposed with density higher than expected density)
        Ps: The estimation of the density is performed at nside=256. Dot not to apply DR9 mask ! :) (Otherwise need to use randoms :))
    """
    import mpytools as mpy

    mean_sky_density = compute_sky_density(pixels, mpicomm, nside=nside)

    # nombre de mocks que l'on peut faire
    nmocks = int(mean_sky_density / expected_density)
    logger_info(logger, f'mean density = {mean_sky_density:2.2f} -- expected density = {expected_density} -- Nmocks available = {nmocks} -- mock density = {mean_sky_density / nmocks:2.2f}', mpicomm.rank)

    # random generator invariant under MPI scaling
    rng = mpy.random.MPIRandomState(pixels.size, seed=seed)

    return rng.uniform(low=0, high=nmocks, dtype=int), nmocks


def extract_expected_density(pixels, mpicomm, expected_density=200, nside=256, seed=5162):
    """ Return mask to have cutsky with the expected sky density.
        Ps: The estimation of the density is performed at nside=256. Dot not to apply DR9 mask ! :) (Otherwise need to use randoms :))
    """
    import mpytools as mpy

    # compute the mean density
    mean_sky_density = compute_sky_density(pixels, mpicomm, nside=nside)

    # random generator invariant under MPI scaling
    rng = mpy.random.MPIRandomState(pixels.size, seed=seed)

    return rng.uniform(low=0, high=1, dtype='f8') <= expected_density / mean_sky_density
