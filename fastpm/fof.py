"""
Credit: This code is copy from https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/fof.py.
        I only remove useless (for me) part of the code and do some adaptations.
"""

from __future__ import print_function

import numpy as np
import logging
from mpi4py import MPI

from .utils import split_size_3d
from .utils import DistributedArray


class FOF(object):
    """
    A friends-of-friends halo finder that computes the label for
    each particle, denoting which halo it belongs to.

    Friends-of-friends was first used by Davis et al 1985 to define
    halos in hierachical structure formation of cosmological simulations.
    The algorithm is also known as DBSCAN in computer science.
    The subroutine here implements a parallel version of the FOF.

    The underlying local FOF algorithm is from :mod:`kdcount.cluster`,
    which is an adaptation of the implementation in Volker Springel's
    Gadget and Martin White's PM.

    Results are computed when the object is inititalized. See the documenation
    of :func:`~FOF.run` for the attributes storing the results.

    For returning a CatalogSource of the FOF halos, see :func:`find_features`
    and for computing a halo catalog with added analytic information for
    a specific redshift and cosmology, see :func:`to_halos`.

    Parameters
    ----------
    source : CatalogSource
        the source to run the FOF algorithm on; must support 'Position'
    linking_length : float
        the linking length, either in absolute units, or relative
        to the mean particle separation
    nmin : int
        halo with fewer particles are ignored
    absolute : bool, optional
        If `True`, the linking length is in absolute units, otherwise it is
        relative to the mean particle separation; default is `False`
    """
    logger = logging.getLogger('FOF')

    def __init__(self, source, linking_length, nmin, absolute=False, periodic=True, domain_factor=1, memory_monitor=None):

        self.comm = source.comm
        self._source = source
        self.memory_monitor = memory_monitor

        if 'Position' not in source:
            raise ValueError("cannot compute FOF without 'Position' column")

        self.attrs = {}
        self.attrs['linking_length'] = linking_length
        self.attrs['nmin'] = nmin
        self.attrs['absolute'] = absolute
        self.attrs['periodic'] = periodic
        self.attrs['domain_factor'] = domain_factor

        if periodic and 'BoxSize' not in source.attrs:
            raise ValueError("Periodic FOF requires BoxSize in .attrs['BoxSize']")

        # linking length relative to mean separation
        if not absolute:
            if 'Nmesh' in source.attrs:
                ndim = len(source.attrs['Nmesh'])
            elif 'Position' in source:
                ndim = source['Position'].shape[1]
            else:
                raise AttributeError('Missing attributes to infer the dimension')
            mean_separation = pow(np.prod(source.attrs['BoxSize']) / source.csize, 1.0 / ndim)
            linking_length *= mean_separation
        self._linking_length = linking_length

        # and run
        self.run()

    def run(self):
        """
        Run the FOF algorithm. This function returns nothing, but does
        attach several attributes to the class instance:

        - attr:`labels`
        - :attr:`max_labels`

        .. note::
            The :attr:`labels` array is scattered evenly across all ranks.

        Attributes
        ----------
        labels : array_like, length: :attr:`size`
            an array the label that specifies which FOF halo each particle
            belongs to
        max_label : int
            the maximum label across all ranks; this represents the total
            number of FOF halos found
        """
        # run the FOF
        minid = fof(self._source, self._linking_length, self.comm, self.attrs['periodic'], self.attrs['domain_factor'], self.logger, memory_monitor=self.memory_monitor)

        # the sorted labels
        self.labels = _assign_labels(minid, comm=self.comm, thresh=self.attrs['nmin'])
        self.max_label = self.comm.allgather(self.labels.max())

    def find_features(self, peakcolumn=None):
        """
        Based on the particle labels, identify the groups, and return
        the center-of-mass ``CMPosition``, ``CMVelocity``, and Length of each
        feature.

        If a ``peakcolumn`` is given, the ``PeakPosition`` and ``PeakVelocity``
        is also calculated for the particle at the peak value of the column.

        Data is scattered evenly across all ranks.

        Returns
        -------
        :class: like `~nbodykit.source.catalog.array.ScatterArray` :
            a source holding the ('CMPosition', 'CMVelocity', 'Length')
            of each feature; optionaly, ``PeakPosition``, ``PeakVelocity`` are
            also included if ``peakcolumn`` is not None

        attrs
        """
        # the center-of-mass (Position, Velocity, Length)
        halos = fof_catalog(self._source, self.labels, self.comm, peakcolumn=peakcolumn, periodic=self.attrs['periodic'], memory_monitor=self.memory_monitor)
        attrs = self._source.attrs.copy()
        attrs.update(self.attrs)
        return halos, attrs


def _assign_labels(minid, comm, thresh):
    """
    Convert minid to sequential labels starting from 0.

    This routine is used to assign halo label to particles with
    the same minid.
    Halos with less than thresh particles are reclassified to 0.

    Parameters
    ----------
    minid : array_like, ('i8')
        The minimum particle id of the halo. All particles of a halo
        have the same minid
    comm : py:class:`MPI.Comm`
        communicator. since this is a collective operation
    thresh : int
        halo with less than thresh particles are merged into halo 0

    Returns
    -------
    labels : array_like ('i8')
        The new labels of particles. Note that this is ordered
        by the size of halo, with the exception 0 represents all
        particles that are in halos that contain less than thresh particles.

    """
    from mpi4py import MPI

    dtype = np.dtype([('origind', 'u8'), ('fofid', 'u8'), ])
    data = np.empty(len(minid), dtype=dtype)
    # assign origind for recovery of ordering, since
    # we need to work in sorted fofid
    data['fofid'] = minid
    data['origind'] = np.arange(len(data), dtype='u4')
    data['origind'] += np.sum(comm.allgather(len(data))[:comm.rank], dtype='intp') \

    data = DistributedArray(data, comm)

    # first attempt is to assign fofid for each group
    data.sort('fofid')
    label = data['fofid'].unique_labels()

    N = label.bincount()

    # now eliminate those with less than thresh particles
    small = N.local <= thresh

    Nlocal = label.bincount(local=True)
    # mask == True for particles in small halos
    mask = np.repeat(small, Nlocal)

    # globally shift halo id by one
    label.local += 1
    label.local[mask] = 0

    data['fofid'].local[:] = label.local[:]
    del label

    data.sort('fofid')

    data['fofid'].local[:] = data['fofid'].unique_labels().local[:]

    # unique_labels may miss the 0 index representing disconnected
    # particles if there are no such particles.
    # shift the fofoid by 1 in that case.
    anysmall = comm.allreduce(small.sum()) != 0
    if not anysmall:
        data['fofid'].local[:] += 1

    data.sort('origind')

    label = data['fofid'].local.view('i8').copy()

    del data

    Nhalo0 = max(comm.allgather(label.max())) + 1
    Nlocal = np.bincount(label, minlength=Nhalo0)
    comm.Allreduce(MPI.IN_PLACE, Nlocal, op=MPI.SUM)

    # sort the labels by halo size
    arg = Nlocal[1:].argsort()[::-1] + 1
    if Nhalo0 > 2**31:
        dtype = 'i8'
    else:
        dtype = 'i4'
    P = np.arange(Nhalo0, dtype=dtype)
    P[arg] = np.arange(len(arg), dtype=dtype) + 1
    label = P[label]
    return label


def _fof_local(layout, pos, boxsize, ll, comm):
    from kdcount import cluster

    N = len(pos)

    pos = layout.exchange(pos)
    if boxsize is not None:
        pos %= boxsize
    data = cluster.dataset(pos, boxsize=boxsize)
    fof = cluster.fof(data, linking_length=ll, np=0)
    labels = fof.labels
    del fof

    PID = np.arange(N, dtype='intp')
    PID += np.sum(comm.allgather(N)[:comm.rank], dtype='intp')

    PID = layout.exchange(PID)
    # initialize global labels
    minid, _, _ = equiv_class(labels, PID, op=np.fmin)
    minid = minid[labels]

    return minid


def _fof_merge(layout, minid, comm):
    # generate global halo id

    while True:
        # merge, if a particle belongs to several ranks
        # use the global label of the minimal
        minid_new = layout.gather(minid, mode=np.fmin)
        minid_new = layout.exchange(minid_new)

        # on my rank, these particles have been merged
        merged = minid_new != minid
        # if no rank has merged any, we are done
        # gl is the global label (albeit with some holes)
        total = comm.allreduce(merged.sum())

        if total == 0:
            del minid_new
            break
        old = minid[merged]
        new = minid_new[merged]
        arg = old.argsort()
        new = new[arg]
        old = old[arg]
        replacesorted(minid, old, new, out=minid)

    minid = layout.gather(minid, mode=np.fmin)
    return minid


def fof(source, linking_length, comm, periodic, domain_factor, logger, memory_monitor=None):
    """
    Run Friends-of-friends halo finder.

    Friends-of-friends was first used by Davis et al 1985 to define
    halos in hierachical structure formation of cosmological simulations.
    The algorithm is also known as DBSCAN in computer science.
    The subroutine here implements a parallel version of the FOF.

    The underlying local FOF algorithm is from `kdcount.cluster`,
    which is an adaptation of the implementation in Volker Springel's
    Gadget and Martin White's PM. It could have been done faster.

    Parameters
    ----------
    source: CatalogSource
        the input source of particles; must support 'Position' column;
        ``source.attrs['BoxSize']`` is also used
    linking_length: float
        linking length in data units. (Usually Mpc/h).
    comm: MPI.Comm
        The mpi communicator.

    Returns
    -------
    minid: array_like
        A unique label of each position. The label is not ranged from 0.
    """
    from pmesh.domain import GridND

    if memory_monitor is not None: memory_monitor()

    n_p = split_size_3d(comm.size)
    nd = n_p * domain_factor

    if periodic:
        BoxSize = source.attrs.get('BoxSize', None)
        if BoxSize is None:
            raise ValueError("cannot compute FOF clustering of source without 'BoxSize' in ``attrs`` dict")
        if np.isscalar(BoxSize):
            BoxSize = [BoxSize, BoxSize, BoxSize]

        left = [0, 0, 0]
        right = BoxSize
    else:
        BoxSize = None
        left = np.min(comm.allgather(source['Position'].min(axis=0).compute()), axis=0)
        right = np.max(comm.allgather(source['Position'].max(axis=0).compute()), axis=0)

    grid = [
        np.linspace(left[0], right[0], nd[0] + 1, endpoint=True),
        np.linspace(left[1], right[1], nd[1] + 1, endpoint=True),
        np.linspace(left[2], right[2], nd[2] + 1, endpoint=True),
    ]
    domain = GridND(grid, comm=comm, periodic=periodic)

    if memory_monitor is not None: memory_monitor()

    Position = source.compute(source['Position'])
    n_p = comm.allgather(len(Position))
    if comm.rank == 0:
        logger.info("Number of particles max/min = %d / %d before spatial decomposition" % (max(n_p), min(n_p)))

    if memory_monitor is not None: memory_monitor()

    # balance the load
    domain.loadbalance(domain.load(Position))

    layout = domain.decompose(Position, smoothing=linking_length * 1)

    n_p = comm.allgather(layout.recvlength)
    if comm.rank == 0:
        logger.info("Number of particles max/min = %d / %d after spatial decomposition" % (max(n_p), min(n_p)))

    if memory_monitor is not None: memory_monitor()

    comm.barrier()
    minid = _fof_local(layout, Position, BoxSize, linking_length, comm)

    if memory_monitor is not None: memory_monitor()

    comm.barrier()
    minid = _fof_merge(layout, minid, comm)

    if memory_monitor is not None: memory_monitor()

    return minid


def fof_catalog(source, label, comm, position='Position', velocity='Velocity', initposition='InitialPosition', peakcolumn=None, periodic=True, memory_monitor=None):
    """
    Catalog of FOF groups based on label from a parent source

    This is a collective operation -- the returned halo catalog will be
    equally distributed across all ranks

    Notes
    -----
    This computes the center-of-mass position and velocity in the same
    units as the corresponding columns ``source``

    Parameters
    ----------
    source: CatalogSource
        the parent source of particles from which the center-of-mass
        position and velocity are computed for each halo
    label : array_like
        the label for each particle that identifies which halo it
        belongs to
    comm: MPI.Comm
        the mpi communicator. Must agree with the datasource
    position : str, optional
        the column name specifying the position
    velocity : str, optional
        the column name specifying the velocity
    initposition : str, optional
        the column name specifying the initial position; this is only
        computed if available
    peakcolumn : str, optional
        if not None, find PeakPostion and PeakVelocity based on the
        value of peakcolumn

    Returns
    -------
    catalog: array_like
        A 1-d array of type 'Position', 'Velocity', 'Length'.
        The center mass position and velocity of the FOF halo, and
        Length is the number of particles in a halo. The catalog is
        sorted such that the most massive halo is first. ``catalog[0]``
        does not correspond to any halo.
    """
    from .utils import ScatterArray

    if memory_monitor is not None: memory_monitor()

    # make sure all of the columns are there
    for col in [position, velocity]:
        if col not in source:
            raise ValueError(f"the column '{col}' is missing from parent source; cannot compute halos")

    dtype = [('CMPosition', ('f4', 3)), ('CMVelocity', ('f4', 3)), ('Length', 'i4')]
    N, nbr_halos = count(label, comm=comm)

    if memory_monitor is not None: memory_monitor()

    if periodic:
        # make sure BoxSize is there
        boxsize = source.attrs.get('BoxSize', None)
        if boxsize is None:
            raise ValueError("cannot compute halo catalog from source without 'BoxSize' in ``attrs`` dict")
    else:
        boxsize = None

    # center of mass position
    hpos = centerofmass(label, source.compute(source[position]), boxsize=boxsize, comm=comm)

    if memory_monitor is not None: memory_monitor()

    # center of mass velocity
    hvel = centerofmass(label, source.compute(source[velocity]), boxsize=None, comm=comm)

    if memory_monitor is not None: memory_monitor()

    # center of mass initial position
    if initposition in source:
        dtype.append(('InitialPosition', ('f4', 3)))
        hpos_init = centerofmass(label, source.compute(source[initposition]), boxsize=boxsize, comm=comm)

    if memory_monitor is not None: memory_monitor()

    if peakcolumn is not None:
        assert peakcolumn in source

        dtype.append(('PeakPosition', ('f4', 3)))
        dtype.append(('PeakVelocity', ('f4', 3)))

        density = source[peakcolumn].compute()
        dmax, unique_label, label_inverse = equiv_class(label, density, op=np.fmax)
        _, dmax_minimal = reduce_sparse(nbr_halos, unique_label, dmax, np.fmax, -np.inf, comm, send_result=True)

        # remove any non-peak particle from the labels
        label1 = label * (density >= dmax_minimal[label_inverse])

        # compute the center of mass on the new labels
        ppos = centerofmass(label1, source.compute(source[position]), boxsize=boxsize, comm=comm)
        pvel = centerofmass(label1, source.compute(source[velocity]), boxsize=None, comm=comm)

        if memory_monitor is not None: memory_monitor()

    dtype = np.dtype(dtype)
    if comm.Get_rank() == 0:
        catalog = np.empty(shape=nbr_halos, dtype=dtype)

        catalog['CMPosition'] = hpos
        catalog['CMVelocity'] = hvel
        catalog['Length'] = N
        catalog['Length'][0] = 0  # label == 0 --> particles without halos
        if 'InitialPosition' in dtype.names:
            catalog['InitialPosition'] = hpos_init

        if peakcolumn is not None:
            catalog['PeakPosition'] = ppos
            catalog['PeakVelocity'] = pvel
    else:
        catalog = None

    if memory_monitor is not None: memory_monitor()

    return ScatterArray(catalog, comm, root=0)


# -----------------------
# Helpers
# -----------------------
def equiv_class(labels, values, op, dense_labels=False, identity=None):
    """
    apply operation to equivalent classes by label, on values

    **MODIFICATION** remove dense argument, work with sparse idea

    Parameters
    ----------
    labels : array_like
        the label of objects, starting from 0.
    values : array_like
        the values of objects (len(labels), ...)
    op : :py:class:`np.ufunc`
        the operation to apply
    # dense_labels : boolean
    #     If the labels are already dense (from 0 to Nobjects - 1)
    #     If False, :py:meth:`np.unique` is used to convert
    #     the labels internally

    Returns
    -------
    result :
        the value of each equivalent class

    Examples
    --------
    >>> x = np.arange(10)
    >>> print equiv_class(x, x, np.fmin) #, dense_labels=True)
    [0 1 2 3 4 5 6 7 8 9]

    >>> x = np.arange(10)
    >>> v = np.arange(20).reshape(10, 2)
    >>> x[1] = 0
    >>> print equiv_class(x, 1.0 * v, np.fmin, identity=np.inf) #, dense_labels=True)
    [[  0.   1.]
     [ inf  inf]
     [  4.   5.]
     [  6.   7.]
     [  8.   9.]
     [ 10.  11.]
     [ 12.  13.]
     [ 14.  15.]
     [ 16.  17.]
     [ 18.  19.]]

    """
    unique_labels, labels, N = np.unique(labels, return_inverse=True, return_counts=True)

    offsets = np.concatenate([[0], N.cumsum()], axis=0)[:-1]
    arg = labels.argsort()
    if identity is None:
        identity = op.identity

    result = op.reduceat(values[arg], offsets)

    if (N == 0).any():
        result[N == 0] = identity

    return result, unique_labels, labels


def replacesorted(arr, sorted, b, out=None):
    """
    replace a with corresponding b in arr

    Parameters
    ----------
    arr : array_like
        input array
    sorted   : array_like
        sorted

    b   : array_like

    out : array_like,
        output array
    Result
    ------
    newarr  : array_like
        arr with a replaced by corresponding b

    Examples
    --------
    >>> print replacesorted(np.arange(10), np.arange(5), np.ones(5))
    [1 1 1 1 1 5 6 7 8 9]

    """
    if out is None:
        out = arr.copy()
    if len(sorted) == 0:
        return out
    ind = sorted.searchsorted(arr)
    ind.clip(0, len(sorted) - 1, out=ind)
    arr = np.array(arr)
    found = sorted[ind] == arr
    out[found] = b[ind[found]]
    return out


def centerofmass(label, pos, boxsize, comm=MPI.COMM_WORLD):
    """
    Calulate the center of mass of particles of the same label.

    The center of mass is defined as the mean of positions of particles,
    but care has to be taken regarding to the periodic boundary.

    This is a collective operation, and after the call, all ranks
    will have the position of halos.

    --> VERY BAD IDEA ...

    --> Now the result is only in rank = 0 and we never create an array of halos size in all the process

    Parameters
    ----------
    label : array_like (integers)
        Halo label of particles, >=0
    pos   : array_like (float, 3)
        position of particles.
    boxsize : float or None
        size of the periodic box, or None if no periodic boundary is assumed.
    comm : :py:class:`MPI.Comm`
        communicator for the collective operation.

    Returns
    -------
    hpos : array_like (float, 3)
        the center of mass position of the halos.

    """
    N, nbr_halos = count(label, comm=comm)

    if boxsize is not None:
        posmin, unique_label, label_inverse = equiv_class(label, pos, op=np.fmin)
        posmin, posmin_minimal = reduce_sparse(nbr_halos, unique_label, posmin, np.fmin, np.inf, comm, send_result=True)
        dpos = pos - posmin_minimal[label_inverse]

        for i in range(dpos.shape[-1]):
            bhalf = boxsize[i] * 0.5
            dpos[..., i][dpos[..., i] < -bhalf] += boxsize[i]
            dpos[..., i][dpos[..., i] >= bhalf] -= boxsize[i]
    else:
        dpos = pos
    dpos, unique_label, _ = equiv_class(label, dpos, op=np.add)
    dpos = reduce_sparse(nbr_halos, unique_label, dpos, np.add, 0, comm, send_result=False)

    if comm.Get_rank() == 0:
        dpos /= N[:, None]

        if boxsize is not None:
            hpos = posmin + dpos
            hpos %= boxsize
        else:
            hpos = dpos
    else:
        hpos = None
    return hpos


def count(label, comm=MPI.COMM_WORLD):
    """
    Count the number of particles of the same label.

    This is a collective operation, and after the call, all ranks
    will have the particle count.

    Parameters
    ----------
    label : array_like (integers)
        Halo label of particles, >=0
    comm : :py:class:`MPI.Comm`
        communicator for the collective operation.

    Returns
    -------
    count : array_like
        the count of number of particles in each halo only for rank 0 to save memory
    Nhalos0 : int
        Number of halos

    """
    Nhalo0 = max(comm.allgather(label.max())) + 1

    # count the number of particle inside the same halos
    unique_label, counts = np.unique(label, return_counts=True)

    N = reduce_sparse(Nhalo0, unique_label, counts, np.add, 0, comm, send_result=False)

    return N, Nhalo0


def reduce_sparse(label_max, unique_label, values, operation, init_value, comm, send_result=False):
    """
    Analog to comm.Allreduce with operation but create array only for root rank and done it with sparse matrix.
    """

    # Send information to the root rank
    nbr_unique_label = comm.gather(unique_label.size, root=0)

    if comm.Get_rank() != 0:  # .Send is a blocking communication
        comm.Send(unique_label, dest=0, tag=1)
        if values.ndim == 1:
            comm.Send(values, dest=0, tag=2)
        else:
            for i in range(values.shape[1]):
                to_send = values[:, i].copy()
                comm.Send(to_send, dest=0, tag=2 + i)

    if comm.Get_rank() == 0:
        # Define used quantity to collect the output from all the ranks
        shape = label_max if values.ndim == 1 else (label_max, values.shape[1])
        res = init_value * np.ones(shape)

        # add rank 0
        res[unique_label] = operation(res[unique_label], values)

        # collect other rank
        for send_rank in range(1, comm.size):
            unique_label_recv = np.zeros(nbr_unique_label[send_rank], dtype=type(unique_label[0]))
            comm.Recv(unique_label_recv, source=send_rank, tag=1)
            if values.ndim == 1:
                values_recv = np.zeros(nbr_unique_label[send_rank], dtype=type(values[0]))
                comm.Recv(values_recv, source=send_rank, tag=2)
                res[unique_label_recv] = operation(res[unique_label_recv], values_recv)
            else:
                for i in range(values.shape[1]):
                    values_recv = np.zeros(nbr_unique_label[send_rank], dtype=type(values[0, 0]))
                    comm.Recv(values_recv, source=send_rank, tag=2 + i)
                    res[unique_label_recv, i] = operation(res[unique_label_recv, i], values_recv)
    else:
        res = None

    if send_result:
        # Will populate the result on all the rank with only the corresponding value of label
        comm.Barrier()  # wait the reduction of all the rank

        if comm.Get_rank() != 0:  # send unique_label to avoid to store it in memory, good choice ?
            # Ask the root rank
            comm.Send(unique_label, dest=0, tag=10)
            # collect the minimal answer to save memory space
            if values.ndim == 1:
                minimal_res = np.zeros(unique_label.size, dtype=type(values[0]))
                comm.Recv(minimal_res, source=0, tag=11)
            else:
                minimal_res = np.zeros((unique_label.size, values.shape[1]), dtype=type(values[0, 0]))
                for i in range(values.shape[1]):
                    recv_tmp = np.zeros(unique_label.size, dtype=type(values[0, 0]))
                    comm.Recv(recv_tmp, source=0, tag=11 + i)
                    minimal_res[:, i] = recv_tmp

        if comm.Get_rank() == 0:
            # For root (rank 0):
            minimal_res = res[unique_label]

            # collect other rank
            for send_rank in range(1, comm.size):
                # Collect the label from send_rank
                unique_label_recv = np.zeros(nbr_unique_label[send_rank], dtype=type(unique_label[0]))
                comm.Recv(unique_label_recv, source=send_rank, tag=10)
                # Send the corresponding part of res to the label to send_rank
                if res.ndim == 1:
                    comm.Send(res[unique_label_recv], dest=send_rank, tag=11)
                else:
                    for i in range(res.shape[1]):
                        to_send = res[unique_label_recv, i].copy()
                        comm.Send(to_send, dest=send_rank, tag=11 + i)

        return res, minimal_res

    else:
        return res
