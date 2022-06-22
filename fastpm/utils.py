"""
Credit: This code is copy from
             https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py.
             https://github.com/bccp/nbodykit/blob/master/nbodykit/batch.py.
             https://github.com/bccp/nbodykit/blob/master/nbodykit/__init__.py
             https://github.com/cosmodesi/mpytools/blob/main/mpytools/random.py

        To remove the dependence at nbodykit.
"""

import numpy as np
from mpi4py import MPI
import os
import mpsort


def mkdir(dirname):
    """Try to create ``dirnm`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname)  # MPI...
    except OSError:
        return


def split_size_3d(s):
    """
    Split `s` into three integers, a, b, c, such
    that a * b * c == s and a <= b <= c

    Note: a <= b <= c is not always true, we can sometime get 3, 1, X

    Parameters
    -----------
    s : int
        integer to split

    Returns
    -------
    a, b, c: int
        integers such that a * b * c == s and a <= b <= c
    """
    a = int(s ** 0.3333333) + 1
    while a > 1:
        if s % a == 0:
            s = s // a
            break
        a = a - 1
    b = int(s ** 0.5) + 1
    while b > 1:
        if s % b == 0:
            s = s // b
            break
        b = b - 1
    c = s
    return a, b, c


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


def GatherArray(data, comm, root=0):
    """
    Gather the input data array from all ranks to the specified ``root``.

    This uses `Gatherv`, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype

    Parameters
    ----------
    data : array_like
        the data on each rank to gather
    comm : MPI communicator
        the MPI communicator
    root : int, or Ellipsis
        the rank number to gather the data to. If root is Ellipsis,
        broadcast the result to all ranks.

    Returns
    -------
    recvbuffer : array_like, None
        the gathered data on root, and `None` otherwise
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("`data` must by numpy array in GatherArray")

    # need C-contiguous order
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    local_length = data.shape[0]

    # check dtypes and shapes
    shapes = comm.allgather(data.shape)
    dtypes = comm.allgather(data.dtype)

    # check for structured data
    if dtypes[0].char == 'V':

        # check for structured data mismatch
        names = set(dtypes[0].names)
        if any(set(dt.names) != names for dt in dtypes[1:]):
            raise ValueError("mismatch between data type fields in structured data")

        # check for 'O' data types
        if any(dtypes[0][name] == 'O' for name in dtypes[0].names):
            raise ValueError("object data types ('O') not allowed in structured data in GatherArray")

        # compute the new shape for each rank
        newlength = comm.allreduce(local_length)
        newshape = list(data.shape)
        newshape[0] = newlength

        # the return array
        if root is Ellipsis or comm.rank == root:
            recvbuffer = np.empty(newshape, dtype=dtypes[0], order='C')
        else:
            recvbuffer = None

        for name in dtypes[0].names:
            d = GatherArray(data[name], comm, root=root)
            if root is Ellipsis or comm.rank == root:
                recvbuffer[name] = d

        return recvbuffer

    # check for 'O' data types
    if dtypes[0] == 'O':
        raise ValueError("object data types ('O') not allowed in structured data in GatherArray")

    # check for bad dtypes and bad shapes
    if root is Ellipsis or comm.rank == root:
        bad_shape = any(s[1:] != shapes[0][1:] for s in shapes[1:])
        bad_dtype = any(dt != dtypes[0] for dt in dtypes[1:])
    else:
        bad_shape, bad_dtype = None, None

    bad_shape, bad_dtype = comm.bcast((bad_shape, bad_dtype))

    if bad_shape:
        raise ValueError("mismatch between shape[1:] across ranks in GatherArray")
    if bad_dtype:
        raise ValueError("mismatch between dtypes across ranks in GatherArray")

    shape = data.shape
    dtype = data.dtype

    # setup the custom dtype
    duplicity = np.product(np.array(shape[1:], 'intp'))
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    # compute the new shape for each rank
    newlength = comm.allreduce(local_length)
    newshape = list(shape)
    newshape[0] = newlength

    # the return array
    if root is Ellipsis or comm.rank == root:
        recvbuffer = np.empty(newshape, dtype=dtype, order='C')
    else:
        recvbuffer = None

    # the recv counts
    counts = comm.allgather(local_length)
    counts = np.array(counts, order='C')

    # the recv offsets
    offsets = np.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    # gather to root
    if root is Ellipsis:
        comm.Allgatherv([data, dt], [recvbuffer, (counts, offsets), dt])
    else:
        comm.Gatherv([data, dt], [recvbuffer, (counts, offsets), dt], root=root)

    dt.Free()

    return recvbuffer


def ScatterArray(data, comm, root=0, counts=None):
    """
    Scatter the input data array across all ranks, assuming `data` is
    initially only on `root` (and `None` on other ranks).

    This uses ``Scatterv``, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype

    Parameters
    ----------
    data : array_like or None
        on `root`, this gives the data to split and scatter
    comm : MPI communicator
        the MPI communicator
    root : int
        the rank number that initially has the data
    counts : list of int
        list of the lengths of data to send to each rank

    Returns
    -------
    recvbuffer : array_like
        the chunk of `data` that each rank gets
    """

    if counts is not None:
        counts = np.asarray(counts, order='C')
        if len(counts) != comm.size:
            raise ValueError("counts array has wrong length!")

    # check for bad input
    if comm.rank == root:
        bad_input = not isinstance(data, np.ndarray)
    else:
        bad_input = None
    bad_input = comm.bcast(bad_input)
    if bad_input:
        raise ValueError("`data` must by numpy array on root in ScatterArray")

    if comm.rank == 0:
        # need C-contiguous order
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        shape_and_dtype = (data.shape, data.dtype)
    else:
        shape_and_dtype = None

    # each rank needs shape/dtype of input data
    shape, dtype = comm.bcast(shape_and_dtype)

    # object dtype is not supported
    fail = False
    if dtype.char == 'V':
        fail = any(dtype[name] == 'O' for name in dtype.names)
    else:
        fail = dtype == 'O'
    if fail:
        raise ValueError("'object' data type not supported in ScatterArray; please specify specific data type")

    # initialize empty data on non-root ranks
    if comm.rank != root:
        np_dtype = np.dtype((dtype, shape[1:]))
        data = np.empty(0, dtype=np_dtype)

    # setup the custom dtype
    duplicity = np.product(np.array(shape[1:], 'intp'))
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    # compute the new shape for each rank
    newshape = list(shape)

    if counts is None:
        newlength = shape[0] // comm.size
        if comm.rank < shape[0] % comm.size:
            newlength += 1
        newshape[0] = newlength
    else:
        if counts.sum() != shape[0]:
            raise ValueError("the sum of the `counts` array needs to be equal to data length")
        newshape[0] = counts[comm.rank]

    # the return array
    recvbuffer = np.empty(newshape, dtype=dtype, order='C')

    # the send counts, if not provided
    if counts is None:
        counts = comm.allgather(newlength)
        counts = np.array(counts, order='C')

    # the send offsets
    offsets = np.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    # do the scatter
    comm.Barrier()
    comm.Scatterv([data, (counts, offsets), dt], [recvbuffer, dt])
    dt.Free()
    return recvbuffer


class DistributedArray(object):
    """
    Distributed Array Object

    A distributed array is striped along ranks, along first dimension

    Attributes
    ----------
    comm : :py:class:`mpi4py.MPI.Comm`
        the communicator

    local : array_like
        the local data

    """

    @staticmethod
    def _find_dtype(dtype, comm):
        # guess the dtype
        dtypes = comm.allgather(dtype)
        dtypes = set([dtype for dtype in dtypes if dtype is not None])

        if len(dtypes) > 1:
            raise TypeError("Type of local array is inconsistent between ranks; got %s" % dtypes)
        return next(iter(dtypes))

    @staticmethod
    def _find_cshape(shape, comm):
        # guess the dtype
        shapes = comm.allgather(shape)

        shapes = set(shape[1:] for shape in shapes)

        if len(shapes) > 1:
            raise TypeError("Shape of local array is inconsistent between ranks; got %s" % shapes)

        clen = comm.allreduce(shape[0])
        cshape = tuple([clen] + list(shape[1:]))
        return cshape

    def __init__(self, local, comm):
        self.comm = comm

        shape = np.array(local, copy=False).shape
        dtype = np.array(local, copy=False).dtype if len(local) else None

        self.dtype = DistributedArray._find_dtype(dtype, comm)
        self.cshape = DistributedArray._find_cshape(shape, comm)

        # directly use the original local array.
        self.local = local
        self.topology = LinearTopology(local, comm)

        self.coffset = sum(comm.allgather(shape[0])[:comm.rank])

    @classmethod
    def cempty(kls, cshape, dtype, comm):
        """ Create an empty array collectively """
        dtype = DistributedArray._find_dtype(dtype, comm)
        cshape = tuple(cshape)
        llen = cshape[0] * (comm.rank + 1) // comm.size - cshape[0] * (comm.rank) // comm.size
        shape = tuple([llen] + list(cshape[1:]))
        cshape1 = DistributedArray._find_cshape(shape, comm)
        if cshape != cshape1:
            raise ValueError("input cshape is inconsistent %s %s" % (cshape, cshape1))

        local = np.empty(shape, dtype=dtype)
        return DistributedArray(local, comm=comm)

    @classmethod
    def concat(kls, *args, **kwargs):
        """
        Append several distributed arrays into one.

        Parameters
        ----------
        localsize : None

        """

        localsize = kwargs.pop('localsize', None)

        comm = args[0].comm

        localsize_in = sum([len(arg.local) for arg in args])

        if localsize is None:
            localsize = sum([len(arg.local) for arg in args])

        eldtype = np.result_type(*[arg.local for arg in args])

        dtype = [('index', 'intp'), ('el', eldtype)]

        inp = np.empty(localsize_in, dtype=dtype)
        out = np.empty(localsize, dtype=dtype)

        go = 0
        o = 0
        for arg in args:
            inp['index'][o:o + len(arg.local)] = go + arg.coffset + np.arange(len(arg.local), dtype='intp')
            inp['el'][o:o + len(arg.local)] = arg.local
            o = o + len(arg.local)
            go = go + arg.cshape[0]
        mpsort.sort(inp, orderby='index', out=out, comm=comm)
        return DistributedArray(out['el'].copy(), comm=comm)

    def sort(self, orderby=None):
        """
        Sort array globally by key orderby.

        Due to a limitation of mpsort, self[orderby] must be u8.

        """
        mpsort.sort(self.local, orderby, comm=self.comm)

    def __getitem__(self, key):
        return DistributedArray(self.local[key], self.comm)

    def unique_labels(self):
        """
        Assign unique labels to sorted local.

        .. warning ::

            local data must be globally sorted, and of simple type. (numpy.unique)

        Returns
        -------
        label   :  :py:class:`DistributedArray`
            the new labels, starting from 0

        """
        # prev, next: -> _ to avoid a warning
        _, next = self.topology.prev(), self.topology.next()

        junk, label = np.unique(self.local, return_inverse=True)

        if len(label) == 0:
            # work around numpy bug (<=1.13.3) when label is empty it
            # spits out booleans?? booleans!!
            # this causes issues when type cast rules in numpy
            # are tighten up.
            label = np.int64(label)

        if len(self.local) == 0:
            Nunique = 0
        else:
            # watch out: this is to make sure after shifting first
            # labels on the next rank is the same as my last label
            # when there is a spill-over.
            if next == self.local[-1]:
                Nunique = len(junk) - 1
            else:
                Nunique = len(junk)

        label += np.sum(self.comm.allgather(Nunique)[:self.comm.rank], dtype='intp')
        return DistributedArray(label, self.comm)

    def bincount(self, weights=None, local=False, shared_edges=True):
        """
        Assign count numbers from sorted local data.

        .. warning ::

            local data must be globally sorted, and of integer type. (numpy.bincount)

        Parameters
        ----------
        weights: array-like
            if given, count the weight instead of the number of objects.
        local : boolean
            if local is True, only count the local array.
        shared_edges : boolean
            if True, keep the counts at edges that are shared between ranks on both ranks.
            if False, keep the counts at shared edges to the rank on the left.

        Returns
        -------
        N :  :py:class:`DistributedArray`
            distributed counts array. If items of the same value spans other
            chunks of array, they are added to N as well.

        Examples
        --------
        if the local array is [ (0, 0), (0, 1)],
        Then the counts array is [ (3, ), (3, 1)]

        """
        prev = self.topology.prev()
        if prev is not EmptyRank:
            offset = prev
            # two cases: either start counting from the last bin
            # of prev rank, or from next bin, depending on the
            # first value of my local data.
            if len(self.local) > 0:
                if prev != self.local[0]:
                    offset = prev + 1
        else:
            offset = 0

        # locally, we will bincount from offset to whereever we end

        N = np.bincount(self.local - offset, weights)

        if local:
            return N

        heads = self.topology.heads()
        tails = self.topology.tails()

        distN = DistributedArray(N, self.comm)
        headsN, tailsN = distN.topology.heads(), distN.topology.tails()

        if len(N) > 0:
            anyshared = False
            for i in reversed(range(self.comm.rank)):
                if tails[i] == self.local[0]:
                    N[0] += tailsN[i]
                    anyshared = True

            for i in range(self.comm.rank + 1, self.comm.size):
                if heads[i] == self.local[-1]:
                    N[-1] += headsN[i]

            if not shared_edges:
                # remove the edge from me, as it s already on the left rank.
                if anyshared:
                    N = N[1:]

        return DistributedArray(N, self.comm)


def _get_empty_rank():
    return EmptyRank


class EmptyRankType(object):
    def __repr__(self):
        return "EmptyRank"

    def __reduce__(self):
        return (_get_empty_rank, ())


EmptyRank = EmptyRankType()


class LinearTopology(object):
    """ Helper object for the topology of a distributed array
    """
    def __init__(self, local, comm):
        self.local = local
        self.comm = comm

    def heads(self):
        """
        The first items on each rank.

        Returns
        -------
        heads : list
            a list of first items, EmptyRank is used for empty ranks
        """

        head = EmptyRank
        if len(self.local) > 0:
            head = self.local[0]

        return self.comm.allgather(head)

    def tails(self):
        """
        The last items on each rank.

        Returns
        -------
        tails: list
            a list of last items, EmptyRank is used for empty ranks
        """
        tail = EmptyRank
        if len(self.local) > 0:
            tail = self.local[-1]

        return self.comm.allgather(tail)

    def prev(self):
        """
        The item before the local data.

        This method fetches the last item before the local data.
        If the rank before is empty, the rank before is used.

        If no item is before this rank, EmptyRank is returned

        Returns
        -------
        prev : scalar
            Item before local data, or EmptyRank if all ranks before this rank is empty.

        """

        tails = self.tails()
        for prev in reversed(tails[:self.comm.rank]):
            if prev is not EmptyRank:
                return prev
        return EmptyRank

    def next(self):
        """
        The item after the local data.

        This method the first item after the local data.
        If the rank after current rank is empty,
        item after that rank is used.

        If no item is after local data, EmptyRank is returned.

        Returns
        -------
        next : scalar
            Item after local data, or EmptyRank if all ranks after this rank is empty.

        """
        heads = self.heads()
        for next in heads[self.comm.rank + 1:]:
            if next is not EmptyRank:
                return next
        return EmptyRank
