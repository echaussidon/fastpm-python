"""

Easy way to read and write BigFile with MPI

BaseFile is taken from: https://github.com/adematti/mockfactory/blob/main/mockfactory/catalog.py
I/O for bifgile is adapted from https://github.com/bccp/nbodykit/blob/master/nbodykit/io/bigfile.py

"""

import os
import logging
import numpy as np

from .utils import mkdir, ScatterArray


logger = logging.getLogger('I/O')


def _dict_to_array(data, struct=True):
    """
    Return dict as numpy array.
    Parameters
    ----------
    data : dict
        Data dictionary of name: array.
    struct : bool, default=True
        Whether to return structured array, with columns accessible through e.g. ``array['Position']``.
        If ``False``, numpy will attempt to cast types of different columns.
    Returns
    -------
    array : array
    """
    array = [(name, data[name]) for name in data]
    if struct:
        array = np.empty(array[0][1].shape[0], dtype=[(name, col.dtype, col.shape[1:]) for name, col in array])
        for name in data: array[name] = data[name]
    else:
        array = np.array([col for _, col in array])
    return array


class BaseFile(object):
    """
    Base class to read/write a file from/to disk.
    File handlers should extend this class, by (at least) implementing :meth:`read`, :meth:`get` and :meth:`write`.
    """
    _want_array = None

    def __init__(self, filename, attrs=None, mode='', mpicomm=None):
        """
        Initialize :class:`BaseFile`.
        Parameters
        ----------
        filename : string, list of strings
            File name(s).
        attrs : dict, default=None
            File attributes. Will be complemented by those read from disk.
            These will eventually be written to disk.
        mode : string, default=''
            If 'r', read file header (necessary for further reading of file columns).
        mpicomm : MPI communicator, default=None
            The current MPI communicator.
        """
        mode = mode.lower()
        allowed_modes = ['r', 'w', 'rw', '']
        if mode not in allowed_modes:
            raise ValueError('mode must be one of {}'.format(allowed_modes))
        if not isinstance(filename, (list, tuple)):
            filename = [filename]
        self.filenames = list(filename)
        self.attrs = attrs or {}
        self.mpicomm = mpicomm
        self.mpiroot = 0
        if 'r' in mode:
            self._read_header()

    def __enter__(self):  # to be used with Class() as c:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def __getitem__(self, item):  # to call the attributs of the class as a dictionnary
        return self.__dict__[item]

    def is_mpi_root(self):
        """Whether current rank is root."""
        return self.mpicomm.rank == self.mpiroot

    def _read_header(self):
        if self.is_mpi_root():
            basenames = ['size', 'columns', 'attrs']
            self.csizes, self.columns, names = [], None, None
            for filename in self.filenames:
                if self.is_mpi_root():
                    logger.info('Loading {}.'.format(filename))
                di = self._read_file_header(filename)
                self.csizes.append(di['size'])
                if self.columns is None:
                    self.columns = list(di['columns'])
                elif not set(di['columns']).issubset(self.columns):
                    raise ValueError('{} does not contain columns {}'.format(filename, set(di['columns']) - set(self.columns)))
                self.attrs = {**di.get('attrs', {}), **self.attrs}
                if names is None:
                    names = [name for name in di if name not in basenames]
                for name in names:  # typically extension name
                    setattr(self, name, di[name])
            state = {name: getattr(self, name) for name in ['csizes'] + basenames[1:] + names}
        self.__dict__.update(self.mpicomm.bcast(state if self.is_mpi_root() else None, root=self.mpiroot))
        # self.mpicomm.Barrier() # necessary to avoid blocking due to file not found
        self.csize = sum(self.csizes)
        self.start = self.mpicomm.rank * self.csize // self.mpicomm.size
        self.stop = (self.mpicomm.rank + 1) * self.csize // self.mpicomm.size
        self.size = self.stop - self.start

    def read(self, column):
        """Read column of name ``column``."""
        if not hasattr(self, 'csizes'):
            self._read_header()
        cumsizes = np.cumsum(self.csizes)
        ifile_start = np.searchsorted(cumsizes, self.start, side='left')  # cumsizes[i-1] < self.start <= cumsizes[i]
        ifile_stop = np.searchsorted(cumsizes, self.stop, side='left')
        toret = []
        for ifile in range(ifile_start, ifile_stop + 1):
            cumstart = 0 if ifile == 0 else cumsizes[ifile - 1]
            rows = slice(max(self.start - cumstart, 0), min(self.stop - cumstart, self.csizes[ifile]))
            toret.append(self._read_file_slice(self.filenames[ifile], column, rows=rows))
        return np.concatenate(toret, axis=0)

    def write(self, data, mpiroot=None):
        """
        Write input data to file(s).
        Parameters
        ----------
        data : array, dict
            Data to write.
        mpiroot : int, default=None
            If ``None``, input array is assumed to be scattered across all ranks.
            Else the MPI rank where input array is gathered.
        """
        isdict = None
        if self.mpicomm.rank == mpiroot or mpiroot is None:
            isdict = isinstance(data, dict)
        if mpiroot is not None:
            isdict = self.mpicomm.bcast(isdict, root=mpiroot)
            if isdict:
                columns = self.mpicomm.bcast(list(data.keys()) if self.mpicomm.rank == mpiroot else None, root=mpiroot)
                data = {name: ScatterArray(data[name] if self.mpicomm.rank == mpiroot else None, mpicomm=self.mpicomm, root=self.mpiroot) for name in columns}
            else:
                data = ScatterArray(data, mpicomm=self.mpicomm, root=self.mpiroot)
        if isdict:
            for name in data:
                size = len(data[name])
                break
        else:
            size = len(data)
        sizes = self.mpicomm.allgather(size)
        cumsizes = np.cumsum(sizes)
        csize = cumsizes[-1]
        nfiles = len(self.filenames)
        mpicomm = self.mpicomm  # store current communicator
        for ifile, filename in enumerate(self.filenames):
            if self.is_mpi_root():
                logger.info('Saving to {}.'.format(filename))
                mkdir(os.path.dirname(filename))
        for ifile, filename in enumerate(self.filenames):
            start, stop = ifile * csize // nfiles, (ifile + 1) * csize // nfiles
            irank_start = np.searchsorted(cumsizes, start, side='left')  # cumsizes[i-1] < self.start <= cumsizes[i]
            irank_stop = np.searchsorted(cumsizes, stop, side='left')
            rows = slice(0, 0)
            if irank_start <= self.mpicomm.rank <= irank_stop:
                cumstart = 0 if mpicomm.rank == 0 else cumsizes[mpicomm.rank - 1]
                rows = slice(max(start - cumstart, 0), min(stop - cumstart, sizes[mpicomm.rank]))
            if isdict:
                tmp = {name: data[name][rows] for name in data}
                if self._want_array:
                    tmp = _dict_to_array(tmp)
            else:
                tmp = data[rows]
                if not self._want_array:
                    tmp = {name: tmp[name] for name in tmp.dtype.names}
            self._write_file_slice(filename, tmp)
        self.mpicomm = mpicomm

    def _read_file_header(self, filename):
        """Return a dictionary of 'size', 'columns' at least for input ``filename``."""
        raise NotImplementedError('Implement method "_read_file" in your "{}"-inherited file handler'.format(self.__class__.___name__))

    def _read_file_slice(self, filename, column, rows):
        """
        Read rows ``rows`` of column ``column`` from file ``filename``.
        To be implemented in your file handler.
        """
        raise NotImplementedError('Implement method "_read_file_slice" in your "{}"-inherited file handler'.format(self.__class__.___name__))

    def _write_file_slice(self, filename, data):
        """
        Write ``data`` (``np.ndarray`` or ``dict``) to file ``filename``.
        To be implemented in your file handler.
        """
        raise NotImplementedError('Implement method "_write_file_slice" in your "{}"-inherited file handler'.format(self.__class__.___name__))


try: import bigfile
except ImportError: bigfile = None


class BigFile(BaseFile):
    """
    Class to read/write a Bigfile file from/to disk. Adapted from https://github.com/bccp/nbodykit/blob/master/nbodykit/io/bigfile.py
    Note
    ----
    In some circumstances (e.g. catalog has just been written), :meth:`get` fails with a file not found error.
    We have tried making sure processes read the file one after the other, but that does not solve the issue.
    A similar issue happens with nbodykit - though at a lower frequency.
    """
    _want_array = None

    def __init__(self, filename, dataset='1/', **kwargs):
        """
        Initialize :class:`BigFile`.
        Parameters
        ----------
        filename : string
            File name.
        dataset : string, default='1/'
            dataset where columns are located.
        kwargs : dict
            Arguments for :class:`BaseFile`.
        """
        if bigfile is None:
            raise ImportError('Install bigfile')

        if not dataset.endswith('/'): dataset = dataset + '/'
        self.dataset = dataset

        super(BigFile, self).__init__(filename=filename, **kwargs)

    def _read_file_header(self, filename):
        """ Find header from the file block by default. """
        with bigfile.File(filename=filename) as file:
            # collect minimal information:
            dataset = file[self.dataset]
            columns = dataset.keys()
            size = dataset[columns[0]].size
            for name in columns:
                if dataset[name].size != size:
                    raise IOError(f'Column {name} has different length (expected {size}, found {dataset[name].size})')

            # collect attrs from the header: suppose that header is saved in Header dataset and nothing is saved with JSON representation
            # --> otherwise see https://github.com/bccp/nbodykit/blob/master/nbodykit/io/bigfile.py
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                attrs = dict(file['Header'].attrs)

        return {'size': size, 'columns': columns, 'attrs': attrs}

    def _read_file_slice(self, filename, column, rows):
        """
        Read the specified column(s) over the given range,
        as a dictionary

        'start' and 'stop' should be between 0 and :attr:`size`,
        which is the total size of the binary file (in particles)
        """
        with bigfile.File(filename=filename)[self.dataset] as file:
            return file[column][rows.start:rows.stop]

    def _write_file_slice(self, filename, data):
        """
        Write data associated to the self.columns and the header saved in self.attrs
        """
        with bigfile.FileMPI(self.mpicomm, filename, create=True) as file:
            # write the header:
            with file.create('Header') as bb:
                for key in self.attrs.keys():
                    try:
                        bb.attrs[key] = self.attrs[key]
                    except KeyError:
                        pass
            # write the data
            for name in data.keys():
                file.create_from_array(self.dataset + name, data[name])
