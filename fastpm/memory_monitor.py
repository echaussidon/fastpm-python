import os
import psutil
import time

import numpy as np
import matplotlib.pyplot as plt


class MemoryMonitor(object):
    """
    Class that monitors memory usage and clock, useful to check for memory leaks.
    >>> with MemoryMonitor() as mem:
            '''do something'''
            mem()
            '''do something else'''
    """
    def __init__(self, pid=None, log_file=None):
        """
        Initalize :class:`MemoryMonitor` and register current memory usage.
        Parameters
        ----------
        pid : int, default=None
            Process identifier. If ``None``, use the identifier of the current process.
        """
        self.proc = psutil.Process(os.getpid() if pid is None else pid)
        self.mem = self.proc.memory_info().rss / 1e6
        self.start_time = time.time()
        self.time = self.start_time
        msg = 'using {:.3f} [Mb]'.format(self.mem)
        if log_file is None:
            self.log_file = None
            print(msg, flush=True)
        else:
            self.log_file = open(log_file, 'w')
            self.log_file.write("# time [s] -- memory [Mb] -- increase of memory [Mb]\n")

    def __enter__(self):
        """Enter context."""
        return self

    def __call__(self, log_format=None):
        """Update memory usage."""
        mem = self.proc.memory_info().rss / 1e6
        t = time.time()

        if self.log_file is None:
            msg = 'using {:.3f} [Mb] (increase of {:.3f} [Mb]) after {:.3f} [s]'.format(mem, mem - self.mem, t - self.time)
            if log_format:
                msg = '[{}] {}'.format(log_format, msg)
            print(msg, flush=True)
        else:
            msg = '{:.3f} {:.3f} {:.3f}'.format(t - self.start_time, mem, mem - self.mem)
            self.log_file.write(msg + '\n')
        self.mem = mem
        self.time = t

    def stop_monitoring(self):
        if self.log_file is not None:
            self.log_file.close()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit context."""
        self()


def plot_memory(path, prefix=''):
    import glob
    list_files = glob.glob(os.path.join(path, "memory-monitor", f"{prefix}memory_monitor_rank_*.txt"))
    nbr_files = len(list_files)
    nbr_max_files = 1000 if nbr_files > 1000 else nbr_files

    tab = np.loadtxt(list_files[0])[:, :2].T
    t, mem = tab[0], tab[1]

    if nbr_max_files > 1:
        for file in list_files[1:nbr_max_files]:
            tab = np.loadtxt(file)[:, :2].T
            t = t + tab[0]
            mem = mem + tab[1]

    t = t / nbr_max_files
    mem_per_proc = mem / nbr_max_files

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(t, mem / 1e3)
    plt.xlabel('t [s]')
    plt.ylabel('Global Memory [Gb]')
    plt.title(f'Max Memory per proc = {np.max(mem_per_proc) / 1e3:2.1f} [Gb] (for nproc={nbr_files}) ')

    plt.subplot(122)
    tab = np.loadtxt(os.path.join(path, "memory-monitor", f"{prefix}memory_monitor_rank_0.txt"))[:, :2].T
    t, mem = tab[0], tab[1]
    plt.plot(t, mem / 1e3, label='rank==0')
    if nbr_files > 1:
        tab = np.loadtxt(os.path.join(path, "memory-monitor", f"{prefix}memory_monitor_rank_1.txt"))[:, :2].T
        t, mem = tab[0], tab[1]
        plt.plot(t, mem / 1e3, label='rank==1')
    plt.xlabel('t [s]')
    plt.ylabel('Memory [Gb]')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{prefix}memory-monitor.png"))
    plt.close()
