#!/usr/bin/env python
# coding: utf-8

import sys
import time
import logging
import traceback


logger = logging.getLogger("Utils")

_logging_handler = None


def exception_handler(exc_type, exc_value, exc_traceback):
    """Print exception with a logger."""
    # Do not print traceback if the exception has been handled and logged
    _logger_name = 'Exception'
    log = logging.getLogger(_logger_name)
    line = '='*100
    #log.critical(line[len(_logger_name) + 5:] + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    log.critical('\n' + line + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    if exc_type is KeyboardInterrupt:
        log.critical('Interrupted by the user.')
    else:
        log.critical('An error occured.')


def setup_logging(log_level="info", stream=sys.stdout, log_file=None):
    """
    Turn on logging with specific configuration.
    Parameters
    ----------
    log_level : 'info', 'debug', 'warning', 'error'
        Logging level, message below this level are not logged.
    stream : sys.stdout or sys.stderr
        Where to stream.
    log_file : str, default=None
        If not ``None`` stream to file name.
    """
    levels = {"info" : logging.INFO,
              "debug" : logging.DEBUG,
              "warning" : logging.WARNING,
              "error" : logging.ERROR}

    logger = logging.getLogger();
    t0 = time.time()

    class Formatter(logging.Formatter):
        def format(self, record):
            self._style._fmt = '[%09.2f]' % (time.time() - t0) + ' %(asctime)s %(name)-10s %(levelname)-8s %(message)s'
            return super(Formatter,self).format(record)
    fmt = Formatter(datefmt='%y-%m-%d %H:%M ')

    global _logging_handler
    if _logging_handler is None:
        _logging_handler = logging.StreamHandler(stream=stream)
        logger.addHandler(_logging_handler)

    _logging_handler.setFormatter(fmt)
    logger.setLevel(levels[log_level.lower()])

    # SAVE LOG INTO A LOG FILE
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    sys.excepthook = exception_handler
