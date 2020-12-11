# -*- coding: utf-8 -*-
"""
pysteps.io.archive
==================

Utilities for finding archived files that match the given criteria.

.. autosummary::
    :toctree: ../generated/

    find_by_date
"""

from datetime import datetime, timedelta
import fnmatch
import os


def find_by_date(
    date,
    root_path,
    path_fmt,
    fn_pattern,
    fn_ext,
    timestep,
    num_prev_files=0,
    num_next_files=0,
    silent=False,
):
    """List input files whose timestamp matches the given date.

    Parameters
    ----------
    date: datetime.datetime
        The given date.
    root_path: str
        The root path to search the input files.
    path_fmt: str
        Path format. It may consist of directory names separated by '/',
        date/time specifiers beginning with '%' (e.g. %Y/%m/%d) and wildcards
        (?) that match any single character.
    fn_pattern: str
        The name pattern of the input files without extension. The pattern can
        contain time specifiers (e.g. %H, %M and %S).
    fn_ext: str
        Extension of the input files.
    timestep: float
        Time step between consecutive input files (minutes).
    num_prev_files: int
        Optional, number of previous files to find before the given timestamp.
    num_next_files: int
        Optional, number of future files to find after the given timestamp.
    silent: bool
        Optional, whether to suppress all messages from the method.

    Returns
    -------
    out: tuple
        If num_prev_files=0 and num_next_files=0, return a pair containing the
        found file name and the corresponding timestamp as a datetime.datetime
        object. Otherwise, return a tuple of two lists, the first one for the
        file names and the second one for the corresponding timestemps. The lists
        are sorted in ascending order with respect to timestamp. A None value is
        assigned if a file name corresponding to a given timestamp is not found.

    """
    filenames = []
    timestamps = []

    for i in range(num_prev_files + num_next_files + 1):
        curdate = (
            date
            + timedelta(minutes=num_next_files * timestep)
            - timedelta(minutes=i * timestep)
        )
        fn = _find_matching_filename(
            curdate, root_path, path_fmt, fn_pattern, fn_ext, silent
        )
        filenames.append(fn)

        timestamps.append(curdate)

    if all(filename is None for filename in filenames):
        raise IOError("no input data found in %s" % root_path)

    if (num_prev_files + num_next_files) > 0:
        return filenames[::-1], timestamps[::-1]
    else:
        return filenames, timestamps


def _find_matching_filename(
    date, root_path, path_fmt, fn_pattern, fn_ext, silent=False
):
    path = _generate_path(date, root_path, path_fmt)
    fn = None

    if os.path.exists(path):
        fn = datetime.strftime(date, fn_pattern) + "." + fn_ext

        # test for wildcars
        if "?" in fn:
            filenames = os.listdir(path)
            if len(filenames) > 0:
                for filename in filenames:
                    if fnmatch.fnmatch(filename, fn):
                        fn = filename
                        break

        fn = os.path.join(path, fn)

        if os.path.exists(fn):
            fn = fn
        else:
            fn = None
            if not silent:
                print("file not found: %s" % fn)
    elif not silent:
        print("path", path, "not found.")

    return fn


def _generate_path(date, root_path, path_format):
    """Generate file path."""
    if not isinstance(date, datetime):
        raise TypeError("The input 'date' argument must be a datetime object")

    if path_format != "":
        sub_path = date.strftime(path_format)
        return os.path.join(root_path, sub_path)
    else:
        return root_path
