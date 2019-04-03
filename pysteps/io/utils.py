"""
pysteps.io.utils
================

Miscellaneous utility functions for the io module.

.. autosummary::
    :toctree: ../generated/

    get_pysteps_data_path
    get_pysteps_data_file

"""

import os

def get_pysteps_data_path():
    stp_data_path = os.environ.get('PYSTEPS_DATA', None)
    if stp_data_path is None:
        raise EnvironmentError("'PYSTEPS_DATA' environment variable not set")
    if not os.path.isdir(stp_data_path):
        raise EnvironmentError("'PYSTEPS_DATA' path '{0}' "
                               "does not exist".format(stp_data_path))
    return stp_data_path

def get_pysteps_data_file(relfile):
    data_file = os.path.join(get_pysteps_data_path(), relfile)
    if not os.path.exists(data_file):
        raise EnvironmentError("PYSTEPS_DATA file '{0}' "
                               "does not exist".format(data_file))
    return data_file
