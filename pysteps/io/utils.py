"""
pysteps.io.utils
================

Miscellaneous utility functions for the io module.

.. autosummary::
    :toctree: ../generated/

    get_pysteps_data_rootpath

"""

import os


def get_pysteps_data_rootpath():
    stp_data_path = os.environ.get("PYSTEPS_DATA", None)
    if stp_data_path is None:
        raise EnvironmentError("'PYSTEPS_DATA' environment variable not set")
    if not os.path.isdir(stp_data_path):
        raise EnvironmentError(
            "'PYSTEPS_DATA' path '{0}' " "does not exist".format(stp_data_path)
        )
    return stp_data_path
