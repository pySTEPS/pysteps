# -*- coding: utf-8 -*-
"""
pysteps.noise.interface
=======================

Interface for the noise module.

.. autosummary::
    :toctree: ../generated/

    get_method
"""

from pysteps.noise.fftgenerators import (
    initialize_param_2d_fft_filter,
    generate_noise_2d_fft_filter,
    initialize_nonparam_2d_fft_filter,
    initialize_nonparam_2d_ssft_filter,
    generate_noise_2d_ssft_filter,
    initialize_nonparam_2d_nested_filter,
)
from pysteps.noise.motion import initialize_bps, generate_bps

_noise_methods = dict()

_noise_methods["parametric"] = (
    initialize_param_2d_fft_filter,
    generate_noise_2d_fft_filter,
)

_noise_methods["nonparametric"] = (
    initialize_nonparam_2d_fft_filter,
    generate_noise_2d_fft_filter,
)
_noise_methods["ssft"] = (
    initialize_nonparam_2d_ssft_filter,
    generate_noise_2d_ssft_filter,
)

_noise_methods["nested"] = (
    initialize_nonparam_2d_nested_filter,
    generate_noise_2d_ssft_filter,
)

_noise_methods["bps"] = (initialize_bps, generate_bps)


def get_method(name):
    """
    Return two callable functions to initialize and generate 2d perturbations
    of precipitation or velocity fields.\n

    Methods for precipitation fields:

    +-------------------+------------------------------------------------------+
    |     Name          |              Description                             |
    +===================+======================================================+
    |  parametric       | this global generator uses parametric Fourier        |
    |                   | filtering (power-law model)                          |
    +-------------------+------------------------------------------------------+
    |  nonparametric    | this global generator uses nonparametric Fourier     |
    |                   | filtering                                            |
    +-------------------+------------------------------------------------------+
    |  ssft             | this local generator uses the short-space Fourier    |
    |                   | filtering                                            |
    +-------------------+------------------------------------------------------+
    |  nested           | this local generator uses a nested Fourier filtering |
    +-------------------+------------------------------------------------------+

    Methods for velocity fields:

    +-------------------+------------------------------------------------------+
    |     Name          |              Description                             |
    +===================+======================================================+
    |  bps              | The method described in :cite:`BPS2006`, where       |
    |                   | time-dependent velocity perturbations are sampled    |
    |                   | from the exponential distribution                    |
    +-------------------+------------------------------------------------------+

    """
    if isinstance(name, str):
        name = name.lower()
    else:
        raise TypeError(
            "Only strings supported for the method's names.\n"
            + "Available names:"
            + str(list(_noise_methods.keys()))
        ) from None

    try:
        return _noise_methods[name]
    except KeyError:
        raise ValueError(
            "Unknown method {}\n".format(name)
            + "The available methods are:"
            + str(list(_noise_methods.keys()))
        ) from None
