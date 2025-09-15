# -*- coding: utf-8 -*-
"""
pysteps.extrapolation.interface
===============================

The functions in the extrapolation module implement the following interface::

    ``extrapolate(extrap, precip, velocity, timesteps, outval=np.nan, **keywords)``

where *extrap* is an extrapolator object returned by the initialize function,
*precip* is a (m,n) array with input precipitation field to be advected and
*velocity* is a (2,m,n) array containing  the x- and y-components of
the m x n advection field.
timesteps is an integer or list specifying the time steps to extrapolate. If
an integer is given, a range of uniformly spaced steps 1,2,...,timesteps is
created. If a list is given, it is assumed to represent a sequence of
monotonously increasing time steps. One time unit is assumed to represent the
time step of the advection field.
The optional argument *outval* specifies the value for pixels advected
from outside the domain.
Optional keyword arguments that are specific to a given extrapolation
method are passed as a dictionary.

The output of each method is an array that contains the time series of
extrapolated fields of shape (num_timesteps, m, n).

.. currentmodule:: pysteps.extrapolation.interface

.. autosummary::
    :toctree: ../generated/

    get_method
    eulerian_persistence
"""

import numpy as np
from pysteps.extrapolation import semilagrangian, eulerian_persistence


def _do_nothing(precip, velocity, timesteps, outval=np.nan, **kwargs):
    """Return None."""
    del precip, velocity, timesteps, outval, kwargs  # Unused
    return None


def _return_none(**kwargs):
    del kwargs  # Not used
    return None


_extrapolation_methods = dict()
_extrapolation_methods["eulerian"] = eulerian_persistence.extrapolate
_extrapolation_methods["semilagrangian"] = semilagrangian.extrapolate
_extrapolation_methods[None] = _do_nothing
_extrapolation_methods["none"] = _do_nothing


def get_method(name):
    """
    Return two-element tuple for the extrapolation method corresponding to
    the given name. The elements of the tuple are callable functions for the
    initializer of the extrapolator and the extrapolation method, respectively.
    The available options are:\n

    +-----------------+--------------------------------------------------------+
    |     Name        |              Description                               |
    +=================+========================================================+
    |  None           | returns None                                           |
    +-----------------+--------------------------------------------------------+
    |  eulerian       | this methods does not apply any advection to the input |
    |                 | precipitation field (Eulerian persistence)             |
    +-----------------+--------------------------------------------------------+
    | semilagrangian  | implementation of the semi-Lagrangian method described |
    |                 | in :cite:`GZ2002`                                      |
    +-----------------+--------------------------------------------------------+

    """
    if isinstance(name, str):
        name = name.lower()

    try:
        return _extrapolation_methods[name]

    except KeyError:
        raise ValueError(
            "Unknown method {}\n".format(name)
            + "The available methods are:"
            + str(list(_extrapolation_methods.keys()))
        ) from None
