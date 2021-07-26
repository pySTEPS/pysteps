# -*- coding: utf-8 -*-
"""
pysteps.noise.motion
====================

Methods for generating perturbations of two-dimensional motion fields.

The methods in this module implement the following interface for
initialization::

  inizialize_xxx(V, pixelsperkm, timestep, optional arguments)

where V (2,m,n) is the motion field and pixelsperkm and timestep describe the
spatial and temporal resolution of the motion vectors.
The output of each initialization method is a dictionary containing the
perturbator that can be supplied to generate_xxx.

The methods in this module implement the following interface for the generation
of a motion perturbation field::

  generate_xxx(perturbator, t, randstate=np.random, seed=None)

where perturbator is a dictionary returned by an initialize_xxx method.
Optional random generator can be specified with the randstate and seed
arguments, respectively.
The output of each generator method is an array of shape (2,m,n) containing the
x- and y-components of the motion vector perturbations, where m and n are
determined from the perturbator.

.. autosummary::
    :toctree: ../generated/

    get_default_params_bps_par
    get_default_params_bps_perp
    initialize_bps
    generate_bps
"""

import numpy as np
from scipy import linalg


def get_default_params_bps_par():
    """Return a tuple containing the default velocity perturbation parameters
    given in :cite:`BPS2006` for the parallel component."""
    return (10.88, 0.23, -7.68)


def get_default_params_bps_perp():
    """Return a tuple containing the default velocity perturbation parameters
    given in :cite:`BPS2006` for the perpendicular component."""
    return (5.76, 0.31, -2.72)


def initialize_bps(
    V, pixelsperkm, timestep, p_par=None, p_perp=None, randstate=None, seed=None
):
    """Initialize the motion field perturbator described in :cite:`BPS2006`.
    For simplicity, the bias adjustment procedure described there has not been
    implemented. The perturbator generates a field whose magnitude increases
    with respect to lead time.

    Parameters
    ----------
    V: array_like
      Array of shape (2,m,n) containing the x- and y-components of the m*n
      motion field to perturb.
    p_par: tuple
      Tuple containing the parameters a,b and c for the standard deviation of
      the perturbations in the direction parallel to the motion vectors. The
      standard deviations are modeled by the function f_par(t) = a*t**b+c,
      where t is lead time. The default values are taken from :cite:`BPS2006`.
    p_perp: tuple
      Tuple containing the parameters a,b and c for the standard deviation of
      the perturbations in the direction perpendicular to the motion vectors.
      The standard deviations are modeled by the function f_par(t) = a*t**b+c,
      where t is lead time. The default values are taken from :cite:`BPS2006`.
    pixelsperkm: float
      Spatial resolution of the motion field (pixels/kilometer).
    timestep: float
      Time step for the motion vectors (minutes).
    randstate: mtrand.RandomState
      Optional random generator to use. If set to None, use numpy.random.
    seed: int
      Optional seed number for the random generator.

    Returns
    -------
    out: dict
      A dictionary containing the perturbator that can be supplied to
      generate_motion_perturbations_bps.

    See also
    --------
    pysteps.noise.motion.generate_bps

    """
    if len(V.shape) != 3:
        raise ValueError("V is not a three-dimensional array")
    if V.shape[0] != 2:
        raise ValueError("the first dimension of V is not 2")

    if p_par is None:
        p_par = get_default_params_bps_par()
    if p_perp is None:
        p_perp = get_default_params_bps_perp()

    if len(p_par) != 3:
        raise ValueError("the length of p_par is not 3")
    if len(p_perp) != 3:
        raise ValueError("the length of p_perp is not 3")

    perturbator = {}
    if randstate is None:
        randstate = np.random

    if seed is not None:
        randstate.seed(seed)

    eps_par = randstate.laplace(scale=1.0 / np.sqrt(2))
    eps_perp = randstate.laplace(scale=1.0 / np.sqrt(2))

    # scale factor for converting the unit of the advection velocities
    # into km/h
    vsf = 60.0 / (timestep * pixelsperkm)

    N = linalg.norm(V, axis=0)
    mask = N > 1e-12
    V_n = np.empty(V.shape)
    V_n[:, mask] = V[:, mask] / np.stack([N[mask], N[mask]])
    V_n[:, ~mask] = 0.0

    perturbator["randstate"] = randstate
    perturbator["vsf"] = vsf
    perturbator["p_par"] = p_par
    perturbator["p_perp"] = p_perp
    perturbator["eps_par"] = eps_par
    perturbator["eps_perp"] = eps_perp
    perturbator["V_par"] = V_n
    perturbator["V_perp"] = np.stack([-V_n[1, :, :], V_n[0, :, :]])

    return perturbator


def generate_bps(perturbator, t):
    """Generate a motion perturbation field by using the method described in
    :cite:`BPS2006`.

    Parameters
    ----------
    perturbator: dict
      A dictionary returned by initialize_motion_perturbations_bps.
    t: float
      Lead time for the perturbation field (minutes).

    Returns
    -------
    out: ndarray
      Array of shape (2,m,n) containing the x- and y-components of the motion
      vector perturbations, where m and n are determined from the perturbator.

    See also
    --------
    pysteps.noise.motion.initialize_bps

    """
    vsf = perturbator["vsf"]
    p_par = perturbator["p_par"]
    p_perp = perturbator["p_perp"]
    eps_par = perturbator["eps_par"]
    eps_perp = perturbator["eps_perp"]
    V_par = perturbator["V_par"]
    V_perp = perturbator["V_perp"]

    g_par = p_par[0] * pow(t, p_par[1]) + p_par[2]
    g_perp = p_perp[0] * pow(t, p_perp[1]) + p_perp[2]

    return (g_par * eps_par * V_par + g_perp * eps_perp * V_perp) / vsf
