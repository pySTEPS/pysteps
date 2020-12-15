# -*- coding: utf-8 -*-
"""
pysteps.noise.utils
===================

Miscellaneous utility functions related to generation of stochastic perturbations.

.. autosummary::
    :toctree: ../generated/

    compute_noise_stddev_adjs
"""

import numpy as np

try:
    import dask

    dask_imported = True
except ImportError:
    dask_imported = False


def compute_noise_stddev_adjs(
    R,
    R_thr_1,
    R_thr_2,
    F,
    decomp_method,
    noise_filter,
    noise_generator,
    num_iter,
    conditional=True,
    num_workers=1,
    seed=None,
):
    """Apply a scale-dependent adjustment factor to the noise fields used in STEPS.

    Simulates the effect of applying a precipitation mask to a Gaussian noise
    field obtained by the nonparametric filter method. The idea is to decompose
    the masked noise field into a cascade and compare the standard deviations
    of each level into those of the observed precipitation intensity field.
    This gives correction factors for the standard deviations :cite:`BPS2006`.
    The calculations are done for n realizations of the noise field, and the
    correction factors are calculated from the average values of the standard
    deviations.

    Parameters
    ----------
    R: array_like
        The input precipitation field, assumed to be in logarithmic units
        (dBR or reflectivity).
    R_thr_1: float
        Intensity threshold for precipitation/no precipitation.
    R_thr_2: float
        Intensity values below R_thr_1 are set to this value.
    F: dict
        A bandpass filter dictionary returned by a method defined in
        pysteps.cascade.bandpass_filters. This defines the filter to use and
        the number of cascade levels.
    decomp_method: function
        A function defined in pysteps.cascade.decomposition. Specifies the
        method to use for decomposing the observed precipitation field and
        noise field into different spatial scales.
    num_iter: int
        The number of noise fields to generate.
    conditional: bool
        If set to True, compute the statistics conditionally by excluding areas
        of no precipitation.
    num_workers: int
        The number of workers to use for parallel computation. Applicable if
        dask is installed.
    seed: int
        Optional seed number for the random generators.

    Returns
    -------
    out: list
        A list containing the standard deviation adjustment factor for each
        cascade level.
    """

    MASK = R >= R_thr_1

    R = R.copy()
    R[~np.isfinite(R)] = R_thr_2
    R[~MASK] = R_thr_2
    if not conditional:
        mu, sigma = np.mean(R), np.std(R)
    else:
        mu, sigma = np.mean(R[MASK]), np.std(R[MASK])
    R -= mu

    MASK_ = MASK if conditional else None
    decomp_R = decomp_method(R, F, mask=MASK_)

    if dask_imported and num_workers > 1:
        res = []
    else:
        N_stds = []

    randstates = []
    seed = None
    for k in range(num_iter):
        randstates.append(np.random.RandomState(seed=seed))
        seed = np.random.randint(0, high=1e9)

    for k in range(num_iter):

        def worker():
            # generate Gaussian white noise field, filter it using the chosen
            # method, multiply it with the standard deviation of the observed
            # field and apply the precipitation mask
            N = noise_generator(noise_filter, randstate=randstates[k], seed=seed)
            N = N / np.std(N) * sigma + mu
            N[~MASK] = R_thr_2

            # subtract the mean and decompose the masked noise field into a
            # cascade
            N -= mu
            decomp_N = decomp_method(N, F, mask=MASK_)

            return decomp_N["stds"]

        if dask_imported and num_workers > 1:
            res.append(dask.delayed(worker)())
        else:
            N_stds.append(worker())

    if dask_imported and num_workers > 1:
        N_stds = dask.compute(*res, num_workers=num_workers)

    # for each cascade level, compare the standard deviations between the
    # observed field and the masked noise field, which gives the correction
    # factors
    return decomp_R["stds"] / np.mean(np.vstack(N_stds), axis=0)
