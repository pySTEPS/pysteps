"""Miscellaneous utility functions related to generation of stochastic perturbations."""

import numpy as np
try:
    import dask
    dask_imported = True
except ImportError:
    dask_imported = False
# Use the pyfftw interface if it is installed. If not, fall back to the fftpack
# interface provided by SciPy, and finally to numpy if SciPy is not installed.
try:
    import pyfftw.interfaces.numpy_fft as fft
    import pyfftw
    # TODO: Caching and multithreading currently disabled because they give a
    # segfault with dask.
    #pyfftw.interfaces.cache.enable()
    fft_kwargs = {"threads":1, "planner_effort":"FFTW_ESTIMATE"}
except ImportError:
    import scipy.fftpack as fft
    fft_kwargs = {}
except ImportError:
    import numpy.fft as fft
    fft_kwargs = {}

def compute_noise_stddev_adjs(R, R_thr_1, R_thr_2, F, decomp_method, num_iter,
                              conditional=True, num_workers=None):
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
    R : array_like
        The input precipitation field, assumed to be in logarithmic units
        (dBR or reflectivity).
    R_thr_1 : float
        Intensity threshold for precipitation/no precipitation.
    R_thr_2 : float
        Intensity values below R_thr_1 are set to this value.
    F : dict
        A bandpass filter dictionary returned by a method defined in
        pysteps.cascade.bandpass_filters. This defines the filter to use and
        the number of cascade levels.
    decomp_method : function
        A function defined in pysteps.cascade.decomposition. Specifies the
        method to use for decomposing the observed precipitation field and
        noise field into different spatial scales.
    num_iter : int
        The number of noise fields to generate.
    conditional : bool
        If set to True, compute the statistics conditionally by excluding areas
        of no precipitation.
    num_workers : int
        The number of workers to use for parallel computation. Set to None to
        use all available CPUs. Applicable if dask is enabled.

    Returns
    -------
    out : list
        A list containing the standard deviation adjustment factor for each
        cascade level.

    """

    MASK = R >= R_thr_1

    R = R.copy()
    R[~np.isfinite(R)] = R_thr_2
    R[~MASK] = R_thr_2
    if not conditional:
        mu,sigma = np.mean(R),np.std(R)
    else:
        mu,sigma = np.mean(R[MASK]),np.std(R[MASK])
    R -= mu

    MASK_ = MASK if conditional else None
    decomp_R = decomp_method(R, F, MASK=MASK_)

    if not dask_imported:
        N_stds = []
    else:
        res = []

    randstates = []
    seed = None
    for k in range(num_iter):
        randstates.append(np.random.RandomState(seed=seed))
        seed = np.random.randint(0, high=1e9)

    R_fft = abs(fft.fft2(R))

    for k in range(num_iter):
        def worker():
            # generate Gaussian white noise field, multiply it with the standard
            # deviation of the observed field and apply the precipitation mask
            N = randstates[k].randn(R.shape[0], R.shape[1])
            N = np.real(fft.ifft2(fft.fft2(N) * R_fft))
            N = N / np.std(N) * sigma + mu
            N[~MASK] = R_thr_2

            # subtract the mean and decompose the masked noise field into a
            # cascade
            N -= mu
            decomp_N = decomp_method(N, F, MASK=MASK_)

            return decomp_N["stds"]

        if dask_imported:
            res.append(dask.delayed(worker)())
        else:
            N_stds.append(worker())

    if dask_imported:
        N_stds = dask.compute(*res, num_workers=num_workers)

    # for each cascade level, compare the standard deviations between the
    # observed field and the masked noise field, which gives the correction
    # factors
    return decomp_R["stds"] / np.mean(np.vstack(N_stds), axis=0)
