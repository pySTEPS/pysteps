"""Implementations of cascade decompositions for separating two-dimensional
images into multiple spatial scales.

The methods in this module implement the following interface:

  decomposition_xxx(X, filter, optional arguments)

where X is the input field and filter is a dictionary returned by a filter
method implemented in bandpass_filters.py. The output of each method is a
dictionary with the following key-value pairs:

+-------------------+----------------------------------------------------------+
|        Key        |                      Value                               |
+===================+==========================================================+
|  cascade_levels   | three-dimensional array of shape (k,m,n), where k is the |
|                   | number of cascade levels and the input fields have shape |
|                   | (m,n)                                                    |
+-------------------+----------------------------------------------------------+
|  means            | list of mean values for each cascade level               |
+-------------------+----------------------------------------------------------+
|  stds             | list of standard deviations for each cascade level       |
+-------------------+----------------------------------------------------------+
"""

import numpy as np
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

def decomposition_fft(X, filter, **kwargs):
    """Decompose a 2d input field into multiple spatial scales by using the Fast
    Fourier Transform (FFT) and a bandpass filter.

    Parameters
    ----------
    X : array_like
      Two-dimensional array containing the input field. All values are required
      to be finite.
    filter : dict
      A filter returned by any method implemented in bandpass_filters.py.

    Other Parameters
    ----------------
    MASK : array_like
      Optional mask to use for computing the statistics for the cascade levels.
      Pixels with MASK==False are excluded from the computations.

    Returns
    -------
    out : ndarray
      A dictionary described in the module documentation. The parameter n is
      determined from the filter (see bandpass_filters.py).

    """
    MASK = kwargs.get("MASK", None)

    if len(X.shape) != 2:
        raise ValueError("the input is not two-dimensional array")
    if MASK is not None and MASK.shape != X.shape:
      raise ValueError("dimension mismatch between X and MASK: X.shape=%s, MASK.shape=%s" % \
        (str(X.shape), str(MASK.shape)))
    if X.shape[0] != filter["weights_2d"].shape[1]:
        raise ValueError("dimension mismatch between X and filter: X.shape[0]=%d, filter['weights_2d'].shape[1]=%d" % \
            (X.shape[0], filter["weights_2d"].shape[1]))
    if int(X.shape[1]/2)+1 != filter["weights_2d"].shape[2]:
        raise ValueError("dimension mismatch between X and filter: int(X.shape[1]/2)+1=%d, filter['weights_2d'].shape[2]=%d" % \
            (int(X.shape[1]/2)+1, filter["weights_2d"].shape[2]))
    if np.any(~np.isfinite(X)):
      raise ValueError("X contains non-finite values")

    result = {}
    means  = []
    stds   = []

    F = fft.rfft2(X, **fft_kwargs)
    X_decomp = []
    for k in range(len(filter["weights_1d"])):
        W_k = filter["weights_2d"][k, :, :]
        X_ = fft.irfft2(F*W_k, s=X.shape, **fft_kwargs)
        X_decomp.append(X_)

        if MASK is not None:
            X_ = X_[MASK]
        means.append(np.mean(X_))
        stds.append(np.std(X_))

    result["cascade_levels"] = np.stack(X_decomp)
    result["means"] = means
    result["stds"]  = stds

    return result
