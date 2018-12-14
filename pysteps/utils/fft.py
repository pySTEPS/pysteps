"""Interface module for different FFT methods."""

try:
    import pyfftw.interfaces.numpy_fft as pyfftw_fft
    import pyfftw
    # TODO: Caching and multithreading are currently disabled because they give 
    # a segfault with dask.
    pyfftw.interfaces.cache.disable()
    pyfftw_kwargs = {"threads":1, "planner_effort":"FFTW_ESTIMATE"}
    pyfftw_imported = True
except ImportError:
    pyfftw_imported = False
try:
    import scipy.fftpack as scipy_fft
    scipy_fft_kwargs = {}
except ImportError:
    scipy_imported = False
try:
    import numpy.fft as numpy_fft
    numpy_fft_kwargs = {}
except ImportError:
    numpy_imported = False

def get_method(name):
    if name == "numpy":
        return numpy_fft
    elif name == "scipy":
        return scipy_fft
    elif name == "pyfftw":
        return pyfftw_fft
    else:
        raise ValueError("unknown method %s, the available methods are 'numpy', 'scipy' and 'pyfftw'" % name)
