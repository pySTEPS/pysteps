"""Interface module for different FFT methods."""

try:
    import pyfftw.interfaces.numpy_fft as pyfftw_fft
    import pyfftw
    # TODO: Caching and multithreading are currently disabled because they give 
    # a segfault with dask.
    pyfftw.interfaces.cache.disable()
    pyfftw_imported = True
except ImportError:
    pyfftw_imported = False
import scipy.fftpack as scipy_fft
import numpy.fft as numpy_fft

from pysteps.exceptions import MissingOptionalDependency

# use numpy implementation of rfft2/irfft2 because they have not been
# implemented in scipy.fftpack
scipy_fft.rfft2  = numpy_fft.rfft2
scipy_fft.irfft2 = numpy_fft.irfft2

def get_method(name):
    """Return a callable function for the FFT method corresponding to the given
    name.
    
    Parameters
    ----------
    name : str
        The name of the method. The available options are 'numpy', 'scipy' and 
        'pyfftw'
    
    Returns
    -------
    out : tuple
        A two-element tuple containing the FFT module and a dictionary of 
        default keyword arguments for calling the FFT method. Each module 
        implements the numpy.fft interface.
    """
    if name == "numpy":
        return numpy_fft,{}
    elif name == "scipy":
        return scipy_fft,{}
    elif name == "pyfftw":
        if not pyfftw_imported:
            raise MissingOptionalDependency("pyfftw is required but it is not installed")
        return pyfftw_fft,{"threads":1, "planner_effort":"FFTW_ESTIMATE"}
    else:
        raise ValueError("unknown method %s, the available methods are 'numpy', 'scipy' and 'pyfftw'" % name)
