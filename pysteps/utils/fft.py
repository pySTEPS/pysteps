"""Interface module for different FFT methods."""

try:
    import pyfftw.interfaces.numpy_fft as pyfftw_fft
    import pyfftw
    # TODO: Caching is currently disabled because it gives segfault with dask.
    pyfftw.interfaces.cache.disable()
    pyfftw_imported = True
except ImportError:
    pyfftw_imported = False
import scipy.fftpack as scipy_fft
import numpy.fft as numpy_fft

# use numpy implementation of rfft2/irfft2 because they have not been
# implemented in scipy.fftpack
scipy_fft.rfft2  = numpy_fft.rfft2
scipy_fft.irfft2 = numpy_fft.irfft2
