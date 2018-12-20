"""Methods for processing and analyzing precipitation fields in the Fourier 
domain."""

import numpy as np
from . import arrays

def rapsd(Z, fft_method=None, return_freq=False, **fft_kwargs):
    """Compute radially averaged power spectral density (RAPSD) from the given 
    2D input field.
    
    Parameters
    ----------
    Z : array_like
      A 2d array of shape (M,N) containing the input field.
    fft_method : object
      A module or object implementing the same methods as numpy.fft and 
      scipy.fftpack. If set to None, Z is assumed to represent the shifted 
      discrete Fourier transform of the input field, where the origin is at 
      the center of the array (see numpy.fft.fftshift or scipy.fftpack.fftshift).
    return_freq: bool
      Whether to also return the Fourier frequencies.
    
    Returns
    -------
    out : ndarray
      One-dimensional array containing the RAPSD. The length of the array is 
      int(L/2)+1 (if L is even) or int(L/2) (if L is odd), where L=max(M,N).
    freq : ndarray
      One-dimensional array containing the Fourier frequencies.
    
    References
    ----------
    :cite:`RC2011`
    
    """

    if len(Z.shape) != 2:
        raise ValueError("%i dimensions are found, but the number of dimensions should be 2" % \
                         len(Z.shape))

    M,N = Z.shape

    YC,XC = arrays.compute_centred_coord_array(M, N)
    R = np.sqrt(XC*XC + YC*YC).round()
    L = max(Z.shape[0], Z.shape[1])

    if L % 2 == 0:
        r_range = np.arange(0, int(L/2)+1)
    else:
        r_range = np.arange(0, int(L/2))

    if fft_method is not None:
      F = fft_method.fftshift(fft_method.fft2(Z, **fft_kwargs))
      F = np.abs(F)**2/F.size
    else:
      F = Z

    result = []
    for r in r_range:
        MASK = R == r
        F_vals = F[MASK]
        result.append(np.mean(F_vals))
    
    if return_freq:
        freq = np.fft.fftfreq(L)
        freq = freq[r_range]
        return np.array(result), freq
    else:
        return np.array(result)