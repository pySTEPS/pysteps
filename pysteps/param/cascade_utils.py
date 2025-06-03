import numpy as  np 

def get_cascade_wavelengths(n_levels, domain_size_km, d=1.0, gauss_scale=0.5):
    """
    Compute the central wavelengths (in km) for each cascade level.

    Parameters
    ----------
    n_levels : int
        Number of cascade levels.
    domain_size_km : int or float
        The larger of the two spatial dimensions (in km) of the domain.
    d : float
        Sample spacing (inverse of sampling rate). Default is 1.
    gauss_scale : float
        The Gaussian filter scaling parameter.

    Returns
    -------
    wavelengths_km : np.ndarray
        Central wavelengths in km for each cascade level (length = n_levels).
    """
    # Compute q as in _gaussweights_1d
    q = pow(0.5 * domain_size_km, 1.0 / n_levels)
    
    # Compute central wavenumbers (in grid units)
    r = [(pow(q, k - 1), pow(q, k)) for k in range(1, n_levels + 1)]
    central_wavenumbers = np.array([0.5 * (r0 + r1) for r0, r1 in r])
    
    # Convert to frequency
    central_freqs = central_wavenumbers / domain_size_km
    central_freqs[0] = 1.0 / domain_size_km  # enforce first freq > 0
    central_freqs[-1] = 0.5  # Nyquist limit
    central_freqs *= d

    # Convert to wavelength (in km)
    central_wavelengths_km = 1.0 / central_freqs

    return central_wavelengths_km
