import numpy as np
from pysteps import extrapolation


def calculate_wavelengths(n_levels: int, domain_size: float, d: float = 1.0):
    """
    Compute the central wavelengths (in km) for each cascade level.

    Parameters
    ----------
    n_levels : int
        Number of cascade levels.
    domain_size : int or float
        The larger of the two spatial dimensions of the domain in pixels.
    d : float
        Sample frequency in pixels per km. Default is 1.

    Returns
    -------
    wavelengths_km : np.ndarray
        Central wavelengths in km for each cascade level (length = n_levels).
    """
    # Compute q
    q = pow(0.5 * domain_size, 1.0 / n_levels)

    # Compute central wavenumbers (in grid units)
    r = [(pow(q, k - 1), pow(q, k)) for k in range(1, n_levels + 1)]
    central_wavenumbers = np.array([0.5 * (r0 + r1) for r0, r1 in r])

    # Convert to frequency
    central_freqs = central_wavenumbers / domain_size
    central_freqs[0] = 1.0 / domain_size
    central_freqs[-1] = 0.5  # Nyquist limit

    # Convert wavelength to km, d is pixels per km
    central_freqs = central_freqs * d
    central_wavelengths_km = 1.0 / central_freqs
    return central_wavelengths_km


def lagr_auto_cor(data: np.ndarray, oflow: np.ndarray):
    """
    Generate the Lagrangian auto correlations for STEPS cascades.

    Args:
        data (np.ndarray): [T, L, M, N] where:
            - T = ar_order + 1 (number of time steps)
            - L = number of cascade levels
            - M, N = spatial dimensions.
        oflow (np.ndarray): [2, M, N] Optical flow vectors.

    Returns:
        np.ndarray: Autocorrelation coefficients of shape (L, ar_order).
    """
    ar_order = 2
    if data.shape[0] < (ar_order + 1):
        raise ValueError(
            f"Insufficient time steps. Expected at least {ar_order + 1}, got {data.shape[0]}."
        )

    n_cascade_levels = data.shape[1]
    extrapolation_method = extrapolation.get_method("semilagrangian")

    autocorrelation_coefficients = np.full((n_cascade_levels, ar_order), np.nan)

    for level in range(n_cascade_levels):
        lag_1 = extrapolation_method(data[-2, level], oflow, 1)[0]
        lag_1 = np.where(np.isfinite(lag_1), lag_1, 0)

        data_t = np.where(np.isfinite(data[-1, level]), data[-1, level], 0)
        if np.std(lag_1) > 1e-1 and np.std(data_t) > 1e-1:
            autocorrelation_coefficients[level, 0] = np.corrcoef(
                lag_1.flatten(), data_t.flatten()
            )[0, 1]

        if ar_order == 2:
            lag_2 = extrapolation_method(data[-3, level], oflow, 1)[0]
            lag_2 = np.where(np.isfinite(lag_2), lag_2, 0)

            lag_1 = extrapolation_method(lag_2, oflow, 1)[0]
            lag_1 = np.where(np.isfinite(lag_1), lag_1, 0)

            if np.std(lag_1) > 1e-1 and np.std(data_t) > 1e-1:
                autocorrelation_coefficients[level, 1] = np.corrcoef(
                    lag_1.flatten(), data_t.flatten()
                )[0, 1]

    return autocorrelation_coefficients
