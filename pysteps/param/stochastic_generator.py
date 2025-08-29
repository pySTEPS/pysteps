# Contains: gen_stoch_field, normalize_db_field, pl_filter
from typing import Optional
import numpy as np
from scipy import interpolate, stats
from steps_params import StepsParameters


def gen_stoch_field(
    steps_params: StepsParameters,
    nx: int,
    ny: int,
    pixel_size: float,
    scale_break: float,
    threshold: float,
):
    """
    Generate a rain field with normal distribution and a power law power spectrum
    Args:
        steps_params (StepsParameters): The dataclass with the steps parameters
        nx (int): x dimension of the output field
        ny (int): y dimension of the output field
        kmperpixel (float): pixel size
        scale_break (float): scale break in km
        threshold (float): rain threshold in db

    Returns:
        np.ndarray: Output field with shape (ny,nx)
    """

    beta_1 = steps_params.beta_1
    beta_2 = steps_params.beta_2

    # generate uniform random numbers in the range 0,1
    y = np.random.uniform(low=0, high=1, size=(ny, nx))

    # Power law filter the field
    fft = np.fft.fft2(y, (ny, nx))
    filter = pl_filter(beta_1, nx, ny, pixel_size, beta_2, scale_break)
    out_fft = fft * filter
    out_field = np.fft.ifft2(out_fft).real

    nbins = 500
    eps = 0.001

    res = stats.cumfreq(out_field, numbins=nbins)
    bins = [res.lowerlimit + ia * res.binsize for ia in range(1 + res.cumcount.size)]
    count = res.cumcount / res.cumcount[nbins - 1]

    # find the threshold value for this non-rain probability
    rain_bin = 0
    for ia in range(nbins):
        if count[ia] <= 1 - steps_params.rain_fraction:
            rain_bin = ia
        else:
            break
    rain_threshold = bins[rain_bin]

    # Shift the data to have the correct probability > 0
    norm_data = out_field - rain_threshold

    # Now we need to transform the "raining" samples to have the desired distribution
    rain_mask = norm_data > threshold
    rain_obs = norm_data[rain_mask]
    rain_res = stats.cumfreq(rain_obs, numbins=nbins)
    rain_bins = [
        rain_res.lowerlimit + ia * rain_res.binsize
        for ia in range(1 + rain_res.cumcount.size)
    ]
    rain_cdf = rain_res.cumcount / rain_res.cumcount[nbins - 1]

    # rain_bins are the bin edges; use bin centers for interpolation
    bin_centers = 0.5 * (np.array(rain_bins[:-1]) + np.array(rain_bins[1:]))

    # Step 1: Build LUT: map empirical CDF → target normal quantiles
    # Make sure rain_cdf values are in (0,1) to avoid issues with extreme tails
    rain_cdf_clipped = np.clip(rain_cdf, eps, 1 - eps)

    # Map rain_cdf quantiles to corresponding values in the target normal distribution
    target_mu = steps_params.nonzero_mean_db
    target_sigma = steps_params.nonzero_stdev_db
    normal_values = stats.norm.ppf(rain_cdf_clipped, loc=target_mu, scale=target_sigma)

    # Create interpolation function from observed rain values to target normal values
    cdf_transform = interpolate.interp1d(
        bin_centers,
        normal_values,
        kind="linear",
        bounds_error=False,
        fill_value=(normal_values[0], normal_values[-1]),  # type: ignore
    )

    # Transform pdf of the raining pixels
    norm_data[rain_mask] = cdf_transform(norm_data[rain_mask])

    return norm_data


def normalize_db_field(data, params, threshold, zerovalue):
    if params.rain_fraction < 0.025:
        return np.full_like(data, zerovalue)

    nbins = 500
    eps = 0.0001

    res = stats.cumfreq(data, numbins=nbins)
    bins = [res.lowerlimit + ia * res.binsize for ia in range(1 + res.cumcount.size)]
    count = res.cumcount / res.cumcount[nbins - 1]

    # find the threshold value for this non-rain probability
    rain_bin = 0
    for ia in range(nbins):
        if count[ia] <= 1 - params.rain_fraction:
            rain_bin = ia
        else:
            break
    rain_threshold = bins[rain_bin + 1]

    # Shift the data to have the correct probability of rain
    norm_data = data + (threshold - rain_threshold)

    # Now we need to transform the raining samples to have the desired distribution
    # Get the sample distribution
    rain_mask = norm_data > threshold
    rain_obs = norm_data[rain_mask]
    rain_res = stats.cumfreq(rain_obs, numbins=nbins)

    rain_bins = [
        rain_res.lowerlimit + ia * rain_res.binsize
        for ia in range(1 + rain_res.cumcount.size)
    ]
    rain_cdf = rain_res.cumcount / rain_res.cumcount[nbins - 1]

    # rain_bins are the bin edges; use bin centers for interpolation
    bin_centers = 0.5 * (np.array(rain_bins[:-1]) + np.array(rain_bins[1:]))

    # Step 1: Build LUT: map empirical CDF → target normal quantiles
    # Make sure rain_cdf values are in (0,1) to avoid issues with extreme tails
    rain_cdf_clipped = np.clip(rain_cdf, eps, 1 - eps)

    # Map rain_cdf quantiles to corresponding values in the target normal distribution
    # We need to reduce the bias in the output fields
    target_mu = params.nonzero_mean_db
    target_sigma = params.nonzero_stdev_db
    normal_values = stats.norm.ppf(rain_cdf_clipped, loc=target_mu, scale=target_sigma)

    # Create interpolation function from observed rain values to target normal values
    fill_value = (normal_values[0], normal_values[-1])
    cdf_transform = interpolate.interp1d(
        bin_centers,
        normal_values,
        kind="linear",
        bounds_error=False,
        fill_value=fill_value,  # type: ignore
    )

    # Transform raining pixels
    norm_data[rain_mask] = cdf_transform(norm_data[rain_mask])

    # Check if we have nans and return zerovalue if yes
    has_nan = np.isnan(norm_data).any()
    if has_nan:
        return np.full_like(data, zerovalue)
    else:
        return norm_data


def pl_filter(
    beta_1: float,
    nx: int,
    ny: int,
    pixel_size: float,
    beta_2: Optional[float] = None,
    scale_break: Optional[float] = None,
):
    """
    Generate a 2D low-pass power-law filter for FFT filtering.

    Parameters:
    beta_1 (float): Power law exponent for frequencies < f1 (low frequencies)
    nx (int): Number of columns (width) in the 2D field
    ny (int): Number of rows (height) in the 2D field
    pixel_size (float): Pixel size in km
    beta_2 (float): Power law exponent for frequencies > f1 (high frequencies) Optional
    scale_break (float): Break scale in km Optional

    Returns:
    np.ndarray: 2D FFT low-pass filter
    """

    # Compute the frequency grid
    freq_x = np.fft.fftfreq(nx, d=pixel_size)  # Frequency in x-direction
    freq_y = np.fft.fftfreq(ny, d=pixel_size)  # Frequency in y-direction

    # 2D array with radial frequency
    freq_r = np.sqrt(freq_x[:, None] ** 2 + freq_y[None, :] ** 2)

    # Initialize the radial 2D filter
    filter_r = np.ones_like(freq_r)  # Initialize with ones
    f_zero = freq_x[1]

    if beta_2 is not None and scale_break is not None:
        b1 = beta_1 / 2.0
        b2 = beta_2 / 2.0

        f1 = 1 / scale_break  # Convert scale break to frequency domain
        weight = (f1 / f_zero) ** b1

        # Apply the power-law function for a **low-pass filter**
        # Handle division by zero at freq = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            mask_low = freq_r < f1  # Frequencies lower than the break
            mask_high = ~mask_low  # Frequencies higher than or equal to the break

            filter_r[mask_low] = (freq_r[mask_low] / f_zero) ** b1
            filter_r[mask_high] = weight * (freq_r[mask_high] / f1) ** b2

            # Ensure DC component (zero frequency) is handled properly
            filter_r[freq_r == 0] = 1  # Preserve the mean component
    else:
        b1 = beta_1 / 2.0
        mask = freq_r > 0
        filter_r[mask] = (freq_r[mask] / f_zero) ** b1
        filter_r[freq_r == 0] = 1  # Preserve the mean component

    return filter_r
