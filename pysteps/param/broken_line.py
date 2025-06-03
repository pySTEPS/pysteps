
import numpy as np
from typing import Optional


def broken_line(rain_mean: float, rain_std: float, time_step: int, duration: int,
                h: Optional[float] = 0.60, q: Optional[float] = 0.85,
                a_zero_min: Optional[float] = 1500, transform: Optional[bool] = True):
    """
    Generate a time series of rainfall using the broken line model.
    Based on Seed et al. (2000), WRR.

    Args:
        rain_mean (float): Mean of time series (must be > 0)
        rain_std (float): Standard deviation of time series (must be > 0)
        time_step (int): Time step in minutes (must be > 0)
        duration (int): Duration of time series in minutes (must be > time_step)
        h (float): Scaling exponent (0 < h < 1)
        q (float): Scale change ratio between lines (0 < q < 1)
        a_zero_min (float): Maximum time scale in minutes (must be > 0)
        transform (bool): Use log transformation to generate the time series  

    Returns:
        np.ndarray: Rainfall time series of specified length, or None on error
    """

    # Validate input parameters
    if not isinstance(rain_mean, (float, int)) or rain_mean <= 0:
        print("Error: rain_mean must be a positive number.")
        return None
    if not isinstance(rain_std, (float, int)) or rain_std <= 0:
        print("Error: rain_std must be a positive number.")
        return None
    if not isinstance(time_step, int) or time_step <= 0:
        print("Error: time_step must be a positive integer.")
        return None
    if not isinstance(duration, int) or duration <= time_step:
        print("Error: duration must be an integer greater than time_step.")
        return None
    if not isinstance(h, (float, int)) or not (0 < h < 1):
        print("Error: h must be a float in the range (0,1).")
        return None
    if not isinstance(q, (float, int)) or not (0 < q < 1):
        print("Error: q must be a float in the range (0,1).")
        return None
    if not isinstance(a_zero_min, (float, int)) or a_zero_min <= 0:
        print("Error: a_zero_min must be a positive number.")
        return None

    # Number of time steps to generate
    length = duration // time_step  # Ensure integer division

    # Calculate the lognormal mean and variance
    if transform:
        ratio = rain_std / rain_mean
        bl_mean = np.log(rain_mean) - 0.5 * np.log(ratio**2 + 1)
        bl_var = np.log(ratio**2 + 1)
    else:
        bl_mean = rain_mean
        bl_var = rain_std ** 2.0

    # Compute number of broken lines
    a_zero = a_zero_min / time_step
    N = max(1, int(np.log(1.0 / a_zero) / np.log(q)) + 1)  # Prevents N=0

    # Compute variance at the outermost scale
    var_zero = bl_var * (1 - q**h) / (1 - q**(N * h))

    # Initialize the time series with mean
    model = np.full(length, bl_mean)

    # Add broken lines at different scales
    for p in range(N):
        break_step = a_zero * q**p
        line_stdev = np.sqrt(var_zero * q**(p * h))
        line = make_line(line_stdev, break_step, length)
        model += line

    # Transform back to rainfall space if needed
    if transform:
        rain = np.exp(model)
        return rain
    else:
        return model


def make_line(std_dev, break_step, length):
    """
    Generate a piecewise linear process with random breakpoints.

    Args:
        std_dev (float): Standard deviation for generating y-values.
        break_step (float): Distance between breakpoints.
        length (int): Length of the output array.

    Returns:
        np.ndarray: Interpolated line of given length.
    """

    # Generate random breakpoints
    rng = np.random.default_rng(None)

    if break_step < 1:
        y = rng.normal(0, std_dev, length)  # Scaled correctly
        return y

    # Number of breakpoints
    n_points = 3 + int(length / break_step)
    y = rng.normal(0, 1.5 * std_dev, n_points)  # Scaled correctly

    # Generate x-coordinates with random offset
    offset = rng.uniform(-break_step, 0)
    x = [offset + break_step*ia for ia in range(n_points)]

    # Interpolate onto full time series
    x_out = np.arange(length)
    line = np.interp(x_out, x, y)

    # Normalize the standard deviation
    line_std = np.std(line)
    if line_std > 0:
        line = (line - np.mean(line)) * (std_dev / line_std)

    return line
