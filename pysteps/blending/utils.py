# -*- coding: utf-8 -*-
"""
pysteps.blending.utils
======================

Module with common utilities used by the blending methods.

.. autosummary::
    :toctree: ../generated/

    stack_cascades
    blend_cascades
    recompose_cascade
    blend_optical_flows
    decompose_NWP
    compute_store_nwp_motion
    load_NWP
    compute_smooth_dilated_mask
"""

import datetime
from typing import Any, Callable
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from pysteps.cascade import get_method as cascade_get_method
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps.exceptions import MissingOptionalDependency
from pysteps.utils import get_method as utils_get_method
from pysteps.utils.check_norain import check_norain as new_check_norain

try:
    import netCDF4

    NETCDF4_IMPORTED = True
except ImportError:
    NETCDF4_IMPORTED = False

try:
    import cv2

    CV2_IMPORTED = True
except ImportError:
    CV2_IMPORTED = False


def blend_cascades(cascades_norm, weights):
    """Calculate blended normalized cascades using STEPS weights following eq.
    10 in :cite:`BPS2006`.

    Parameters
    ----------
    cascades_norm : array-like
        Array of shape [number_components + 1, scale_level, ...]
        with the cascade for each component (NWP, nowcasts, noise) and scale level,
        obtained by calling a method implemented in pysteps.blending.utils.stack_cascades

    weights : array-like
        An array of shape [number_components + 1, scale_level, ...]
        containing the weights to be used in this routine
        for each component plus noise, scale level, and optionally [y, x]
        dimensions, obtained by calling a method implemented in
        pysteps.blending.steps.calculate_weights

    Returns
    -------
    combined_cascade : array-like
        An array of shape [scale_level, y, x]
        containing per scale level (cascade) the weighted combination of
        cascades from multiple components (NWP, nowcasts and noise) to be used
        in STEPS blending.
    """
    # check inputs
    if isinstance(cascades_norm, (list, tuple)):
        cascades_norm = np.stack(cascades_norm)

    if isinstance(weights, (list, tuple)):
        weights = np.asarray(weights)

    # check weights dimensions match number of sources
    num_sources = cascades_norm.shape[0]
    num_sources_klevels = cascades_norm.shape[1]
    num_weights = weights.shape[0]
    num_weights_klevels = weights.shape[1]

    if num_weights != num_sources:
        raise ValueError(
            "dimension mismatch between cascades and weights.\n"
            "weights dimension must match the number of components in cascades.\n"
            f"number of models={num_sources}, number of weights={num_weights}"
        )
    if num_weights_klevels != num_sources_klevels:
        raise ValueError(
            "dimension mismatch between cascades and weights.\n"
            "weights cascade levels dimension must match the number of cascades in cascades_norm.\n"
            f"number of cascade levels={num_sources_klevels}, number of weights={num_weights_klevels}"
        )

    # cascade_norm component, scales, y, x
    # weights component, scales, ....
    # Reshape weights to make the calculation possible with numpy
    all_c_wn = weights.reshape(num_weights, num_weights_klevels, 1, 1) * cascades_norm
    combined_cascade = np.sum(all_c_wn, axis=0)
    # combined_cascade [scale, ...]
    return combined_cascade


def recompose_cascade(combined_cascade, combined_mean, combined_sigma):
    """Recompose the cascades into a transformed rain rate field.


    Parameters
    ----------
    combined_cascade : array-like
        An array of shape [scale_level, y, x]
        containing per scale level (cascade) the weighted combination of
        cascades from multiple components (NWP, nowcasts and noise) to be used
        in STEPS blending.
    combined_mean : array-like
        An array of shape [scale_level, ...]
        similar to combined_cascade, but containing the normalization parameter
        mean.
    combined_sigma : array-like
        An array of shape [scale_level, ...]
        similar to combined_cascade, but containing the normalization parameter
        standard deviation.

    Returns
    -------
    out: array-like
        A two-dimensional array containing the recomposed cascade.

    """
    # Renormalize with the blended sigma and mean values
    renorm = (
        combined_cascade * combined_sigma.reshape(combined_cascade.shape[0], 1, 1)
    ) + combined_mean.reshape(combined_mean.shape[0], 1, 1)
    # print(renorm.shape)
    out = np.sum(renorm, axis=0)
    # print(out.shape)
    return out


def blend_optical_flows(flows, weights):
    """Combine advection fields using given weights. Following :cite:`BPS2006`
    the second level of the cascade is used for the weights

    Parameters
    ----------
    flows : array-like
        A stack of multiple advenction fields having shape
        (S, 2, m, n), where flows[N, :, :, :] contains the motion vectors
        for source N.
        Advection fields for each source can be obtanined by
        calling any of the methods implemented in
        pysteps.motion and then stack all together
    weights : array-like
        An array of shape [number_sources]
        containing the weights to be used to combine
        the advection fields of each source.
        weights are modified to make their sum equal to one.
    Returns
    -------
    out: ndarray
        Return the blended advection field having shape
        (2, m, n), where out[0, :, :] contains the x-components of
        the blended motion vectors and out[1, :, :] contains the y-components.
        The velocities are in units of pixels / timestep.
    """

    # check inputs
    if isinstance(flows, (list, tuple)):
        flows = np.stack(flows)

    if isinstance(weights, (list, tuple)):
        weights = np.asarray(weights)

    # check weights dimensions match number of sources
    num_sources = flows.shape[0]
    num_weights = weights.shape[0]

    if num_weights != num_sources:
        raise ValueError(
            "dimension mismatch between flows and weights.\n"
            "weights dimension must match the number of flows.\n"
            f"number of flows={num_sources}, number of weights={num_weights}"
        )
    # normalize weigths
    weights = weights / np.sum(weights)

    # flows dimension sources, 2, m, n
    # weights dimension sources
    # move source axis to last to allow broadcasting
    # TODO: Check if broadcasting has worked well
    all_c_wn = weights * np.moveaxis(flows, 0, -1)
    # sum uses last axis
    combined_flows = np.sum(all_c_wn, axis=-1)
    # combined_flows [2, m, n]
    return combined_flows


def decompose_NWP(
    precip_nwp_dataset: xr.Dataset,
    num_cascade_levels=6,
    num_workers=1,
    decomp_method="fft",
    fft_method="numpy",
    domain="spatial",
    normalize=True,
    compute_stats=True,
    compact_output=True,
) -> xr.Dataset:
    """Decomposes the NWP forecast data into cascades and saves it in
    a netCDF file

    Parameters
    ----------
    R_NWP: array-like
        Array of dimension (n_timesteps, x, y) containing the precipitation forecast
        from some NWP model.
    NWP_model: str
        The name of the NWP model
    analysis_time: numpy.datetime64
        The analysis time of the NWP forecast. The analysis time is assumed to be a
        numpy.datetime64 type as imported by the pysteps importer
    output_path: str
        The location where to save the file with the NWP cascade. Defaults to the
        path_workdir specified in the rcparams file.
    num_cascade_levels: int, optional
        The number of frequency bands to use. Must be greater than 2. Defaults to 8.
    num_workers: int, optional
        The number of workers to use for parallel computation. Applicable if dask
        is enabled or pyFFTW is used for computing the FFT. When num_workers>1, it
        is advisable to disable OpenMP by setting the environment variable
        OMP_NUM_THREADS to 1. This avoids slowdown caused by too many simultaneous
        threads.

    Other Parameters
    ----------------
    decomp_method: str, optional
        A string defining the decomposition method to use. Defaults to "fft".
    fft_method: str or tuple, optional
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see :py:func:`pysteps.utils.interface.get_method`).
        Defaults to "numpy". This option is not used if input_domain and
        output_domain are both set to "spectral".
    domain: {"spatial", "spectral"}, optional
        If "spatial", the output cascade levels are transformed back to the
        spatial domain by using the inverse FFT. If "spectral", the cascade is
        kept in the spectral domain. Defaults to "spatial".
    normalize: bool, optional
        If True, normalize the cascade levels to zero mean and unit variance.
        Requires that compute_stats is True. Implies that compute_stats is True.
        Defaults to False.
    compute_stats: bool, optional
        If True, the output dictionary contains the keys "means" and "stds"
        for the mean and standard deviation of each output cascade level.
        Defaults to False.
    compact_output: bool, optional
        Applicable if output_domain is "spectral". If set to True, only the
        parts of the Fourier spectrum with non-negligible filter weights are
        stored. Defaults to False.


    Returns
    -------
    xarray.Dataset
        The same dataset as was passed in but with the precip data replaced
        with decomposed precip data and means and stds added
    """

    nwp_precip_var = precip_nwp_dataset.attrs["precip_var"]
    precip_nwp = precip_nwp_dataset[nwp_precip_var].values
    # Decompose the NWP data
    filter_g = filter_gaussian(precip_nwp.shape[1:], num_cascade_levels)
    fft = utils_get_method(
        fft_method, shape=precip_nwp.shape[1:], n_threads=num_workers
    )
    decomp_method, _ = cascade_get_method(decomp_method)

    pr_decomposed = np.zeros(
        (
            precip_nwp.shape[0],
            num_cascade_levels,
            precip_nwp.shape[1],
            precip_nwp.shape[2],
        ),
        dtype=np.float32,
    )
    means = np.zeros(
        (precip_nwp.shape[0], num_cascade_levels),
        dtype=np.float64,
    )
    stds = np.zeros(
        (precip_nwp.shape[0], num_cascade_levels),
        dtype=np.float64,
    )

    for i in range(precip_nwp.shape[0]):
        decomposed_precip_nwp = decomp_method(
            field=precip_nwp[i, :, :],
            bp_filter=filter_g,
            fft_method=fft,
            input_domain=domain,
            output_domain=domain,
            normalize=normalize,
            compute_stats=compute_stats,
            compact_output=compact_output,
        )

        pr_decomposed[i, :, :, :] = decomposed_precip_nwp["cascade_levels"]
        means[i, :] = decomposed_precip_nwp["means"]
        stds[i, :] = decomposed_precip_nwp["stds"]

    precip_nwp_dataset = precip_nwp_dataset.assign_coords(
        cascade_level=(
            "cascade_level",
            np.arange(num_cascade_levels),
            {"long_name": "cascade level", "units": ""},
        )
    )
    precip_nwp_dataset = precip_nwp_dataset.drop_vars(nwp_precip_var)
    precip_nwp_dataset[nwp_precip_var] = (
        ["time", "cascade_level", "y", "x"],
        pr_decomposed,
    )
    precip_nwp_dataset["means"] = (["time", "cascade_level"], means)
    precip_nwp_dataset["stds"] = (["time", "cascade_level"], stds)

    precip_nwp_dataset[nwp_precip_var].attrs["domain"] = domain
    precip_nwp_dataset[nwp_precip_var].attrs["normalized"] = int(normalize)
    precip_nwp_dataset[nwp_precip_var].attrs["compact_output"] = int(compact_output)

    return precip_nwp_dataset


def _preprocess_nwp_data_single_member(
    precip_nwp_dataset: xr.Dataset,
    oflow_method: Callable[..., Any],
    decompose_nwp: bool,
    decompose_kwargs: dict[str, Any] = {},
) -> xr.Dataset:
    nwp_precip_var = precip_nwp_dataset.attrs["precip_var"]
    precip_nwp = precip_nwp_dataset[nwp_precip_var].values

    # Get the velocity field per time step
    v_nwp_x = np.zeros((precip_nwp.shape[0], precip_nwp.shape[1], precip_nwp.shape[2]))
    v_nwp_y = np.zeros((precip_nwp.shape[0], precip_nwp.shape[1], precip_nwp.shape[2]))
    # Loop through the timesteps. We need two images to construct a motion
    # field, so we can start from timestep 1.
    for t in range(1, precip_nwp.shape[0]):
        v_nwp_dataset = oflow_method(precip_nwp_dataset.isel(time=slice(t - 1, t + 1)))
        v_nwp_x[t] = v_nwp_dataset.velocity_x
        v_nwp_y[t] = v_nwp_dataset.velocity_y

    # Make timestep 0 the same as timestep 1.
    v_nwp_x[0] = v_nwp_x[1]
    v_nwp_y[0] = v_nwp_y[1]
    precip_nwp_dataset["velocity_x"] = (["time", "y", "x"], v_nwp_x)
    precip_nwp_dataset["velocity_y"] = (["time", "y", "x"], v_nwp_y)

    if decompose_nwp:
        precip_nwp_dataset = decompose_NWP(precip_nwp_dataset, **decompose_kwargs)

    return precip_nwp_dataset


def preprocess_nwp_data(
    precip_nwp_dataset: xr.Dataset,
    oflow_method: Callable[..., Any],
    nwp_model: str,
    output_path: str | None,
    decompose_nwp: bool,
    decompose_kwargs: dict[str, Any] = {},
):
    """Computes, per forecast lead time, the velocity field of an NWP model field.

    Parameters
    ----------
    precip_nwp_dataset: xarray.Dataset
        xarray Dataset containing the precipitation forecast
        from some NWP model.
    oflow_method: {'constant', 'darts', 'lucaskanade', 'proesmans', 'vet'}, optional
        An optical flow method from pysteps.motion.get_method.
    nwp_model: str
        The name of the NWP model.
    output_path: str, optional
        The location where to save the netcdf file with the NWP velocity fields. Defaults
        to the path_workdir specified in the rcparams file.
    decompose_nwp: bool
        Defines wether or not the NWP needs to be decomposed before storing. This can
        be beneficial for performance, because then the decomposition does not need
        to happen during the blending anymore. It can however also be detrimental because
        this increases the amount of storage and RAM required for the blending.
    decompose_kwargs: dict
        Keyword arguments passed to the decompose_NWP method.

    Returns
    -------
    Nothing
    """

    if not NETCDF4_IMPORTED:
        raise MissingOptionalDependency(
            "netCDF4 package is required to save the NWP data, "
            "but it is not installed"
        )

    if "ens_number" in precip_nwp_dataset.dims:
        preprocessed_nwp_datasets = []
        for ens_number in precip_nwp_dataset["ens_number"]:
            preprocessed_nwp_datasets.append(
                _preprocess_nwp_data_single_member(
                    precip_nwp_dataset.sel(ens_number=ens_number),
                    oflow_method,
                    decompose_nwp,
                    decompose_kwargs,
                ).expand_dims({"ens_number": [ens_number]}, axis=0)
            )
        precip_nwp_dataset = xr.concat(preprocessed_nwp_datasets, "ens_number")
    else:
        precip_nwp_dataset = _preprocess_nwp_data_single_member(
            precip_nwp_dataset, oflow_method, decompose_nwp, decompose_kwargs
        )

    # Save it as a numpy array
    if output_path:
        analysis_time = precip_nwp_dataset.time.values[0]
        output_date = f"{analysis_time.astype('datetime64[us]').astype(datetime.datetime):%Y%m%d%H%M%S}"
        outfn = Path(output_path) / f"preprocessed_{nwp_model}_{output_date}.nc"
        precip_nwp_dataset.to_netcdf(outfn)
        return None
    else:
        return precip_nwp_dataset


def check_norain(precip_arr, precip_thr=None, norain_thr=0.0):
    """
    DEPRECATED use :py:mod:`pysteps.utils.check_norain.check_norain` in stead
    Parameters
    ----------
    precip_arr:  array-like
        Array containing the input precipitation field
    precip_thr: float, optional
        Specifies the threshold value for minimum observable precipitation intensity. If None, the
        minimum value over the domain is taken.
    norain_thr: float, optional
        Specifies the threshold value for the fraction of rainy pixels in precip_arr below which we consider there to be
        no rain. Standard set to 0.0
    Returns
    -------
    norain: bool
        Returns whether the fraction of rainy pixels is below the norain_thr threshold.

    """
    warnings.warn(
        "pysteps.blending.utils.check_norain has been deprecated, use pysteps.utils.check_norain.check_norain instead"
    )
    return new_check_norain(precip_arr, precip_thr, norain_thr, None)


def compute_smooth_dilated_mask(
    original_mask,
    max_padding_size_in_px=0,
    gaussian_kernel_size=9,
    inverted=False,
    non_linear_growth_kernel_sizes=False,
):
    """
    Compute a smooth dilated mask using Gaussian blur and dilation with varying kernel sizes.

    Parameters
    ----------
    original_mask : array_like
        Two-dimensional boolean array containing the input mask.
    max_padding_size_in_px : int
        The maximum size of the padding in pixels. Default is 100.
    gaussian_kernel_size : int, optional
        Size of the Gaussian kernel to use for blurring, this should be an uneven number. This option ensures
        that the nan-fields are large enough to start the smoothing. Without it, the method will also be applied
        to local nan-values in the radar domain. Default is 9, which is generally a recommended number to work
        with.
    inverted : bool, optional
        Typically, the smoothed mask works from the outside of the radar domain inward, using the
        max_padding_size_in_px. If set to True, it works from the edge of the radar domain outward
        (generally not recommended). Default is False.
    non_linear_growth_kernel_sizes : bool, optional
        If True, use non-linear growth for kernel sizes. Default is False.

    Returns
    -------
    final_mask : array_like
        The smooth dilated mask normalized to the range [0,1].
    """
    if not CV2_IMPORTED:
        raise MissingOptionalDependency(
            "CV2 package is required to transform the mask into a smoot mask."
            " Please install it using `pip install opencv-python`."
        )

    if max_padding_size_in_px < 0:
        raise ValueError("max_padding_size_in_px must be greater than or equal to 0.")

    # Check if gaussian_kernel_size is an uneven number
    assert gaussian_kernel_size % 2

    # Convert the original mask to uint8 numpy array and invert if needed
    array_2d = np.array(original_mask, dtype=np.uint8)
    if inverted:
        array_2d = np.bitwise_not(array_2d)

    # Rescale the 2D array values to 0-255 (black or white)
    rescaled_array = array_2d * 255

    # Apply Gaussian blur to the rescaled array
    blurred_image = cv2.GaussianBlur(
        rescaled_array, (gaussian_kernel_size, gaussian_kernel_size), 0
    )

    # Apply binary threshold to negate the blurring effect
    _, binary_image = cv2.threshold(blurred_image, 128, 255, cv2.THRESH_BINARY)

    # Define kernel sizes
    if non_linear_growth_kernel_sizes:
        lin_space = np.linspace(0, np.sqrt(max_padding_size_in_px), 10)
        non_lin_space = np.power(lin_space, 2)
        kernel_sizes = list(set(non_lin_space.astype(np.uint8)))
    else:
        kernel_sizes = np.linspace(0, max_padding_size_in_px, 10, dtype=np.uint8)

    # Process each kernel size
    final_mask = np.zeros_like(binary_image, dtype=np.float64)
    for kernel_size in kernel_sizes:
        if kernel_size == 0:
            dilated_image = binary_image
        else:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            dilated_image = cv2.dilate(binary_image, kernel)

        # Convert the dilated image to a binary array
        _, binary_array = cv2.threshold(dilated_image, 128, 1, cv2.THRESH_BINARY)
        final_mask += binary_array

    final_mask = final_mask / final_mask.max()

    return final_mask
