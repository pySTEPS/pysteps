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
"""

import datetime
from pathlib import Path

import numpy as np

from pysteps.cascade import get_method as cascade_get_method
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps.exceptions import MissingOptionalDependency
from pysteps.utils import get_method as utils_get_method

try:
    import netCDF4

    NETCDF4_IMPORTED = True
except ImportError:
    NETCDF4_IMPORTED = False


def stack_cascades(R_d, donorm=True):
    """Stack the given cascades into a larger array.

    Parameters
    ----------
    R_d : dict
      Dictionary containing a list of cascades obtained by calling a method
      implemented in pysteps.cascade.decomposition.
    donorm : bool
      If True, normalize the cascade levels before stacking.

    Returns
    -------
    out : tuple
      A three-element tuple containing a four-dimensional array of stacked
      cascade levels and arrays of mean values and standard deviations for each
      cascade level.
    """
    R_c = []
    mu_c = []
    sigma_c = []

    for cascade in R_d:
        R_ = []
        R_i = cascade["cascade_levels"]
        n_levels = R_i.shape[0]
        mu_ = np.asarray(cascade["means"])
        sigma_ = np.asarray(cascade["stds"])
        if donorm:
            for j in range(n_levels):
                R__ = (R_i[j, :, :] - mu_[j]) / sigma_[j]
                R_.append(R__)
        else:
            R_ = R_i
        R_c.append(np.stack(R_))
        mu_c.append(mu_)
        sigma_c.append(sigma_)
    return np.stack(R_c), np.stack(mu_c), np.stack(sigma_c)


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
    R_NWP,
    NWP_model,
    analysis_time,
    timestep,
    valid_times,
    output_path,
    num_cascade_levels=8,
    num_workers=1,
    decomp_method="fft",
    fft_method="numpy",
    domain="spatial",
    normalize=True,
    compute_stats=True,
    compact_output=True,
):
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
    timestep: int
      Timestep in minutes between subsequent NWP forecast fields
    valid_times: array_like
      Array containing the valid times of the NWP forecast fields. The times are
      assumed to be numpy.datetime64 types as imported by the pysteps importer.
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
    None
    """

    if not NETCDF4_IMPORTED:
        raise MissingOptionalDependency(
            "netCDF4 package is required to save the decomposed NWP data, "
            "but it is not installed"
        )

    # Make a NetCDF file
    output_date = f"{analysis_time.astype('datetime64[us]').astype(datetime.datetime):%Y%m%d%H%M%S}"
    outfn = Path(output_path) / f"cascade_{NWP_model}_{output_date}.nc"
    ncf = netCDF4.Dataset(outfn, "w", format="NETCDF4")

    # Express times relative to the zero time
    zero_time = np.datetime64("1970-01-01T00:00:00", "ns")
    valid_times = np.array(valid_times) - zero_time
    analysis_time = analysis_time - zero_time

    # Set attributes of decomposition method
    ncf.domain = domain
    ncf.normalized = int(normalize)
    ncf.compact_output = int(compact_output)
    ncf.analysis_time = int(analysis_time)
    ncf.timestep = int(timestep)

    # Create dimensions
    ncf.createDimension("time", R_NWP.shape[0])
    ncf.createDimension("cascade_levels", num_cascade_levels)
    ncf.createDimension("x", R_NWP.shape[2])
    ncf.createDimension("y", R_NWP.shape[1])

    # Create variables (decomposed cascade, means and standard deviations)
    R_d = ncf.createVariable(
        "pr_decomposed",
        np.float32,
        ("time", "cascade_levels", "y", "x"),
        zlib=True,
        complevel=4,
    )
    means = ncf.createVariable("means", np.float64, ("time", "cascade_levels"))
    stds = ncf.createVariable("stds", np.float64, ("time", "cascade_levels"))
    v_times = ncf.createVariable("valid_times", np.float64, ("time",))
    v_times.units = "nanoseconds since 1970-01-01 00:00:00"

    # The valid times are saved as an array of floats, because netCDF files can't handle datetime types
    v_times[:] = np.array([np.float64(valid_times[i]) for i in range(len(valid_times))])

    # Decompose the NWP data
    filter_g = filter_gaussian(R_NWP.shape[1:], num_cascade_levels)
    fft = utils_get_method(fft_method, shape=R_NWP.shape[1:], n_threads=num_workers)
    decomp_method, _ = cascade_get_method(decomp_method)

    for i in range(R_NWP.shape[0]):
        R_ = decomp_method(
            field=R_NWP[i, :, :],
            bp_filter=filter_g,
            fft_method=fft,
            input_domain=domain,
            output_domain=domain,
            normalize=normalize,
            compute_stats=compute_stats,
            compact_output=compact_output,
        )

        # Save data to netCDF file
        # print(R_["cascade_levels"])
        R_d[i, :, :, :] = R_["cascade_levels"]
        means[i, :] = R_["means"]
        stds[i, :] = R_["stds"]

    # Close the file
    ncf.close()


def compute_store_nwp_motion(
    precip_nwp,
    oflow_method,
    analysis_time,
    nwp_model,
    output_path,
):
    """Computes, per forecast lead time, the velocity field of an NWP model field.

    Parameters
    ----------
    precip_nwp: array-like
      Array of dimension (n_timesteps, x, y) containing the precipitation forecast
      from some NWP model.
    oflow_method: {'constant', 'darts', 'lucaskanade', 'proesmans', 'vet'}, optional
      An optical flow method from pysteps.motion.get_method.
    analysis_time: numpy.datetime64
      The analysis time of the NWP forecast. The analysis time is assumed to be a
      numpy.datetime64 type as imported by the pysteps importer.
    nwp_model: str
      The name of the NWP model.
    output_path: str, optional
      The location where to save the file with the NWP velocity fields. Defaults
      to the path_workdir specified in the rcparams file.

    Returns
    -------
    Nothing
    """

    # Set the output file
    output_date = f"{analysis_time.astype('datetime64[us]').astype(datetime.datetime):%Y%m%d%H%M%S}"
    outfn = Path(output_path) / f"motion_{nwp_model}_{output_date}.npy"

    # Get the velocity field per time step
    v_nwp = np.zeros((precip_nwp.shape[0], 2, precip_nwp.shape[1], precip_nwp.shape[2]))
    # Loop through the timesteps. We need two images to construct a motion
    # field, so we can start from timestep 1.
    for t in range(1, precip_nwp.shape[0]):
        v_nwp[t] = oflow_method(precip_nwp[t - 1 : t + 1, :, :])

    # Make timestep 0 the same as timestep 1.
    v_nwp[0] = v_nwp[1]

    assert v_nwp.ndim == 4, "v_nwp must be a four-dimensional array"

    # Save it as a numpy array
    np.save(outfn, v_nwp)


def load_NWP(input_nc_path_decomp, input_path_velocities, start_time, n_timesteps):
    """Loads the decomposed NWP and velocity data from the netCDF files

    Parameters
    ----------
    input_nc_path_decomp: str
      Path to the saved netCDF file containing the decomposed NWP data.
    input_path_velocities: str
        Path to the saved numpy binary file containing the estimated velocity
        fields from the NWP data.
    start_time: numpy.datetime64
      The start time of the nowcasting. Assumed to be a numpy.datetime64 type
    n_timesteps: int
      Number of time steps to forecast

    Returns
    -------
    R_d: list
      A list of dictionaries with each element in the list corresponding to
      a different time step. Each dictionary has the same structure as the
      output of the decomposition function
    uv: array-like
        Array of shape (timestep,2,m,n) containing the x- and y-components
      of the advection field for the (NWP) model field per forecast lead time.
    """

    if not NETCDF4_IMPORTED:
        raise MissingOptionalDependency(
            "netCDF4 package is required to load the decomposed NWP data, "
            "but it is not installed"
        )

    # Open the file
    ncf_decomp = netCDF4.Dataset(input_nc_path_decomp, "r", format="NETCDF4")
    velocities = np.load(input_path_velocities)

    # Initialise the decomposition dictionary
    decomp_dict = dict()
    decomp_dict["domain"] = ncf_decomp.domain
    decomp_dict["normalized"] = bool(ncf_decomp.normalized)
    decomp_dict["compact_output"] = bool(ncf_decomp.compact_output)

    # Convert the start time and the timestep to datetime64 and timedelta64 type
    zero_time = np.datetime64("1970-01-01T00:00:00", "ns")
    analysis_time = np.timedelta64(int(ncf_decomp.analysis_time), "ns") + zero_time

    timestep = ncf_decomp.timestep
    timestep = np.timedelta64(timestep, "m")

    valid_times = ncf_decomp.variables["valid_times"][:]
    valid_times = np.array(
        [np.timedelta64(int(valid_times[i]), "ns") for i in range(len(valid_times))]
    )
    valid_times = valid_times + zero_time

    # Find the indices corresponding with the required start and end time
    start_i = (start_time - analysis_time) // timestep
    assert analysis_time + start_i * timestep == start_time
    end_i = start_i + n_timesteps + 1

    # Add the valid times to the output
    decomp_dict["valid_times"] = valid_times[start_i:end_i]

    # Slice the velocity fields with the start and end indices
    uv = velocities[start_i:end_i, :, :, :]

    # Initialise the list of dictionaries which will serve as the output (cf: the STEPS function)
    R_d = list()

    for i in range(start_i, end_i):
        decomp_dict_ = decomp_dict.copy()

        cascade_levels = ncf_decomp.variables["pr_decomposed"][i, :, :, :]

        # In the netcdf file this is saved as a masked array, so we're checking if there is no mask
        assert not cascade_levels.mask

        means = ncf_decomp.variables["means"][i, :]
        assert not means.mask

        stds = ncf_decomp.variables["stds"][i, :]
        assert not stds.mask

        # Save the values in the dictionary as normal arrays with the filled method
        decomp_dict_["cascade_levels"] = np.ma.filled(cascade_levels)
        decomp_dict_["means"] = np.ma.filled(means)
        decomp_dict_["stds"] = np.ma.filled(stds)

        # Append the output list
        R_d.append(decomp_dict_)

    return R_d, uv
