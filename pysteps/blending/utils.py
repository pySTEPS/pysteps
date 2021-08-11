"""
pysteps.blending.utils
======================

Module with common utilities used by blending methods.

.. autosummary::
    :toctree: ../generated/

    stack_cascades
    recompose_cascade
"""

import numpy as np
from datetime import datetime, timedelta
from pysteps import cascade
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps import utils
import os
import netCDF4


def stack_cascades(R_d, donorm=True):
    """Stack the given cascades into a larger array.

    Parameters
    ----------
    R_d : list
      List of cascades obtained by calling a method implemented in
      pysteps.cascade.decomposition.
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
        mu_ = np.ones(n_levels) * 0.0
        sigma_ = np.ones(n_levels) * 1.0
        if donorm:
            mu_ = np.asarray(cascade["means"])
            sigma_ = np.asarray(cascade["stds"])
        for j in range(n_levels):
            R__ = (R_i[j, :, :] - mu_[j]) / sigma_[j]
            R_.append(R__)
        R_c.append(np.stack(R_))
        mu_c.append(mu_)
        sigma_c.append(sigma_)
    return np.stack(R_c), np.stack(mu_c), np.stack(sigma_c)


def blend_optical_flows(flows, weights):
    """Combine advection fields using given weights.

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
    out: ndarray_
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
    all_c_wn = weights * np.moveaxis(flows, 0, -1)
    # sum uses last axis
    combined_flows = np.sum(all_c_wn, axis=-1)
    # combined_flows [2, m, n]
    return combined_flows


def decompose_NWP(
    NWP_output,
    R_NWP,
    start_time,
    timestep,
    num_cascade_levels,
    decomp_method="fft",
    fft_method="numpy",
    domain="spatial",
    normalize=True,
    compute_stats=True,
    compact_output=True,
):

    # Make a NetCDF file
    outfn = os.path.join(NWP_output, "NWP_cascade" + ".nc")
    ncf = netCDF4.Dataset(outfn, "w", format="NETCDF4")

    # Set attributes of decomposition method
    ncf.domain = domain
    ncf.normalized = int(normalize)
    ncf.compact_output = int(compact_output)
    ncf.start_time = start_time.strftime("%Y%m%d%H%M%S")
    ncf.timestep = timestep

    # Create dimensions
    time_dim = ncf.createDimension('time', R_NWP.shape[0])
    casc_dim = ncf.createDimension('cascade levels', num_cascade_levels)
    x_dim = ncf.createDimension('x', R_NWP.shape[1])
    y_dim = ncf.createDimension('y', R_NWP.shape[2])
    means_dim = ncf.createDimension('means', num_cascade_levels)
    stds_dim = ncf.createDimension('stds', num_cascade_levels)

    # Create variable (decomposed cascade)
    R_d = ncf.createVariable('R_d', np.float64, ('time', 'cascade levels', 'x', 'y'))
    means = ncf.createVariable('means', np.float64, ('time', 'means'))
    stds = ncf.createVariable('stds', np.float64, ('time', 'stds'))

    # Decompose the NWP data
    filter = filter_gaussian(R_NWP.shape[1:], num_cascade_levels)
    fft = utils.get_method(fft_method, shape=R_NWP.shape[1:], n_threads=1)
    decomp_method, _ = cascade.get_method(decomp_method)

    for i in range(R_NWP.shape[0]):
        R_ = decomp_method(
            R_NWP[i, :, :],
            filter,
            fft_method=fft,
            output_domain=domain,
            normalize=normalize,
            compute_stats=compute_stats,
            compact_output=compact_output,
        )

        # Save data to netCDF file
        R_d[i, :, :, :] = R_["cascade_levels"]
        means[i, :] = R_["means"]
        stds[i, :] = R_["stds"]

    ncf.close()


def load_NWP(NWP_output, analysis_time, n_timesteps):
    outfn = os.path.join(NWP_output, "NWP_cascade" + ".nc")
    ncf = netCDF4.Dataset(outfn, "r", format="NETCDF4")

    decomp_dict = dict()
    decomp_dict["domain"] = ncf.domain
    decomp_dict["normalized"] = bool(ncf.normalized)
    decomp_dict["compact_output"] = bool(ncf.compact_output)

    start_time = ncf.start_time
    start_time = datetime.strptime(start_time, "%Y%m%d%H%M%S")
    timestep = ncf.timestep
    timestep = timedelta(minutes=int(timestep))

    start_i = (analysis_time - start_time) // timestep + 1
    end_i = start_i + n_timesteps

    R_d = list()

    for i in range(start_i, end_i):
        decomp_dict_ = decomp_dict.copy()
        cascade_levels = ncf.variables["R_d"][i, :, :, :]
        assert not cascade_levels.mask
        decomp_dict_["cascade_levels"] = np.ma.filled(cascade_levels, np.nan)
        R_d.append(decomp_dict_)

    return R_d
