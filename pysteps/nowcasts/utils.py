"""
pysteps.nowcasts.utils
======================

Module with common utilities used by nowcasts methods.

.. autosummary::
    :toctree: ../generated/

    binned_timesteps
    compute_dilated_mask
    compute_percentile_mask
    nowcast_main_loop
    print_ar_params
    print_corrcoefs
    stack_cascades
"""

import time
import numpy as np
import scipy.ndimage
from pysteps import extrapolation


try:
    import dask

    DASK_IMPORTED = True
except ImportError:
    DASK_IMPORTED = False


def binned_timesteps(timesteps):
    """
    Compute a binning of the given irregular time steps.

    Parameters
    ----------
    timesteps: array_like
        List or one-dimensional array containing the time steps in ascending
        order.

    Returns
    -------
    out: list
        List of length int(np.ceil(timesteps[-1]))+1 containing the bins. Each
        element is a list containing the indices of the time steps falling in
        the bin (excluding the right edge).
    """
    timesteps = list(timesteps)
    if not sorted(timesteps) == timesteps:
        raise ValueError("timesteps is not in ascending order")

    if np.any(np.array(timesteps) < 0):
        raise ValueError("negative time steps are not allowed")

    num_bins = int(np.ceil(timesteps[-1]))
    timestep_range = np.arange(num_bins + 1)
    bin_idx = np.digitize(timesteps, timestep_range, right=False)

    out = [[] for _ in range(num_bins + 1)]
    for i, bi in enumerate(bin_idx):
        out[bi - 1].append(i)

    return out


def compute_dilated_mask(input_mask, kr, r):
    """Buffer the input rain mask using the given kernel. Add a grayscale rim
    for smooth rain/no-rain transition by iteratively dilating the mask.

    Parameters
    ----------
    input_mask : array_like
        Two-dimensional boolean array containing the input mask.
    kr : array_like
        Structuring element for the dilation.
    r : int
        The number of iterations for the dilation.

    Returns
    -------
    out : array_like
        The dilated mask normalized to the range [0,1].
    """
    # buffer the input mask
    input_mask = np.ndarray.astype(input_mask.copy(), "uint8")
    mask_dilated = scipy.ndimage.morphology.binary_dilation(input_mask, kr)

    # add grayscale rim
    kr1 = scipy.ndimage.generate_binary_structure(2, 1)
    mask = mask_dilated.astype(float)
    for _ in range(r):
        mask_dilated = scipy.ndimage.morphology.binary_dilation(mask_dilated, kr1)
        mask += mask_dilated

    # normalize between 0 and 1
    return mask / mask.max()


def compute_percentile_mask(precip, pct):
    """Compute a precipitation mask, where True/False values are assigned for
    pixels above/below the given percentile.

    Parameters
    ----------
    precip: array_like
        Two-dimensional array of shape (m,n) containing the input precipitation
        field.
    pct: float
        The percentile value.

    Returns
    -------
    out: ndarray_
        Array of shape (m,n), where True/False values are assigned for pixels
        above/below the precipitation intensity corresponding to the given
        percentile.
    """
    # obtain the CDF from the input precipitation field
    precip_s = precip.flatten()

    # compute the precipitation intensity threshold corresponding to the given
    # percentile
    precip_s.sort(kind="quicksort")
    x = 1.0 * np.arange(1, len(precip_s) + 1)[::-1] / len(precip_s)
    i = np.argmin(np.abs(x - pct))
    # handle ties
    if precip_s[i] == precip_s[i + 1]:
        i = np.where(precip_s == precip_s[i])[0][-1]
    precip_pct_thr = precip_s[i]

    # determine the mask using the above threshold value
    return precip >= precip_pct_thr


def nowcast_main_loop(
    precip,
    velocity,
    state,
    timesteps,
    extrap_method,
    func,
    extrap_kwargs=None,
    velocity_pert_gen=None,
    params=None,
    ensemble=False,
    num_ensemble_members=1,
    callback=None,
    return_output=True,
    num_workers=1,
    measure_time=False,
):
    """Utility method for advection-based nowcast models that are applied in
    the Lagrangian coordinates. In addition, this method allows the case, where
    one or more components of the model (e.g. an autoregressive process) require
    using regular integer time steps but the user-supplied values are irregular
    or non-integer.

    Parameters
    ----------
    precip : array_like
        Array of shape (m,n) containing the most recently observed precipitation
        field.
    velocity : array_like
        Array of shape (2,m,n) containing the x- and y-components of the
        advection field.
    state : object
        The initial state of the nowcast model.
    timesteps : int or list of floats
        Number of time steps to forecast or a list of time steps for which the
        forecasts are computed. The elements of the list are required to be in
        ascending order.
    extrap_method : str, optional
        Name of the extrapolation method to use. See the documentation of
        :py:mod:`pysteps.extrapolation.interface`.
    ensemble : bool
        Set to True to produce a nowcast ensemble.
    num_ensemble_members : int
        Number of ensemble members. Applicable if ensemble is set to True.
    func : function
        A function that takes the current state of the nowcast model and its
        parameters and returns a forecast field and the new state. The shape of
        the forecast field is expected to be (m,n) for a deterministic nowcast
        and (n_ens_members,m,n) for an ensemble.
    extrap_kwargs : dict, optional
        Optional dictionary containing keyword arguments for the extrapolation
        method. See the documentation of pysteps.extrapolation.
    velocity_pert_gen : list, optional
        Optional list of functions that generate velocity perturbations. The
        length of the list is expected to be n_ens_members. The functions
        are expected to take lead time (relative to timestep index) as input
        argument and return a perturbation field of shape (2,m,n).
    params : dict, optional
        Optional dictionary containing keyword arguments for func.
    callback : function, optional
        Optional function that is called after computation of each time step of
        the nowcast. The function takes one argument: the nowcast array. This
        can be used, for instance, writing output files.
    return_output : bool, optional
        Set to False to disable returning the output forecast fields and return
        None instead. This can save memory if the intermediate results are
        instead written to files using the callback function.
    num_workers : int, optional
        Number of parallel workers to use. Applicable if a nowcast ensemble is
        generated.
    measure_time : bool, optional
        If set to True, measure, print and return the computation time.

    Returns
    -------
    out : list
        List of forecast fields for the given time steps. If measure_time is
        True, return a pair, where the second element is the total computation
        time in the loop.
    """
    precip_forecast_out = None

    # create a range of time steps
    # if an integer time step is given, create a simple range iterator
    # otherwise, assing the time steps to integer bins so that each bin
    # contains a list of time steps belonging to that bin
    if isinstance(timesteps, int):
        timesteps = range(timesteps + 1)
        timestep_type = "int"
    else:
        original_timesteps = [0] + list(timesteps)
        timesteps = binned_timesteps(original_timesteps)
        timestep_type = "list"

    state_cur = state
    if not ensemble:
        precip_forecast_prev = precip[np.newaxis, :]
    else:
        precip_forecast_prev = np.stack([precip for _ in range(num_ensemble_members)])
    displacement = None
    t_prev = 0.0
    t_total = 0.0

    # initialize the extrapolator
    extrapolator = extrapolation.get_method(extrap_method)

    x_values, y_values = np.meshgrid(
        np.arange(precip.shape[1]), np.arange(precip.shape[0])
    )

    xy_coords = np.stack([x_values, y_values])

    if extrap_kwargs is None:
        extrap_kwargs = dict()
    else:
        extrap_kwargs = extrap_kwargs.copy()
    extrap_kwargs["xy_coords"] = xy_coords
    extrap_kwargs["return_displacement"] = True

    if measure_time:
        starttime_total = time.time()

    # loop through the integer time steps or bins if non-integer time steps
    # were given
    for t, subtimestep_idx in enumerate(timesteps):
        if timestep_type == "list":
            subtimesteps = [original_timesteps[t_] for t_ in subtimestep_idx]
        else:
            subtimesteps = [t]

        if (timestep_type == "list" and subtimesteps) or (
            timestep_type == "int" and t > 0
        ):
            is_nowcast_time_step = True
        else:
            is_nowcast_time_step = False

        # print a message if nowcasts are computed for the current integer time
        # step (this is not necessarily the case, since the current bin might
        # not contain any time steps)
        if is_nowcast_time_step:
            print(
                f"Computing nowcast for time step {t}... ",
                end="",
                flush=True,
            )

            if measure_time:
                starttime = time.time()

        # call the function to iterate the integer-timestep part of the model
        # for one time step
        precip_forecast_new, state_new = func(state_cur, params)

        if not ensemble:
            precip_forecast_new = precip_forecast_new[np.newaxis, :]

        # advect the currect forecast field to the subtimesteps in the current
        # timestep bin and append the results to the output list
        # apply temporal interpolation to the forecasts made between the
        # previous and the next integer time steps
        for t_sub in subtimesteps:
            if t_sub > 0:
                t_diff_prev_int = t_sub - int(t_sub)
                if t_diff_prev_int > 0.0:
                    precip_forecast_ip = (
                        1.0 - t_diff_prev_int
                    ) * precip_forecast_prev + t_diff_prev_int * precip_forecast_new
                else:
                    precip_forecast_ip = precip_forecast_prev

                t_diff_prev = t_sub - t_prev
                t_total += t_diff_prev

                if displacement is None:
                    displacement = [None for _ in range(precip_forecast_ip.shape[0])]

                if precip_forecast_out is None and return_output:
                    precip_forecast_out = [
                        [] for _ in range(precip_forecast_ip.shape[0])
                    ]

                precip_forecast_out_cur = [
                    None for _ in range(precip_forecast_ip.shape[0])
                ]

                def worker1(i):
                    extrap_kwargs_ = extrap_kwargs.copy()
                    extrap_kwargs_["displacement_prev"] = displacement[i]
                    extrap_kwargs_["allow_nonfinite_values"] = (
                        True if np.any(~np.isfinite(precip_forecast_ip[i])) else False
                    )

                    if velocity_pert_gen is not None:
                        velocity_ = velocity + velocity_pert_gen[i](t_total)
                    else:
                        velocity_ = velocity

                    precip_forecast_ep, displacement[i] = extrapolator(
                        precip_forecast_ip[i],
                        velocity_,
                        [t_diff_prev],
                        **extrap_kwargs_,
                    )

                    precip_forecast_out_cur[i] = precip_forecast_ep[0]
                    if return_output:
                        precip_forecast_out[i].append(precip_forecast_ep[0])

                if DASK_IMPORTED and ensemble and num_ensemble_members > 1:
                    res = []
                    for i in range(precip_forecast_ip.shape[0]):
                        res.append(dask.delayed(worker1)(i))
                    dask.compute(*res, num_workers=num_workers)
                else:
                    for i in range(precip_forecast_ip.shape[0]):
                        worker1(i)

                if callback is not None:
                    precip_forecast_out_cur = np.stack(precip_forecast_out_cur)
                    callback(precip_forecast_out_cur)

                precip_forecast_out_cur = None
                t_prev = t_sub

        # advect the forecast field by one time step if no subtimesteps in the
        # current interval were found
        if not subtimesteps:
            t_diff_prev = t + 1 - t_prev
            t_total += t_diff_prev

            if displacement is None:
                displacement = [None for _ in range(precip_forecast_new.shape[0])]

            def worker2(i):
                extrap_kwargs_ = extrap_kwargs.copy()
                extrap_kwargs_["displacement_prev"] = displacement[i]

                if velocity_pert_gen is not None:
                    velocity_ = velocity + velocity_pert_gen[i](t_total)
                else:
                    velocity_ = velocity

                _, displacement[i] = extrapolator(
                    None,
                    velocity_,
                    [t_diff_prev],
                    **extrap_kwargs_,
                )

            if DASK_IMPORTED and ensemble and num_ensemble_members > 1:
                res = []
                for i in range(precip_forecast_new.shape[0]):
                    res.append(dask.delayed(worker2)(i))
                dask.compute(*res, num_workers=num_workers)
            else:
                for i in range(precip_forecast_new.shape[0]):
                    worker2(i)

            t_prev = t + 1

        precip_forecast_prev = precip_forecast_new
        state_cur = state_new

        if is_nowcast_time_step:
            if measure_time:
                print(f"{time.time() - starttime:.2f} seconds.")
            else:
                print("done.")

    if return_output:
        precip_forecast_out = np.stack(precip_forecast_out)
        if not ensemble:
            precip_forecast_out = precip_forecast_out[0, :]

    if measure_time:
        return precip_forecast_out, time.time() - starttime_total
    else:
        return precip_forecast_out


def print_ar_params(phi):
    """
    Print the parameters of an AR(p) model.

    Parameters
    ----------
    phi: array_like
        Array of shape (n, p) containing the AR(p) parameters for n cascade
        levels.
    """
    print("****************************************")
    print("* AR(p) parameters for cascade levels: *")
    print("****************************************")

    n = phi.shape[1]

    hline_str = "---------"
    for _ in range(n):
        hline_str += "---------------"

    title_str = "| Level |"
    for i in range(n - 1):
        title_str += "    Phi-%d     |" % (i + 1)
    title_str += "    Phi-0     |"

    print(hline_str)
    print(title_str)
    print(hline_str)

    fmt_str = "| %-5d |"
    for _ in range(n):
        fmt_str += " %-12.6f |"

    for i in range(phi.shape[0]):
        print(fmt_str % ((i + 1,) + tuple(phi[i, :])))
        print(hline_str)


def print_corrcoefs(gamma):
    """
    Print the parameters of an AR(p) model.

    Parameters
    ----------
    gamma: array_like
      Array of shape (m, n) containing n correlation coefficients for m cascade
      levels.
    """
    print("************************************************")
    print("* Correlation coefficients for cascade levels: *")
    print("************************************************")

    m = gamma.shape[0]
    n = gamma.shape[1]

    hline_str = "---------"
    for _ in range(n):
        hline_str += "----------------"

    title_str = "| Level |"
    for i in range(n):
        title_str += "     Lag-%d     |" % (i + 1)

    print(hline_str)
    print(title_str)
    print(hline_str)

    fmt_str = "| %-5d |"
    for _ in range(n):
        fmt_str += " %-13.6f |"

    for i in range(m):
        print(fmt_str % ((i + 1,) + tuple(gamma[i, :])))
        print(hline_str)


def stack_cascades(precip_decomp, n_levels, convert_to_full_arrays=False):
    """
    Stack the given cascades into a larger array.

    Parameters
    ----------
    precip_decomp: list
        List of cascades obtained by calling a method implemented in
        pysteps.cascade.decomposition.
    n_levels: int
        The number of cascade levels.

    Returns
    -------
    out: tuple
        A list of three-dimensional arrays containing the stacked cascade levels.
    """
    out = []

    n_inputs = len(precip_decomp)

    for i in range(n_levels):
        precip_cur_level = []
        for j in range(n_inputs):
            precip_cur_input = precip_decomp[j]["cascade_levels"][i]
            if precip_decomp[j]["compact_output"] and convert_to_full_arrays:
                precip_tmp = np.zeros(
                    precip_decomp[j]["weight_masks"].shape[1:], dtype=complex
                )
                precip_tmp[precip_decomp[j]["weight_masks"][i]] = precip_cur_input
                precip_cur_input = precip_tmp
            precip_cur_level.append(precip_cur_input)
        out.append(np.stack(precip_cur_level))

    if not np.any(
        [precip_decomp[i]["compact_output"] for i in range(len(precip_decomp))]
    ):
        out = np.stack(out)

    return out
