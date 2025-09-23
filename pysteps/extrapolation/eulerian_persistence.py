import numpy as np
import xarray as xr
from pysteps.xarray_helpers import convert_output_to_xarray_dataset


def extrapolate(precip_dataset: xr.Dataset, timesteps, outval=np.nan, **kwargs):
    """
    A dummy extrapolation method to apply Eulerian persistence to a
    two-dimensional precipitation field. The method returns the a sequence
    of the same initial field with no extrapolation applied (i.e. Eulerian
    persistence).

    Parameters
    ----------
    precip_dataset : xarray.Dataset
        xarray dataset containing the input precipitation field. All
        values are required to be finite.
    timesteps : int or list of floats
        Number of time steps or a list of time steps.
    outval : float, optional
        Not used by the method.

    Other Parameters
    ----------------
    return_displacement : bool
        If True, return the total advection velocity (displacement) between the
        initial input field and the advected one integrated along
        the trajectory. Default : False

    Returns
    -------
    out : array or tuple
        If return_displacement=False, return a sequence of the same initial field
        of shape (num_timesteps,m,n). Otherwise, return a tuple containing the
        replicated fields and a (2,m,n) array of zeros.

    References
    ----------
    :cite:`GZ2002`

    """
    del outval  # Unused by _eulerian_persistence
    precip_var = precip_dataset.attrs["precip_var"]
    precip = precip_dataset[precip_var].values[-1]

    if isinstance(timesteps, int):
        num_timesteps = timesteps
    else:
        num_timesteps = len(timesteps)

    return_displacement = kwargs.get("return_displacement", False)

    extrapolated_precip = np.repeat(precip[np.newaxis, :, :], num_timesteps, axis=0)
    extrapolated_precip_dataset = convert_output_to_xarray_dataset(
        precip_dataset, timesteps, extrapolated_precip
    )

    if not return_displacement:
        return extrapolated_precip_dataset
    else:
        return extrapolated_precip_dataset, np.zeros((2,) + extrapolated_precip.shape)
