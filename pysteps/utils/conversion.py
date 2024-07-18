# -*- coding: utf-8 -*-
"""
pysteps.utils.conversion
========================

Methods for converting physical units.

.. autosummary::
    :toctree: ../generated/

    to_rainrate
    to_raindepth
    to_reflectivity
"""

import warnings

import xarray as xr

from . import transformation

# TODO: This should not be done. Instead fix the code so that it doesn't
# produce the warnings.
# to deactivate warnings for comparison operators with NaNs
warnings.filterwarnings("ignore", category=RuntimeWarning)


def cf_parameters_from_unit(unit: str) -> tuple[str, dict[str, str | None]]:
    if unit == "mm/h":
        var_name = "precip_intensity"
        var_standard_name = None
        var_long_name = "instantaneous precipitation rate"
        var_unit = "mm/h"
    elif unit == "mm":
        var_name = "precip_accum"
        var_standard_name = None
        var_long_name = "accumulated precipitation"
        var_unit = "mm"
    elif unit == "dBZ":
        var_name = "reflectivity"
        var_long_name = "equivalent reflectivity factor"
        var_standard_name = "equivalent_reflectivity_factor"
        var_unit = "dBZ"
    else:
        raise ValueError(f"unknown unit {unit}")

    return var_name, {
        "standard_name": var_standard_name,
        "long_name": var_long_name,
        "units": var_unit,
    }


def _change_unit(dataset: xr.Dataset, precip_var: str, new_unit: str) -> xr.Dataset:
    new_var, new_attrs = cf_parameters_from_unit(new_unit)
    dataset = dataset.rename_vars({precip_var: new_var})
    dataset.attrs["precip_var"] = new_var

    dataset[new_var].attrs = {
        **dataset[new_var].attrs,
        **new_attrs,
    }

    return dataset


def to_rainrate(dataset: xr.Dataset, zr_a=None, zr_b=None):
    """
    Convert to rain rate [mm/h].

    Parameters
    ----------
    dataset: Dataset
        Dataset to be (back-)transformed.

        Additionally, in case of conversion to/from reflectivity units, the
        zr_a and zr_b attributes are also required,
        but only if zr_a = zr_b = None.
        If missing, it defaults to Marshall–Palmer relation,
        that is, zr_a = 200.0 and zr_b = 1.6.
    zr_a, zr_b: float, optional
        The a and b coefficients of the Z-R relationship (Z = a*R^b).

    Returns
    -------
    dataset: Dataset
        Dataset containing the converted units.
    """

    dataset = dataset.copy(deep=True)
    precip_var = dataset.attrs["precip_var"]
    metadata = dataset[precip_var].attrs

    if metadata["transform"] is not None:
        if metadata["transform"] == "dB":
            dataset = transformation.dB_transform(dataset, inverse=True)

        elif metadata["transform"] in ["BoxCox", "log"]:
            dataset = transformation.boxcox_transform(dataset, inverse=True)

        elif metadata["transform"] == "NQT":
            dataset = transformation.NQ_transform(dataset, inverse=True)

        elif metadata["transform"] == "sqrt":
            dataset = transformation.sqrt_transform(dataset, inverse=True)

        else:
            raise ValueError(f'Unknown transformation {metadata["transform"]}')

    precip_var = dataset.attrs["precip_var"]
    metadata = dataset[precip_var].attrs
    precip_data = dataset[precip_var].values

    if metadata["units"] == "mm/h":
        pass

    elif metadata["units"] == "mm":
        threshold = metadata["threshold"]  # convert the threshold, too
        zerovalue = metadata["zerovalue"]  # convert the zerovalue, too

        precip_data = precip_data / float(metadata["accutime"]) * 60.0
        threshold = threshold / float(metadata["accutime"]) * 60.0
        zerovalue = zerovalue / float(metadata["accutime"]) * 60.0

        metadata["threshold"] = threshold
        metadata["zerovalue"] = zerovalue

    elif metadata["units"] == "dBZ":
        threshold = metadata["threshold"]  # convert the threshold, too
        zerovalue = metadata["zerovalue"]  # convert the zerovalue, too

        # Z to R
        if zr_a is None:
            zr_a = metadata.get("zr_a", 200.0)  # default to Marshall–Palmer
        if zr_b is None:
            zr_b = metadata.get("zr_b", 1.6)  # default to Marshall–Palmer
        precip_data = (precip_data / zr_a) ** (1.0 / zr_b)
        threshold = (threshold / zr_a) ** (1.0 / zr_b)
        zerovalue = (zerovalue / zr_a) ** (1.0 / zr_b)

        metadata["zr_a"] = zr_a
        metadata["zr_b"] = zr_b
        metadata["threshold"] = threshold
        metadata["zerovalue"] = zerovalue

    else:
        raise ValueError(
            f'Cannot convert unit {metadata["units"]} and transform {metadata["transform"]} to mm/h'
        )

    dataset[precip_var].data[:] = precip_data
    dataset = _change_unit(dataset, precip_var, "mm/h")
    return dataset


def to_raindepth(dataset: xr.Dataset, zr_a=None, zr_b=None):
    """
    Convert to rain depth [mm].

    Parameters
    ----------
    dataset: Dataset
        Dataset to be (back-)transformed.

        Additionally, in case of conversion to/from reflectivity units, the
        zr_a and zr_b attributes are also required,
        but only if zr_a = zr_b = None.
        If missing, it defaults to Marshall–Palmer relation, that is,
        zr_a = 200.0 and zr_b = 1.6.
    zr_a, zr_b: float, optional
        The a and b coefficients of the Z-R relationship (Z = a*R^b).

    Returns
    -------
    dataset: Dataset
        Dataset containing the converted units.
    """

    dataset = dataset.copy(deep=True)
    precip_var = dataset.attrs["precip_var"]
    metadata = dataset[precip_var].attrs

    if metadata["transform"] is not None:
        if metadata["transform"] == "dB":
            dataset = transformation.dB_transform(dataset, inverse=True)

        elif metadata["transform"] in ["BoxCox", "log"]:
            dataset = transformation.boxcox_transform(dataset, inverse=True)

        elif metadata["transform"] == "NQT":
            dataset = transformation.NQ_transform(dataset, inverse=True)

        elif metadata["transform"] == "sqrt":
            dataset = transformation.sqrt_transform(dataset, inverse=True)

        else:
            raise ValueError(f'Unknown transformation {metadata["transform"]}')

    precip_var = dataset.attrs["precip_var"]
    metadata = dataset[precip_var].attrs
    precip_data = dataset[precip_var].values

    if metadata["units"] == "mm" and metadata["transform"] is None:
        pass

    elif metadata["units"] == "mm/h":
        threshold = metadata["threshold"]  # convert the threshold, too
        zerovalue = metadata["zerovalue"]  # convert the zerovalue, too

        precip_data = precip_data / 60.0 * metadata["accutime"]
        threshold = threshold / 60.0 * metadata["accutime"]
        zerovalue = zerovalue / 60.0 * metadata["accutime"]

        metadata["threshold"] = threshold
        metadata["zerovalue"] = zerovalue

    elif metadata["units"] == "dBZ":
        threshold = metadata["threshold"]  # convert the threshold, too
        zerovalue = metadata["zerovalue"]  # convert the zerovalue, too

        # Z to R
        if zr_a is None:
            zr_a = metadata.get("zr_a", 200.0)  # Default to Marshall–Palmer
        if zr_b is None:
            zr_b = metadata.get("zr_b", 1.6)  # Default to Marshall–Palmer
        precip_data = (precip_data / zr_a) ** (1.0 / zr_b) / 60.0 * metadata["accutime"]
        threshold = (threshold / zr_a) ** (1.0 / zr_b) / 60.0 * metadata["accutime"]
        zerovalue = (zerovalue / zr_a) ** (1.0 / zr_b) / 60.0 * metadata["accutime"]

        metadata["zr_a"] = zr_a
        metadata["zr_b"] = zr_b
        metadata["threshold"] = threshold
        metadata["zerovalue"] = zerovalue

    else:
        raise ValueError(
            f'Cannot convert unit {metadata["units"]} and transform {metadata["transform"]} to mm'
        )

    dataset[precip_var].data[:] = precip_data
    dataset = _change_unit(dataset, precip_var, "mm")
    return dataset


def to_reflectivity(dataset: xr.Dataset, zr_a=None, zr_b=None):
    """
    Convert to reflectivity [dBZ].

    Parameters
    ----------
    dataset: Dataset
        Dataset to be (back-)transformed.

        Additionally, in case of conversion to/from reflectivity units, the
        zr_a and zr_b attributes are also required,
        but only if zr_a = zr_b = None.
        If missing, it defaults to Marshall–Palmer relation, that is,
        zr_a = 200.0 and zr_b = 1.6.
    zr_a, zr_b: float, optional
        The a and b coefficients of the Z-R relationship (Z = a*R^b).

    Returns
    -------
    dataset: Dataset
        Dataset containing the converted units.
    """

    dataset = dataset.copy(deep=True)
    precip_var = dataset.attrs["precip_var"]
    metadata = dataset[precip_var].attrs

    if metadata["transform"] is not None:
        if metadata["transform"] == "dB":
            dataset = transformation.dB_transform(dataset, inverse=True)

        elif metadata["transform"] in ["BoxCox", "log"]:
            dataset = transformation.boxcox_transform(dataset, inverse=True)

        elif metadata["transform"] == "NQT":
            dataset = transformation.NQ_transform(dataset, inverse=True)

        elif metadata["transform"] == "sqrt":
            dataset = transformation.sqrt_transform(dataset, inverse=True)

        else:
            raise ValueError(f'Unknown transformation {metadata["transform"]}')

    precip_var = dataset.attrs["precip_var"]
    metadata = dataset[precip_var].attrs
    precip_data = dataset[precip_var].values

    if metadata["units"] == "mm/h":
        # Z to R
        if zr_a is None:
            zr_a = metadata.get("zr_a", 200.0)  # Default to Marshall–Palmer
        if zr_b is None:
            zr_b = metadata.get("zr_b", 1.6)  # Default to Marshall–Palmer

        precip_data = zr_a * precip_data**zr_b
        metadata["threshold"] = zr_a * metadata["threshold"] ** zr_b
        metadata["zerovalue"] = zr_a * metadata["zerovalue"] ** zr_b
        metadata["zr_a"] = zr_a
        metadata["zr_b"] = zr_b

        # Z to dBZ
        dataset = transformation.dB_transform(dataset)

    elif metadata["units"] == "mm":
        # depth to rate
        dataset = to_rainrate(dataset)

        # Z to R
        if zr_a is None:
            zr_a = metadata.get("zr_a", 200.0)  # Default to Marshall-Palmer
        if zr_b is None:
            zr_b = metadata.get("zr_b", 1.6)  # Default to Marshall-Palmer
        precip_data = zr_a * precip_data**zr_b
        metadata["threshold"] = zr_a * metadata["threshold"] ** zr_b
        metadata["zerovalue"] = zr_a * metadata["zerovalue"] ** zr_b
        metadata["zr_a"] = zr_a
        metadata["zr_b"] = zr_b

        # Z to dBZ
        dataset = transformation.dB_transform(dataset)

    elif metadata["units"] == "dBZ":
        # Z to dBZ
        dataset = transformation.dB_transform(dataset)

    else:
        raise ValueError(
            f'Cannot convert unit {metadata["units"]} and transform {metadata["transform"]} to dBZ'
        )

    precip_var = dataset.attrs["precip_var"]
    metadata = dataset[precip_var].attrs
    precip_data = dataset[precip_var].values

    dataset[precip_var].data[:] = precip_data
    dataset = _change_unit(dataset, precip_var, "dBZ")
    return dataset
