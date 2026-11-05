"""Compatibility reader for pySTEPS forecast NetCDF datasets.

New physical CF files use :func:`pysteps.io.import_netcdf_cf`. Both readers
return an xarray Dataset. Only the compatibility reader offers a finite-fill
view and the historical warn/return-None error policy.

.. autosummary::
    :toctree: ../generated/

    import_netcdf_pysteps
"""

import numpy as np

__all__ = ["import_netcdf_pysteps"]


def import_netcdf_pysteps(
    filename, dtype="double", fillna=np.nan, onerror="warn", **kwargs
):
    """Compatibility entry point over :func:`import_netcdf_cf`.

    Return a physical Dataset with the requested float32/64 dtype. Scientific
    metadata missing from old files must be supplied through ``metadata``.
    ``legacy=True`` is the default here. A finite ``fillna`` view carries a
    linked validity mask; the CF exporter restores its missing values before
    validation. ``onerror='warn'`` warns and returns ``(None, None)`` on errors;
    ``onerror='raise'`` propagates the error. Eager/lazy ownership follows the
    core importer, and lazy datasets must be closed by the caller.
    """
    import warnings
    from pysteps.dataset_metadata import physical_variables
    from pysteps.io.netcdf_cf import import_netcdf_cf

    if not isinstance(onerror, str) or onerror.lower() not in ("warn", "raise"):
        raise ValueError("'onerror' keyword must be 'warn' or 'raise'.")
    try:
        if not np.isscalar(fillna) or np.isinf(fillna):
            raise ValueError("fillna must be a finite scalar or NaN")
        kwargs.setdefault("legacy", True)
        dataset = import_netcdf_cf(filename, dtype=dtype, **kwargs)
        try:
            if not np.isnan(fillna):
                for name in physical_variables(dataset):
                    with np.errstate(over="ignore", invalid="ignore"):
                        replacement = np.asarray(
                            fillna, dtype=dataset[name].dtype
                        ).item()
                    if not np.isfinite(replacement):
                        raise ValueError(
                            "fillna cannot be represented by the requested floating dtype"
                        )
                    mask_name = name + "_validity"
                    if mask_name in dataset:
                        raise ValueError(f"Validity mask name collision: {mask_name!r}")
                    valid = dataset[name].notnull().astype("int8")
                    dataset[mask_name] = (
                        valid.dims,
                        valid.data,
                        {
                            "long_name": "validity of " + name,
                            "flag_values": np.array([0, 1], dtype="int8"),
                            "flag_meanings": "missing valid",
                            "units": "1",
                        },
                    )
                    attrs = dataset[name].attrs.copy()
                    values = dataset[name].fillna(fillna)
                    dataset[name] = (values.dims, values.data, attrs)
                    dataset[name].attrs.update(
                        pysteps_validity_mask=mask_name,
                        pysteps_fillna_value=float(fillna),
                        ancillary_variables=" ".join(
                            filter(None, [attrs.get("ancillary_variables"), mask_name])
                        ),
                    )
            return dataset
        except Exception:
            dataset.close()
            raise
    except Exception as error:
        if onerror.lower() == "raise":
            raise
        warnings.warn(
            f"There was an error processing {filename}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None
