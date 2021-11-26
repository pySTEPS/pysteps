# -*- coding: utf-8 -*-
"""
pysteps.decorators
==================

Decorators used to define reusable building blocks that can change or extend
the behavior of some functions in pysteps.

.. autosummary::
    :toctree: ../generated/

    postprocess_import
    check_input_frames
    prepare_interpolator
    memoize
"""
import inspect
import uuid
import xarray as xr
from collections import defaultdict
from functools import wraps

import numpy as np


def _add_extra_kwrds_to_docstrings(target_func, extra_kwargs_doc_text):
    """
    Update the functions docstrings by replacing the `{extra_kwargs_doc}` occurences in
    the docstring by the `extra_kwargs_doc_text` value.
    """
    # Clean up indentation from docstrings for the
    # docstrings to be merged correctly.
    extra_kwargs_doc = inspect.cleandoc(extra_kwargs_doc_text)
    target_func.__doc__ = inspect.cleandoc(target_func.__doc__)

    # Add extra kwargs docstrings
    target_func.__doc__ = target_func.__doc__.format_map(
        defaultdict(str, extra_kwargs_doc=extra_kwargs_doc)
    )
    return target_func


def postprocess_import(fillna=np.nan, dtype="float32"):
    """
    Postprocess the imported precipitation data.
    Operations:

    - Allow type casting (dtype keyword)
    - Set invalid or missing data to predefined value (fillna keyword)

    This decorator replaces the text "{extra_kwargs}" in the function's
    docstring with the documentation of the keywords used in the postprocessing.
    The additional docstrings are added as "Other Parameters" in the importer function.

    Parameters
    ----------
    dtype: str
        Default data type for precipitation. Float32 precision by default.
    fillna: float or np.nan
        Default value used to represent the missing data ("No Coverage").
        By default, np.nan is used.
        If the importer returns a MaskedArray, all the masked values are set to the
        fillna value. If a numpy array is returned, all the invalid values (nan and inf)
        are set to the fillna value.
    """

    def _postprocess_import(importer):
        @wraps(importer)
        def _import_with_postprocessing(*args, **kwargs):
            data_array = importer(*args, **kwargs)

            if not isinstance(data_array, xr.DataArray):
                array, _, metadata = data_array
                data_array = _to_xarray(array, metadata)

            _dtype = kwargs.get("dtype", dtype)

            accepted_precisions = ["float32", "float64", "single", "double"]
            if _dtype not in accepted_precisions:
                raise ValueError(
                    "The selected precision does not correspond to a valid value."
                    "The accepted values are: " + str(accepted_precisions)
                )

            _fillna = kwargs.get("fillna", fillna)
            if _fillna is not np.nan:
                data_array = data_array.fillna(_fillna)

            data_array = data_array.astype(_dtype)

            if kwargs.get("legacy", False):
                return _xarray2legacy(data_array)
            return data_array

        extra_kwargs_doc = """
            Other Parameters
            ----------------
            dtype: str
                Data-type to which the array is cast.
                Valid values:  "float32", "float64", "single", and "double".
            fillna: float or np.nan
                Value used to represent the missing data ("No Coverage").
                By default, np.nan is used.
            """

        _add_extra_kwrds_to_docstrings(_import_with_postprocessing, extra_kwargs_doc)

        return _import_with_postprocessing

    return _postprocess_import


def check_input_frames(
    minimum_input_frames=2, maximum_input_frames=np.inf, just_ndim=False
):
    """
    Check that the input_images used as inputs in the optical-flow
    methods have the correct shape (t, x, y ).
    """

    def _check_input_frames(motion_method_func):
        @wraps(motion_method_func)
        def new_function(*args, **kwargs):
            """
            Return new function with the checks prepended to the
            target motion_method_func function.
            """

            input_images = args[0]
            if input_images.ndim != 3:
                raise ValueError(
                    "input_images dimension mismatch.\n"
                    f"input_images.shape: {str(input_images.shape)}\n"
                    "(t, x, y ) dimensions expected"
                )

            if not just_ndim:
                num_of_frames = input_images.shape[0]

                if minimum_input_frames < num_of_frames > maximum_input_frames:
                    raise ValueError(
                        f"input_images frames {num_of_frames} mismatch.\n"
                        f"Minimum frames: {minimum_input_frames}\n"
                        f"Maximum frames: {maximum_input_frames}\n"
                    )

            return motion_method_func(*args, **kwargs)

        return new_function

    return _check_input_frames


def prepare_interpolator(nchunks=4):
    """
    Check that all the inputs have the correct shape, and that all values are
    finite. It also split the destination grid in  `nchunks` parts, and process each
    part independently.
    """

    def _preamble_interpolation(interpolator):
        @wraps(interpolator)
        def _interpolator_with_preamble(sparse_data, xgrid, ygrid, **kwargs):
            nonlocal nchunks  # https://stackoverflow.com/questions/5630409/

            if not isinstance(sparse_data, (xr.Dataset, xr.DataArray)):
                raise ValueError(
                    "sparse_data must be an instance of xarray's Dataset or DataArray"
                )

            if "sample" not in sparse_data.dims:
                raise ValueError("missing dimension 'sample'")

            if not all([d in ("sample", "variable") for d in sparse_data.dims]):
                raise ValueError(
                    "sparse_data must have either dimension ('sample') or "
                    f"('sample', 'variable'), but it has {sparse_data.dims}"
                )

            if not ("x" in sparse_data.coords and "y" in sparse_data.coords):
                raise ValueError("missing coordinates 'x' and 'y'")

            if np.isnan(xgrid).any() or np.isnan(ygrid).any():
                raise ValueError("Target grid coordinates contain missing values")

            sparse_data = sparse_data.copy()

            if isinstance(sparse_data, xr.Dataset):
                sparse_data = sparse_data.to_array(name="convert_to_dataset")
            elif "variable" not in sparse_data.dims:
                sparse_data = sparse_data.expand_dims("variable")

            sparse_data = sparse_data.transpose("sample", "variable")

            # drop missing values
            sparse_data = sparse_data.dropna(dim="sample")
            sparse_data = sparse_data.drop_isel(sample=np.isnan(sparse_data.x))
            sparse_data = sparse_data.drop_isel(sample=np.isnan(sparse_data.y))

            input_nvars = sparse_data.sizes["variable"]
            input_nsamples = sparse_data.sizes["sample"]
            input_dtype = sparse_data.dtype

            grid_shape = (ygrid.size, xgrid.size)
            output_grid = xr.DataArray(
                np.zeros((input_nvars,) + grid_shape, dtype=input_dtype),
                dims=("variable", "y", "x"),
                coords={"y": ("y", ygrid), "x": ("x", xgrid)},
            )
            output_grid = output_grid.assign_coords(
                sparse_data.drop_vars(("x", "y", "sample"), errors="ignore").coords
            )
            output_grid = output_grid.assign_attrs(sparse_data.attrs)

            # only one sample, return uniform output
            if input_nsamples == 1:
                for n, v in enumerate(sparse_data.isel(sample=0)):
                    output_grid[dict(variable=n)] += v

            # all equal elements, return uniform output
            elif sparse_data.max() == sparse_data.min():
                output_grid += sparse_data.values.ravel()[0]

            # actual interpolation
            else:
                # split grid in n chunks
                nchunks = int(kwargs.get("nchunks", nchunks) ** 0.5)
                if nchunks > 1:
                    subxgrids = np.array_split(xgrid, nchunks)
                    subxgrids = [x for x in subxgrids if x.size > 0]
                    subygrids = np.array_split(ygrid, nchunks)
                    subygrids = [y for y in subygrids if y.size > 0]

                    # generate a unique identifier to be used for caching
                    # intermediate results
                    kwargs["hkey"] = uuid.uuid1().int
                else:
                    subxgrids = [xgrid]
                    subygrids = [ygrid]

                indx = 0
                for subxgrid in subxgrids:
                    deltax = subxgrid.size
                    indy = 0
                    for subygrid in subygrids:
                        deltay = subygrid.size
                        output_grid[
                            :, indy : (indy + deltay), indx : (indx + deltax)
                        ] = interpolator(sparse_data, subxgrid, subygrid, **kwargs)
                        indy += deltay
                    indx += deltax

            if "convert_to_dataset" == sparse_data.name:
                output_grid = output_grid.to_dataset(dim="variable")

            return output_grid.squeeze()

        extra_kwargs_doc = """
            nchunks: int, optional
                Split and process the destination grid in nchunks.
                Useful for large grids to limit the memory footprint.
            """

        _add_extra_kwrds_to_docstrings(_interpolator_with_preamble, extra_kwargs_doc)

        return _interpolator_with_preamble

    return _preamble_interpolation


def memoize(maxsize=10):
    """
    Add a Least Recently Used (LRU) cache to any function.
    Caching is purely based on the optional keyword argument 'hkey', which needs
    to be a hashable.

    Parameters
    ----------
    maxsize: int, optional
        The maximum number of elements stored in the LRU cache.
    """

    def _memoize(func):
        cache = dict()
        hkeys = []

        @wraps(func)
        def _func_with_cache(*args, **kwargs):
            hkey = kwargs.pop("hkey", None)
            if hkey in cache:
                return cache[hkey]
            result = func(*args, **kwargs)
            if hkey is not None:
                cache[hkey] = result
                hkeys.append(hkey)
                if len(hkeys) > maxsize:
                    cache.pop(hkeys.pop(0))

            return result

        return _func_with_cache

    return _memoize


def _to_xarray(array, metadata):
    """Convert to xarray DataArray."""
    x1, x2 = metadata["x1"], metadata["x2"]
    y1, y2 = metadata["y1"], metadata["y2"]
    xsize, ysize = metadata["xpixelsize"], metadata["ypixelsize"]
    # x_coords = np.arange(x1, x2, xsize) + xsize / 2
    # y_coords = np.arange(y1, y2, ysize) + ysize / 2
    x_coords = np.arange(x1, x1 + xsize * array.shape[1], xsize) + xsize / 2
    y_coords = np.arange(y1, y1 + ysize * array.shape[0], ysize) + ysize / 2

    data_array = xr.DataArray(
        data=array,
        dims=("y", "x"),
        coords=dict(
            x=("x", x_coords),
            y=("y", y_coords),
        ),
    )

    data_array.attrs.update(
        {
            # TODO: Revise this list before final 2.0 version ?
            "unit": metadata["unit"],
            "accutime": metadata["accutime"],
            "transform": metadata["transform"],
            "zerovalue": metadata["zerovalue"],
            "threshold": metadata["threshold"],
            "zr_a": metadata.get("zr_a", None),
            "zr_b": metadata.get("zr_b", None),
            "institution": metadata.get("institution", None),
            "projection": metadata["projection"],
            "bounding_box": (x1, x2, y1, y2),
            "yorigin": metadata["yorigin"],
            "xpixelsize": metadata["xpixelsize"],
            "ypixelsize": metadata["ypixelsize"],
            "cartesian_unit": metadata["cartesian_unit"],
        }
    )

    data_array.x.attrs.update(
        {
            "standard_name": "projection_x_coordinate",
            "units": metadata["cartesian_unit"],
        }
    )

    data_array.y.attrs.update(
        {
            "standard_name": "projection_y_coordinate",
            "units": metadata["cartesian_unit"],
        }
    )

    return data_array


# TODO: Remove before final 2.0 version
def _xarray2legacy(data_array):
    """
    Convert the new DataArrays to the legacy format used in pysteps v1.*
    """
    _array = data_array.values

    metadata = data_array.x.attrs.copy()
    metadata.update(**data_array.y.attrs)
    metadata.update(**data_array.attrs)

    if "t" in data_array.coords:
        print(data_array["t"])
        metadata["timestamps"] = data_array["t"]

    return _array, None, metadata
