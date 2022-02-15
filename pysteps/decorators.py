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


def postprocess_import(fillna=np.nan, dtype="double"):
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
        Default data type for precipitation. Double precision by default.
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

            precip, *other_args = importer(*args, **kwargs)

            _dtype = kwargs.get("dtype", dtype)

            accepted_precisions = ["float32", "float64", "single", "double"]
            if _dtype not in accepted_precisions:
                raise ValueError(
                    "The selected precision does not correspond to a valid value."
                    "The accepted values are: " + str(accepted_precisions)
                )

            if isinstance(precip, np.ma.MaskedArray):
                invalid_mask = np.ma.getmaskarray(precip)
                precip.data[invalid_mask] = fillna
            else:
                # If plain numpy arrays are used, the importers should indicate
                # the invalid values with np.nan.
                _fillna = kwargs.get("fillna", fillna)
                if _fillna is not np.nan:
                    mask = ~np.isfinite(precip)
                    precip[mask] = _fillna

            return (precip.astype(_dtype),) + tuple(other_args)

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
        def _interpolator_with_preamble(xy_coord, values, xgrid, ygrid, **kwargs):
            nonlocal nchunks  # https://stackoverflow.com/questions/5630409/

            values = values.copy()
            xy_coord = xy_coord.copy()

            input_ndims = values.ndim
            input_nvars = 1 if input_ndims == 1 else values.shape[1]
            input_nsamples = values.shape[0]

            coord_ndims = xy_coord.ndim
            coord_nsamples = xy_coord.shape[0]

            grid_shape = (ygrid.size, xgrid.size)

            if np.any(~np.isfinite(values)):
                raise ValueError("argument 'values' contains non-finite values")
            if np.any(~np.isfinite(xy_coord)):
                raise ValueError("argument 'xy_coord' contains non-finite values")

            if input_ndims > 2:
                raise ValueError(
                    "argument 'values' must have 1 (n) or 2 dimensions (n, m), "
                    f"but it has {input_ndims}"
                )
            if not coord_ndims == 2:
                raise ValueError(
                    "argument 'xy_coord' must have 2 dimensions (n, 2), "
                    f"but it has {coord_ndims}"
                )

            if not input_nsamples == coord_nsamples:
                raise ValueError(
                    "the number of samples in argument 'values' does not match the "
                    f"number of coordinates {input_nsamples}!={coord_nsamples}"
                )

            # only one sample, return uniform output
            if input_nsamples == 1:
                output_array = np.ones((input_nvars,) + grid_shape)
                for n, v in enumerate(values[0, ...]):
                    output_array[n, ...] *= v
                return output_array.squeeze()

            # all equal elements, return uniform output
            if values.max() == values.min():
                return np.ones((input_nvars,) + grid_shape) * values.ravel()[0]

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

            interpolated = np.zeros((input_nvars,) + grid_shape)
            indx = 0
            for subxgrid in subxgrids:
                deltax = subxgrid.size
                indy = 0
                for subygrid in subygrids:
                    deltay = subygrid.size
                    interpolated[
                        :, indy : (indy + deltay), indx : (indx + deltax)
                    ] = interpolator(xy_coord, values, subxgrid, subygrid, **kwargs)
                    indy += deltay
                indx += deltax

            return interpolated.squeeze()

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
