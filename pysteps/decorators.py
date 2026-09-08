# -*- coding: utf-8 -*-
"""
pysteps.decorators
==================

Decorators used to define reusable building blocks that can change or extend
the behavior of some functions in pysteps.

.. autosummary::
    :toctree: ../generated/

    check_input_frames
    prepare_interpolator
    memoize
"""
import inspect
import uuid
import warnings
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
    if target_func.__doc__ is None:
        return target_func
    extra_kwargs_doc = inspect.cleandoc(extra_kwargs_doc_text)
    target_func.__doc__ = inspect.cleandoc(target_func.__doc__)

    # Add extra kwargs docstrings
    target_func.__doc__ = target_func.__doc__.format_map(
        defaultdict(str, extra_kwargs_doc=extra_kwargs_doc)
    )
    return target_func


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

            dataset = args[0]
            precip_var = dataset.attrs["precip_var"]
            input_images = dataset[precip_var].values
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
                    interpolated[:, indy : (indy + deltay), indx : (indx + deltax)] = (
                        interpolator(xy_coord, values, subxgrid, subygrid, **kwargs)
                    )
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


def deprecate_args(old_new_args, deprecation_release):
    """
    Support deprecated argument names while issuing deprecation warnings.

    Parameters
    ----------
    old_new_args: dict[str, str]
        Mapping from old to new argument names.
    deprecation_release: str
        Specify which future release will convert this warning into an error.
    """

    def _deprecate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs_names = list(kwargs.keys())
            for key_old in kwargs_names:
                if key_old in old_new_args:
                    key_new = old_new_args[key_old]
                    kwargs[key_new] = kwargs.pop(key_old)
                    warnings.warn(
                        f"Argument '{key_old}' has been renamed to '{key_new}'. "
                        f"This will raise a TypeError in pysteps {deprecation_release}.",
                        FutureWarning,
                    )
            return func(*args, **kwargs)

        return wrapper

    return _deprecate
