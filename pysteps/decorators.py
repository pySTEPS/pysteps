"""
pysteps.decorators
==================

Decorators used to define reusable building blocks that can change or extend
the behavior of some functions in pysteps.

.. autosummary::
    :toctree: ../generated/

    postprocess_import
    check_motion_input_image
"""
import inspect
from collections import defaultdict
from functools import wraps

import numpy as np


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
    dtype : str
        Default data type for precipitation. Double precision by default.
    fillna : float or np.nan
        Default value used to represent the missing data ("No Coverage").
        By default, np.nan is used.
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
            dtype : str
                Data-type to which the array is cast.
                Valid values:  "float32", "float64", "single", and "double".
            fillna : float or np.nan
                Value used to represent the missing data ("No Coverage").
                By default, np.nan is used.
            """

        # Clean up indentation from docstrings for the
        # docstrings to be merged correctly.
        extra_kwargs_doc = inspect.cleandoc(extra_kwargs_doc)
        _import_with_postprocessing.__doc__ = inspect.cleandoc(
            _import_with_postprocessing.__doc__
        )

        # Add extra kwargs docstrings
        _import_with_postprocessing.__doc__ = _import_with_postprocessing.__doc__.format_map(
            defaultdict(str, extra_kwargs_doc=extra_kwargs_doc)
        )

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
