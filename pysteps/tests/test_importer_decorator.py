# -*- coding: utf-8 -*-
from functools import partial

import numpy as np
import pytest

from pysteps.tests.helpers import get_precipitation_fields

default_dtypes = dict(
    fmi="double",
    knmi="double",
    mch="double",
    opera="double",
    saf="double",
    mrms="single",
)


@pytest.mark.parametrize("source, default_dtype", default_dtypes.items())
def test_postprocess_import_decorator(source, default_dtype):
    """Test the postprocessing decorator for the importers."""
    import_data = partial(get_precipitation_fields, return_raw=True, source=source)

    precip = import_data()
    invalid_mask = ~np.isfinite(precip)

    assert precip.dtype == default_dtype

    if default_dtype == "single":
        dtype = "double"
    else:
        dtype = "single"

    precip = import_data(dtype=dtype)

    assert precip.dtype == dtype

    # Test that invalid types are handled correctly
    for dtype in ["int", "int64"]:
        with pytest.raises(ValueError):
            _ = import_data(dtype=dtype)

    precip = import_data(fillna=-1000)
    new_invalid_mask = precip == -1000
    assert (new_invalid_mask == invalid_mask).all()
