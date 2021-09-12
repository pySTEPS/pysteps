# -*- coding: utf-8 -*-
from functools import partial

import numpy as np
import pytest

from pysteps.tests.helpers import get_precipitation_fields

DEFAULT_DTYPE = "float32"

def test_postprocess_import_decorator():
    """Test the postprocessing decorator for the importers."""
    import_data = partial(get_precipitation_fields, source="mch")

    precip = import_data()
    invalid_mask = ~np.isfinite(precip)

    assert precip.dtype == DEFAULT_DTYPE

    if DEFAULT_DTYPE == "float32":
        dtype = "float64"
    else:
        dtype = "float32"

    precip = import_data(importer_kwargs=dict(dtype=dtype))

    assert precip.dtype == dtype

    # Test that equivalent dtypes are handled correctly
    for dtype1, dtype2 in zip(["float64", "float32",], ["double", "single"]):
        precip = import_data(importer_kwargs=dict(dtype=dtype1))
        assert precip.dtype == dtype2

    # Test that invalid types are handled correctly
    for dtype in ["int", "int64"]:
        with pytest.raises(ValueError):
            import_data(importer_kwargs=dict(dtype=dtype))

    precip = import_data(importer_kwargs=dict(fillna=-1000))
    new_invalid_mask = precip == -1000
    assert (new_invalid_mask == invalid_mask).all()
