import pytest
import rioxarray  # noqa

from pysteps.tests.helpers import get_precipitation_fields
from pysteps.xarray_helpers import validate_dataset


def test_validate_dataset() -> None:
    precip_dataset = get_precipitation_fields(
        num_prev_files=0,
        num_next_files=0,
        return_raw=True,
        source="knmi",
        log_transform=False,
        qty="ACRR",
    )
    validate_dataset(precip_dataset)


def _create_xradar_dataset():
    """Create a precipitation dataset for testing.

    This creates an xarray dataset from KNMI data,
    but prepare as it would be coming from xradar.
    """

    precip_dataset = get_precipitation_fields(
        num_prev_files=0,
        num_next_files=0,
        return_raw=True,
        source="knmi",
        log_transform=False,
        qty="ACRR",
    )

    # remove dataset attrs
    precip_dataset.precip_accum.attrs = {}

    # give precip a generic name
    precip_dataset = precip_dataset.rename(precip_accum="ACRR", time="vcp_time")

    # drop projection variable
    precip_dataset = precip_dataset.drop("polar_stereographic")

    # provide data attrs
    precip_dataset["ACRR"].attrs = {
        "standard_name": "precipitation_amount",
        "units": "kg s-2",
        "grid_mapping": "spatial_ref",
    }

    # provide coordinate attrs
    precip_dataset["x"].attrs = {
        "standard_name": "projection_x_coordinate",
        "long_name": "Easting",
        "units": "m",
        "axis": "X",
    }
    precip_dataset["y"].attrs = {
        "standard_name": "projection_y_coordinate",
        "long_name": "Northing",
        "units": "m",
        "axis": "Y",
    }
    # rename time axis
    precip_dataset["vcp_time"].attrs = {
        "standard_name": "time",
        "long_name": "Volume time",
        "axis": "T",
    }

    # rioxarray writes the CF grid mapping variable (grid_mapping_name, crs_wkt, datum
    # parameters) so the projection is carried inside the store.
    projstring = "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +a=6378137 +b=6356752 +x_0=0 +y_0=0 +units=km"
    precip_dataset = precip_dataset.rio.write_crs(
        projstring, grid_mapping_name="spatial_ref"
    )

    # rewrite attrs
    precip_dataset.attrs = {
        "Conventions": "CF-1.10",
        "source": "pysteps source data with xradar attributes",
        "history": "some history here",
    }

    return precip_dataset


def test_validate_xradar_dataset() -> None:

    precip_dataset = _create_xradar_dataset()
    with pytest.raises(ValueError, match="requires a time coordinate"):
        validate_dataset(precip_dataset)

    precip_dataset = precip_dataset.rename(vcp_time="time")

    with pytest.raises(ValueError, match="requires a to level 'precip_var'"):
        validate_dataset(precip_dataset)

    precip_dataset.attrs["precip_var"] = "ACRR"

    validate_dataset(precip_dataset)
