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
