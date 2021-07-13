# -*- coding: utf-8 -*-

import pytest
from datetime import datetime

from pysteps.io.archive import _generate_path

test_argvalues = [
    ("20190130_1200", "%Y/foo/%m", "./2019/foo/01"),
    ("20190225_1200", "%Y/foo/%m", "./2019/foo/02"),
    ("20190122_2222", "%Y/foo/%m", "./2019/foo/01"),
    ("20190130_1200", "%Y/foo/%m", "./2019/foo/01"),
    ("20190130_1205", "%Y%m%d/foo/bar/%H%M", "./20190130/foo/bar/1205"),
    ("20190130_1205", "foo/bar/%H%M", "./foo/bar/1205"),
]


@pytest.mark.parametrize("timestamp, path_fmt, expected_path", test_argvalues)
def test_generate_path(timestamp, path_fmt, expected_path):
    date = datetime.strptime(timestamp, "%Y%m%d_%H%M")
    assert _generate_path(date, "./", path_fmt) == expected_path
