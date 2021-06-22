# -*- coding: utf-8 -*-
import time

from pysteps.decorators import memoize


def test_memoize():
    @memoize(maxsize=1)
    def _slow_function(x, **kwargs):
        time.sleep(1)
        return x

    for i in range(2):
        out = _slow_function(i, hkey=i)
        assert out == i

    # cached result
    t0 = time.monotonic()
    out = _slow_function(1, hkey=1)
    assert time.monotonic() - t0 < 1
    assert out == 1

    # maxsize exceeded
    t0 = time.monotonic()
    out = _slow_function(0, hkey=0)
    assert time.monotonic() - t0 >= 1
    assert out == 0

    # no hash
    t0 = time.monotonic()
    out = _slow_function(1)
    assert time.monotonic() - t0 >= 1
    assert out == 1
