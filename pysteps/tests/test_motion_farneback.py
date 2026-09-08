import pytest
import numpy as np

from pysteps.motion import farneback
from pysteps.exceptions import MissingOptionalDependency
from pysteps.tests.helpers import get_precipitation_fields

fb_arg_names = (
    "pyr_scale",
    "levels",
    "winsize",
    "iterations",
    "poly_n",
    "poly_sigma",
    "flags",
    "size_opening",
    "sigma",
    "verbose",
)

fb_arg_values = [
    (0.5, 3, 15, 3, 5, 1.1, 0, 3, 60.0, False),  # default
    (0.5, 1, 5, 1, 7, 1.5, 0, 0, 0.0, True),  # minimal settings, sigma=0, verbose
    (
        0.3,
        5,
        30,
        10,
        7,
        1.5,
        1,
        5,
        10.0,
        False,
    ),  # maximal settings, flags=1, big opening
    (0.5, 3, 15, 3, 5, 1.1, 0, 0, 60.0, True),  # no opening, verbose
]


@pytest.mark.parametrize(fb_arg_names, fb_arg_values)
def test_farneback_params(
    pyr_scale,
    levels,
    winsize,
    iterations,
    poly_n,
    poly_sigma,
    flags,
    size_opening,
    sigma,
    verbose,
):
    """Test Farneback with various parameters and input types."""
    pytest.importorskip("cv2")
    # Input: realistic precipitation fields
    precip, metadata = get_precipitation_fields(
        num_prev_files=2,
        num_next_files=0,
        return_raw=False,
        metadata=True,
        upscale=2000,
    )
    precip = precip.filled()

    output = farneback.farneback(
        precip,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=flags,
        size_opening=size_opening,
        sigma=sigma,
        verbose=verbose,
    )

    assert isinstance(output, np.ndarray)
    assert output.shape[0] == 2
    assert output.shape[1:] == precip[0].shape
    assert np.isfinite(output).all() or np.isnan(output).any()


def test_farneback_invalid_shape():
    """Test error when input is wrong shape."""
    pytest.importorskip("cv2")
    arr = np.random.rand(64, 64)
    with pytest.raises(ValueError):
        farneback.farneback(arr)


def test_farneback_nan_input():
    """Test NaN handling in input."""
    pytest.importorskip("cv2")
    arr = np.random.rand(2, 64, 64)
    arr[0, 0, 0] = np.nan
    arr[1, 10, 10] = np.inf
    result = farneback.farneback(arr)
    assert result.shape == (2, 64, 64)


def test_farneback_cv2_missing(monkeypatch):
    """Test MissingOptionalDependency when cv2 is not injected."""
    monkeypatch.setattr(farneback, "CV2_IMPORTED", False)
    arr = np.random.rand(2, 64, 64)
    with pytest.raises(MissingOptionalDependency):
        farneback.farneback(arr)
    monkeypatch.setattr(farneback, "CV2_IMPORTED", True)  # restore


def test_farneback_sigma_zero():
    """Test sigma=0 disables smoothing logic."""
    pytest.importorskip("cv2")
    arr = np.random.rand(2, 32, 32)
    result = farneback.farneback(arr, sigma=0.0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 32, 32)


def test_farneback_small_window():
    """Test winsize edge case behavior."""
    pytest.importorskip("cv2")
    arr = np.random.rand(2, 16, 16)
    result = farneback.farneback(arr, winsize=3)
    assert result.shape == (2, 16, 16)


def test_farneback_verbose(capsys):
    """Test that verbose produces output (side-effect only)."""
    pytest.importorskip("cv2")
    arr = np.random.rand(2, 16, 16)
    farneback.farneback(arr, verbose=True)
    out = capsys.readouterr().out
    assert "Farneback method" in out or "mult factor" in out or "---" in out
