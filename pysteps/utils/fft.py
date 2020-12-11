# -*- coding: utf-8 -*-
"""
pysteps.utils.fft
=================

Interface module for different FFT methods.

.. autosummary::
    :toctree: ../generated/

    get_numpy
    get_scipy
    get_pyfftw
"""

from pysteps.exceptions import MissingOptionalDependency
from types import SimpleNamespace


def get_numpy(shape, fftn_shape=None, **kwargs):
    import numpy.fft as numpy_fft

    f = {
        "fft2": numpy_fft.fft2,
        "ifft2": numpy_fft.ifft2,
        "rfft2": numpy_fft.rfft2,
        "irfft2": lambda X: numpy_fft.irfft2(X, s=shape),
        "fftshift": numpy_fft.fftshift,
        "ifftshift": numpy_fft.ifftshift,
        "fftfreq": numpy_fft.fftfreq,
    }
    if fftn_shape is not None:
        f["fftn"] = numpy_fft.fftn
    fft = SimpleNamespace(**f)

    return fft


def get_scipy(shape, fftn_shape=None, **kwargs):
    import numpy.fft as numpy_fft
    import scipy.fftpack as scipy_fft

    # use numpy implementation of rfft2/irfft2 because they have not been
    # implemented in scipy.fftpack
    f = {
        "fft2": scipy_fft.fft2,
        "ifft2": scipy_fft.ifft2,
        "rfft2": numpy_fft.rfft2,
        "irfft2": lambda X: numpy_fft.irfft2(X, s=shape),
        "fftshift": scipy_fft.fftshift,
        "ifftshift": scipy_fft.ifftshift,
        "fftfreq": scipy_fft.fftfreq,
    }
    if fftn_shape is not None:
        f["fftn"] = scipy_fft.fftn
    fft = SimpleNamespace(**f)

    return fft


def get_pyfftw(shape, fftn_shape=None, n_threads=1, **kwargs):
    try:
        import pyfftw.interfaces.numpy_fft as pyfftw_fft
        import pyfftw

        pyfftw.interfaces.cache.enable()
    except ImportError:
        raise MissingOptionalDependency("pyfftw is required but not installed")

    X = pyfftw.empty_aligned(shape, dtype="complex128")
    F = pyfftw.empty_aligned(shape, dtype="complex128")

    fft_obj = pyfftw.FFTW(
        X,
        F,
        flags=["FFTW_ESTIMATE"],
        direction="FFTW_FORWARD",
        axes=(0, 1),
        threads=n_threads,
    )
    ifft_obj = pyfftw.FFTW(
        F,
        X,
        flags=["FFTW_ESTIMATE"],
        direction="FFTW_BACKWARD",
        axes=(0, 1),
        threads=n_threads,
    )

    if fftn_shape is not None:
        X = pyfftw.empty_aligned(fftn_shape, dtype="complex128")
        F = pyfftw.empty_aligned(fftn_shape, dtype="complex128")

        fftn_obj = pyfftw.FFTW(
            X,
            F,
            flags=["FFTW_ESTIMATE"],
            direction="FFTW_FORWARD",
            axes=list(range(len(fftn_shape))),
            threads=n_threads,
        )

    X = pyfftw.empty_aligned(shape, dtype="float64")
    output_shape = list(shape[:-1])
    output_shape.append(int(shape[-1] / 2) + 1)
    output_shape = tuple(output_shape)
    F = pyfftw.empty_aligned(output_shape, dtype="complex128")

    rfft_obj = pyfftw.FFTW(
        X,
        F,
        flags=["FFTW_ESTIMATE"],
        direction="FFTW_FORWARD",
        axes=(0, 1),
        threads=n_threads,
    )
    irfft_obj = pyfftw.FFTW(
        F,
        X,
        flags=["FFTW_ESTIMATE"],
        direction="FFTW_BACKWARD",
        axes=(0, 1),
        threads=n_threads,
    )

    f = {
        "fft2": lambda X: fft_obj(input_array=X.copy()).copy(),
        "ifft2": lambda X: ifft_obj(input_array=X.copy()).copy(),
        "rfft2": lambda X: rfft_obj(input_array=X.copy()).copy(),
        "irfft2": lambda X: irfft_obj(input_array=X.copy()).copy(),
        "fftshift": pyfftw_fft.fftshift,
        "ifftshift": pyfftw_fft.ifftshift,
        "fftfreq": pyfftw_fft.fftfreq,
    }

    if fftn_shape is not None:
        f["fftn"] = lambda X: fftn_obj(input_array=X).copy()
    fft = SimpleNamespace(**f)

    return fft
