# -*- coding: utf-8 -*-

import os
import numpy as np

import pytest

import pysteps
from pysteps.timeseries import autoregression, correlation

pytest.importorskip("pyproj")


def test_estimate_ar_params_ols():
    R = _create_data_univariate()

    for p in range(1, 4):
        phi = autoregression.estimate_ar_params_ols(R[-(p + 1) :], p)
        assert len(phi) == p + 1
        for i in range(len(phi)):
            assert np.isscalar(phi[i])

        phi = autoregression.estimate_ar_params_ols(
            R[-(p + 1) :], p, include_constant_term=True
        )
        assert len(phi) == p + 2
        for i in range(len(phi)):
            assert np.isscalar(phi[i])

        phi = autoregression.estimate_ar_params_ols(
            R[-(p + 2) :], p, include_constant_term=True, d=1
        )
        assert len(phi) == p + 3
        for i in range(len(phi)):
            assert np.isscalar(phi[i])


def test_estimate_ar_params_yw():
    R = _create_data_univariate()

    for p in range(1, 4):
        gamma = correlation.temporal_autocorrelation(R[-(p + 1) :])
        phi = autoregression.estimate_ar_params_yw(gamma)
        assert len(phi) == p + 1
        for i in range(len(phi)):
            assert np.isscalar(phi[i])


def test_estimate_ar_params_yw_localized():
    R = _create_data_univariate()

    for p in range(1, 4):
        gamma = correlation.temporal_autocorrelation(
            R[-(p + 1) :], window="gaussian", window_radius=25
        )
        phi = autoregression.estimate_ar_params_yw_localized(gamma)
        assert len(phi) == p + 1
        for i in range(len(phi)):
            assert phi[i].shape == R.shape[1:]


def test_estimate_ar_params_ols_localized():
    R = _create_data_univariate()

    for p in range(1, 4):
        phi = autoregression.estimate_ar_params_ols_localized(R[-(p + 1) :], p, 25)
        assert len(phi) == p + 1
        for i in range(len(phi)):
            assert phi[i].shape == R.shape[1:]

        phi = autoregression.estimate_ar_params_ols_localized(
            R[-(p + 1) :], p, 25, include_constant_term=True
        )
        assert len(phi) == p + 2
        for i in range(len(phi)):
            assert phi[i].shape == R.shape[1:]

        phi = autoregression.estimate_ar_params_ols_localized(
            R[-(p + 2) :], p, 25, include_constant_term=True, d=1
        )
        assert len(phi) == p + 3
        for i in range(len(phi)):
            assert phi[i].shape == R.shape[1:]


def test_estimate_var_params_ols():
    R = _create_data_multivariate()
    q = R.shape[1]

    for p in range(1, 4):
        phi = autoregression.estimate_var_params_ols(R[-(p + 1) :], p)
        assert len(phi) == p + 1
        for i in range(len(phi)):
            assert phi[i].shape == (q, q)

        phi = autoregression.estimate_var_params_ols(
            R[-(p + 1) :], p, include_constant_term=True
        )
        assert len(phi) == p + 2
        assert phi[0].shape == (q,)
        for i in range(1, len(phi)):
            assert phi[i].shape == (q, q)

        phi = autoregression.estimate_var_params_ols(
            R[-(p + 2) :], p, include_constant_term=True, d=1
        )
        assert len(phi) == p + 3
        assert phi[0].shape == (q,)
        for i in range(1, len(phi)):
            assert phi[i].shape == (q, q)


def test_estimate_var_params_ols_localized():
    R = _create_data_multivariate()
    q = R.shape[1]

    for p in range(1, 4):
        phi = autoregression.estimate_var_params_ols_localized(R[-(p + 1) :], p, 25)
        assert len(phi) == p + 1
        for i in range(len(phi)):
            assert phi[i].shape == (R.shape[2], R.shape[3], q, q)

        phi = autoregression.estimate_var_params_ols_localized(
            R[-(p + 1) :], p, 25, include_constant_term=True
        )
        assert len(phi) == p + 2
        assert phi[0].shape == (R.shape[2], R.shape[3], q)
        for i in range(1, len(phi)):
            assert phi[i].shape == (R.shape[2], R.shape[3], q, q)

        phi = autoregression.estimate_var_params_ols_localized(
            R[-(p + 2) :], p, 25, include_constant_term=True, d=1
        )
        assert len(phi) == p + 3
        assert phi[0].shape == (R.shape[2], R.shape[3], q)
        for i in range(1, len(phi)):
            assert phi[i].shape == (R.shape[2], R.shape[3], q, q)


def test_estimate_var_params_yw():
    R = _create_data_multivariate()

    for p in range(1, 4):
        gamma = correlation.temporal_autocorrelation_multivariate(R[-(p + 1) :])
        phi = autoregression.estimate_var_params_yw(gamma)
        assert len(phi) == p + 1
        for i in range(len(phi)):
            assert phi[i].shape == (R.shape[1], R.shape[1])


def test_estimate_var_params_yw_localized():
    R = _create_data_multivariate()
    q = R.shape[1]

    for p in range(1, 4):
        gamma = correlation.temporal_autocorrelation_multivariate(
            R[-(p + 1) :], window="gaussian", window_radius=25
        )
        phi = autoregression.estimate_var_params_yw_localized(gamma)
        assert len(phi) == p + 1
        for i in range(len(phi)):
            assert phi[i].shape == (R.shape[2], R.shape[3], q, q)


def test_iterate_ar():
    R = _create_data_univariate()
    p = 2

    phi = autoregression.estimate_ar_params_ols(R[-(p + 1) :], p)
    autoregression.iterate_ar_model(R, phi)


def test_iterate_ar_localized():
    R = _create_data_univariate()
    p = 2

    phi = autoregression.estimate_ar_params_ols_localized(R[-(p + 1) :], p, 25)
    R_ = autoregression.iterate_ar_model(R, phi)
    assert R_.shape == R.shape


def test_iterate_var():
    R = _create_data_multivariate()
    p = 2

    phi = autoregression.estimate_var_params_ols(R[-(p + 1) :], p)
    R_ = autoregression.iterate_var_model(R, phi)
    assert R_.shape == R.shape


def test_iterate_var_localized():
    R = _create_data_multivariate()
    p = 2

    phi = autoregression.estimate_var_params_ols_localized(R[-(p + 1) :], p, 25)
    R_ = autoregression.iterate_var_model(R, phi)
    assert R_.shape == R.shape


def _create_data_multivariate():
    root_path = pysteps.rcparams.data_sources["fmi"]["root_path"]

    filenames = [
        "201609281600_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz",
        "201609281605_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz",
        "201609281610_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz",
        "201609281615_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz",
        "201609281620_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz",
    ]

    R = []
    for fn in filenames:
        filename = os.path.join(root_path, "20160928", fn)
        R_, _, _ = pysteps.io.import_fmi_pgm(filename, gzipped=True)
        R_[~np.isfinite(R_)] = 0.0
        R.append(np.stack([R_, np.roll(R_, 5, axis=0)]))

    R = np.stack(R)
    R = R[:, :, 575:800, 255:480]

    return R


def _create_data_univariate():
    root_path = pysteps.rcparams.data_sources["fmi"]["root_path"]

    filenames = [
        "201609281600_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz",
        "201609281605_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz",
        "201609281610_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz",
        "201609281615_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz",
        "201609281620_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz",
    ]

    R = []
    for fn in filenames:
        filename = os.path.join(root_path, "20160928", fn)
        R_, _, _ = pysteps.io.import_fmi_pgm(filename, gzipped=True)
        R_[~np.isfinite(R_)] = 0.0
        R.append(R_)

    R = np.stack(R)
    R = R[:, 575:800, 255:480]

    return R
