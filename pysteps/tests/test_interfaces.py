# -*- coding: utf-8 -*-

import numpy
import pytest

import pysteps


def _generic_interface_test(method_getter,
                            valid_names_func_pair,
                            invalid_names):
    for name, expected_function in valid_names_func_pair:
        error_message = "Error getting '{}' function.".format(name)
        assert method_getter(name) == expected_function, error_message
        if isinstance(name, str):
            assert method_getter(name.upper()) == expected_function, error_message

    # test invalid names
    for invalid_name in invalid_names:
        with pytest.raises(ValueError):
            method_getter(invalid_name)


def test_cascade_interface():
    """Test the cascade module interface."""

    from pysteps.cascade import decomposition, bandpass_filters

    method_getter = pysteps.cascade.interface.get_method

    valid_names_func_pair = [('fft', decomposition.decomposition_fft),
                             ('gaussian', bandpass_filters.filter_gaussian),
                             ('uniform', bandpass_filters.filter_uniform)]

    invalid_names = ['gauss', 'fourier']
    _generic_interface_test(method_getter, valid_names_func_pair, invalid_names)


def test_extrapolation_interface():
    """Test the extrapolation module interface."""

    from pysteps import extrapolation
    from pysteps.extrapolation import semilagrangian

    from pysteps.extrapolation.interface import eulerian_persistence as eulerian
    from pysteps.extrapolation.interface import _do_nothing as do_nothing
    from pysteps.extrapolation.interface import _return_none

    method_getter = extrapolation.interface.get_method

    valid_returned_objs = dict()
    valid_returned_objs['semilagrangian'] = semilagrangian.extrapolate
    valid_returned_objs['eulerian'] = eulerian
    valid_returned_objs[None] = do_nothing
    valid_returned_objs['None'] = do_nothing

    valid_names_func_pair = list(valid_returned_objs.items())

    invalid_names = ['euler', 'LAGRANGIAN']
    _generic_interface_test(method_getter, valid_names_func_pair, invalid_names)

    # Test eulerian persistence method
    precip = numpy.random.rand(100, 100)
    velocity = numpy.random.rand(100, 100)
    num_timesteps = 10
    for name in ["eulerian", "EULERIAN"]:
        forecaster = method_getter(name)
        forecast = forecaster(precip, velocity, num_timesteps)
        for i in range(num_timesteps):
            assert numpy.all(forecast[i] == precip)

    forecaster = method_getter(None)
    assert forecaster(precip, velocity, num_timesteps) is None


def test_io_interface():
    """Test the io module interface."""

    from pysteps.io import import_bom_rf3
    from pysteps.io import import_fmi_pgm
    from pysteps.io import import_mch_gif
    from pysteps.io import import_mch_hdf5
    from pysteps.io import import_mch_metranet
    from pysteps.io import import_odim_hdf5
    from pysteps.io import initialize_forecast_exporter_netcdf

    # Test importers
    valid_names_func_pair = [('bom_rf3', import_bom_rf3),
                             ('fmi_pgm', import_fmi_pgm),
                             ('mch_gif', import_mch_gif),
                             ('mch_hdf5', import_mch_hdf5),
                             ('mch_metranet', import_mch_metranet),
                             ('odim_hdf5', import_odim_hdf5),
                             ('mch_gif', import_mch_gif),
                             ('mch_gif', import_mch_gif),
                             ('mch_gif', import_mch_gif), ]

    def method_getter(name):
        return pysteps.io.interface.get_method(name, 'importer')

    invalid_names = ['odim', 'mch', 'fmi']
    _generic_interface_test(method_getter, valid_names_func_pair, invalid_names)

    # Test exporters
    def method_getter(name):
        return pysteps.io.interface.get_method(name, 'exporter')

    valid_names_func_pair = [('netcdf', initialize_forecast_exporter_netcdf)]
    invalid_names = ['hdf']

    _generic_interface_test(method_getter, valid_names_func_pair, invalid_names)

    # Test for invalid argument type
    with pytest.raises(TypeError):
        pysteps.io.interface.get_method('mch_gif', None)
        pysteps.io.interface.get_method(None, 'importer')

    # Test for invalid method types
    with pytest.raises(ValueError):
        pysteps.io.interface.get_method('mch_gif', 'io')


def test_motion_interface():
    """Test the motion module interface."""

    from pysteps.motion.darts import DARTS
    from pysteps.motion.lucaskanade import dense_lucaskanade
    from pysteps.motion.vet import vet

    method_getter = pysteps.motion.interface.get_method

    valid_names_func_pair = [('lk', dense_lucaskanade),
                             ('lucaskanade', dense_lucaskanade),
                             ('vet', vet),
                             ('DARTS', DARTS)]

    invalid_names = ['dart', 'pyvet', 'no_method']

    _generic_interface_test(method_getter, valid_names_func_pair, invalid_names)

    # Test default dummy method
    precip_field = method_getter(None)(numpy.random.random([2, 300, 500]))
    assert numpy.max(numpy.abs(precip_field)) == pytest.approx(0)

    # test not implemented names
    for name in ["brox", "clg"]:
        with pytest.raises(NotImplementedError):
            method_getter(name)  # last s missing


def test_noise_interface():
    """Test the noise module interface."""

    from pysteps.noise.fftgenerators import (initialize_param_2d_fft_filter,
                                             generate_noise_2d_fft_filter,
                                             initialize_nonparam_2d_fft_filter,
                                             initialize_nonparam_2d_ssft_filter,
                                             generate_noise_2d_ssft_filter,
                                             initialize_nonparam_2d_nested_filter)

    from pysteps.noise.motion import initialize_bps, generate_bps

    method_getter = pysteps.noise.interface.get_method

    valid_names_func_pair = [
        ('parametric', (initialize_param_2d_fft_filter,
                        generate_noise_2d_fft_filter)),
        ('nonparametric', (initialize_nonparam_2d_fft_filter,
                           generate_noise_2d_fft_filter)),
        ('ssft', (initialize_nonparam_2d_ssft_filter,
                  generate_noise_2d_ssft_filter)),
        ('nested', (initialize_nonparam_2d_nested_filter,
                    generate_noise_2d_ssft_filter)),
        ('bps', (initialize_bps, generate_bps))]

    invalid_names = ['nest', 'sft']

    _generic_interface_test(method_getter, valid_names_func_pair, invalid_names)


def test_nowcasts_interface():
    """Test the nowcasts module interface."""

    from pysteps.nowcasts import steps
    from pysteps.nowcasts import extrapolation
    method_getter = pysteps.nowcasts.interface.get_method

    valid_names_func_pair = [('extrapolation', extrapolation.forecast),
                             ('steps', steps.forecast)]

    invalid_names = ['extrap', 'step']
    _generic_interface_test(method_getter, valid_names_func_pair, invalid_names)

    # Test eulerian persistence method
    precip = numpy.random.rand(100, 100)
    velocity = numpy.random.rand(100, 100)
    num_timesteps = 10
    for name in ["eulerian", "EULERIAN"]:
        forecast = method_getter(name)(precip, velocity, num_timesteps)
        for i in range(num_timesteps):
            assert numpy.all(forecast[i] == precip)


def test_utils_interface():
    """Test utils module interface."""

    from pysteps.utils import conversion
    from pysteps.utils import transformation
    from pysteps.utils import dimension

    method_getter = pysteps.utils.interface.get_method

    valid_names_func_pair = [('mm/h', conversion.to_rainrate),
                             ('rainrate', conversion.to_rainrate),
                             ('mm', conversion.to_raindepth),
                             ('raindepth', conversion.to_raindepth),
                             ('dbz', conversion.to_reflectivity),
                             ('reflectivity', conversion.to_reflectivity),
                             ('rainrate', conversion.to_rainrate),
                             ('boxcox', transformation.boxcox_transform),
                             ('box-cox', transformation.boxcox_transform),
                             ('db', transformation.dB_transform),
                             ('decibel', transformation.dB_transform),
                             ('log', transformation.boxcox_transform),
                             ('nqt', transformation.NQ_transform),
                             ('sqrt', transformation.sqrt_transform),
                             ('accumulate', dimension.aggregate_fields_time),
                             ('clip', dimension.clip_domain),
                             ('square', dimension.square_domain),
                             ('upscale', dimension.aggregate_fields_space),
                             ]

    invalid_names = ['random', 'invalid']
    _generic_interface_test(method_getter, valid_names_func_pair, invalid_names)
