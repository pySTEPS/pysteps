import numpy as np
import pytest
from pysteps.postprocessing.probmatching import resample_distributions

class TestResampleDistributions:

    @pytest.fixture(autouse=True)
    def setup(self):
        # Set the seed for reproducibility
        np.random.seed(42)
    
    def test_valid_inputs(self):
        first_array = np.array([1, 3, 5, 7, 9])
        second_array = np.array([2, 4, 6, 8, 10])
        probability_first_array = 0.6
        result = resample_distributions(first_array, second_array, probability_first_array)
        expected_result = np.array([9, 8, 6, 3, 1])  # Expected result based on the seed
        assert result.shape == first_array.shape
        assert np.array_equal(result, expected_result)

    def test_probability_zero(self):
        first_array = np.array([1, 3, 5, 7, 9])
        second_array = np.array([2, 4, 6, 8, 10])
        probability_first_array = 0.0
        result = resample_distributions(first_array, second_array, probability_first_array)
        assert np.array_equal(result, np.sort(second_array)[::-1])

    def test_probability_one(self):
        first_array = np.array([1, 3, 5, 7, 9])
        second_array = np.array([2, 4, 6, 8, 10])
        probability_first_array = 1.0
        result = resample_distributions(first_array, second_array, probability_first_array)
        assert np.array_equal(result, np.sort(first_array)[::-1])

    def test_nan_in_first_array(self):
        first_array_with_nan = np.array([1, 3, np.nan, 7, 9])
        second_array = np.array([2, 4, 6, 8, 10])
        probability_first_array = 0.6
        result = resample_distributions(first_array_with_nan, second_array, probability_first_array)
        assert result.shape == first_array_with_nan.shape
        assert not np.any(np.isnan(result))

    def test_nan_in_second_array(self):
        first_array = np.array([1, 3, 5, 7, 9])
        second_array_with_nan = np.array([2, 4, 6, 8, np.nan])
        probability_first_array = 0.6
        result = resample_distributions(first_array, second_array_with_nan, probability_first_array)
        assert result.shape == first_array.shape
        assert not np.any(np.isnan(result))

    def test_nan_in_both_arrays(self):
        first_array_with_nan = np.array([1, np.nan, 5, np.nan, 9])
        second_array_with_nan = np.array([np.nan, 4, np.nan, 8, 10])
        probability_first_array = 0.6
        result = resample_distributions(first_array_with_nan, second_array_with_nan, probability_first_array)
        assert result.shape == first_array_with_nan.shape
        assert not np.any(np.isnan(result))
