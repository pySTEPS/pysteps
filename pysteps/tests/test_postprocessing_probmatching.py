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
        result = resample_distributions(
            first_array, second_array, probability_first_array
        )
        expected_result = np.array([9, 8, 6, 3, 1])  # Expected result based on the seed
        assert result.shape == first_array.shape
        assert np.array_equal(result, expected_result)

    def test_probability_zero(self):
        first_array = np.array([1, 3, 5, 7, 9])
        second_array = np.array([2, 4, 6, 8, 10])
        probability_first_array = 0.0
        result = resample_distributions(
            first_array, second_array, probability_first_array
        )
        assert np.array_equal(result, np.sort(second_array)[::-1])

    def test_probability_one(self):
        first_array = np.array([1, 3, 5, 7, 9])
        second_array = np.array([2, 4, 6, 8, 10])
        probability_first_array = 1.0
        result = resample_distributions(
            first_array, second_array, probability_first_array
        )
        assert np.array_equal(result, np.sort(first_array)[::-1])

    def test_nan_in_inputs(self):
        array_with_nan = np.array([1, 3, np.nan, 7, 9])
        array_without_nan = np.array([2, 4, 6, 8, 10])
        probability_first_array = 0.6
        with pytest.raises(ValueError, match="Input arrays must not contain NaNs"):
            resample_distributions(
                array_with_nan, array_without_nan, probability_first_array
            )
        with pytest.raises(ValueError, match="Input arrays must not contain NaNs"):
            resample_distributions(
                array_without_nan, array_with_nan, probability_first_array
            )
        with pytest.raises(ValueError, match="Input arrays must not contain NaNs"):
            resample_distributions(
                array_with_nan, array_with_nan, probability_first_array
            )
