import numpy as np
import pytest

from pysteps.postprocessing.probmatching import (
    nonparam_match_empirical_cdf,
    resample_distributions,
)


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

    def test_nan_in_arr1_prob_1(self):
        array_with_nan = np.array([1, 3, np.nan, 7, 9])
        array_without_nan = np.array([2.0, 4, 6, 8, 10])
        probability_first_array = 1.0
        result = resample_distributions(
            array_with_nan, array_without_nan, probability_first_array
        )
        expected_result = np.array([np.nan, 9, 7, 3, 1], dtype=float)
        assert np.allclose(result, expected_result, equal_nan=True)

    def test_nan_in_arr1_prob_0(self):
        array_with_nan = np.array([1, 3, np.nan, 7, 9])
        array_without_nan = np.array([2, 4, 6, 8, 10])
        probability_first_array = 0.0
        result = resample_distributions(
            array_with_nan, array_without_nan, probability_first_array
        )
        expected_result = np.array([np.nan, 10, 8, 4, 2], dtype=float)
        assert np.allclose(result, expected_result, equal_nan=True)

    def test_nan_in_arr2_prob_1(self):
        array_without_nan = np.array([1, 3, 5, 7, 9])
        array_with_nan = np.array([2.0, 4, 6, np.nan, 10])
        probability_first_array = 1.0
        result = resample_distributions(
            array_without_nan, array_with_nan, probability_first_array
        )
        expected_result = np.array([np.nan, 9, 5, 3, 1], dtype=float)
        assert np.allclose(result, expected_result, equal_nan=True)

    def test_nan_in_arr2_prob_0(self):
        array_without_nan = np.array([1, 3, 5, 7, 9])
        array_with_nan = np.array([2, 4, 6, np.nan, 10])
        probability_first_array = 0.0
        result = resample_distributions(
            array_without_nan, array_with_nan, probability_first_array
        )
        expected_result = np.array([np.nan, 10, 6, 4, 2], dtype=float)
        assert np.allclose(result, expected_result, equal_nan=True)

    def test_nan_in_both_prob_1(self):
        array1_with_nan = np.array([1, np.nan, np.nan, 7, 9])
        array2_with_nan = np.array([2.0, 4, np.nan, np.nan, 10])
        probability_first_array = 1.0
        result = resample_distributions(
            array1_with_nan, array2_with_nan, probability_first_array
        )
        expected_result = np.array([np.nan, np.nan, np.nan, 9, 1], dtype=float)
        assert np.allclose(result, expected_result, equal_nan=True)

    def test_nan_in_both_prob_0(self):
        array1_with_nan = np.array([1, np.nan, np.nan, 7, 9])
        array2_with_nan = np.array([2.0, 4, np.nan, np.nan, 10])
        probability_first_array = 0.0
        result = resample_distributions(
            array1_with_nan, array2_with_nan, probability_first_array
        )
        expected_result = np.array([np.nan, np.nan, np.nan, 10, 2], dtype=float)
        assert np.allclose(result, expected_result, equal_nan=True)


class TestNonparamMatchEmpiricalCDF:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Set the seed for reproducibility
        np.random.seed(42)

    def test_ignore_indices_with_nans_both(self):
        initial_array = np.array([np.nan, np.nan, 6, 2, 0, 0, 0, 0, 0, 0])
        target_array = np.array([np.nan, np.nan, 9, 5, 4, 0, 0, 0, 0, 0])
        result = nonparam_match_empirical_cdf(
            initial_array, target_array, ignore_indices=np.isnan(initial_array)
        )
        expected_result = np.array([np.nan, np.nan, 9, 5, 0, 0, 0, 0, 0, 0])
        assert np.allclose(result, expected_result, equal_nan=True)

    def test_zeroes_initial(self):
        initial_array = np.zeros(10)
        target_array = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = nonparam_match_empirical_cdf(initial_array, target_array)
        expected_result = np.zeros(10)
        assert np.allclose(result, expected_result)

    def test_nans_initial(self):
        initial_array = np.array(
            [0, 1, 2, 3, 4, np.nan, np.nan, np.nan, np.nan, np.nan]
        )
        target_array = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        with pytest.raises(
            ValueError,
            match="Initial array contains non-finite values outside ignore_indices mask.",
        ):
            nonparam_match_empirical_cdf(initial_array, target_array)

    def test_all_nans_initial(self):
        initial_array = np.full(10, np.nan)
        target_array = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        with pytest.raises(ValueError, match="Initial array contains only nans."):
            nonparam_match_empirical_cdf(initial_array, target_array)

    def test_ignore_indices_nans_initial(self):
        initial_array = np.array(
            [0, 1, 2, 3, 4, np.nan, np.nan, np.nan, np.nan, np.nan]
        )
        target_array = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = nonparam_match_empirical_cdf(
            initial_array, target_array, ignore_indices=np.isnan(initial_array)
        )
        expected_result = np.array(
            [0, 7, 8, 9, 10, np.nan, np.nan, np.nan, np.nan, np.nan]
        )
        assert np.allclose(result, expected_result, equal_nan=True)

    def test_ignore_indices_nans_target(self):
        # We expect the initial_array values for which ignore_indices is true to be conserved as-is.
        initial_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        target_array = np.array(
            [0, 2, 3, 4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        )
        result = nonparam_match_empirical_cdf(
            initial_array, target_array, ignore_indices=np.isnan(target_array)
        )
        expected_result = np.array([0, 2, 3, 4, 4, 5, 6, 7, 8, 9])
        assert np.allclose(result, expected_result, equal_nan=True)

    def test_more_zeroes_in_initial(self):
        initial_array = np.array([1, 4, 0, 0, 0, 0, 0, 0, 0, 0])
        target_array = np.array([10, 8, 6, 4, 2, 0, 0, 0, 0, 0])
        result = nonparam_match_empirical_cdf(
            initial_array, target_array, ignore_indices=np.isnan(initial_array)
        )
        expected_result = np.array([8, 10, 0, 0, 0, 0, 0, 0, 0, 0])
        assert np.allclose(result, expected_result, equal_nan=True)

    def test_more_zeroes_in_initial_unsrt(self):
        initial_array = np.array([1, 4, 0, 0, 0, 0, 0, 0, 0, 0])
        target_array = np.array([6, 4, 2, 0, 0, 0, 0, 0, 10, 8])
        result = nonparam_match_empirical_cdf(
            initial_array, target_array, ignore_indices=np.isnan(initial_array)
        )
        expected_result = np.array([8, 10, 0, 0, 0, 0, 0, 0, 0, 0])
        assert np.allclose(result, expected_result, equal_nan=True)

    def test_more_zeroes_in_target(self):
        initial_array = np.array([1, 3, 7, 5, 0, 0, 0, 0, 0, 0])
        target_array = np.array([10, 8, 0, 0, 0, 0, 0, 0, 0, 0])
        result = nonparam_match_empirical_cdf(
            initial_array, target_array, ignore_indices=np.isnan(initial_array)
        )
        expected_result = np.array([0, 0, 10, 8, 0, 0, 0, 0, 0, 0])
        assert np.allclose(result, expected_result, equal_nan=True)

    def test_2dim_array(self):
        initial_array = np.array([[1, 3, 5], [11, 9, 7]])
        target_array = np.array([[2, 4, 6], [8, 10, 12]])
        result = nonparam_match_empirical_cdf(initial_array, target_array)
        expected_result = np.array([[2, 4, 6], [12, 10, 8]])
        assert np.allclose(result, expected_result, equal_nan=True)
