# -*- coding: utf-8 -*-
"""Class for an ensemble Kalman filter in which the observations are also available as
a full ensemble. The class contains the update step, the computation of the covariance
matrices and several filter/masking functions."""

import numpy as np


class EnsembleKalmanFilter:

    def __init__(self, config, params):

        self._config = config

        # Check for combination kwargs in params
        self.__n_tapering = params.combination_kwargs.get("n_tapering", 0)
        self.__non_precip_mask = params.combination_kwargs.get("non_precip_mask", True)
        self.__n_ens_prec = params.combination_kwargs.get("n_ens_prec", 1)
        self.__lien_criterion = params.combination_kwargs.get("lien_criterion", True)
        self.__n_lien = params.combination_kwargs.get(
            "n_lien", self._config.n_ens_members // 2
        )

        print("Initialize ensemble Kalman filter")
        print("=================================")
        print("")

        print(f"Non-tapered diagonals:              {self.__n_tapering}")
        print(f"Non precip mask:                    {self.__non_precip_mask}")
        print(f"No. ens mems with precipitation:    {self.__n_ens_prec}")
        print(f"Lien Criterion:                     {self.__lien_criterion}")
        print(f"No. ens mems with precip (Lien):    {self.__n_lien}")
        print("")

        return

    def update(
        self,
        X_bg: np.ndarray,
        Y_obs: np.ndarray,
        inflation_factor_bg: float,
        inflation_factor_obs: float,
        offset_bg: float,
        offset_obs: float,
        X_bg_lien: np.ndarray | None = None,
        Y_obs_lien: np.ndarray | None = None,
    ):
        """
        Compute the ensemble Kalman filter update step.

        Parameters
        ----------
        X_bg: np.ndarray
            Two-dimensional array of shape (n_ens, n_pc) containing the background
            ensemble that corresponds to the Nowcast ensemble forecast.
        Y_obs: np.ndarray
            Two-dimensional array of shape (n_ens, n_pc) containing the observations
            that correspond to the NWP ensemble forecast.
        inflation_factor_bg: float
            Inflation factor of the background ensemble covariance matrix.
        inflation_factor_obs: float
            Inflation factor of the observation covariance matrix.
        offset_bg: float
            Offset of the background ensemble covariance matrix.
        offset_obs: float
            Offset of the observation covariance matrix.

        Other Parameters
        ----------------
        X_bg_lien: np.ndarray
            Two-dimensional array of shape (n_ens, n_pc) containing the background
            ensemble that consists only of grid boxes at which the Lien criterion is
            satisfied.
        Y_obs_lien: np.ndarray
            Two-dimensional array of shape (n_ens, n_pc) containing the observations
            that consists only of grid boxes at which the Lien criterion is satisfied.

        Returns
        -------
        X_ana: np.ndarray
            Two-dimensional array of shape (n_ens, n_pc) containing the updated
            analysis matrix.
        """

        # If the masked background and observation arrays are given, compute the
        # covariance matrices P and R only on these values...
        if X_bg_lien is not None and Y_obs_lien is not None:

            P = self.get_covariance_matrix(
                X_bg_lien, inflation_factor=inflation_factor_bg, offset=offset_bg
            )

            R = self.get_covariance_matrix(
                Y_obs_lien, inflation_factor=inflation_factor_obs, offset=offset_obs
            )
        # ...otherwise use the complete arrays.
        else:

            P = self.get_covariance_matrix(
                X_bg, inflation_factor=inflation_factor_bg, offset=offset_bg
            )

            R = self.get_covariance_matrix(
                Y_obs, inflation_factor=inflation_factor_obs, offset=offset_obs
            )

        # Estimate the Kalman gain
        K = np.dot(P, np.linalg.inv(P + R))

        # Update the background ensemble
        X_ana = X_bg.T + np.dot(K, (Y_obs - X_bg).T)

        return X_ana, K

    def get_covariance_matrix(
        self, M: np.ndarray, inflation_factor: float, offset: float
    ):
        """
        Compute the covariance matrix of a given ensemble forecast along the grid boxes
        or principal components.

        Parameters
        ----------
        M: np.ndarray
            Two-dimensional array of shape (n_ens, n_pc) containing an ensemble
            forecast of one lead time.
        inflation_factor: float
            Factor to increase the covariance and therefore the ensemble spread.
        offset: float
            Offset to shift the covariance.

        Returns
        -------
        Cov: np.ndarray
            Two-dimensional array of shape (n_pc, n_pc) containg the covariance matrix
            of the given ensemble forecast.
        """

        # Compute the ensemble mean
        M_mean = np.mean(M, axis=0)
        # Center the ensemble forecast and multiply with the given inflation factor
        M_centered = (M - M_mean) * inflation_factor
        # Compute the covariance matrix and add the respective offset and filter
        # unwanted diagonals, respectively.
        Cov = (
            1 / (M.shape[0] - 1) * np.dot(M_centered.T, M_centered) + offset
        ) * self.get_tapering(M.shape[1])

        return Cov

    def get_tapering(self, n: int):
        """
        Create a window function to clip unwanted diagonals of the covariance matrix.

        Parameters
        ----------
        n: integer
            Number of grid boxes/principal components of the ensemble forecast for that
            the covariance matrix is computed.

        Returns
        -------
        window_function: np.ndarray
            Two-dimensional array of shape (n_pc, n_pc) containing the window function
            to filter unwanted diagonals of the covariance matrix.
        """

        # Create an n-dimensional I-matrix as basis of the window function
        window_function = np.eye(n)
        # Get the weightings of a hanning window function with respect to the number of
        # diagonals that on want to keep
        hanning_values = np.hanning(self.__n_tapering * 2 + 1)[
            (self.__n_tapering + 1) :
        ]

        # Add the respective values to I
        for d in range(self.__n_tapering):

            window_function += np.diag(np.ones(n - d - 1) * hanning_values[d], k=d + 1)
            window_function += np.diag(np.ones(n - d - 1) * hanning_values[d], k=-d - 1)

        return window_function

    def get_precipitation_mask(self, X: np.ndarray):
        """
        Create the set of grid boxes where at least a minimum number (configurable)
        ensemble members forecast precipitation.

        Parameters
        ----------
        X: np.ndarray
            Two-dimensional array of shape (n_ens, n_grid) containg the ensemble
            forecast for one lead time.

        Returns
        -------
        idx_prec: np.ndarray
            One-dimensional array of shape (n_grid) that is set to True if the minimum
            number of ensemble members predict precipitation.
        """

        # Check the number of ensemble members forecast precipitation at each grid box.
        X_sum = np.sum(X >= self._config.precip_threshold, axis=0)

        # If the masking of areas without precipitation requested, mask grid boxes
        # where less ensemble members predict precipitation as the limit
        # n_ens_prec...
        if self.__non_precip_mask == True:

            idx_prec = X_sum >= self.__n_ens_prec

        # ...otherwise set all to True.
        else:

            idx_prec = np.ones_like(X_sum).astype(bool)

        return idx_prec

    def get_lien_criterion(self, X_nwc: np.ndarray, X_nwp: np.ndarray):
        """
        Create the set of grid boxes where the Lien criterion is satisfied (Lien et
        al., 2013) and thus, at least half (configurable) of the ensemble members of
        each forecast (Nowcast and NWP) predict precipitation.

        Parameters
        ----------
        X_nwc: np.ndarray
            Two-dimensional array (n_ens, n_grid) containing the nowcast ensemble
            forecast for one lead time.
        X_nwp: np.ndarray
            Two-dimensional array (n_ens, n_grid) containg the NWP ensemble forecast
            for one lead time.

        Returns
        -------
        idx_lien: np.ndarray
            One-dimensional array of shape (n_grid) that is set to True at grid boxes
            where the Lien criterion is satisfied.
        """

        # Check the number of ensemble members forecast precipitation at each grid box.
        X_nwc_sum = np.sum(X_nwc >= self._config.precip_threshold, axis=0)
        X_nwp_sum = np.sum(X_nwp >= self._config.precip_threshold, axis=0)

        # If the masking of areas without precipitation requested, mask grid boxes
        # where less ensemble members predict precipitation as the limit
        # n_ens_fc_prec...
        if self.__lien_criterion == True:

            idx_lien = np.logical_and(
                X_nwc_sum >= self.__n_lien, X_nwp_sum >= self.__n_lien
            )

        # ...otherwise set all to True.
        else:

            idx_lien = np.ones_like(X_nwc_sum).astype(bool)

        return idx_lien

    def get_weighting_for_probability_matching(
        self, X_bg: np.ndarray, X_ana: np.ndarray, Y_obs: np.ndarray
    ):
        """
        Compute the weighting between background and observation ensemble that results
        to the updated analysis ensemble in physical space for an optional probability
        matching.

        Parameters
        ----------
        X_bg: np.ndarray
            Two-dimensional array of shape (n_ens, n_grid) containing the background
            ensemble.
        X_ana: np.ndarray
            Two-dimensional array of shape (n_ens, n_grid) containing the updated
            analysis ensemble.
        Y_obs: np.ndarray
            Two-dimensional array of shape (n_ens, n_grid) containing the observation
            ensemble.

        Returns
        -------
        input_weight: float
            A weighting of which elements of the input ensemble contributed to the
            updated analysis ensemble with respect to Y_obs. Therefore, 0 means that
            the contribution comes entirely from X_bg. 1 means that the contribution
            comes entirely from Y_obs.
        """

        # Compute the sum of differences between X_ana and X_bg as well as Y_obs and
        # X_bg along the grid boxes.
        w1 = np.sum(X_ana - X_bg, axis=0)
        w2 = np.sum(Y_obs - X_bg, axis=0)

        # Check for infinitesimal differences between w1 and w2 as well as 0.
        w_close = np.isclose(w1, w2)
        w_zero = np.logical_and(w_close, np.isclose(w2, 0.0))

        # Compute the fraction of w1 and w2 and set values on grid boxes marked by
        # w_close or w_zero to 1 and 0, respectively.
        input_weight = w1 / w2
        input_weight[w_close] = 1.0
        input_weight[w_zero] = 0.0

        # Even now we have at some grid boxes weightings outside the range between 0
        # and 1. Therefore, we leave them out in the calculation of the averaged
        # weighting.
        valid_values = np.logical_and(input_weight >= 0.0, input_weight <= 1.0)
        input_weight = np.nanmean(input_weight[valid_values])

        # If there is no finite input_weight, we are switching to the NWP
        if not np.isfinite(input_weight):
            input_weight = 1.0

        return input_weight
