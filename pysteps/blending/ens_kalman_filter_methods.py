# -*- coding: utf-8 -*-
"""
pysteps.blending.ens_kalman_filter_methods
=============================================
Methods to calculate the ensemble Kalman filter based correction methods for blending
between nowcast and NWP.
The core of the method occurs in the EnsembleKalmanFilter class. The specific method
to use this core class can be selected. Currently, only the implementation of the
ensemble Kalman filter from :cite:`Nerini2019` is available.

Additional keyword arguments for the ensemble Kalman filter are:

n_tapering: int, (0)
    Number of grid boxes/principal components of the ensemble forecast for that
    the covariance matrix is computed. Defaults to 0.
non_precip_mask: bool, (True)
    Flag to specify whether the computation should be truncated on grid boxes where at
    least a minimum number (configurable) of ensemble members forecast precipitation.
    Defaults to True.
n_ens_prec: int, (1)
    Minimum number of ensemble members that forecast precipitation for the above
    mentioned mask. Defaults to 1.
lien_criterion: bool, (True)
    Flag to specify whether Lien criterion (Lien et al., 2013) should be applied for
    the computation of the update step within the ensemble Kalman filter. Defaults to
    True.
n_lien: int, (n_ens_members/2)
    Minimum number of ensemble members that forecast precipitation for the Lien
    criterion. Defaults to half of the ensemble members.
prob_matching: str, {'iterative','post_forecast','none'}
    Specify the probability matching method that should be applied as an additional
    processing step of the forecast computation. Defaults to 'iterative'.
inflation_factor_bg: float, (1.0)
    Inflation factor of the background (NWC) covariance matrix. This factor increases
    the covariances of the background ensemble and, thus, supports a faster convergence
    towards the observation ensemble. Defaults to 1.0.
inflation_factor_obs: float, (1.0)
    Inflation factor of the observation (NWP) covariance matrix. This factor increases
    the covariances of the observation ensemble and, thus, supports a slower convergence
    towards the observation ensemble. Defaults to 1.0.
offset_bg: float, (0.0)
    Offset of the background (NWC) covariance matrix. This offset supports a faster
    convergence towards the observation ensemble by linearly increasing all elements
    of the background covariance matrix. Defaults to 0.0.
offset_obs: float, (0.0)
    Offset of the observation (NWP) covariance matrix. This offset supports a slower convergence towards the observation ensemble by linearly incerasing all elements
    of the observation covariance matrix. Defaults to 0.0.
nwp_hres_eff: float
    Effective horizontal resolution of the utilized NWP model.
sampling_prob_source: str, {'ensemble','explained_var'}
    Computation method of the sampling probability for the probability matching.
    'ensemble' computes this probability as the ratio between the ensemble differences
    of X_ana - X_bg and X_obs - X_bg. 'explained_var' uses the sum of the Kalman gain
    weighted by the explained variance ratio.
"""


import numpy as np

from pysteps import utils
from pysteps.postprocessing import probmatching

try:
    import dask

    DASK_IMPORTED = True
except ImportError:
    DASK_IMPORTED = False


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
        # covariance matrices P and R only on these values.
        if X_bg_lien is not None and Y_obs_lien is not None:
            # Equation 13 in Nerini et al. (2019)
            P = self.get_covariance_matrix(
                X_bg_lien, inflation_factor=inflation_factor_bg, offset=offset_bg
            )
            # Equation 14 in Nerini et al. (2019)
            R = self.get_covariance_matrix(
                Y_obs_lien, inflation_factor=inflation_factor_obs, offset=offset_obs
            )
        # Otherwise use the complete arrays.
        else:
            # Equation 13 in Nerini et al. (2019)
            P = self.get_covariance_matrix(
                X_bg, inflation_factor=inflation_factor_bg, offset=offset_bg
            )
            # Equation 14 in Nerini et al. (2019)
            R = self.get_covariance_matrix(
                Y_obs, inflation_factor=inflation_factor_obs, offset=offset_obs
            )

        # Estimate the Kalman gain (eq. 15 in Nerini et al., 2019)
        self.K = np.dot(P, np.linalg.inv(P + R))

        # Update the background ensemble (eq. 16 in Nerini et al., 2019)
        X_ana = X_bg.T + np.dot(self.K, (Y_obs - X_bg).T)

        return X_ana

    def get_covariance_matrix(
        self, forecast_array: np.ndarray, inflation_factor: float, offset: float
    ):
        """
        Compute the covariance matrix of a given ensemble forecast along the grid boxes
        or principal components as it is done by Eq. 13 and 14 in Nerini et al., 2019.

        Parameters
        ----------
        forecast_array: np.ndarray
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
        ensemble_mean = np.mean(forecast_array, axis=0)
        # Center the ensemble forecast and multiply with the given inflation factor
        centered_ensemble = (forecast_array - ensemble_mean) * inflation_factor
        # Compute the covariance matrix and add the respective offset and filter
        # unwanted diagonals, respectively.
        Cov = (
            1
            / (forecast_array.shape[0] - 1)
            * np.dot(centered_ensemble.T, centered_ensemble)
            + offset
        ) * self.get_tapering(forecast_array.shape[1])

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

    def get_precipitation_mask(self, forecast_array: np.ndarray):
        """
        Create the set of grid boxes where at least a minimum number (configurable)
        of ensemble members forecast precipitation.

        Parameters
        ----------
        forecast_array: np.ndarray
            Two-dimensional array of shape (n_ens, n_grid) containg the ensemble
            forecast for one lead time.

        Returns
        -------
        idx_prec: np.ndarray
            One-dimensional array of shape (n_grid) that is set to True if the minimum
            number of ensemble members predict precipitation.
        """

        # Check the number of ensemble members forecast precipitation at each grid box.
        forecast_array_sum = np.sum(
            forecast_array >= self._config.precip_threshold, axis=0
        )

        # If the masking of areas without precipitation is requested, mask grid boxes
        # where less ensemble members predict precipitation than the set limit n_ens_prec.
        if self.__non_precip_mask == True:
            idx_prec = forecast_array_sum >= self.__n_ens_prec
        # Else, set all to True.
        else:
            idx_prec = np.ones_like(forecast_array_sum).astype(bool)

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


class MaskedEnKF(EnsembleKalmanFilter):

    def __init__(self, config, params):

        EnsembleKalmanFilter.__init__(self, config, params)
        self.__params = params

        # Read arguments from combination kwargs or set standard values if kwargs not
        # given
        self.__iterative_prob_matching = self.__params.combination_kwargs.get(
            "iterative_prob_matching", True
        )
        self.__inflation_factor_bg = self.__params.combination_kwargs.get(
            "inflation_factor_bg", 1.0
        )
        self.__inflation_factor_obs = self.__params.combination_kwargs.get(
            "inflation_factor_obs", 1.0
        )
        self.__offset_bg = self.__params.combination_kwargs.get("offset_bg", 0.0)
        self.__offset_obs = self.__params.combination_kwargs.get("offset_obs", 0.0)
        self.__sampling_prob_source = self.__params.combination_kwargs.get(
            "sampling_prob_source", "ensemble"
        )
        self.__use_accum_sampling_prob = self.__params.combination_kwargs.get(
            "use_accum_sampling_prob", False
        )

        self.__sampling_probability = 0.0

        print("Initialize masked ensemble Kalman filter")
        print("========================================")
        print("")

        print(f"Iterative probability matching:     {self.__iterative_prob_matching}")
        print(f"Background inflation factor:        {self.__inflation_factor_bg}")
        print(f"Observation inflation factor:       {self.__inflation_factor_obs}")
        print(f"Background offset:                  {self.__offset_bg}")
        print(f"Observation offset:                 {self.__offset_obs}")
        print(f"Sampling probability source:        {self.__sampling_prob_source}")
        print(f"Use accum. sampling probability:    {self.__use_accum_sampling_prob}")
        print("")

        return

    def correct_step(self, X_nwc, X_nwp, X_resampled):
        """
        Prepare input ensembles of Nowcast and NWP for the ensemble Kalman filter
        update step.

        Parameters
        ----------
        X_nwc: np.ndarray
            Three-dimensional array of shape (n_ens, m, n) containing the Nowcast
            ensemble forecast for one timestep. This data is used as background
            ensemble in the ensemble Kalman filter.
        X_nwp: np.ndarray
            Three-dimensional array of shape (n_ens, m, n) containing the NWP ensemble
            forecast for one timestep. This data is used as observation ensemble in the
            ensemble Kalman filter.

        Returns
        -------
        X_ana: np.ndarray
            Three-dimensional array of shape (n_ens, m, n) containing the Nowcast
            ensemble forecast corrected by NWP ensemble data.
        """

        # Get indices with predicted precipitation.
        idx_prec = np.logical_or(
            self.get_precipitation_mask(X_nwc), self.get_precipitation_mask(X_nwp)
        )

        # Get indices with satisfied Lien criterion and truncate the array onto the
        # precipitation area.
        idx_lien = self.get_lien_criterion(X_nwc, X_nwp)[idx_prec]

        # Stack both ensemble forecasts and truncate them onto the precipitation area.
        X_ens_stacked = np.vstack((X_nwc, X_nwp))[:, idx_prec]

        # Remove possible non-finite values
        X_ens_stacked[~np.isfinite(X_ens_stacked)] = self._config.norain_threshold

        # Check whether there are more rainy grid boxes as two times the ensemble
        # members
        if np.sum(idx_prec) <= X_ens_stacked.shape[0]:
            # If this is the case, the NWP ensemble forecast is returned
            return X_nwp

        # Transform both ensemble forecasts into the PC space.
        kwargs = {"n_components": X_ens_stacked.shape[0], "svd_solver": "full"}
        X_ens_stacked_pc, pca_params = utils.pca.pca_transform(
            X=X_ens_stacked, get_params=True, **kwargs
        )

        # And do that transformation also for the Lien criterion masked values.
        X_lien_pc = utils.pca.pca_transform(
            X=X_ens_stacked, mask=idx_lien, pca_params=pca_params, **kwargs
        )

        # Get the updated background ensemble (Nowcast ensemble) in PC space.
        X_ana_pc = self.update(
            X_bg=X_ens_stacked_pc[: X_nwc.shape[0]],
            Y_obs=X_ens_stacked_pc[X_nwc.shape[0] :],
            inflation_factor_bg=self.__inflation_factor_bg,
            inflation_factor_obs=self.__inflation_factor_obs,
            offset_bg=self.__offset_bg,
            offset_obs=self.__offset_obs,
            X_bg_lien=X_lien_pc[: X_nwc.shape[0]],
            Y_obs_lien=X_lien_pc[X_nwc.shape[0] :],
        )

        # Transform the analysis ensemble back into physical space.
        X_ana = utils.pca.pca_backtransform(X_pc=X_ana_pc, pca_params=pca_params)

        # Get the sampling probability either based on the ensembles...
        if self.__sampling_prob_source == "ensemble":
            sampling_probability_single_step = (
                self.get_weighting_for_probability_matching(
                    X_bg=X_ens_stacked[: X_nwc.shape[0]][:, idx_lien],
                    X_ana=X_ana[:, idx_lien],
                    Y_obs=X_ens_stacked[X_nwc.shape[0] :][:, idx_lien],
                )
            )
        # ...or based on the explained variance weighted Kalman gain.
        elif self.__sampling_prob_source == "explained_var":
            sampling_probability_single_step = np.sum(
                np.diag(self.K) * pca_params["explained_variance"]
            )
        else:
            raise ValueError(
                f"Sampling probability source should be either 'ensemble' or 'explained_var', but is {self.__sampling_prob_source}!"
            )

        if self.__use_accum_sampling_prob == True:
            self.__sampling_probability = (
                1 - sampling_probability_single_step
            ) * self.__sampling_probability + sampling_probability_single_step
        else:
            self.__sampling_probability = sampling_probability_single_step

        print(f"Sampling probability:               {self.__sampling_probability:1.4f}")

        # Apply probability matching to the analysis ensemble
        if self.__iterative_prob_matching == True:

            def worker(j):

                # Get the combined distribution based on the input weight
                X_resampled[j] = probmatching.resample_distributions(
                    first_array=X_nwc[j],
                    second_array=X_nwp[j],
                    probability_first_array=1 - self.__sampling_probability,
                ).reshape(self.__params.len_y, self.__params.len_x)

            dask_worker_collection = []

            if DASK_IMPORTED and self._config.n_ens_members > 1:
                for j in range(self._config.n_ens_members):
                    dask_worker_collection.append(dask.delayed(worker)(j))
                dask.compute(
                    *dask_worker_collection,
                    num_workers=self.__params.num_ensemble_workers,
                )
            else:
                for j in range(self._config.n_ens_members):
                    worker(j)

            dask_worker_collection = None

        # Set analysis ensemble into the Nowcast ensemble
        X_nwc[:, idx_prec] = X_ana

        return X_nwc, X_resampled
