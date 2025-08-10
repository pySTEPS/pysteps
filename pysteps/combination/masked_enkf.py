# -*- coding: utf-8 -*-
"""
pysteps.combination.masked_enkf
=========================

Implementation of the ensemble Kalman filter as it is utilized in :cite:`Nerini2019`.
"""

import numpy as np

from pysteps import utils
from pysteps.postprocessing import probmatching
from pysteps.combination.ensemble_kalman_filter import EnsembleKalmanFilter

try:
    import dask

    DASK_IMPORTED = True
except ImportError:
    DASK_IMPORTED = False


class MaskedEnKF(EnsembleKalmanFilter):

    def __init__(self, config, params):

        EnsembleKalmanFilter.__init__(self, config)
        self.__params = params

        return

    def correct_step(self, X_nwc, X_nwp):

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

        # TODO: Add the Kalman gain adjustment to reduce smoothing effects due to
        # different horizontal resolutions

        # Get the updated background ensemble (Nowcast ensemble) in PC space.
        X_ana_pc = self.update(
            X_bg=X_ens_stacked_pc[: X_nwc.shape[0]],
            Y_obs=X_ens_stacked_pc[X_nwc.shape[0] :],
            inflation_factor_bg=self.__params.inflation_factor_bg,
            inflation_factor_obs=self.__params.inflation_factor_obs,
            offset_bg=self.__params.offset_bg,
            offset_obs=self.__params.offset_obs,
            X_bg_lien=X_lien_pc[: X_nwc.shape[0]],
            Y_obs_lien=X_lien_pc[X_nwc.shape[0] :],
        )

        # Transform the analysis ensemble back into physical space.
        X_ana = utils.pca.pca_backtransform(X_pc=X_ana_pc, pca_params=pca_params)

        # Get the weighting for the probability matching.
        input_weight = self.get_weighting_for_probability_matching(
            X_bg=X_ens_stacked[: X_nwc.shape[0]],
            X_ana=X_ana,
            Y_obs=X_ens_stacked[X_nwc.shape[0] :],
        )

        if self._config.iter_probability_matching == True:

            def worker(j):

                precip_forecast_probability_matching_resampled = (
                    probmatching.resample_distributions(
                        first_array=X_ens_stacked[j],
                        second_array=X_ens_stacked[j + X_nwc.shape[0]],
                        probability_first_array=1 - input_weight,
                    )
                )

                X_ana[j] = probmatching.nonparam_match_empirical_cdf(
                    X_ana[j],
                    precip_forecast_probability_matching_resampled,
                )

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

        X_nwc[:, idx_prec] = X_ana

        return X_nwc
