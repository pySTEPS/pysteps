# -*- coding: utf-8 -*-

import pytest
import numpy as np
from pysteps.utils import pca


pca_arg_values = (
    (10, 10),
    (20, 20),
    (10, 5),
    (20, 15),
)

pca_arg_names = ("len_y", "n_components")


@pytest.mark.parametrize(pca_arg_names, pca_arg_values)
def test_pca(len_y, n_components):

    pytest.importorskip("sklearn")

    precip_field = np.zeros((len_y, 200, 200))
    for i in range(len_y):
        a = 3 * i
        b = 2 * i
        precip_field[i, 20 + b : 160 - b, 30 + a : 180 - a] = 0.1
        precip_field[i, 22 + b : 162 - b, 35 + a : 178 - a] = 0.1
        precip_field[i, 24 + b : 164 - b, 40 + a : 176 - a] = 1.0
        precip_field[i, 26 + b : 166 - b, 45 + a : 174 - a] = 5.0
        precip_field[i, 28 + b : 168 - b, 50 + a : 172 - a] = 5.0
        precip_field[i, 30 + b : 170 - b, 35 + a : 170 - a] = 4.5
        precip_field[i, 32 + b : 172 - b, 40 + a : 168 - a] = 4.5
        precip_field[i, 34 + b : 174 - b, 45 + a : 166 - a] = 4.0
        precip_field[i, 36 + b : 176 - b, 50 + a : 164 - a] = 2.0
        precip_field[i, 38 + b : 178 - b, 55 + a : 162 - a] = 1.0
        precip_field[i, 40 + b : 180 - b, 60 + a : 160 - a] = 0.5
        precip_field[i, 42 + b : 182 - b, 65 + a : 158 - a] = 0.1

    precip_field = precip_field.reshape(
        len_y, precip_field.shape[1] * precip_field.shape[2]
    )

    kwargs = {"n_components": n_components, "svd_solver": "full"}
    precip_field_pc, pca_params = pca.pca_transform(
        forecast_ens=precip_field, get_params=True, **kwargs
    )

    assert precip_field_pc.shape == (len_y, n_components)
    assert pca_params["principal_components"].shape[1] == precip_field.shape[1]
    assert pca_params["mean"].shape[0] == precip_field.shape[1]

    precip_field_backtransformed = pca.pca_backtransform(
        precip_field_pc, pca_params=pca_params
    )

    # These fields are only equal if the full PCA is computed
    if len_y == n_components:
        assert np.sum(np.abs(precip_field_backtransformed - precip_field)) < 1e-6
