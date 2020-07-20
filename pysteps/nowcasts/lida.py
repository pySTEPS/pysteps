"""
pysteps.nowcasts.lida
=====================

Implements the Lagrangian Integro-Difference equation with Autoregression (LIDA) model. It consists of the following components:

- advection-based nowcast
- autoregressive integrated (ARI) process for growth and decay
- convolution to account for loss of predictability
- stochastic perturbations

Advection is determined first, and the other components are applied in the Lagrangian coordinates. The model combines an advection-based nowcast, the ANVIL method developed in :cite:`PCLH2020` and the integro-difference equation (IDE) methodology developed in :cite:`?` and :cite:`?`. The idea of using a convolution is taken from the IDE model. Combined with the AR process, the convolution essentially replaces the cascade decomposition. Using a convolution has several advantages such as the ability to handle anisotropic structure, domain boundaries and missing data. The model parameters are localized, and an ellipitical convolution kernel is used. Perturbations are generated in a localized fashion by using the SSFT approach developed in :cite:`NBSG2017`.

.. autosummary::
    :toctree: ../generated/

    forecast
"""


def forecast():
    pass
