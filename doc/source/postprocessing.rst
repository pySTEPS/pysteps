.. _pysteps.postprocessing:

Post-processing of forecasts (:mod:`pysteps.postprocessing`)
************************************************************

Methods for post-processing of forecasts. Currently the module contains 
cumulative density function (CDF)-based matching between a forecast and the 
target distribution and computation of mean value and exceedance probabilities 
from forecast ensembles.

pysteps\.postprocessing\.ensemblestats
--------------------------------------

.. currentmodule:: pysteps.postprocessing.ensemblestats

.. autosummary::
    mean
    excprob

.. automodule:: pysteps.postprocessing.ensemblestats
    :members:

pysteps\.postprocessing\.probmatching
-------------------------------------

.. currentmodule:: pysteps.postprocessing.probmatching

.. autosummary::
    compute_empirical_cdf
    nonparam_match_empirical_cdf
    pmm_init
    pmm_compute
    shift_scale

.. automodule:: pysteps.postprocessing.probmatching
    :members:
