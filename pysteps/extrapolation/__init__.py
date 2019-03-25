"""
.. _pysteps.extrapolation:

Advection-based extrapolation (:mod:`pysteps.extrapolation`)
************************************************************

Methods for advection-based extrapolation of precipitation fields. Currently 
the module contains an implementation of the semi-Lagrangian method described 
in :cite:`GZ2002` and the eulerian persistence.

pysteps\.extrapolation\.interface
---------------------------------

.. automodule:: pysteps.extrapolation.interface
    :members:

pysteps\.extrapolation\.semilagrangian
--------------------------------------

.. automodule:: pysteps.extrapolation.semilagrangian
    :members:
    
"""

from pysteps.extrapolation.interface import get_method


