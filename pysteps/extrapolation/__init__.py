# -*- coding: utf-8 -*-
"""
    Methods for advection-based extrapolation of precipitation fields.
Currently the module contains an implementation of the
semi-Lagrangian method described in :cite:`GZ2002` and the
eulerian persistence."""

from pysteps.extrapolation.interface import get_method
