# -- coding: utf-8 --
"""
pysteps.verification.lifetime
=============================

Estimation of precipitation lifetime from
a decaying verification score function
(e.g. autocorrelation function).

.. autosummary::
    :toctree: ../generated/

    lifetime
    lifetime_init
    lifetime_accum
    lifetime_compute
"""

from math import exp
import numpy as np
from scipy.integrate import simps


def lifetime(X_s, X_t, rule="1/e"):
    """
    Compute the average lifetime by integrating the correlation function
    as a function of lead time. When not using the 1/e rule, the correlation
    function must be long enough to converge to 0, otherwise the lifetime is
    underestimated. The correlation function can be either empirical or
    theoretical, e.g. derived using the function 'ar_acf'
    in timeseries/autoregression.py.

    Parameters
    ----------
    X_s: array-like
        Array with the correlation function.
        Works also with other decaying scores that are defined
        in the range [0,1]=[min_skill,max_skill].
    X_t: array-like
        Array with the forecast lead times in the desired unit,
        e.g. [min, hour].
    rule: str {'1/e', 'trapz', 'simpson'}, optional
        Name of the method to integrate the correlation curve. \n
        '1/e' uses the 1/e rule and assumes an exponential decay. It linearly
        interpolates the time when the correlation goes below the value 1/e.
        When all values are > 1/e it returns the max lead time.
        When all values are < 1/e it returns the min lead time. \n
        'trapz' uses the trapezoidal rule for integration.\n
        'simpson' uses the Simpson's rule for integration.

    Returns
    -------
    lf: float
        Estimated lifetime with same units of X_t.
    """
    X_s = X_s.copy()
    X_t = X_t.copy()
    life = lifetime_init(rule)
    lifetime_accum(life, X_s, X_t)
    return lifetime_compute(life)


def lifetime_init(rule="1/e"):
    """
    Initialize a lifetime object.

    Parameters
    ----------
    rule: str {'1/e', 'trapz', 'simpson'}, optional
        Name of the method to integrate the correlation curve. \n
        '1/e' uses the 1/e rule and assumes an exponential decay. It linearly
        interpolates the time when the correlation goes below the value 1/e.
        When all values are > 1/e it returns the max lead time.
        When all values are < 1/e it returns the min lead time.\n
        'trapz' uses the trapezoidal rule for integration.\n
        'simpson' uses the Simpson's rule for integration.

    Returns
    -------
    out: dict
      The lifetime object.
    """
    list_rules = ["trapz", "simpson", "1/e"]
    if rule not in list_rules:
        raise ValueError(
            "Unknown rule %s for integration.\n" % rule
            + "The available methods are: "
            + str(list_rules)
        )

    lifetime = {}
    lifetime["lifetime_sum"] = 0.0
    lifetime["n"] = 0.0
    lifetime["rule"] = rule
    return lifetime


def lifetime_accum(lifetime, X_s, X_t):
    """
    Compute the lifetime by integrating the correlation function
    and accumulate the result into the given lifetime object.

    Parameters
    ----------
    X_s: array-like
        Array with the correlation function.
        Works also with other decaying scores that are defined
        in the range [0,1]=[min_skill,max_skill].
    X_t: array-like
        Array with the forecast lead times in the desired unit,
        e.g. [min, hour].
    """
    if lifetime["rule"] == "trapz":
        lf = np.trapz(X_s, x=X_t)
    elif lifetime["rule"] == "simpson":
        lf = simps(X_s, x=X_t)
    elif lifetime["rule"] == "1/e":
        euler_number = 1.0 / exp(1.0)
        X_s_ = np.array(X_s)

        is_euler_reached = np.sum(X_s_ <= euler_number) > 0
        if is_euler_reached:
            idx_b = np.argmax(X_s_ <= euler_number)
            if idx_b > 0:
                idx_a = idx_b - 1
                fraction_score = (
                    (euler_number - X_s[idx_b])
                    * (X_t[idx_a] - X_t[idx_b])
                    / (X_s[idx_a] - X_s[idx_b])
                )
                lf = X_t[idx_b] + fraction_score
            else:
                # if all values are below the 1/e value, return min lead time
                lf = np.min(X_t)
        else:
            # if all values are above the 1/e value, return max lead time
            lf = np.max(X_t)

    lifetime["lifetime_sum"] += lf
    lifetime["n"] += 1


def lifetime_compute(lifetime):
    """
    Compute the average value from the lifetime object.

    Parameters
    ----------
    lifetime: dict
      A lifetime object created with lifetime_init.

    Returns
    -------
    out: float
      The computed lifetime.
    """
    return 1.0 * lifetime["lifetime_sum"] / lifetime["n"]
