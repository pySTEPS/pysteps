"""
pysteps.utils.conversion
========================

Methods for converting physical units.

.. autosummary::
    :toctree: ../generated/

    to_rainrate
    to_raindepth
    to_reflectivity
"""

import numpy as np
import warnings

# TODO: This should not be done. Instead fix the code so that it doesn't
# produce the warnings.
# to deactivate warnings for comparison operators with NaNs
warnings.filterwarnings("ignore", category=RuntimeWarning)

from . import transformation

def to_rainrate(R, metadata, a=None, b=None):
    """Convert to rain rate [mm/h].

    Parameters
    ----------
    R : array-like
        Array of any shape to be (back-)transformed.
    metadata : dict
        Metadata dictionary containing the accutime, transform, unit, threshold
        and zerovalue attributes as described in the documentation of
        :py:mod:`pysteps.io.importers`.

        Additionally, in case of conversion to/from reflectivity units, the
        zr_a and zr_b attributes are also required, but only if a=b=None.
    a,b : float, optional
        The a and b coefficients of the Z-R relationship.

    Returns
    -------
    R : array-like
        Array of any shape containing the converted units.
    metadata : dict
        The metadata with updated attributes.

    """

    R = R.copy()
    metadata = metadata.copy()

    if metadata["transform"] is not None:

        if metadata["transform"] is "dB":

            R, metadata = transformation.dB_transform(R, metadata, inverse=True)

        elif metadata["transform"] in  ["BoxCox", "log"]:

            R, metadata = transformation.boxcox_transform(R, metadata, inverse=True)

        elif metadata["transform"] is "NQT":

            R, metadata = transformation.NQ_transform(R, metadata, inverse=True)

        elif metadata["transform"] is "sqrt":

            R, metadata = transformation.sqrt_transform(R, metadata, inverse=True)

        else:

            raise ValueError("Unknown transformation %s" % metadata["transform"])


    if metadata["unit"] == "mm/h":

        pass

    elif metadata["unit"] == "mm":

        threshold = metadata["threshold"] # convert the threshold, too
        zerovalue = metadata["zerovalue"] # convert the zerovalue, too

        R = R/float(metadata["accutime"])*60.0
        threshold = threshold/float(metadata["accutime"])*60.0
        zerovalue = zerovalue/float(metadata["accutime"])*60.0

        metadata["threshold"] = threshold
        metadata["zerovalue"] = zerovalue

    elif metadata["unit"] == "dBZ":

        threshold = metadata["threshold"] # convert the threshold, too
        zerovalue = metadata["zerovalue"] # convert the zerovalue, too

        # Z to R
        if a is None:
            a = metadata.get("zr_a", 316.0)
        if b is None:
            b = metadata.get("zr_b", 1.5)
        R = (R/a)**(1.0/b)
        threshold = (threshold/a)**(1.0/b)
        zerovalue = (zerovalue/a)**(1.0/b)

        metadata["zr_a"] = a
        metadata["zr_b"] = b
        metadata["threshold"] = threshold
        metadata["zerovalue"] = zerovalue

    else:
        raise ValueError("Cannot convert unit %s and transform %s to mm/h" % (metadata["unit"], metadata["transform"]))

    metadata["unit"] = "mm/h"

    return R, metadata

def to_raindepth(R, metadata, a=None, b=None):
    """Convert to rain depth [mm].

    Parameters
    ----------
    R : array-like
        Array of any shape to be (back-)transformed.
    metadata : dict
        Metadata dictionary containing the accutime, transform, unit, threshold
        and zerovalue attributes as described in the documentation of
        :py:mod:`pysteps.io.importers`.

        Additionally, in case of conversion to/from reflectivity units, the
        zr_a and zr_b attributes are also required, but only if a=b=None.
    a,b : float, optional
        The a and b coefficients of the Z-R relationship.

    Returns
    -------
    R : array-like
        Array of any shape containing the converted units.
    metadata : dict
        The metadata with updated attributes.

    """

    R = R.copy()
    metadata = metadata.copy()

    if metadata["transform"] is not None:

        if metadata["transform"] is "dB":

            R, metadata = transformation.dB_transform(R, metadata, inverse=True)

        elif metadata["transform"] in ["BoxCox", "log"]:

            R, metadata = transformation.boxcox_transform(R, metadata, inverse=True)

        elif metadata["transform"] is "NQT":

            R, metadata = transformation.NQ_transform(R, metadata, inverse=True)

        elif metadata["transform"] is "sqrt":

            R, metadata = transformation.sqrt_transform(R, metadata, inverse=True)

        else:

            raise ValueError("Unknown transformation %s" % metadata["transform"])

    if metadata["unit"] == "mm" and metadata["transform"] is None:
        pass

    elif metadata["unit"] == "mm/h":

        threshold = metadata["threshold"] # convert the threshold, too
        zerovalue = metadata["zerovalue"] # convert the zerovalue, too

        R = R/60.0*metadata["accutime"]
        threshold = threshold/60.0*metadata["accutime"]
        zerovalue = zerovalue/60.0*metadata["accutime"]

        metadata["threshold"] = threshold
        metadata["zerovalue"] = zerovalue

    elif metadata["unit"] == "dBZ":

        threshold = metadata["threshold"] # convert the threshold, too
        zerovalue = metadata["zerovalue"] # convert the zerovalue, too

        # Z to R
        if a is None:
            a = metadata.get("zr_a", 316.0)
        if b is None:
            b = metadata.get("zr_b", 1.5)
        R = (R/a)**(1.0/b)/60.0*metadata["accutime"]
        threshold = (threshold/a)**(1.0/b)/60.0*metadata["accutime"]
        zerovalue = (zerovalue/a)**(1.0/b)/60.0*metadata["accutime"]

        metadata["zr_a"] = a
        metadata["zr_b"] = b
        metadata["threshold"] = threshold
        metadata["zerovalue"] = zerovalue

    else:
        raise ValueError("Cannot convert unit %s and transform %s to mm" % (metadata["unit"], metadata["transform"]))

    metadata["unit"] = "mm"

    return R, metadata

def to_reflectivity(R, metadata, a=None, b=None):
    """Convert to reflectivity [dBZ].

    Parameters
    ----------
    R : array-like
        Array of any shape to be (back-)transformed.
    metadata : dict
        Metadata dictionary containing the accutime, transform, unit, threshold
        and zerovalue attributes as described in the documentation of
        :py:mod:`pysteps.io.importers`.

        Additionally, in case of conversion to/from reflectivity units, the
        zr_a and zr_b attributes are also required, but only if a=b=None.
    a,b : float, optional
        The a and b coefficients of the Z-R relationship.

    Returns
    -------
    R : array-like
        Array of any shape containing the converted units.
    metadata : dict
        The metadata with updated attributes.

    """

    R = R.copy()
    metadata = metadata.copy()

    if metadata["transform"] is not None:

        if metadata["transform"] is "dB":

            R, metadata = transformation.dB_transform(R, metadata, inverse=True)

        elif metadata["transform"] in ["BoxCox", "log"]:

            R, metadata = transformation.boxcox_transform(R, metadata, inverse=True)

        elif metadata["transform"] is "NQT":

            R, metadata = transformation.NQ_transform(R, metadata, inverse=True)

        elif metadata["transform"] is "sqrt":

            R, metadata = transformation.sqrt_transform(R, metadata, inverse=True)

        else:

            raise ValueError("Unknown transformation %s" % metadata["transform"])

    if metadata["unit"] == "mm/h":

        # R to Z
        if a is None:
            a = metadata.get("zr_a", 316.0)
        if b is None:
            b = metadata.get("zr_b", 1.5)

        R = a*R**b
        metadata["threshold"] = a*metadata["threshold"]**b
        metadata["zerovalue"] = a*metadata["zerovalue"]**b

        # Z to dBZ
        R, metadata = transformation.dB_transform(R, metadata)

    elif metadata["unit"] == "mm":

        # depth to rate
        R, metadata = to_rainrate(R, metadata)

        # R to Z
        if a is None:
            a = metadata.get("zr_a", 316.0)
        if b is None:
            b = metadata.get("zr_b", 1.5)
        R = a*R**b
        metadata["threshold"] = a*metadata["threshold"]**b
        metadata["zerovalue"] = a*metadata["zerovalue"]**b

        # Z to dBZ
        R, metadata = transformation.dB_transform(R, metadata)

    elif metadata["unit"] == "dBZ":

        # Z to dBZ
        R, metadata = transformation.dB_transform(R, metadata)

    else:

        raise ValueError("Cannot convert unit %s and transform %s to mm/h" % (metadata["unit"], metadata["transform"]))

    metadata["unit"] = "dBZ"

    return R, metadata
