"""
pysteps.feature.interface
=========================

Interface for the feature detection module. It returns a callable function for
detecting features from two-dimensional images.

The feature detectors implement the following interface:

    ``detection(input_image, **keywords)``

The input is a two-dimensional image. Additional arguments to the specific
method can be given via **keywords. The output is an array of shape (n, m),
where each row corresponds to one of the n features. The first two columns
contain the coordinates (x, y) of the features, and additional information can
be specified in the remaining columns.

All implemented methods support the following keyword arguments:

+------------------+-----------------------------------------------------+
|       Key        |                Value                                |
+==================+=====================================================+
| max_num_features | maximum number of features to detect                |
+------------------+-----------------------------------------------------+

.. autosummary::
    :toctree: ../generated/

    get_method
"""

from pysteps.feature import blob
from pysteps.feature import tstorm
from pysteps.feature import shitomasi

_detection_methods = dict()
_detection_methods["blob"] = blob.detection
_detection_methods["tstorm"] = tstorm.detection
_detection_methods["shitomasi"] = shitomasi.detection


def get_method(name):
    """Return a callable function for feature detection.

    Implemented methods:

    +-----------------+-------------------------------------------------------+
    |     Name        |              Description                              |
    +=================+=======================================================+
    |  blob           | blob detection in scale space                         |
    +-----------------+-------------------------------------------------------+
    |  tstorm         | Thunderstorm cell detection                           |
    +-----------------+-------------------------------------------------------+
    |  shitomasi      | Shi-Tomasi corner detection                           |
    +-----------------+-------------------------------------------------------+
    """
    if isinstance(name, str):
        name = name.lower()
    else:
        raise TypeError(
            "Only strings supported for the method's names.\n"
            + "Available names:"
            + str(list(_detection_methods.keys()))
        ) from None

    try:
        return _detection_methods[name]
    except KeyError:
        raise ValueError(
            "Unknown detection method {}\n".format(name)
            + "The available methods are:"
            + str(list(_detection_methods.keys()))
        ) from None
