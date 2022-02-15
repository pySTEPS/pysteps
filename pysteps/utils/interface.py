# -*- coding: utf-8 -*-
"""
pysteps.utils.interface
=======================

Interface for the utils module.

.. autosummary::
    :toctree: ../generated/

    get_method
"""

from . import arrays
from . import cleansing
from . import conversion
from . import dimension
from . import fft
from . import images
from . import interpolate
from . import reprojection
from . import spectral
from . import tapering
from . import transformation


def get_method(name, **kwargs):
    """
    Return a callable function for the utility method corresponding to the
    given name.\n\

    Arrays methods:

    +-------------------+-----------------------------------------------------+
    |     Name          |              Description                            |
    +===================+=====================================================+
    | centred_coord     | compute a 2D coordinate array                       |
    +-------------------+-----------------------------------------------------+

    Cleansing methods:

    +-------------------+-----------------------------------------------------+
    |     Name          |              Description                            |
    +===================+=====================================================+
    | decluster         | decluster a set of sparse data points               |
    +-------------------+-----------------------------------------------------+
    | detect_outliers   | detect outliers in a dataset                        |
    +-------------------+-----------------------------------------------------+

    Conversion methods:

    +-------------------+-----------------------------------------------------+
    |     Name          |              Description                            |
    +===================+=====================================================+
    | mm/h or rainrate  | convert to rain rate [mm/h]                         |
    +-------------------+-----------------------------------------------------+
    | mm or raindepth   | convert to rain depth [mm]                          |
    +-------------------+-----------------------------------------------------+
    | dbz or            | convert to reflectivity [dBZ]                       |
    | reflectivity      |                                                     |
    +-------------------+-----------------------------------------------------+

    Dimension methods:

    +-------------------+-----------------------------------------------------+
    |     Name          |              Description                            |
    +===================+=====================================================+
    |  accumulate       | aggregate fields in time                            |
    +-------------------+-----------------------------------------------------+
    |  clip             | resize the field domain by geographical coordinates |
    +-------------------+-----------------------------------------------------+
    |  square           | either pad or crop the data to get a square domain  |
    +-------------------+-----------------------------------------------------+
    |  upscale          | upscale the field                                   |
    +-------------------+-----------------------------------------------------+

    FFT methods (wrappers to different implementations):

    +-------------------+-----------------------------------------------------+
    |     Name          |              Description                            |
    +===================+=====================================================+
    |  numpy            | numpy.fft                                           |
    +-------------------+-----------------------------------------------------+
    |  scipy            | scipy.fftpack                                       |
    +-------------------+-----------------------------------------------------+
    |  pyfftw           | pyfftw.interfaces.numpy_fft                         |
    +-------------------+-----------------------------------------------------+

    Image processing methods:

    +-------------------+-----------------------------------------------------+
    |     Name          |              Description                            |
    +===================+=====================================================+
    |  morph_opening    | filter small scale noise                            |
    +-------------------+-----------------------------------------------------+

    Interpolation methods:

    +-------------------+-----------------------------------------------------+
    |     Name          |              Description                            |
    +===================+=====================================================+
    |  rbfinterp2d      | Radial Basis Function (RBF) interpolation of a      |
    |                   | (multivariate) array over a 2D grid.                |
    +-------------------+-----------------------------------------------------+
    |  idwinterp2d      | Inverse distance weighting (IDW) interpolation of a |
    |                   | (multivariate) array over a 2D grid.                |
    +-------------------+-----------------------------------------------------+

    Additional keyword arguments are passed to the initializer of the FFT
    methods, see utils.fft.

    Reprojection methods:

    +-------------------+-----------------------------------------------------+
    |     Name          |              Description                            |
    +===================+=====================================================+
    |  reproject_grids  | Reproject grids to a destination grid.              |
    +-------------------+-----------------------------------------------------+

    Spectral methods:

    +-------------------+-----------------------------------------------------+
    |     Name          |              Description                            |
    +===================+=====================================================+
    |  rapsd            | Compute radially averaged power spectral density    |
    +-------------------+-----------------------------------------------------+
    |  rm_rdisc         | remove the rain / no-rain discontinuity             |
    +-------------------+-----------------------------------------------------+

    Tapering methods:

    +-------------------------------+-----------------------------------------+
    |     Name                      |              Description                |
    +===============================+=========================================+
    |  compute_mask_window_function | Compute window function for a           |
    |                               | two-dimensional area defined by a       |
    |                               | non-rectangular mask.                   |
    +-------------------------------+-----------------------------------------+
    |  compute_window_function      | Compute window function for a           |
    |                               | two-dimensional rectangular area.       |
    +-------------------------------+-----------------------------------------+

    Transformation methods:

    +-------------------+-----------------------------------------------------+
    |     Name          |              Description                            |
    +===================+=====================================================+
    | boxcox or box-cox | one-parameter Box-Cox transform                     |
    +-------------------+-----------------------------------------------------+
    | db or decibel     | transform to units of decibel                       |
    +-------------------+-----------------------------------------------------+
    | log               | log transform                                       |
    +-------------------+-----------------------------------------------------+
    | nqt               | Normal Quantile Transform                           |
    +-------------------+-----------------------------------------------------+
    | sqrt              | square-root transform                               |
    +-------------------+-----------------------------------------------------+

    """

    if name is None:
        name = "none"

    name = name.lower()

    def donothing(R, metadata=None, *args, **kwargs):
        return R.copy(), {} if metadata is None else metadata.copy()

    methods_objects = dict()
    methods_objects["none"] = donothing

    # arrays methods
    methods_objects["centred_coord"] = arrays.compute_centred_coord_array

    # cleansing methods
    methods_objects["decluster"] = cleansing.decluster
    methods_objects["detect_outliers"] = cleansing.detect_outliers

    # conversion methods
    methods_objects["mm/h"] = conversion.to_rainrate
    methods_objects["rainrate"] = conversion.to_rainrate
    methods_objects["mm"] = conversion.to_raindepth
    methods_objects["raindepth"] = conversion.to_raindepth
    methods_objects["dbz"] = conversion.to_reflectivity
    methods_objects["reflectivity"] = conversion.to_reflectivity

    # dimension methods
    methods_objects["accumulate"] = dimension.aggregate_fields_time
    methods_objects["clip"] = dimension.clip_domain
    methods_objects["square"] = dimension.square_domain
    methods_objects["upscale"] = dimension.aggregate_fields_space

    # image processing methods
    methods_objects["morph_opening"] = images.morph_opening

    # interpolation methods
    methods_objects["rbfinterp2d"] = interpolate.rbfinterp2d
    methods_objects["idwinterp2d"] = interpolate.idwinterp2d

    # reprojection methods
    methods_objects["reproject_grids"] = reprojection.reproject_grids

    # spectral methods
    methods_objects["rapsd"] = spectral.rapsd
    methods_objects["rm_rdisc"] = spectral.remove_rain_norain_discontinuity

    # tapering methods
    methods_objects[
        "compute_mask_window_function"
    ] = tapering.compute_mask_window_function
    methods_objects["compute_window_function"] = tapering.compute_window_function

    # transformation methods
    methods_objects["boxcox"] = transformation.boxcox_transform
    methods_objects["box-cox"] = transformation.boxcox_transform
    methods_objects["db"] = transformation.dB_transform
    methods_objects["decibel"] = transformation.dB_transform
    methods_objects["log"] = transformation.boxcox_transform
    methods_objects["nqt"] = transformation.NQ_transform
    methods_objects["sqrt"] = transformation.sqrt_transform

    # FFT methods
    if name in ["numpy", "pyfftw", "scipy"]:
        if "shape" not in kwargs.keys():
            raise KeyError("mandatory keyword argument shape not given")
        return _get_fft_method(name, **kwargs)
    else:
        try:
            return methods_objects[name]
        except KeyError as e:
            raise ValueError(
                "Unknown method %s\n" % e
                + "Supported methods:%s" % str(methods_objects.keys())
            )


def _get_fft_method(name, **kwargs):
    kwargs = kwargs.copy()
    shape = kwargs["shape"]
    kwargs.pop("shape")

    if name == "numpy":
        return fft.get_numpy(shape, **kwargs)
    elif name == "scipy":
        return fft.get_scipy(shape, **kwargs)
    elif name == "pyfftw":
        return fft.get_pyfftw(shape, **kwargs)
    else:
        raise ValueError(
            "Unknown method {}\n".format(name)
            + "The available methods are:"
            + str(["numpy", "pyfftw", "scipy"])
        ) from None
