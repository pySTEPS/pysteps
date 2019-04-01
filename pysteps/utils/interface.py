"""
pysteps.utils.interface
=======================

Interface for the utils module.

.. autosummary::
    :toctree: ../generated/

    get_method
"""

from . import arrays
from . import conversion
from . import transformation
from . import dimension
from . import fft
from . import spectral

def get_method(name, **kwargs):
    """Return a callable function for the utility method corresponding to the
    given name.\n\

    Arrays methods:

    +-------------------+--------------------------------------------------------+
    |     Name          |              Description                               |
    +===================+========================================================+
    | centred_coord     | compute a 2D coordinate array                          |
    +-------------------+--------------------------------------------------------+

    Conversion methods:

    +-------------------+--------------------------------------------------------+
    |     Name          |              Description                               |
    +===================+========================================================+
    | mm/h or rainrate  | convert to rain rate [mm/h]                            |
    +-------------------+--------------------------------------------------------+
    | mm or raindepth   | convert to rain depth [mm]                             |
    +-------------------+--------------------------------------------------------+
    | dbz or            | convert to reflectivity [dBZ]                          |
    | reflectivity      |                                                        |
    +-------------------+--------------------------------------------------------+

    Transformation methods:

    +-------------------+--------------------------------------------------------+
    |     Name          |              Description                               |
    +===================+========================================================+
    | boxcox or box-cox | one-parameter Box-Cox transform                        |
    +-------------------+--------------------------------------------------------+
    | db or decibel     | transform to units of decibel                          |
    +-------------------+--------------------------------------------------------+
    | log               | log transform                                          |
    +-------------------+--------------------------------------------------------+
    | nqt               | Normal Quantile Transform                              |
    +-------------------+--------------------------------------------------------+
    | sqrt              | square-root transform                                  |
    +-------------------+--------------------------------------------------------+

    Dimension methods:

    +-------------------+--------------------------------------------------------+
    |     Name          |              Description                               |
    +===================+========================================================+
    |  accumulate       | aggregate fields in time                               |
    +-------------------+--------------------------------------------------------+
    |  clip             | resize the field domain by geographical coordinates    |
    +-------------------+--------------------------------------------------------+
    |  square           | either pad or crop the data to get a square domain     |
    +-------------------+--------------------------------------------------------+
    |  upscale          | upscale the field                                      |
    +-------------------+--------------------------------------------------------+

    FFT methods (wrappers to different implementations):

    +-------------------+--------------------------------------------------------+
    |     Name          |              Description                               |
    +===================+========================================================+
    |  numpy            | numpy.fft                                              |
    +-------------------+--------------------------------------------------------+
    |  scipy            | scipy.fftpack                                          |
    +-------------------+--------------------------------------------------------+
    |  pyfftw           | pyfftw.interfaces.numpy_fft                            |
    +-------------------+--------------------------------------------------------+

    Additional keyword arguments are passed to the initializer of the FFT
    methods, see utils.fft.

    Spectral methods:

    +-------------------+-----------------------------------------------------+
    |     Name          |              Description                            |
    +===================+=====================================================+
    |  rapsd            | Compute radially averaged power spectral density    |
    +-------------------+-----------------------------------------------------+
    |  rm_rdisc         | remove the rain / no-rain discontinuity             |
    +-------------------+-----------------------------------------------------+

    """

    if name is None:
        name = "none"

    name = name.lower()

    def donothing(R, metadata=None, *args, **kwargs):
        return R.copy(), {} if metadata is None else metadata.copy()

    methods_objects                  = dict()
    methods_objects["none"]          = donothing
    # arrays methods
    methods_objects["centred_coord"] = arrays.compute_centred_coord_array
    # conversion methods
    methods_objects["mm/h"]          = conversion.to_rainrate
    methods_objects["rainrate"]      = conversion.to_rainrate
    methods_objects["mm"]            = conversion.to_raindepth
    methods_objects["raindepth"]     = conversion.to_raindepth
    methods_objects["dbz"]           = conversion.to_reflectivity
    methods_objects["reflectivity"]  = conversion.to_reflectivity
    # transformation methods
    methods_objects["boxcox"]        = transformation.boxcox_transform
    methods_objects["box-cox"]       = transformation.boxcox_transform
    methods_objects["db"]            = transformation.dB_transform
    methods_objects["decibel"]       = transformation.dB_transform
    methods_objects["log"]           = transformation.boxcox_transform
    methods_objects["nqt"]           = transformation.NQ_transform
    methods_objects["sqrt"]          = transformation.sqrt_transform
    # dimension methods
    methods_objects["accumulate"]    = dimension.aggregate_fields_time
    methods_objects["clip"]          = dimension.clip_domain
    methods_objects["square"]        = dimension.square_domain
    methods_objects["upscale"]       = dimension.aggregate_fields_space
    # FFT methods
    if name in ["numpy", "pyfftw", "scipy"]:
        if "shape" not in kwargs.keys():
            raise KeyError("mandatory keyword argument shape not given")
        return _get_fft_method(name, **kwargs)
    else:
        try:
            return methods_objects[name]
        except KeyError as e:
            raise ValueError("Unknown method %s\n" % e +
                             "Supported methods:%s" % str(methods_objects.keys()))
    # spectral methods
    methods_objects["rapsd"]        = spectral.rapsd
    methods_objects["rm_rdisc"]     = spectral.remove_rain_norain_discontinuity

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
        raise ValueError("Unknown method {}\n".format(name)
                         + "The available methods are:"
                         + str(["numpy", "pyfftw", "scipy"])) from None
