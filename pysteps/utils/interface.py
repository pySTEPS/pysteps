
from . import conversion
from . import transformation
from . import dimension

def get_method(name):
    """Return a callable function for the bandpass filter or decomposition method 
    corresponding to the given name.\n\
    
    Conversion methods:
    +-------------------+--------------------------------------------------------+
    |     Name          |              Description                               |
    +===================+========================================================+
    | mm/h or rainrate  | convert to rain rate [mm/h]                            |
    +-------------------+--------------------------------------------------------+
    | mm or raindepth   | convert to rain depth [mm]                             |
    +-------------------+--------------------------------------------------------+
    | dBZ or reflectivity  | convert to reflectivity [dBZ]                       |
    +-------------------+--------------------------------------------------------+
    
    Transformation methods:
    +-------------------+--------------------------------------------------------+
    |     Name          |              Description                               |
    +===================+========================================================+
    |  dB, decibel      | transform to units of decibel                          |
    +-------------------+--------------------------------------------------------+
    |  BoxCox           | apply one-parameter Box-Cox transform                  |
    +-------------------+--------------------------------------------------------+
    
    Dimension methods:
    +-------------------+--------------------------------------------------------+
    |     Name          |              Description                               |
    +===================+========================================================+
    |  aggregate        | aggregate fields in time                               |
    +-------------------+--------------------------------------------------------+
    |  square           | either pad or crop the data to get a square domain     |
    +-------------------+--------------------------------------------------------+
    """
    if name is None:
        def donothing(R, metadata, *args, **kwargs):
            return R.copy(), metadata.copy()
        return donothing
    elif name.lower() in ["mm/h", "rainrate"]:
        return conversion.to_rainrate
    elif name.lower() in ["mm", "raindepth"]:
        return conversion.to_raindepth
    elif name.lower() in ["dbz", "reflectivity"]:
        return conversion.to_reflectivity
    elif name.lower() in ["db", "decibel"]:
        return transformation.dB_transform
    elif name.lower() in ["boxcox", "box-cox"]:
        return transformation.boxcox_transform
    elif name.lower() in ["aggregate", "accumulate"]:
        return dimension.aggregate_fields_time
    elif name.lower() in ["square"]:
        return dimension.square_domain
    else:
        raise ValueError("unknown method %s" % name)
