
from . import fftgenerators
from . import motion

def get_method(name):
    """Return two callable functions to initialize and generate 2d perturbations
    of precipitation  or velocity fields.\n\

    Methods for precipitation fields:

    +-------------------+-------------------------------------------------------+
    |     Name          |              Description                              |
    +===================+=======================================================+
    |  parametric       | this global generator uses parametric Fourier         |
    |                   | filering (power-law model)                            |
    +-------------------+-------------------------------------------------------+
    |  nonparametric    | this global generator uses nonparametric Fourier      |
    |                   | filering                                              |
    +-------------------+-------------------------------------------------------+
    |  ssft             | this local generator uses the short-space Fourier     |
    |                   | filtering                                             |
    +-------------------+-------------------------------------------------------+
    |  nested           | this local generator uses a nested Fourier filtering  |
    +-------------------+-------------------------------------------------------+

    Methods for velocity fields:

    +-------------------+-----------------------------------------------------+
    |     Name          |              Description                            |
    +===================+=====================================================+
    |  bps              | The method described in :cite:`BPS2006`, where      |
    |                   | time-dependent velocity perturbations are sampled   |
    |                   | from the exponential distribution                   |
    +-------------------+-----------------------------------------------------+

    """
    if name.lower() == "parametric":
        return fftgenerators.initialize_param_2d_fft_filter, \
            fftgenerators.generate_noise_2d_fft_filter
    elif name.lower() == "nonparametric":
        return fftgenerators.initialize_nonparam_2d_fft_filter, \
            fftgenerators.generate_noise_2d_fft_filter
    elif name.lower() == "ssft":
        return fftgenerators.initialize_nonparam_2d_ssft_filter, \
            fftgenerators.generate_noise_2d_ssft_filter
    elif name.lower() == "nested":
        return fftgenerators.initialize_nonparam_2d_nested_filter, \
            fftgenerators.generate_noise_2d_ssft_filter
    elif name.lower() == "bps":
        return motion.initialize_bps, motion.generate_bps
    else:
        raise ValueError("unknown perturbation method %s" % name)
