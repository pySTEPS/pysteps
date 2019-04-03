"""
pysteps.noise.fftgenerators
===========================

Methods for noise generators based on FFT filtering of white noise.

The methods in this module implement the following interface for filter
initialization depending on their parametric or nonparametric nature::

  initialize_param_2d_xxx_filter(X, **kwargs)

or::

  initialize_nonparam_2d_xxx_filter(X, **kwargs)

where X is an array of shape (m, n) or (t, m, n) that defines the target field 
and optional parameters are supplied as keyword arguments.

The output of each initialization method is a dictionary containing the keys F
and input_shape. The first is a two-dimensional array of shape (m, int(n/2)+1)
that defines the filter. The second one is the shape of the input field for the
filter.

The methods in this module implement the following interface for the generation
of correlated noise::

  generate_noise_2d_xxx_filter(F, randstate=np.random, seed=None, **kwargs)

where F (m, n) is a filter returned from the corresponding initialization
method, and randstate and seed can be used to set the random generator and
its seed. Additional keyword arguments can be included as a dictionary.

The output of each generator method is a two-dimensional array containing the
field of correlated noise cN of shape (m, n).

.. autosummary::
    :toctree: ../generated/

    initialize_param_2d_fft_filter
    initialize_nonparam_2d_fft_filter
    initialize_nonparam_2d_nested_filter
    initialize_nonparam_2d_ssft_filter
    generate_noise_2d_fft_filter
    generate_noise_2d_ssft_filter
"""

import numpy as np
from scipy import optimize
from .. import utils


def initialize_param_2d_fft_filter(X, **kwargs):
    """Takes one ore more 2d input fields, fits two spectral slopes, beta1 and beta2,
    to produce one parametric, global and isotropic fourier filter.

    Parameters
    ----------
    X : array-like
        Two- or three-dimensional array containing one or more input fields.
        All values are required to be finite. If more than one field are passed,
        the average fourier filter is returned. It assumes that fields are stacked
        by the first axis: [nr_fields, y, x].

    Other Parameters
    ----------------
    win_type : {'hanning', 'flat-hanning' or None}
        Optional tapering function to be applied to X, generated with
        :py:func:`pysteps.noise.fftgenerators.build_2D_tapering_function`
        (default None).
    model : {'power-law'}
        The name of the parametric model to be used to fit the power spectrum of 
        X (default 'power-law').
    weighted : bool
        Whether or not to apply 1/sqrt(power) as weight in the numpy.polyfit() 
        function (default False).
    rm_rdisc : bool
        Whether or not to remove the rain/no-rain disconituity (default False). 
        It assumes no-rain pixels are assigned with lowest value.
    fft_method : str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see :py:func:`pysteps.utils.fft.get_method`). Defaults to "numpy".

    Returns
    -------
    out : dict
        A a dictionary containing the keys F, input_shape, model and pars.
        The first is a two-dimensional array of shape (m, int(n/2)+1) that 
        defines the filter. The second one is the shape of the input field for 
        the filter. The last two are the model and fitted parameters, 
        respectively.

        This dictionary can be passed to 
        :py:func:`pysteps.noise.fftgenerators.generate_noise_2d_fft_filter` to
        generate noise fields.
    """

    if len(X.shape) < 2 or len(X.shape) > 3:
        raise ValueError("the input is not two- or three-dimensional array")
    if np.any(~np.isfinite(X)):
        raise ValueError("X contains non-finite values")

    # defaults
    win_type = kwargs.get("win_type", None)
    model = kwargs.get("model", "power-law")
    weighted = kwargs.get("weighted", False)
    rm_rdisc = kwargs.get("rm_rdisc", False)
    fft = kwargs.get("fft_method", "numpy")
    if type(fft) == str:
        fft_shape = X.shape if len(X.shape) == 2 else X.shape[1:]
        fft = utils.get_method(fft, shape=fft_shape)

    X = X.copy()

    # remove rain/no-rain discontinuity
    if rm_rdisc:
        X[X > X.min()] -= X[X > X.min()].min() - X.min()

    # dims
    if len(X.shape) == 2:
        X = X[None, :, :]
    nr_fields = X.shape[0]
    M, N = X.shape[1:]

    if win_type is not None:
        tapering = build_2D_tapering_function((M, N), win_type)

        # make sure non-rainy pixels are set to zero
        X -= X.min(axis=(1, 2))[:, None, None]
    else:
        tapering = np.ones((M, N))

    if model.lower() == "power-law":

        # compute average 2D PSD
        F = np.zeros((M, N), dtype=complex)
        for i in range(nr_fields):
            F += fft.fftshift(fft.fft2(X[i, :, :] * tapering))
        F /= nr_fields
        F = abs(F) ** 2 / F.size

        # compute radially averaged 1D PSD
        psd = utils.spectral.rapsd(F)
        L = max(M, N)

        # wavenumbers
        if L % 2 == 0:
            wn = np.arange(0, int(L / 2) + 1)
        else:
            wn = np.arange(0, int(L / 2))

        # compute single spectral slope beta as first guess
        if weighted:
            p0 = np.polyfit(np.log(wn[1:]), np.log(psd[1:]), 1, w=np.sqrt(psd[1:]))
        else:
            p0 = np.polyfit(np.log(wn[1:]), np.log(psd[1:]), 1)
        beta = p0[0]

        # create the piecewise function with two spectral slopes beta1 and beta2
        # and scaling break x0
        def piecewise_linear(x, x0, y0, beta1, beta2):
            return np.piecewise(
                x,
                [x < x0, x >= x0],
                [
                    lambda x: beta1 * x + y0 - beta1 * x0,
                    lambda x: beta2 * x + y0 - beta2 * x0,
                ],
            )

        # fit the two betas and the scaling break
        p0 = [2.0, 0, beta, beta]  # first guess
        bounds = (
            [2.0, 0, -4, -4],
            [5.0, 20, -1.0, -1.0],
        )  # TODO: provide better bounds
        if weighted:
            p, e = optimize.curve_fit(
                piecewise_linear,
                np.log(wn[1:]),
                np.log(psd[1:]),
                p0=p0,
                bounds=bounds,
                sigma=1 / np.sqrt(psd[1:]),
            )
        else:
            p, e = optimize.curve_fit(
                piecewise_linear, np.log(wn[1:]), np.log(psd[1:]), p0=p0, bounds=bounds
            )

        # compute 2d filter
        YC, XC = utils.arrays.compute_centred_coord_array(M, N)
        R = np.sqrt(XC * XC + YC * YC)
        R = fft.fftshift(R)
        pf = p.copy()
        pf[2:] = pf[2:] / 2
        F = np.exp(piecewise_linear(np.log(R), *pf))
        F[~np.isfinite(F)] = 1

        f = piecewise_linear

    else:
        raise ValueError("unknown parametric model %s" % model)

    return {
        "F": F,
        "input_shape": X.shape[1:],
        "use_full_fft": True,
        "model": f,
        "pars": p,
    }


def initialize_nonparam_2d_fft_filter(X, **kwargs):
    """Takes one ore more 2d input fields and produces one non-paramtric, global
    and anasotropic fourier filter.

    Parameters
    ----------
    X : array-like
        Two- or three-dimensional array containing one or more input fields.
        All values are required to be finite. If more than one field are passed,
        the average fourier filter is returned. It assumes that fields are stacked
        by the first axis: [nr_fields, y, x].

    Other Parameters
    ----------------
    win_type : {'hanning', 'flat-hanning'}
        Optional tapering function to be applied to X, generated with
        :py:func:`pysteps.noise.fftgenerators.build_2D_tapering_function`
        (default 'flat-hanning').
    donorm : bool
       Option to normalize the real and imaginary parts.
       Default : False
    rm_rdisc : bool
        Whether or not to remove the rain/no-rain disconituity (default True). 
        It assumes no-rain pixels are assigned with lowest value.
    fft_method : str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see :py:func:`pysteps.utils.fft.get_method`). Defaults to "numpy".

    Returns
    -------
    out : dict
      A dictionary containing the keys F and input_shape. The first is a 
      two-dimensional array of shape (m, int(n/2)+1) that defines the filter. 
      The second one is the shape of the input field for the filter.
      
      It can be passed to 
      :py:func:`pysteps.noise.fftgenerators.generate_noise_2d_fft_filter`.
    """
    if len(X.shape) < 2 or len(X.shape) > 3:
        raise ValueError("the input is not two- or three-dimensional array")
    if np.any(~np.isfinite(X)):
        raise ValueError("X contains non-finite values")

    # defaults
    win_type = kwargs.get("win_type", "flat-hanning")
    donorm = kwargs.get("donorm", False)
    rm_rdisc = kwargs.get("rm_rdisc", True)
    use_full_fft = kwargs.get("use_full_fft", False)
    fft = kwargs.get("fft_method", "numpy")
    if type(fft) == str:
        fft_shape = X.shape if len(X.shape) == 2 else X.shape[1:]
        fft = utils.get_method(fft, shape=fft_shape)

    X = X.copy()

    # remove rain/no-rain discontinuity
    if rm_rdisc:
        X[X > X.min()] -= X[X > X.min()].min() - X.min()

    # dims
    if len(X.shape) == 2:
        X = X[None, :, :]
    nr_fields = X.shape[0]
    field_shape = X.shape[1:]
    if use_full_fft:
        fft_shape = (X.shape[1], X.shape[2])
    else:
        fft_shape = (X.shape[1], int(X.shape[2] / 2) + 1)

    # make sure non-rainy pixels are set to zero
    X -= X.min(axis=(1, 2))[:, None, None]

    if win_type is not None:
        tapering = build_2D_tapering_function(field_shape, win_type)
    else:
        tapering = np.ones(field_shape)

    F = np.zeros(fft_shape, dtype=complex)
    for i in range(nr_fields):
        if use_full_fft:
            F += fft.fft2(X[i, :, :] * tapering)
        else:
            F += fft.rfft2(X[i, :, :] * tapering)
    F /= nr_fields

    # normalize the real and imaginary parts
    if donorm:
        if np.std(F.imag) > 0:
            F.imag = (F.imag - np.mean(F.imag)) / np.std(F.imag)
        if np.std(F.real) > 0:
            F.real = (F.real - np.mean(F.real)) / np.std(F.real)

    return {"F": np.abs(F), "input_shape": X.shape[1:], "use_full_fft": use_full_fft}


def generate_noise_2d_fft_filter(F, randstate=None, seed=None, fft_method=None):
    """Produces a field of correlated noise using global Fourier filtering.

    Parameters
    ----------
    F : dict
        A filter object returned by 
        :py:func:`pysteps.noise.fftgenerators.initialize_param_2d_fft_filter` or
        :py:func:`pysteps.noise.fftgenerators.initialize_nonparam_2d_fft_filter`. 
        All values in the filter array are required to be finite.
    randstate : mtrand.RandomState
        Optional random generator to use. If set to None, use numpy.random.
    seed : int
        Value to set a seed for the generator. None will not set the seed.
    fft_method : str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see :py:func:`pysteps.utils.fft.get_method`). Defaults to "numpy".

    Returns
    -------
    N : array-like
        A two-dimensional numpy array of stationary correlated noise.
    """
    input_shape = F["input_shape"]
    use_full_fft = F["use_full_fft"]
    F = F["F"]

    if len(F.shape) != 2:
        raise ValueError("F is not two-dimensional array")
    if np.any(~np.isfinite(F)):
        raise ValueError("F contains non-finite values")

    if randstate is None:
        randstate = np.random

    # set the seed
    if seed is not None:
        randstate.seed(seed)

    if fft_method is None:
        fft = utils.get_method("numpy", shape=input_shape)
    else:
        if type(fft_method) == str:
            fft = utils.get_method(fft_method, shape=input_shape)
        else:
            fft = fft_method

    # produce fields of white noise
    N = randstate.randn(input_shape[0], input_shape[1])

    # apply the global Fourier filter to impose a correlation structure
    if use_full_fft:
        fN = fft.fft2(N)
    else:
        fN = fft.rfft2(N)
    fN *= F
    if use_full_fft:
        N = np.array(fft.ifft2(fN).real)
    else:
        N = np.array(fft.irfft2(fN))
    N = (N - N.mean()) / N.std()

    return N


def initialize_nonparam_2d_ssft_filter(X, **kwargs):
    """Function to compute the local Fourier filters using the Short-Space Fourier
    filtering approach.

    Parameters
    ----------
    X : array-like
        Two- or three-dimensional array containing one or more input fields.
        All values are required to be finite. If more than one field are passed,
        the average fourier filter is returned. It assumes that fields are stacked
        by the first axis: [nr_fields, y, x].

    Other Parameters
    ----------------
    win_size : int or two-element tuple of ints
        Size-length of the window to compute the SSFT (default (128, 128)).
    win_type : {'hanning', 'flat-hanning'}
        Optional tapering function to be applied to X, generated with
        :py:func:`pysteps.noise.fftgenerators.build_2D_tapering_function`
        (default 'flat-hanning').
    overlap : float [0,1[
        The proportion of overlap to be applied between successive windows 
        (default 0.3).
    war_thr : float [0,1]
        Threshold for the minimum fraction of rain needed for computing the FFT
        (default 0.1).
    rm_rdisc : bool
        Whether or not to remove the rain/no-rain disconituity. It assumes no-rain
        pixels are assigned with lowest value.
    fft_method : str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see :py:func:`pysteps.utils.fft.get_method`). Defaults to "numpy".

    Returns
    -------
    F : array-like
        Four-dimensional array containing the 2d fourier filters distributed over
        a 2d spatial grid. 
        It can be passed to 
        :py:func:`pysteps.noise.fftgenerators.generate_noise_2d_ssft_filter`.

    References
    ----------
    :cite:`NBSG2017`

    """

    if len(X.shape) < 2 or len(X.shape) > 3:
        raise ValueError("the input is not two- or three-dimensional array")
    if np.any(np.isnan(X)):
        raise ValueError("X must not contain NaNs")

    # defaults
    win_size = kwargs.get("win_size", (128, 128))
    if type(win_size) == int:
        win_size = (win_size, win_size)
    win_type = kwargs.get("win_type", "flat-hanning")
    overlap = kwargs.get("overlap", 0.3)
    war_thr = kwargs.get("war_thr", 0.1)
    rm_rdisc = kwargs.get("rm_disc", True)
    fft = kwargs.get("fft_method", "numpy")
    if type(fft) == str:
        fft_shape = X.shape if len(X.shape) == 2 else X.shape[1:]
        fft = utils.get_method(fft, shape=fft_shape)

    X = X.copy()

    # remove rain/no-rain discontinuity
    if rm_rdisc:
        X[X > X.min()] -= X[X > X.min()].min() - X.min()

    # dims
    if len(X.shape) == 2:
        X = X[None, :, :]
    nr_fields = X.shape[0]
    dim = X.shape[1:]
    dim_x = dim[1]
    dim_y = dim[0]

    # make sure non-rainy pixels are set to zero
    X -= X.min(axis=(1, 2))[:, None, None]

    # SSFT algorithm

    # prepare indices
    idxi = np.zeros((2, 1), dtype=int)
    idxj = np.zeros((2, 1), dtype=int)

    # number of windows
    num_windows_y = np.ceil(float(dim_y) / win_size[0]).astype(int)
    num_windows_x = np.ceil(float(dim_x) / win_size[1]).astype(int)

    # domain fourier filter
    F0 = initialize_nonparam_2d_fft_filter(
        X, win_type=win_type, donorm=True, use_full_fft=True, fft_method=fft
    )["F"]
    # and allocate it to the final grid
    F = np.zeros((num_windows_y, num_windows_x, F0.shape[0], F0.shape[1]))
    F += F0[np.newaxis, np.newaxis, :, :]

    # loop rows
    for i in range(F.shape[0]):
        # loop columns
        for j in range(F.shape[1]):

            # compute indices of local window
            idxi[0] = int(np.max((i * win_size[0] - overlap * win_size[0], 0)))
            idxi[1] = int(
                np.min((idxi[0] + win_size[0] + overlap * win_size[0], dim_y))
            )
            idxj[0] = int(np.max((j * win_size[1] - overlap * win_size[1], 0)))
            idxj[1] = int(
                np.min((idxj[0] + win_size[1] + overlap * win_size[1], dim_x))
            )

            # build localization mask
            # TODO: the 0.01 rain threshold must be improved
            mask = _get_mask(dim, idxi, idxj, win_type)
            war = float(np.sum((X * mask[None, :, :]) > 0.01)) / (
                (idxi[1] - idxi[0]) * (idxj[1] - idxj[0]) * nr_fields
            )

            if war > war_thr:
                # the new filter
                F[i, j, :, :] = initialize_nonparam_2d_fft_filter(
                    X * mask[None, :, :],
                    win_type=None,
                    donorm=True,
                    use_full_fft=True,
                    fft_method=fft,
                )["F"]

    return {"F": F, "input_shape": X.shape[1:], "use_full_fft": True}


def initialize_nonparam_2d_nested_filter(X, gridres=1.0, **kwargs):
    """Function to compute the local Fourier filters using a nested approach.

    Parameters
    ----------
    X : array-like
        Two- or three-dimensional array containing one or more input fields.
        All values are required to be finite.
        If more than one field are passed, the average fourier filter is returned.
        It assumes that fields are stacked by the first axis: [nr_fields, y, x].
    gridres : float
        Grid resolution in km.

    Other Parameters
    ----------------
    max_level : int
        Localization parameter. 0: global noise, >0: increasing degree of 
        localization (default 3).
    win_type : {'hanning', 'flat-hanning'}
        Optional tapering function to be applied to X, generated with
        :py:func:`pysteps.noise.fftgenerators.build_2D_tapering_function`
        (default 'flat-hanning').
    war_thr : float [0;1]
        Threshold for the minimum fraction of rain needed for computing the FFT
        (default 0.1).
    rm_rdisc : bool
        Whether or not to remove the rain/no-rain disconituity. It assumes no-rain
        pixels are assigned with lowest value.
    fft_method : str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see :py:func:`pysteps.utils.fft.get_method`). Defaults to "numpy".

    Returns
    -------
    F : array-like
        Four-dimensional array containing the 2d fourier filters distributed over
        a 2d spatial grid.
        It can be passed to 
        :py:func:`pysteps.noise.fftgenerators.generate_noise_2d_ssft_filter`.
    """

    if len(X.shape) < 2 or len(X.shape) > 3:
        raise ValueError("the input is not two- or three-dimensional array")
    if np.any(np.isnan(X)):
        raise ValueError("X must not contain NaNs")

    # defaults
    max_level = kwargs.get("max_level", 3)
    win_type = kwargs.get("win_type", "flat-hanning")
    war_thr = kwargs.get("war_thr", 0.1)
    rm_rdisc = kwargs.get("rm_disc", True)
    fft = kwargs.get("fft_method", "numpy")
    if type(fft) == str:
        fft_shape = X.shape if len(X.shape) == 2 else X.shape[1:]
        fft = utils.get_method(fft, shape=fft_shape)

    X = X.copy()

    # remove rain/no-rain discontinuity
    if rm_rdisc:
        X[X > X.min()] -= X[X > X.min()].min() - X.min()

    # dims
    if len(X.shape) == 2:
        X = X[None, :, :]
    nr_fields = X.shape[0]
    dim = X.shape[1:]
    dim_x = dim[1]
    dim_y = dim[0]

    # make sure non-rainy pixels are set to zero
    X -= X.min(axis=(1, 2))[:, None, None]

    # Nested algorithm

    # prepare indices
    Idxi = np.array([[0, dim_y]])
    Idxj = np.array([[0, dim_x]])
    Idxipsd = np.array([[0, 2 ** max_level]])
    Idxjpsd = np.array([[0, 2 ** max_level]])

    # generate the FFT sample frequencies
    freqx = fft.fftfreq(dim_x, gridres)
    freqy = fft.fftfreq(dim_y, gridres)
    fx, fy = np.meshgrid(freqx, freqy)
    freq_grid = np.sqrt(fx ** 2 + fy ** 2)

    # domain fourier filter
    F0 = initialize_nonparam_2d_fft_filter(
        X, win_type=win_type, donorm=True, use_full_fft=True, fft_method=fft
    )["F"]
    # and allocate it to the final grid
    F = np.zeros((2 ** max_level, 2 ** max_level, F0.shape[0], F0.shape[1]))
    F += F0[np.newaxis, np.newaxis, :, :]

    # now loop levels and build composite spectra
    level = 0
    while level < max_level:

        for m in range(len(Idxi)):

            # the indices of rainfall field
            Idxinext, Idxjnext = _split_field(Idxi[m, :], Idxj[m, :], 2)
            # the indices of the field of fourier filters
            Idxipsdnext, Idxjpsdnext = _split_field(Idxipsd[m, :], Idxjpsd[m, :], 2)

            for n in range(len(Idxinext)):

                mask = _get_mask(dim, Idxinext[n, :], Idxjnext[n, :], win_type)
                war = np.sum((X * mask[None, :, :]) > 0.01) / float(
                    (Idxinext[n, 1] - Idxinext[n, 0])
                    * (Idxjnext[n, 1] - Idxjnext[n, 0])
                    * nr_fields
                )

                if war > war_thr:
                    # the new filter
                    newfilter = initialize_nonparam_2d_fft_filter(
                        X * mask[None, :, :],
                        win_type=None,
                        donorm=True,
                        use_full_fft=True,
                        fft_method=fft,
                    )["F"]

                    # compute logistic function to define weights as function of frequency
                    # k controls the shape of the weighting function
                    # TODO: optimize parameters
                    k = 0.05
                    x0 = (
                        Idxinext[n, 1] - Idxinext[n, 0]
                    ) / 2.0  # TODO: consider y dimension, too
                    merge_weights = 1 / (1 + np.exp(-k * (1 / freq_grid - x0)))
                    newfilter *= 1 - merge_weights

                    # perform the weighted average of previous and new fourier filters
                    F[
                        Idxipsdnext[n, 0] : Idxipsdnext[n, 1],
                        Idxjpsdnext[n, 0] : Idxjpsdnext[n, 1],
                        :,
                        :,
                    ] *= merge_weights[np.newaxis, np.newaxis, :, :]
                    F[
                        Idxipsdnext[n, 0] : Idxipsdnext[n, 1],
                        Idxjpsdnext[n, 0] : Idxjpsdnext[n, 1],
                        :,
                        :,
                    ] += newfilter[np.newaxis, np.newaxis, :, :]

        # update indices
        level += 1
        Idxi, Idxj = _split_field((0, dim[0]), (0, dim[1]), 2 ** level)
        Idxipsd, Idxjpsd = _split_field(
            (0, 2 ** max_level), (0, 2 ** max_level), 2 ** level
        )

    return {"F": F, "input_shape": X.shape[1:], "use_full_fft": True}


def generate_noise_2d_ssft_filter(F, randstate=None, seed=None, **kwargs):
    """Function to compute the locally correlated noise using a nested approach.

    Parameters
    ----------
    F : array-like
        A filter object returned by 
        :py:func:`pysteps.noise.fftgenerators.initialize_nonparam_2d_ssft_filter` or
        :py:func:`pysteps.noise.fftgenerators.initialize_nonparam_2d_nested_filter`. 
        The filter is a four-dimensional array containing the 2d fourier filters 
        distributed over a 2d spatial grid.
    randstate : mtrand.RandomState
        Optional random generator to use. If set to None, use numpy.random.
    seed : int
        Value to set a seed for the generator. None will not set the seed.

    Other Parameters
    ----------------
    overlap : float
        Percentage overlap [0-1] between successive windows (default 0.2).
    win_type : {'hanning', 'flat-hanning'}
        Optional tapering function to be applied to X, generated with
        :py:func:`pysteps.noise.fftgenerators.build_2D_tapering_function`
        (default 'flat-hanning').
    fft_method : str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see :py:func:`pysteps.utils.fft.get_method`). Defaults to "numpy".

    Returns
    -------
    N : array-like
        A two-dimensional numpy array of non-stationary correlated noise.

    """
    input_shape = F["input_shape"]
    use_full_fft = F["use_full_fft"]
    F = F["F"]

    if len(F.shape) != 4:
        raise ValueError("the input is not four-dimensional array")
    if np.any(~np.isfinite(F)):
        raise ValueError("F contains non-finite values")

    # defaults
    overlap = kwargs.get("overlap", 0.2)
    win_type = kwargs.get("win_type", "flat-hanning")
    fft = kwargs.get("fft_method", "numpy")
    if type(fft) == str:
        fft = utils.get_method(fft, shape=input_shape)

    if randstate is None:
        randstate = np.random

    # set the seed
    if seed is not None:
        randstate.seed(seed)

    dim_y = F.shape[2]
    dim_x = F.shape[3]
    dim = (dim_y, dim_x)

    # produce fields of white noise
    N = randstate.randn(dim_y, dim_x)
    fN = fft.fft2(N)

    # initialize variables
    cN = np.zeros(dim)
    sM = np.zeros(dim)

    idxi = np.zeros((2, 1), dtype=int)
    idxj = np.zeros((2, 1), dtype=int)

    # get the window size
    win_size = (float(dim_y) / F.shape[0], float(dim_x) / F.shape[1])

    # loop the windows and build composite image of correlated noise

    # loop rows
    for i in range(F.shape[0]):
        # loop columns
        for j in range(F.shape[1]):

            # apply fourier filtering with local filter
            lF = F[i, j, :, :]
            flN = fN * lF
            flN = np.array(fft.ifft2(flN).real)

            # compute indices of local window
            idxi[0] = np.max((i * win_size[0] - overlap * win_size[0], 0)).astype(int)
            idxi[1] = np.min(
                (idxi[0] + win_size[0] + overlap * win_size[0], dim_y)
            ).astype(int)
            idxj[0] = np.max((j * win_size[1] - overlap * win_size[1], 0)).astype(int)
            idxj[1] = np.min(
                (idxj[0] + win_size[1] + overlap * win_size[1], dim_x)
            ).astype(int)

            # build mask and add local noise field to the composite image
            M = _get_mask(dim, idxi, idxj, win_type)
            cN += flN * M
            sM += M

    # normalize the field
    cN[sM > 0] /= sM[sM > 0]
    cN = (cN - cN.mean()) / cN.std()

    return cN


def build_2D_tapering_function(win_size, win_type="flat-hanning"):
    """Produces two-dimensional tapering function for rectangular fields.

    Parameters
    ----------
    win_size : tuple of int
        Size of the tapering window as two-element tuple of integers.
    win_type : {'hanning', 'flat-hanning'}
        Name of the tapering window type 

    Returns
    -------
    w2d : array-like
        A two-dimensional numpy array containing the 2D tapering function.
    """

    if len(win_size) != 2:
        raise ValueError("win_size is not a two-element tuple")

    if win_type == "hanning":
        w1dr = np.hanning(win_size[0])
        w1dc = np.hanning(win_size[1])

    elif win_type == "flat-hanning":

        T = win_size[0] / 4.0
        W = win_size[0] / 2.0
        B = np.linspace(-W, W, 2 * W)
        R = np.abs(B) - T
        R[R < 0] = 0.0
        A = 0.5 * (1.0 + np.cos(np.pi * R / T))
        A[np.abs(B) > (2 * T)] = 0.0
        w1dr = A

        T = win_size[1] / 4.0
        W = win_size[1] / 2.0
        B = np.linspace(-W, W, 2 * W)
        R = np.abs(B) - T
        R[R < 0] = 0.0
        A = 0.5 * (1.0 + np.cos(np.pi * R / T))
        A[np.abs(B) > (2 * T)] = 0.0
        w1dc = A

    else:
        raise ValueError("unknown win_type %s" % win_type)

    # Expand to 2-D
    # w2d = np.sqrt(np.outer(w1dr,w1dc))
    w2d = np.outer(w1dr, w1dc)

    # Set nans to zero
    if np.sum(np.isnan(w2d)) > 0:
        w2d[np.isnan(w2d)] = np.min(w2d[w2d > 0])

    return w2d


def _split_field(idxi, idxj, Segments):
    """ Split domain field into a number of equally sapced segments.
    """

    sizei = idxi[1] - idxi[0]
    sizej = idxj[1] - idxj[0]

    winsizei = int(sizei / Segments)
    winsizej = int(sizej / Segments)

    Idxi = np.zeros((Segments ** 2, 2))
    Idxj = np.zeros((Segments ** 2, 2))

    count = -1
    for i in range(Segments):
        for j in range(Segments):
            count += 1
            Idxi[count, 0] = idxi[0] + i * winsizei
            Idxi[count, 1] = np.min((Idxi[count, 0] + winsizei, idxi[1]))
            Idxj[count, 0] = idxj[0] + j * winsizej
            Idxj[count, 1] = np.min((Idxj[count, 0] + winsizej, idxj[1]))

    Idxi = np.array(Idxi).astype(int)
    Idxj = np.array(Idxj).astype(int)

    return Idxi, Idxj


def _get_mask(Size, idxi, idxj, win_type):
    """Compute a mask of zeros with a window at a given position.
    """

    idxi = np.array(idxi).astype(int)
    idxj = np.array(idxj).astype(int)

    win_size = (idxi[1] - idxi[0], idxj[1] - idxj[0])
    wind = build_2D_tapering_function(win_size, win_type)

    mask = np.zeros(Size)
    mask[idxi.item(0) : idxi.item(1), idxj.item(0) : idxj.item(1)] = wind

    return mask
