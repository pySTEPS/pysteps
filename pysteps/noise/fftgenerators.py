"""
pysteps.noise.fftgenerators
===========================

Methods for noise generators based on FFT filtering of white noise.

The methods in this module implement the following interface for filter
initialization depending on their parametric or nonparametric nature::

  initialize_param_2d_xxx_filter(field, **kwargs)

or::

  initialize_nonparam_2d_xxx_filter(field, **kwargs)

where field is an array of shape (m, n) or (t, m, n) that defines the target field
and optional parameters are supplied as keyword arguments.

The output of each initialization method is a dictionary containing the keys field
and input_shape. The first is a two-dimensional array of shape (m, int(n/2)+1)
that defines the filter. The second one is the shape of the input field for the
filter.

The methods in this module implement the following interface for the generation
of correlated noise::

  generate_noise_2d_xxx_filter(field, randstate=np.random, seed=None, **kwargs)

where field (m, n) is a filter returned from the corresponding initialization
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


def initialize_param_2d_fft_filter(field, **kwargs):
    """Takes one ore more 2d input fields, fits two spectral slopes, beta1 and beta2,
    to produce one parametric, global and isotropic fourier filter.

    Parameters
    ----------
    field: array-like
        Two- or three-dimensional array containing one or more input fields.
        All values are required to be finite. If more than one field are passed,
        the average fourier filter is returned. It assumes that fields are stacked
        by the first axis: [nr_fields, y, x].

    Other Parameters
    ----------------
    win_fun: {'hann', 'tukey' or None}
        Optional tapering function to be applied to the input field, generated with
        :py:func:`pysteps.utils.tapering.compute_window_function`.
        (default None).
    model: {'power-law'}
        The name of the parametric model to be used to fit the power spectrum of
        the input field (default 'power-law').
    weighted: bool
        Whether or not to apply 1/sqrt(power) as weight in the numpy.polyfit()
        function (default False).
    rm_rdisc: bool
        Whether or not to remove the rain/no-rain disconituity (default False).
        It assumes no-rain pixels are assigned with lowest value.
    fft_method: str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see "FFT methods" in :py:func:`pysteps.utils.interface.get_method`).
        Defaults to "numpy".

    Returns
    -------
    out: dict
        A dictionary containing the keys field, input_shape, model and pars.
        The first is a two-dimensional array of shape (m, int(n/2)+1) that
        defines the filter. The second one is the shape of the input field for
        the filter. The last two are the model and fitted parameters,
        respectively.

        This dictionary can be passed to
        :py:func:`pysteps.noise.fftgenerators.generate_noise_2d_fft_filter` to
        generate noise fields.
    """

    if len(field.shape) < 2 or len(field.shape) > 3:
        raise ValueError("the input is not two- or three-dimensional array")
    if np.any(~np.isfinite(field)):
        raise ValueError("field contains non-finite values")

    # defaults
    win_fun = kwargs.get("win_fun", None)
    model = kwargs.get("model", "power-law")
    weighted = kwargs.get("weighted", False)
    rm_rdisc = kwargs.get("rm_rdisc", False)
    fft = kwargs.get("fft_method", "numpy")
    if type(fft) == str:
        fft_shape = field.shape if len(field.shape) == 2 else field.shape[1:]
        fft = utils.get_method(fft, shape=fft_shape)

    field = field.copy()

    # remove rain/no-rain discontinuity
    if rm_rdisc:
        field[field > field.min()] -= field[field > field.min()].min() - field.min()

    # dims
    if len(field.shape) == 2:
        field = field[None, :, :]
    nr_fields = field.shape[0]
    M, N = field.shape[1:]

    if win_fun is not None:
        tapering = utils.tapering.compute_window_function(M, N, win_fun)

        # make sure non-rainy pixels are set to zero
        field -= field.min(axis=(1, 2))[:, None, None]
    else:
        tapering = np.ones((M, N))

    if model.lower() == "power-law":

        # compute average 2D PSD
        F = np.zeros((M, N), dtype=complex)
        for i in range(nr_fields):
            F += fft.fftshift(fft.fft2(field[i, :, :] * tapering))
        F /= nr_fields
        F = abs(F) ** 2 / F.size

        # compute radially averaged 1D PSD
        psd = utils.spectral.rapsd(F)
        L = max(M, N)

        # wavenumbers
        if L % 2 == 1:
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
        "field": F,
        "input_shape": field.shape[1:],
        "use_full_fft": True,
        "model": f,
        "pars": p,
    }


def initialize_nonparam_2d_fft_filter(field, **kwargs):
    """Takes one ore more 2d input fields and produces one non-parametric, global
    and anisotropic fourier filter.

    Parameters
    ----------
    field: array-like
        Two- or three-dimensional array containing one or more input fields.
        All values are required to be finite. If more than one field are passed,
        the average fourier filter is returned. It assumes that fields are stacked
        by the first axis: [nr_fields, y, x].

    Other Parameters
    ----------------
    win_fun: {'hann', 'tukey', None}
        Optional tapering function to be applied to the input field, generated with
        :py:func:`pysteps.utils.tapering.compute_window_function`
        (default 'tukey').
    donorm: bool
        Option to normalize the real and imaginary parts.
        Default: False
    rm_rdisc: bool
        Whether or not to remove the rain/no-rain disconituity (default True).
        It assumes no-rain pixels are assigned with lowest value.
    fft_method: str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see "FFT methods" in :py:func:`pysteps.utils.interface.get_method`).
        Defaults to "numpy".

    Returns
    -------
    out: dict
        A dictionary containing the keys field and input_shape. The first is a
        two-dimensional array of shape (m, int(n/2)+1) that defines the filter.
        The second one is the shape of the input field for the filter.

        It can be passed to
        :py:func:`pysteps.noise.fftgenerators.generate_noise_2d_fft_filter`.
    """
    if len(field.shape) < 2 or len(field.shape) > 3:
        raise ValueError("the input is not two- or three-dimensional array")
    if np.any(~np.isfinite(field)):
        raise ValueError("field contains non-finite values")

    # defaults
    win_fun = kwargs.get("win_fun", "tukey")
    donorm = kwargs.get("donorm", False)
    rm_rdisc = kwargs.get("rm_rdisc", True)
    use_full_fft = kwargs.get("use_full_fft", False)
    fft = kwargs.get("fft_method", "numpy")
    if type(fft) == str:
        fft_shape = field.shape if len(field.shape) == 2 else field.shape[1:]
        fft = utils.get_method(fft, shape=fft_shape)

    field = field.copy()

    # remove rain/no-rain discontinuity
    if rm_rdisc:
        field[field > field.min()] -= field[field > field.min()].min() - field.min()

    # dims
    if len(field.shape) == 2:
        field = field[None, :, :]
    nr_fields = field.shape[0]
    field_shape = field.shape[1:]
    if use_full_fft:
        fft_shape = (field.shape[1], field.shape[2])
    else:
        fft_shape = (field.shape[1], int(field.shape[2] / 2) + 1)

    # make sure non-rainy pixels are set to zero
    field -= field.min(axis=(1, 2))[:, None, None]

    if win_fun is not None:
        tapering = utils.tapering.compute_window_function(
            field_shape[0], field_shape[1], win_fun
        )
    else:
        tapering = np.ones(field_shape)

    F = np.zeros(fft_shape, dtype=complex)
    for i in range(nr_fields):
        if use_full_fft:
            F += fft.fft2(field[i, :, :] * tapering)
        else:
            F += fft.rfft2(field[i, :, :] * tapering)
    F /= nr_fields

    # normalize the real and imaginary parts
    if donorm:
        if np.std(F.imag) > 0:
            F.imag = (F.imag - np.mean(F.imag)) / np.std(F.imag)
        if np.std(F.real) > 0:
            F.real = (F.real - np.mean(F.real)) / np.std(F.real)

    return {
        "field": np.abs(F),
        "input_shape": field.shape[1:],
        "use_full_fft": use_full_fft,
    }


def generate_noise_2d_fft_filter(
    F, randstate=None, seed=None, fft_method=None, domain="spatial"
):
    """Produces a field of correlated noise using global Fourier filtering.

    Parameters
    ----------
    F: dict
        A filter object returned by
        :py:func:`pysteps.noise.fftgenerators.initialize_param_2d_fft_filter` or
        :py:func:`pysteps.noise.fftgenerators.initialize_nonparam_2d_fft_filter`.
        All values in the filter array are required to be finite.
    randstate: mtrand.RandomState
        Optional random generator to use. If set to None, use numpy.random.
    seed: int
        Value to set a seed for the generator. None will not set the seed.
    fft_method: str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see "FFT methods" in :py:func:`pysteps.utils.interface.get_method`).
        Defaults to "numpy".
    domain: {"spatial", "spectral"}
        The domain for the computations: If "spatial", the noise is generated
        in the spatial domain and transformed back to spatial domain after the
        Fourier filtering. If "spectral", the noise field is generated and kept
        in the spectral domain.

    Returns
    -------
    N: array-like
        A two-dimensional field of stationary correlated noise. The noise field
        is normalized to zero mean and unit variance.

    """
    if domain not in ["spatial", "spectral"]:
        raise ValueError(
            "invalid value %s for the 'domain' argument: must be 'spatial' or 'spectral'"
            % str(domain)
        )

    input_shape = F["input_shape"]
    use_full_fft = F["use_full_fft"]
    F = F["field"]

    if len(F.shape) != 2:
        raise ValueError("field is not two-dimensional array")
    if np.any(~np.isfinite(F)):
        raise ValueError("field contains non-finite values")

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
    if domain == "spatial":
        N = randstate.randn(input_shape[0], input_shape[1])
    else:
        if use_full_fft:
            size = (input_shape[0], input_shape[1])
        else:
            size = (input_shape[0], int(input_shape[1] / 2) + 1)
        theta = randstate.uniform(low=0.0, high=2.0 * np.pi, size=size)
        if input_shape[0] % 2 == 0:
            theta[int(input_shape[0] / 2) + 1 :, 0] = -theta[
                1 : int(input_shape[0] / 2), 0
            ][::-1]
        else:
            theta[int(input_shape[0] / 2) + 1 :, 0] = -theta[
                1 : int(input_shape[0] / 2) + 1, 0
            ][::-1]
        N = np.cos(theta) + 1.0j * np.sin(theta)

    # apply the global Fourier filter to impose a correlation structure
    if domain == "spatial":
        if use_full_fft:
            fN = fft.fft2(N)
        else:
            fN = fft.rfft2(N)
    else:
        fN = N
    fN *= F
    if domain == "spatial":
        if use_full_fft:
            N = np.array(fft.ifft2(fN).real)
        else:
            N = np.array(fft.irfft2(fN))
        N = (N - N.mean()) / N.std()
    else:
        N = fN
        N[0, 0] = 0.0
        N /= utils.spectral.std(N, input_shape, use_full_fft=use_full_fft)

    return N


def initialize_nonparam_2d_ssft_filter(field, **kwargs):
    """Function to compute the local Fourier filters using the Short-Space Fourier
    filtering approach.

    Parameters
    ----------
    field: array-like
        Two- or three-dimensional array containing one or more input fields.
        All values are required to be finite. If more than one field are passed,
        the average fourier filter is returned. It assumes that fields are stacked
        by the first axis: [nr_fields, y, x].

    Other Parameters
    ----------------
    win_size: int or two-element tuple of ints
        Size-length of the window to compute the SSFT (default (128, 128)).
    win_fun: {'hann', 'tukey', None}
        Optional tapering function to be applied to the input field, generated with
        :py:func:`pysteps.utils.tapering.compute_window_function`
        (default 'tukey').
    overlap: float [0,1[
        The proportion of overlap to be applied between successive windows
        (default 0.3).
    war_thr: float [0,1]
        Threshold for the minimum fraction of rain needed for computing the FFT
        (default 0.1).
    rm_rdisc: bool
        Whether or not to remove the rain/no-rain disconituity. It assumes no-rain
        pixels are assigned with lowest value.
    fft_method: str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see "FFT methods" in :py:func:`pysteps.utils.interface.get_method`).
        Defaults to "numpy".

    Returns
    -------
    field: array-like
        Four-dimensional array containing the 2d fourier filters distributed over
        a 2d spatial grid.
        It can be passed to
        :py:func:`pysteps.noise.fftgenerators.generate_noise_2d_ssft_filter`.

    References
    ----------
    :cite:`NBSG2017`
    """

    if len(field.shape) < 2 or len(field.shape) > 3:
        raise ValueError("the input is not two- or three-dimensional array")
    if np.any(np.isnan(field)):
        raise ValueError("field must not contain NaNs")

    # defaults
    win_size = kwargs.get("win_size", (128, 128))
    if type(win_size) == int:
        win_size = (win_size, win_size)
    win_fun = kwargs.get("win_fun", "tukey")
    overlap = kwargs.get("overlap", 0.3)
    war_thr = kwargs.get("war_thr", 0.1)
    rm_rdisc = kwargs.get("rm_disc", True)
    fft = kwargs.get("fft_method", "numpy")
    if type(fft) == str:
        fft_shape = field.shape if len(field.shape) == 2 else field.shape[1:]
        fft = utils.get_method(fft, shape=fft_shape)

    field = field.copy()

    # remove rain/no-rain discontinuity
    if rm_rdisc:
        field[field > field.min()] -= field[field > field.min()].min() - field.min()

    # dims
    if len(field.shape) == 2:
        field = field[None, :, :]
    nr_fields = field.shape[0]
    dim = field.shape[1:]
    dim_x = dim[1]
    dim_y = dim[0]

    # make sure non-rainy pixels are set to zero
    field -= field.min(axis=(1, 2))[:, None, None]

    # SSFT algorithm

    # prepare indices
    idxi = np.zeros((2, 1), dtype=int)
    idxj = np.zeros((2, 1), dtype=int)

    # number of windows
    num_windows_y = np.ceil(float(dim_y) / win_size[0]).astype(int)
    num_windows_x = np.ceil(float(dim_x) / win_size[1]).astype(int)

    # domain fourier filter
    F0 = initialize_nonparam_2d_fft_filter(
        field, win_fun=win_fun, donorm=True, use_full_fft=True, fft_method=fft
    )["field"]
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
            mask = _get_mask(dim, idxi, idxj, win_fun)
            war = float(np.sum((field * mask[None, :, :]) > 0.01)) / (
                (idxi[1] - idxi[0]) * (idxj[1] - idxj[0]) * nr_fields
            )

            if war > war_thr:
                # the new filter
                F[i, j, :, :] = initialize_nonparam_2d_fft_filter(
                    field * mask[None, :, :],
                    win_fun=None,
                    donorm=True,
                    use_full_fft=True,
                    fft_method=fft,
                )["field"]

    return {"field": F, "input_shape": field.shape[1:], "use_full_fft": True}


def initialize_nonparam_2d_nested_filter(field, gridres=1.0, **kwargs):
    """Function to compute the local Fourier filters using a nested approach.

    Parameters
    ----------
    field: array-like
        Two- or three-dimensional array containing one or more input fields.
        All values are required to be finite.
        If more than one field are passed, the average fourier filter is returned.
        It assumes that fields are stacked by the first axis: [nr_fields, y, x].
    gridres: float
        Grid resolution in km.

    Other Parameters
    ----------------
    max_level: int
        Localization parameter. 0: global noise, >0: increasing degree of
        localization (default 3).
    win_fun: {'hann', 'tukey', None}
        Optional tapering function to be applied to the input field, generated with
        :py:func:`pysteps.utils.tapering.compute_window_function`
        (default 'tukey').
    war_thr: float [0;1]
        Threshold for the minimum fraction of rain needed for computing the FFT
        (default 0.1).
    rm_rdisc: bool
        Whether or not to remove the rain/no-rain disconituity. It assumes no-rain
        pixels are assigned with lowest value.
    fft_method: str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see "FFT methods" in :py:func:`pysteps.utils.interface.get_method`).
        Defaults to "numpy".

    Returns
    -------
    field: array-like
        Four-dimensional array containing the 2d fourier filters distributed over
        a 2d spatial grid.
        It can be passed to
        :py:func:`pysteps.noise.fftgenerators.generate_noise_2d_ssft_filter`.
    """

    if len(field.shape) < 2 or len(field.shape) > 3:
        raise ValueError("the input is not two- or three-dimensional array")
    if np.any(np.isnan(field)):
        raise ValueError("field must not contain NaNs")

    # defaults
    max_level = kwargs.get("max_level", 3)
    win_fun = kwargs.get("win_fun", "tukey")
    war_thr = kwargs.get("war_thr", 0.1)
    rm_rdisc = kwargs.get("rm_disc", True)
    fft = kwargs.get("fft_method", "numpy")
    if type(fft) == str:
        fft_shape = field.shape if len(field.shape) == 2 else field.shape[1:]
        fft = utils.get_method(fft, shape=fft_shape)

    field = field.copy()

    # remove rain/no-rain discontinuity
    if rm_rdisc:
        field[field > field.min()] -= field[field > field.min()].min() - field.min()

    # dims
    if len(field.shape) == 2:
        field = field[None, :, :]
    nr_fields = field.shape[0]
    dim = field.shape[1:]
    dim_x = dim[1]
    dim_y = dim[0]

    # make sure non-rainy pixels are set to zero
    field -= field.min(axis=(1, 2))[:, None, None]

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
        field, win_fun=win_fun, donorm=True, use_full_fft=True, fft_method=fft
    )["field"]
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

                mask = _get_mask(dim, Idxinext[n, :], Idxjnext[n, :], win_fun)
                war = np.sum((field * mask[None, :, :]) > 0.01) / float(
                    (Idxinext[n, 1] - Idxinext[n, 0])
                    * (Idxjnext[n, 1] - Idxjnext[n, 0])
                    * nr_fields
                )

                if war > war_thr:
                    # the new filter
                    newfilter = initialize_nonparam_2d_fft_filter(
                        field * mask[None, :, :],
                        win_fun=None,
                        donorm=True,
                        use_full_fft=True,
                        fft_method=fft,
                    )["field"]

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

    return {"field": F, "input_shape": field.shape[1:], "use_full_fft": True}


def generate_noise_2d_ssft_filter(F, randstate=None, seed=None, **kwargs):
    """Function to compute the locally correlated noise using a nested approach.

    Parameters
    ----------
    F: array-like
        A filter object returned by
        :py:func:`pysteps.noise.fftgenerators.initialize_nonparam_2d_ssft_filter` or
        :py:func:`pysteps.noise.fftgenerators.initialize_nonparam_2d_nested_filter`.
        The filter is a four-dimensional array containing the 2d fourier filters
        distributed over a 2d spatial grid.
    randstate: mtrand.RandomState
        Optional random generator to use. If set to None, use numpy.random.
    seed: int
        Value to set a seed for the generator. None will not set the seed.

    Other Parameters
    ----------------
    overlap: float
        Percentage overlap [0-1] between successive windows (default 0.2).
    win_fun: {'hann', 'tukey', None}
        Optional tapering function to be applied to the input field, generated with
        :py:func:`pysteps.utils.tapering.compute_window_function`
        (default 'tukey').
    fft_method: str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see "FFT methods" in :py:func:`pysteps.utils.interface.get_method`).
        Defaults to "numpy".

    Returns
    -------
    N: array-like
        A two-dimensional numpy array of non-stationary correlated noise.
    """
    input_shape = F["input_shape"]
    use_full_fft = F["use_full_fft"]
    F = F["field"]

    if len(F.shape) != 4:
        raise ValueError("the input is not four-dimensional array")
    if np.any(~np.isfinite(F)):
        raise ValueError("field contains non-finite values")

    if "domain" in kwargs.keys() and kwargs["domain"] == "spectral":
        raise NotImplementedError(
            "SSFT-based noise generator is not implemented in the spectral domain"
        )

    # defaults
    overlap = kwargs.get("overlap", 0.2)
    win_fun = kwargs.get("win_fun", "tukey")
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
            idxi[0] = int(np.max((i * win_size[0] - overlap * win_size[0], 0)))
            idxi[1] = int(
                np.min((idxi[0] + win_size[0] + overlap * win_size[0], dim_y))
            )
            idxj[0] = int(np.max((j * win_size[1] - overlap * win_size[1], 0)))
            idxj[1] = int(
                np.min((idxj[0] + win_size[1] + overlap * win_size[1], dim_x))
            )

            # build mask and add local noise field to the composite image
            M = _get_mask(dim, idxi, idxj, win_fun)
            cN += flN * M
            sM += M

    # normalize the field
    cN[sM > 0] /= sM[sM > 0]
    cN = (cN - cN.mean()) / cN.std()

    return cN


def _split_field(idxi, idxj, Segments):
    """Split domain field into a number of equally sapced segments."""

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


def _get_mask(Size, idxi, idxj, win_fun):
    """Compute a mask of zeros with a window at a given position."""

    idxi = np.array(idxi).astype(int)
    idxj = np.array(idxj).astype(int)

    win_size = (idxi[1] - idxi[0], idxj[1] - idxj[0])
    if win_fun is not None:
        wind = utils.tapering.compute_window_function(win_size[0], win_size[1], win_fun)
        wind += 1e-6  # avoid zero values

    else:
        wind = np.ones(win_size)

    mask = np.zeros(Size)
    mask[idxi.item(0) : idxi.item(1), idxj.item(0) : idxj.item(1)] = wind

    return mask
