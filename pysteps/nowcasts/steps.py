"""Implementation of the STEPS method."""

import numpy as np
import scipy.ndimage
import sys
import time
from .. import extrapolation
from .. import cascade
from .. import noise
from ..postprocessing import probmatching
from ..timeseries import autoregression, correlation
try:
    import dask
    dask_imported = True
except ImportError:
    dask_imported = False

def forecast(R, V, n_timesteps, n_ens_members=24, n_cascade_levels=6, R_thr=None,
             kmperpixel=None, timestep=None, extrap_method="semilagrangian",
             decomp_method="fft", bandpass_filter_method="gaussian",
             noise_method="nonparametric", noise_stddev_adj=False, ar_order=2,
             vel_pert_method="bps", conditional=False, probmatching_method="cdf",
             mask_method="incremental", callback=None, return_output=True,
             seed=None, num_workers=None, fft_method="numpy", extrap_kwargs={},
             filter_kwargs={}, noise_kwargs={}, vel_pert_kwargs={}):
    """Generate a nowcast ensemble by using the Short-Term Ensemble Prediction
    System (STEPS) method.

    Parameters
    ----------
    R : array-like
      Array of shape (ar_order+1,m,n) containing the input precipitation fields
      ordered by timestamp from oldest to newest. The time steps between the inputs
      are assumed to be regular, and the inputs are required to have finite values.
    V : array-like
      Array of shape (2,m,n) containing the x- and y-components of the advection
      field. The velocities are assumed to represent one time step between the
      inputs. All values are required to be finite.
    n_timesteps : int
      Number of time steps to forecast.
    n_ens_members : int
      The number of ensemble members to generate.
    n_cascade_levels : int
      The number of cascade levels to use.

    Other Parameters
    ----------------
    R_thr : float
      Specifies the threshold value for minimum observable precipitation
      intensity. Required if mask_method is not None or conditional is True.
    kmperpixel : float
      Spatial resolution of the input data (kilometers/pixel). Required if
      vel_pert_method is not None or mask_method is 'incremental'.
    timestep : float
      Time step of the motion vectors (minutes). Required if vel_pert_method is
      not None or mask_method is 'incremental'.
    extrap_method : {'semilagrangian'}
      Name of the extrapolation method to use. See the documentation of
      pysteps.extrapolation.interface.
    decomp_method : {'fft'}
      Name of the cascade decomposition method to use. See the documentation
      of pysteps.cascade.interface.
    bandpass_filter_method : {'gaussian', 'uniform'}
      Name of the bandpass filter method to use with the cascade decomposition.
      See the documentation of pysteps.cascade.interface.
    noise_method : {'parametric','nonparametric','ssft','nested',None}
      Name of the noise generator to use for perturbating the precipitation
      field. See the documentation of pysteps.noise.interface. If set to None,
      no noise is generated.
    noise_stddev_adj : bool
      Optional adjustment for the standard deviations of the noise fields added
      to each cascade level. See pysteps.noise.utils.compute_noise_stddev_adjs.
    ar_order : int
      The order of the autoregressive model to use. Must be >= 1.
    vel_pert_method : {'bps',None}
      Name of the noise generator to use for perturbing the advection field. See
      the documentation of pysteps.noise.interface. If set to None, the advection 
      field is not perturbed.
    conditional : bool
      If set to True, compute the statistics of the precipitation field
      conditionally by excluding pixels where the values are below the threshold
      R_thr.
    mask_method : {'obs','sprog','incremental',None}
      The method to use for masking no precipitation areas in the forecast field. 
      The masked pixels are set to the minimum value of the observations.
      'obs' = apply R_thr to the most recently observed precipitation intensity
      field, 'sprog' = use the smoothed forecast field from S-PROG, where the
      AR(p) model has been applied, 'incremental' = iteratively buffer the mask
      with a certain rate (currently it is 1 km/min), None=no masking.
    probmatching_method : {'cdf','mean',None}
      Method for matching the statistics of the forecast field with those of
      the most recently observed one. Requires that mask_method is not None.
      'cdf'=map the forecast CDF to the observed one, 'mean'=adjust only the
      mean value of the forecast field, None=no matching applied.
    callback : function
      Optional function that is called after computation of each time step of
      the nowcast. The function takes one argument: a three-dimensional array
      of shape (n_ens_members,h,w), where h and w are the height and width
      of the input field R, respectively. This can be used, for instance,
      writing the outputs into files.
    return_output : bool
      Set to False to disable returning the outputs as numpy arrays. This can
      save memory if the intermediate results are written to output files using
      the callback function.
    seed : int
      Optional seed number for the random generators.
    num_workers : int
      The number of workers to use for parallel computation. Set to None to use
      all available CPUs. Applicable if dask is enabled.
    fft_method : str or tuple
      A string or a (function,kwargs) tuple defining the FFT method to use
      (see utils.fft.get_method). Defaults to 'numpy'.
    extrap_kwargs : dict
      Optional dictionary containing keyword arguments for the extrapolation
      method. See the documentation of pysteps.extrapolation.
    filter_kwargs : dict
      Optional dictionary containing keyword arguments for the filter method.
      See the documentation of pysteps.cascade.bandpass_filters.py.
    noise_kwargs : dict
      Optional dictionary containing keyword arguments for the initializer of
      the noise generator. See the documentation of pysteps.noise.fftgenerators.
    vel_pert_kwargs : dict
      Optional dictionary containing keyword arguments for the initializer of
      the velocity perturbator. See the documentation of pysteps.noise.motion.

    Returns
    -------
    out : ndarray
      If return_output is True, a four-dimensional array of shape
      (n_ens_members,n_timesteps,m,n) containing a time series of forecast
      precipitation fields for each ensemble member. Otherwise, a None value
      is returned. The time step is taken from the input precipitation fields R.

    See also
    --------
    pysteps.extrapolation.interface, pysteps.cascade.interface,
    pysteps.noise.interface, pysteps.noise.utils.compute_noise_stddev_adjs

    Notes
    -----
    If noise_method and vel_pert_method are None, n_ens_members is 1,
    mask_method is 'sprog' and probmatching_method is 'mean', the deterministic
    S-PROG nowcast is generated, see :cite:`Seed2003`.

    References
    ----------
    :cite:`Seed2003`, :cite:`BPS2006`, :cite:`SPN2013`

    """
    _check_inputs(R, V, ar_order)

    if np.any(~np.isfinite(R)):
        raise ValueError("R contains non-finite values")

    if np.any(~np.isfinite(V)):
        raise ValueError("V contains non-finite values")

    if mask_method not in ["obs", "sprog", "incremental", None]:
        raise ValueError("unknown mask method %s: must be 'obs', 'sprog' or 'incremental' or None" % mask_method)

    if conditional and R_thr is None:
        raise ValueError("conditional=True but R_thr is not set")

    if probmatching_method is not None and R_thr is None:
        raise ValueError("probmatching_method!=None but R_thr is not set")

    if probmatching_method is not None and mask_method is None:
        raise ValueError("probmatching_method!=None but mask_method=None")

    if kmperpixel is None:
        if vel_pert_method is not None:
            raise ValueError("vel_pert_method is set but kmperpixel=None")
        if mask_method == "incremental":
            raise ValueError("mask_method='incremental' but kmperpixel=None")

    if timestep is None:
        if vel_pert_method is not None:
            raise ValueError("vel_pert_method is set but timestep=None")
        if mask_method == "incremental":
            raise ValueError("mask_method='incremental' but timestep=None")

    print("Computing STEPS nowcast:")
    print("------------------------")
    print("")

    print("Inputs:")
    print("-------")
    print("input dimensions: %dx%d" % (R.shape[1], R.shape[2]))
    if kmperpixel is not None:
        print("km/pixel:         %g"    % kmperpixel)
    if timestep is not None:
        print("time step:        %d minutes" % timestep)
    print("")

    print("Methods:")
    print("--------")
    print("extrapolation:          %s" % extrap_method)
    print("bandpass filter:        %s" % bandpass_filter_method)
    print("decomposition:          %s" % decomp_method)
    print("noise generator:        %s" % noise_method)
    print("noise adjustment:       %s" % ("yes" if noise_stddev_adj else "no"))
    print("velocity perturbator:   %s" % vel_pert_method)
    print("conditional statistics: %s" % ("yes" if conditional else "no"))
    print("precip. mask method:    %s" % mask_method)
    print("probability matching:   %s" % probmatching_method)
    print("FFT method:             %s" % fft_method)
    print("")

    print("Parameters:")
    print("-----------")
    print("number of time steps:     %d" % n_timesteps)
    print("ensemble size:            %d" % n_ens_members)
    print("number of cascade levels: %d" % n_cascade_levels)
    print("order of the AR(p) model: %d" % ar_order)
    if vel_pert_method is "bps":
        vp_par  = vel_pert_kwargs.get("p_pert_par",  noise.motion.get_default_params_bps_par())
        vp_perp = vel_pert_kwargs.get("p_pert_perp", noise.motion.get_default_params_bps_perp())
        print("velocity perturbations, parallel:      %g,%g,%g" % \
            (vp_par[0],  vp_par[1],  vp_par[2]))
        print("velocity perturbations, perpendicular: %g,%g,%g" % \
            (vp_perp[0], vp_perp[1], vp_perp[2]))

    if conditional or mask_method is not None:
        print("precip. intensity threshold: %g" % R_thr)

    M,N = R.shape[1:]
    extrap_method = extrapolation.get_method(extrap_method)
    R = R[-(ar_order + 1):, :, :].copy()

    if conditional:
        MASK_thr = np.logical_and.reduce([R[i, :, :] >= R_thr for i in range(R.shape[0])])
    else:
        MASK_thr = None

    # advect the previous precipitation fields to the same position with the
    # most recent one (i.e. transform them into the Lagrangian coordinates)
    extrap_kwargs = extrap_kwargs.copy()
    res = []
    f = lambda R,i: extrap_method(R[i, :, :], V, ar_order-i, "min", **extrap_kwargs)[-1]
    for i in range(ar_order):
        if not dask_imported:
            R[i, :, :] = f(R, i)
        else:
            res.append(dask.delayed(f)(R, i))

    if dask_imported:
        R = np.stack(list(dask.compute(*res, num_workers=num_workers)) + [R[-1, :, :]])

    # initialize the band-pass filter
    filter_method = cascade.get_method(bandpass_filter_method)
    filter = filter_method((M, N), n_cascade_levels, **filter_kwargs)

    # compute the cascade decompositions of the input precipitation fields
    decomp_method = cascade.get_method(decomp_method)
    R_d = []
    for i in range(ar_order+1):
        R_ = decomp_method(R[i, :, :], filter, MASK=MASK_thr, fft_method=fft_method)
        R_d.append(R_)

    # normalize the cascades and rearrange them into a four-dimensional array
    # of shape (n_cascade_levels,ar_order+1,m,n) for the autoregressive model
    R_c,mu,sigma = _stack_cascades(R_d, n_cascade_levels)
    R_d = None

    # compute lag-l temporal autocorrelation coefficients for each cascade level
    GAMMA = np.empty((n_cascade_levels, ar_order))
    for i in range(n_cascade_levels):
        R_c_ = np.stack([R_c[i, j, :, :] for j in range(ar_order+1)])
        GAMMA[i, :] = correlation.temporal_autocorrelation(R_c_, MASK=MASK_thr)
    R_c_ = None

    _print_corrcoefs(GAMMA)

    if ar_order == 2:
        # adjust the lag-2 correlation coefficient to ensure that the AR(p)
        # process is stationary
        for i in range(n_cascade_levels):
            GAMMA[i, 1] = autoregression.adjust_lag2_corrcoef2(GAMMA[i, 0], GAMMA[i, 1])

    # estimate the parameters of the AR(p) model from the autocorrelation
    # coefficients
    PHI = np.empty((n_cascade_levels, ar_order+1))
    for i in range(n_cascade_levels):
        PHI[i, :] = autoregression.estimate_ar_params_yw(GAMMA[i, :])

    _print_ar_params(PHI, False)

    # discard all except the p-1 last cascades because they are not needed for
    # the AR(p) model
    R_c = R_c[:, -ar_order:, :, :]

    # stack the cascades into a five-dimensional array containing all ensemble
    # members
    R_c = np.stack([R_c.copy() for i in range(n_ens_members)])

    # initialize the random generators
    if noise_method is not None:
        randgen_prec   = []
        randgen_motion = []
        np.random.seed(seed)
        for j in range(n_ens_members):
            rs = np.random.RandomState(seed)
            randgen_prec.append(rs)
            seed = rs.randint(0, high=1e9)
            rs = np.random.RandomState(seed)
            randgen_motion.append(rs)
            seed = rs.randint(0, high=1e9)

    R_min = np.min(R)

    if noise_method is not None:
        # get methods for perturbations
        init_noise, generate_noise = noise.get_method(noise_method)

        # initialize the perturbation generator for the precipitation field
        pp = init_noise(R, fft_method=fft_method, **noise_kwargs)

        if noise_stddev_adj:
            print("Computing noise adjustment factors... ", end="")
            sys.stdout.flush()
            starttime = time.time()

            noise_std_coeffs = noise.utils.compute_noise_stddev_adjs(R[-1, :, :],
                R_thr, R_min, filter, decomp_method, 10, conditional=True,
                num_workers=num_workers)

            print("%.2f seconds." % (time.time() - starttime))
        else:
            noise_std_coeffs = np.ones(n_cascade_levels)

    if vel_pert_method is not None:
        init_vel_noise, generate_vel_noise = noise.get_method(vel_pert_method)

        # initialize the perturbation generators for the motion field
        vps = []
        for j in range(n_ens_members):
            kwargs = {"randstate":randgen_motion[j],
                      "p_pert_par":vp_par,
                      "p_pert_perp":vp_perp}
            vp_ = init_vel_noise(V, 1./kmperpixel, timestep, **kwargs)
            vps.append(vp_)

    D = [None for j in range(n_ens_members)]
    R_f = [[] for j in range(n_ens_members)]

    if mask_method is not None:
        MASK_prec = R[-1, :, :] >= R_thr

        if mask_method == "obs":
            pass
            # add a slight buffer to the mask
            # n=5
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n,n))
            # MASK_prec = MASK_prec.astype('uint8')
            # MASK_prec = cv2.dilate(MASK_prec,kernel).astype(bool)
        elif mask_method == "sprog":
            # compute the wet area ratio and the precipitation mask
            war = 1.0*np.sum(MASK_prec) / (R.shape[1]*R.shape[2])
            R_m = R_c[0, :, :, :].copy()
        elif mask_method == "incremental":
            # initialize precip mask for each member
            MASK_prec = [MASK_prec.copy() for j in range(n_ens_members)]
            # initialize the structuring element
            struct = scipy.ndimage.generate_binary_structure(2, 1)
            # iterate it to expand it nxn
            n = timestep/kmperpixel
            struct = scipy.ndimage.iterate_structure(struct, int((n - 1)/2.))

        if probmatching_method == "mean":
            mu_0 = np.mean(R[-1, :, :][MASK_prec])

    R = R[-1, :, :]

    print("Starting nowcast computation.")

    # iterate each time step
    for t in range(n_timesteps):
        print("Computing nowcast for time step %d... " % (t+1), end="")
        sys.stdout.flush()
        starttime = time.time()

        if mask_method == "sprog":
            for i in range(n_cascade_levels):
                # use a separate AR(p) model for the non-perturbed forecast,
                # from which the mask is obtained
                R_m[i, :, :, :] = \
                    autoregression.iterate_ar_model(R_m[i, :, :, :], PHI[i, :])

            R_m_ = _recompose_cascade(R_m, mu, sigma)

            # obtain the CDF from the non-perturbed forecast that is
            # scale-filtered by the AR(p) model
            R_s = R_m_.flatten()

            # compute the threshold value R_pct_thr corresponding to the
            # same fraction of precipitation pixels (forecast values above
            # R_min) as in the most recently observed precipitation field
            R_s.sort(kind="quicksort")
            x = 1.0*np.arange(1, len(R_s)+1)[::-1] / len(R_s)
            i = np.argmin(abs(x - war))
            # handle ties
            if R_s[i] == R_s[i + 1]:
                i = np.where(R_s == R_s[i])[0][-1] + 1
            R_pct_thr = R_s[i]

            # determine a mask using the above threshold value to preserve the
            # wet-area ratio
            MASK_prec = R_m_ < R_pct_thr

        # iterate each ensemble member
        def worker(j):
            if noise_method is not None:
                # generate noise field
                EPS = generate_noise(pp, randstate=randgen_prec[j], 
                                     fft_method=fft_method)
                # decompose the noise field into a cascade
                EPS = decomp_method(EPS, filter, fft_method=fft_method)
            else:
                EPS = None

            # iterate the AR(p) model for each cascade level
            for i in range(n_cascade_levels):
                # normalize the noise cascade
                if EPS is not None:
                    EPS_ = (EPS["cascade_levels"][i, :, :] - EPS["means"][i]) / EPS["stds"][i]
                    EPS_ *= noise_std_coeffs[i]
                else:
                    EPS_ = None
                # apply AR(p) process to cascade level
                R_c[j, i, :, :, :] = \
                    autoregression.iterate_ar_model(R_c[j, i, :, :, :],
                                                    PHI[i, :], EPS=EPS_)

            EPS  = None
            EPS_ = None

            # compute the recomposed precipitation field(s) from the cascades
            # obtained from the AR(p) model(s)
            R_c_ = _recompose_cascade(R_c[j, :, :, :], mu, sigma)

            if mask_method is not None:
                # apply the precipitation mask to prevent generation of new
                # precipitation into areas where it was not originally
                # observed
                if mask_method == "obs":
                    MASK_prec_ = ~MASK_prec
                elif mask_method == "incremental":
                    MASK_prec_ = ~MASK_prec[j]
                elif mask_method == "sprog":
                    MASK_prec_ = MASK_prec

                R_c_[MASK_prec_] = R_c_.min()

            if probmatching_method == "cdf":
                # adjust the conditional CDF of the forecast (precipitation
                # intensity above the threshold R_thr) to match the most
                # recently observed precipitation field
                R_c_ = probmatching.nonparam_match_empirical_cdf(R_c_, R)
            elif probmatching_method == "mean":
                mu_fct = np.mean(R_c_[~MASK_prec_])
                R_c_[~MASK_prec_] = R_c_[~MASK_prec_] - mu_fct + mu_0

            if mask_method == "incremental":
                MASK_prec_ = R_c_ >= R_thr
                MASK_prec_ = scipy.ndimage.morphology.binary_dilation(MASK_prec_, struct)
                MASK_prec[j] = MASK_prec_

            # compute the perturbed motion field
            if vel_pert_method is not None:
                V_ = V + generate_vel_noise(vps[j], t*timestep)
            else:
                V_ = V

            # advect the recomposed precipitation field to obtain the forecast
            # for time step t
            extrap_kwargs.update({"D_prev":D[j], "return_displacement":True})
            R_f_,D_ = extrap_method(R_c_, V_, 1, **extrap_kwargs)
            D[j] = D_
            R_f_ = R_f_[0]

            return R_f_

        res = []
        for j in range(n_ens_members):
            if not dask_imported or n_ens_members == 1:
                res.append(worker(j))
            else:
                res.append(dask.delayed(worker)(j))

        R_f_ = dask.compute(*res, num_workers=num_workers) \
            if dask_imported and n_ens_members > 1 else res
        res = None

        print("%.2f seconds." % (time.time() - starttime))

        if callback is not None:
            callback(np.stack(R_f_))
            R_f_ = None

        if return_output:
            for j in range(n_ens_members):
                R_f[j].append(R_f_[j])

    if return_output:
        return np.stack([np.stack(R_f[j]) for j in range(n_ens_members)])
    else:
        return None

def _check_inputs(R, V, ar_order):
    if len(R.shape) != 3:
        raise ValueError("R must be a three-dimensional array")
    if R.shape[0] < ar_order + 1:
        raise ValueError("R.shape[0] < ar_order+1")
    if len(V.shape) != 3:
        raise ValueError("V must be a three-dimensional array")
    if R.shape[1:3] != V.shape[1:3]:
        raise ValueError("dimension mismatch between R and V: shape(R)=%s, shape(V)=%s" % \
                         (str(R.shape), str(V.shape)))

def _print_ar_params(PHI, include_perturb_term):
    print("****************************************")
    print("* AR(p) parameters for cascade levels: *")
    print("****************************************")

    n = PHI.shape[1]

    hline_str = "---------"
    for k in range(n):
        hline_str += "---------------"

    print(hline_str)
    title_str = "| Level |"
    for k in range(n-1):
        title_str += "    Phi-%d     |" % (k+1)
    title_str += "    Phi-0     |"
    print(title_str)
    print(hline_str)

    fmt_str = "| %-5d |"
    for k in range(n):
        fmt_str += " %-12.6f |"

    for k in range(PHI.shape[0]):
        print(fmt_str % ((k+1,) + tuple(PHI[k, :])))
        print(hline_str)

def _print_corrcoefs(GAMMA):
    print("************************************************")
    print("* Correlation coefficients for cascade levels: *")
    print("************************************************")

    m = GAMMA.shape[0]
    n = GAMMA.shape[1]

    hline_str = "---------"
    for k in range(n):
        hline_str += "----------------"

    print(hline_str)
    title_str = "| Level |"
    for k in range(n):
        title_str += "     Lag-%d     |" % (k+1)
    print(title_str)
    print(hline_str)

    fmt_str = "| %-5d |"
    for k in range(n):
        fmt_str += " %-13.6f |"

    for k in range(m):
        print(fmt_str % ((k+1,) + tuple(GAMMA[k, :])))
        print(hline_str)

def _stack_cascades(R_d, n_levels, donorm=True):
  R_c   = []
  mu    = np.empty(n_levels)
  sigma = np.empty(n_levels)

  n_inputs = len(R_d)

  for i in range(n_levels):
      R_ = []
      mu_    = 0
      sigma_ = 1
      for j in range(n_inputs):
          if donorm:
              mu_    = R_d[j]["means"][i]
              sigma_ = R_d[j]["stds"][i]
          if j == n_inputs - 1:
              mu[i]    = mu_
              sigma[i] = sigma_
          R__ = (R_d[j]["cascade_levels"][i, :, :] - mu_) / sigma_
          R_.append(R__)
      R_c.append(np.stack(R_))

  return np.stack(R_c),mu,sigma

def _recompose_cascade(R, mu, sigma):
    R_rc = [(R[i, -1, :, :] * sigma[i]) + mu[i] for i in range(len(mu))]
    R_rc = np.sum(np.stack(R_rc), axis=0)

    return R_rc
