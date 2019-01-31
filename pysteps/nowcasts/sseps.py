"""
Implementation of the Short-space ensemble prediction system (SSEPS) method.
Essentially, SSEPS is a localized version of STEPS. 

For  localization  we  intend  the  use of  a  subset  of  the  observations  
in  order  to  estimate  model parameters that are distributed in space. 
The short-space approach used in :cite:`NBSG2017` is generalized to 
the whole nowcasting system. This essenially boils down to a moving window 
localization of the nowcasting procedure, whereby all parameters are estimated 
over a subdomain of prescribed size.
"""

import sys
import time
import numpy as np
import scipy.ndimage
from .. import extrapolation
from .. import cascade
from .. import noise
from ..postprocessing import probmatching
from ..timeseries import autoregression, correlation
from .. import utils
try:
    import dask
    dask_imported = True
except ImportError:
    dask_imported = False

def forecast(R, metadata, V, n_timesteps, n_ens_members=24, n_cascade_levels=6,
             win_size=256, overlap = 0.1,
             extrap_method="semilagrangian", decomp_method="fft",
             bandpass_filter_method="gaussian", noise_method="nonparametric", ar_order=2,
             vel_pert_method=None, probmatching_method="cdf",
             mask_method="incremental", callback=None, fft_method="numpy",
             return_output=True, seed=None, num_workers=1, extrap_kwargs={},
             filter_kwargs={}, noise_kwargs={}, vel_pert_kwargs={}):
    """Generate a nowcast ensemble by using the Short-space ensemble prediction
    system (SSEPS) method.
    This is an experimental version of STEPS which allows for localization 
    by means of a window function.

    Parameters
    ----------
    R : array-like
        Array of shape (ar_order+1,m,n) containing the input precipitation fields
        ordered by timestamp from oldest to newest. The time steps between the inputs
        are assumed to be regular, and the inputs are required to have finite values.
    metadata : dict
        The metadata dictionary contains all information related to R.
    V : array-like
        Array of shape (2,m,n) containing the x- and y-components of the advection
        field. The velocities are assumed to represent one time step between the
        inputs. All values are required to be finite.
    win_size : int or two-element sequence of ints
        Size-length of the localization window.
    overlap : float [0,1[
        A float between 0 and 1 prescribing the level of overlap between 
        successive windows. If set to 0, no overlap is used.
    n_timesteps : int
        Number of time steps to forecast.
    n_ens_members : int
        The number of ensemble members to generate.
    n_cascade_levels : int
        The number of cascade levels to use.

    Other Parameters
    ----------------
    extrap_method : {'semilagrangian'}
        Name of the extrapolation method to use. See the documentation of
        pysteps.extrapolation.interface.
    decomp_method : {'fft'}
        Name of the cascade decomposition method to use. See the documentation
        of pysteps.cascade.interface.
    bandpass_filter_method : {'gaussian', 'uniform'}
        Name of the bandpass filter method to use with the cascade decomposition.
    noise_method : {'parametric','nonparametric','ssft','nested',None}
        Name of the noise generator to use for perturbating the precipitation
        field. See the documentation of pysteps.noise.interface. If set to None,
        no noise is generated.
    ar_order : int
        The order of the autoregressive model to use. Must be >= 1.
    vel_pert_method : {'bps',None}
        Name of the noise generator to use for perturbing the advection field. See
        the documentation of pysteps.noise.interface. If set to None, the advection
        field is not perturbed.
    mask_method : {'obs','incremental', None}
        The method to use for masking no precipitation areas in the forecast field.
        The masked pixels are set to the minimum value of the observations.
        'obs' = apply R_thr to the most recently observed precipitation intensity
        field, 'incremental' = iteratively buffer the mask
        with a certain rate (currently it is 1 km/min), None=no masking.
    probmatching_method : {'cdf', None}
        Method for matching the statistics of the forecast field with those of
        the most recently observed one. 'cdf'=map the forecast CDF to the observed
        one, None=no matching applied. Using 'mean' requires
        that mask_method is not None.
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
        The number of workers to use for parallel computation. Applicable if dask
        is enabled or pyFFTW is used for computing the FFT. When num_workers>1, it
        is advisable to disable OpenMP by setting the environment variable
        OMP_NUM_THREADS to 1. This avoids slowdown caused by too many simultaneous
        threads.
    fft_method : str
        A string defining the FFT method to use (see utils.fft.get_method).
        Defaults to 'numpy' for compatibility reasons. If pyFFTW is installed,
        the recommended method is 'pyfftw'.
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
        Optional dictionary containing keyword arguments "p_pert_par" and
        "p_pert_perp" for the initializer of the velocity perturbator.
        See the documentation of pysteps.noise.motion.

    Returns
    -------
    out : ndarray
        If return_output is True, a four-dimensional array of shape
          (n_ens_members,n_timesteps,m,n) containing a time series of forecast
        precipitation fields for each ensemble member. Otherwise, a None value
        is returned. The time series starts from t0+timestep, where timestep is
        taken from the input precipitation fields R.

    See also
    --------
    pysteps.extrapolation.interface, pysteps.cascade.interface,
    pysteps.noise.interface, pysteps.noise.utils.compute_noise_stddev_adjs

    Notes
    -----
    Please be aware that this represents a (very) experimental implementation.

    References
    ----------
    :cite:`Seed2003`, :cite:`BPS2006`, :cite:`SPN2013`, :cite:`NBSG2017`

    """
    _check_inputs(R, V, ar_order)

    if np.any(~np.isfinite(R)):
        raise ValueError("R contains non-finite values")

    if np.any(~np.isfinite(V)):
        raise ValueError("V contains non-finite values")

    if mask_method not in ["obs", "incremental", None]:
        raise ValueError("unknown mask method %s: must be 'obs' or 'incremental' or None" % mask_method)

    if np.isscalar(win_size):
        win_size = (np.int(win_size), np.int(win_size))
    else:
        win_size = tuple([np.int(win_size[i]) for i in range(2)])

    timestep = metadata["accutime"]
    kmperpixel = metadata["xpixelsize"]/1000

    print("Computing SSEPS nowcast:")
    print("------------------------")
    print("")

    print("Inputs:")
    print("-------")
    print("input dimensions: %dx%d" % (R.shape[1], R.shape[2]))
    print("km/pixel:         %g"    % kmperpixel)
    print("time step:        %d minutes" % timestep)
    print("")

    print("Methods:")
    print("--------")
    print("extrapolation:          %s" % extrap_method)
    print("bandpass filter:        %s" % bandpass_filter_method)
    print("decomposition:          %s" % decomp_method)
    print("noise generator:        %s" % noise_method)
    print("velocity perturbator:   %s" % vel_pert_method)
    print("precip. mask method:    %s" % mask_method)
    print("probability matching:   %s" % probmatching_method)
    print("FFT method:             %s" % fft_method)
    print("")

    print("Parameters:")
    print("-----------")
    print("localization window:      %dx%d" % (win_size[0], win_size[1]))
    print("overlap:                  %.1f" % overlap)
    print("number of time steps:     %d" % n_timesteps)
    print("ensemble size:            %d" % n_ens_members)
    print("number of cascade levels: %d" % n_cascade_levels)
    print("order of the AR(p) model: %d" % ar_order)
    print("dask imported:            %s" % ("yes" if dask_imported else "no"))
    print("num workers:              %d" % num_workers)


    if vel_pert_method is "bps":
        vp_par = vel_pert_kwargs.get("p_pert_par", noise.motion.get_default_params_bps_par())
        vp_perp = vel_pert_kwargs.get("p_pert_perp", noise.motion.get_default_params_bps_perp())
        print("velocity perturbations, parallel:      %g,%g,%g" % \
            (vp_par[0], vp_par[1], vp_par[2]))
        print("velocity perturbations, perpendicular: %g,%g,%g" % \
            (vp_perp[0], vp_perp[1], vp_perp[2]))

    R_thr = metadata["threshold"]

    num_ensemble_workers = n_ens_members if num_workers > n_ens_members \
                           else num_workers

    # get methods
    extrap_method = extrapolation.get_method(extrap_method)
    decomp_method = cascade.get_method(decomp_method)
    filter_method = cascade.get_method(bandpass_filter_method)
    if noise_method is not None:
        init_noise, generate_noise = noise.get_method(noise_method)

    # advect the previous precipitation fields to the same position with the
    # most recent one (i.e. transform them into the Lagrangian coordinates)
    R = R[-(ar_order + 1):, :, :]
    extrap_kwargs = extrap_kwargs.copy()
    res = []
    f = lambda R,i: extrap_method(R[i, :, :], V, ar_order-i, "min", **extrap_kwargs)[-1]
    for i in range(ar_order):
        if not dask_imported:
            R[i, :, :] = f(R, i)
        else:
            res.append(dask.delayed(f)(R, i))

    if dask_imported:
        num_workers_ = len(res) if num_workers > len(res) else num_workers
        R = np.stack(list(dask.compute(*res, num_workers=num_workers)) + [R[-1, :, :]])

    if mask_method == "incremental":
        # initialize the structuring element
        struct = scipy.ndimage.generate_binary_structure(2, 1)
        # iterate it to expand it nxn
        n = timestep/kmperpixel
        struct = scipy.ndimage.iterate_structure(struct, int((n - 1)/2.))

    # loop windows
    M,N = R.shape[1:]
    n_windows_M = np.ceil( 1.*M/win_size[0] ).astype(int)
    n_windows_N = np.ceil( 1.*N/win_size[1] ).astype(int)
    idxm = np.zeros((2, 1), dtype=int)
    idxn = np.zeros((2, 1), dtype=int)
    war = np.empty((n_windows_M, n_windows_N))
    GAMMA = np.empty((n_windows_M, n_windows_N, n_cascade_levels, ar_order))
    PHI = np.empty((n_windows_M, n_windows_N, n_cascade_levels, ar_order+1))
    mu = np.empty((n_windows_M, n_windows_N, n_cascade_levels))
    sigma = np.empty((n_windows_M, n_windows_N, n_cascade_levels))
    ff=[]; rc=[]; pp=[]; mm=[]
    for m in range(n_windows_M):
        ff_=[]; rc_=[]; pp_=[]; mm_=[]
        for n in range(n_windows_N):

            # compute indices of local window
            idxm[0] = int(np.max( (m*win_size[0] - overlap*win_size[0], 0) ))
            idxm[1] = int(np.min( (idxm[0] + win_size[0]  + overlap*win_size[0], M) ))
            idxn[0] = int(np.max( (n*win_size[1] - overlap*win_size[1], 0) ))
            idxn[1] = int(np.min( (idxn[0] + win_size[1]  + overlap*win_size[1], N) ))

            R_ = R[:, idxm.item(0):idxm.item(1), idxn.item(0):idxn.item(1)]

            war_ = np.sum(R_ >= R_thr) / R_.size
            war[m,n]= war_ > 0

            if war[m,n]:

                # initialize the perturbation generator for the precipitation field
                if noise_method is not None:
                    pp_.append(init_noise(R_, fft_method=fft_method, **{"rm_rdisc":True, "donorm":True}))
                else:
                    pp_.append(None)

                # initialize the band-pass filter
                ff_.append(filter_method(R_.shape[1:], n_cascade_levels, **filter_kwargs))

                # compute the cascade decompositions of the input precipitation fields
                R_d = []
                for i in range(ar_order+1):
                    R_d_ = decomp_method(R_[i, :, :], ff_[-1], fft_method=fft_method)
                    R_d.append(R_d_)
                R_d_ = None

                # normalize the cascades and rearrange them into a four-dimensional array
                # of shape (n_cascade_levels,ar_order+1,m,n) for the autoregressive model
                R_c,mu_,sigma_ = _stack_cascades(R_d, n_cascade_levels)
                mu[m, n, :] = mu_
                sigma[m, n, :] = sigma_
                R_d = None

                # compute lag-l temporal autocorrelation coefficients for each cascade level
                for i in range(n_cascade_levels):
                    R_c_ = np.stack([R_c[i, j, :, :] for j in range(ar_order+1)])
                    GAMMA[m, n, i, :] = correlation.temporal_autocorrelation(R_c_)
                R_c_ = None

                if ar_order == 2:
                    # adjust the local lag-2 correlation coefficient to ensure that the AR(p)
                    # process is stationary
                    for i in range(n_cascade_levels):
                        GAMMA[m, n, i, 1] = autoregression.adjust_lag2_corrcoef2(
                                            GAMMA[m, n, i, 0], GAMMA[m, n, i, 1])

                # estimate the parameters of the AR(p) model from the autocorrelation
                # coefficients
                for i in range(n_cascade_levels):
                    PHI[m, n, i, :] = autoregression.estimate_ar_params_yw(GAMMA[m, n, i, :])

                # discard all except the p-1 last cascades because they are not needed for
                # the AR(p) model
                R_c = R_c[:, -ar_order:, :, :]

                # stack the cascades into a five-dimensional array containing all ensemble
                # members
                R_c = np.stack([R_c.copy() for i in range(n_ens_members)])
                rc_.append(R_c)
                R_c = None


                if mask_method is not None:
                    MASK_prec = R_[-1, :, :] >= R_thr
                    if mask_method == "obs":
                        pass
                    elif mask_method == "incremental":
                        # initialize precip mask for each member
                        MASK_prec = scipy.ndimage.morphology.binary_dilation(MASK_prec, struct)
                        MASK_prec = [MASK_prec.copy() for j in range(n_ens_members)]
                    mm_.append(MASK_prec)

            else:

                ff_.append(None)
                rc_.append(None)
                pp_.append(None)
                mm_.append(None)

        ff.append(ff_)
        rc.append(rc_)
        pp.append(pp_)
        mm.append(mm_)

    ff_ = None
    rc_ = None
    pp_ = None
    mm_ = None

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

    if vel_pert_method is not None:
        init_vel_noise, generate_vel_noise = noise.get_method(vel_pert_method)

        # initialize the perturbation generators for the motion field
        vps = []
        for j in range(n_ens_members):
            kwargs = {"randstate":randgen_motion[j],
                      "p_par":vp_par,
                      "p_perp":vp_perp}
            vp_ = init_vel_noise(V, 1./kmperpixel, timestep, **kwargs)
            vps.append(vp_)

    D = [None for j in range(n_ens_members)]
    R_f = [[] for j in range(n_ens_members)]

    R = R[-1, :, :]

    print("Starting nowcast computation.")

    # iterate each time step
    scale = np.ones((n_windows_M, n_windows_N))
    for t in range(n_timesteps):
        print("Computing nowcast for time step %d... " % (t+1), end="")
        sys.stdout.flush()
        starttime = time.time()

        # iterate each ensemble member
        def worker(j):
            idxm = np.zeros((2, 1), dtype=int)
            idxn = np.zeros((2, 1), dtype=int)
            R_f = np.zeros((M,N), dtype=float)
            M_s = np.zeros((M,N), dtype=float)
            for m in range(n_windows_M):
                for n in range(n_windows_N):
                    if war[m,n]:
                        # compute indices of local window
                        idxm[0] = int(np.max( (m*win_size[0] - overlap*win_size[0], 0) ))
                        idxm[1] = int(np.min( (idxm[0] + win_size[0]  + overlap*win_size[0], M) ))
                        idxn[0] = int(np.max( (n*win_size[1] - overlap*win_size[1], 0) ))
                        idxn[1] = int(np.min( (idxn[0] + win_size[1]  + overlap*win_size[1], N) ))

                        # build localization mask
                        mask = _build_2D_tapering_function((idxm[1]-idxm[0], idxn[1]-idxn[0]))
                        M_s[idxm.item(0):idxm.item(1), idxn.item(0):idxn.item(1)] += mask

                        if noise_method is not None:
                            # generate noise field
                            EPS = generate_noise(pp[m][n], randstate=randgen_prec[j], fft_method=fft_method)
                            # decompose the noise field into a cascade
                            EPS = decomp_method(EPS, ff[m][n], fft_method=fft_method)
                        else:
                            EPS = None

                        # iterate the AR(p) model for each cascade level
                        R_c = rc[m][n]
                        for i in range(n_cascade_levels):
                            # normalize the noise cascade
                            if EPS is not None:
                                EPS_ = (EPS["cascade_levels"][i, :, :] - EPS["means"][i]) / EPS["stds"][i]
                            else:
                                EPS_ = None
                            # apply AR(p) process to cascade level
                            R_c[j, i, :, :, :] = \
                                autoregression.iterate_ar_model(R_c[j, i, :, :, :],
                                                        PHI[m, n, i, :], EPS=EPS_)
                            EPS_ = None
                        rc[m][n] = R_c
                        EPS = None

                        # compute the recomposed precipitation field(s) from the cascades
                        # obtained from the AR(p) model(s)
                        R_c_ = _recompose_cascade(R_c[j, :, :, :], mu[m, n, :], sigma[m, n, :])

                        if mask_method is not None:
                            # apply the precipitation mask to prevent generation of new
                            # precipitation into areas where it was not originally
                            # observed
                            MASK_prec = mm[m][n]
                            if mask_method == "obs":
                                R_c_[~MASK_prec] = R_c_.min()
                            elif mask_method == "incremental":
                                R_c_[~MASK_prec[j]] = R_c_.min()

                        if probmatching_method == "cdf":
                            # adjust the CDF of the forecast to match the most recently
                            # observed precipitation field
                            R_ = R[idxm.item(0):idxm.item(1), idxn.item(0):idxn.item(1)].copy()
                            R_c_ = probmatching.nonparam_match_empirical_cdf(R_c_, R_)
                            R_ = None

                        if mask_method == "incremental":
                            MASK_prec_ = R_c_ >= R_thr
                            MASK_prec_ = scipy.ndimage.morphology.binary_dilation(MASK_prec_, struct)
                            mm[m][n][j] = MASK_prec_

                        R_f[idxm.item(0):idxm.item(1), idxn.item(0):idxn.item(1)] += R_c_*mask
            
            ind = M_s > 0
            R_f[ind] /= M_s[ind]
            R_f[~ind] = metadata["zerovalue"]

            if probmatching_method == "cdf":
                # adjust the CDF of the forecast to match the most recently
                # observed precipitation field
                R_f = probmatching.nonparam_match_empirical_cdf(R_f, R)

            # compute the perturbed motion field
            if vel_pert_method is not None:
                V_ = V + generate_vel_noise(vps[j], t*timestep)
            else:
                V_ = V

            # advect the recomposed precipitation field to obtain the forecast
            # for time step t
            extrap_kwargs.update({"D_prev":D[j], "return_displacement":True})
            R_f_,D_ = extrap_method(R_f, V_, 1, **extrap_kwargs)
            D[j] = D_
            R_f_ = R_f_[0]

            return R_f_

        res = []
        for j in range(n_ens_members):
            if not dask_imported or n_ens_members == 1:
                res.append(worker(j))
            else:
                res.append(dask.delayed(worker)(j))

        R_f_ = dask.compute(*res, num_workers=num_ensemble_workers) \
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
        if n_ens_members == 1:
            return np.stack(R_f[0])
        else:
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

def _build_2D_tapering_function(win_size, win_type='flat-hanning'):
    """Produces two-dimensional tapering function for rectangular fields.

    Parameters
    ----------
    win_size : tuple of int
        Size of the tapering window as two-element tuple of integers.
    win_type : str
        Name of the tapering window type (hanning, flat-hanning)

    Returns
    -------
    w2d : array-like
        A two-dimensional numpy array containing the 2D tapering function.
    """

    if len(win_size) != 2:
        raise ValueError("win_size is not a two-element tuple")

    if win_type == 'hanning':
        w1dr = np.hanning(win_size[0])
        w1dc = np.hanning(win_size[1])

    elif win_type == 'flat-hanning':

        T = win_size[0]/4.0
        W = win_size[0]/2.0
        B = np.linspace(-W,W,2*W)
        R = np.abs(B)-T
        R[R < 0] = 0.
        A = 0.5*(1.0 + np.cos(np.pi*R/T))
        A[np.abs(B) > (2*T)] = 0.0
        w1dr = A

        T = win_size[1]/4.0
        W = win_size[1]/2.0
        B = np.linspace(-W, W, 2*W)
        R = np.abs(B) - T
        R[R < 0] = 0.
        A = 0.5*(1.0 + np.cos(np.pi*R/T))
        A[np.abs(B) > (2*T)] = 0.0
        w1dc = A

    else:
        raise ValueError("unknown win_type %s" % win_type)

    # Expand to 2-D
    # w2d = np.sqrt(np.outer(w1dr,w1dc))
    w2d = np.outer(w1dr,w1dc)

    # Set nans to zero
    if np.sum(np.isnan(w2d)) > 0:
        w2d[np.isnan(w2d)] = np.min(w2d[w2d > 0])

    return w2d

def _get_mask(Size, idxi, idxj, win_type="flat-hanning"):
    """Compute a mask of zeros with a window at a given position.
    """

    idxi = np.array(idxi).astype(int)
    idxj =  np.array(idxj).astype(int)

    win_size = (idxi[1] - idxi[0] , idxj[1] - idxj[0])
    wind = _build_2D_tapering_function(win_size, win_type)

    mask = np.zeros(Size)
    mask[idxi.item(0):idxi.item(1), idxj.item(0):idxj.item(1)] = wind

    return mask
