"""Implementation of the STEPS method."""

import numpy as np
import sys
import time
from .. import advection
from .. import cascade
from .. import noise
from ..postproc import probmatching
from ..timeseries import autoregression, correlation
try:
    import dask
    dask_imported = True
except ImportError:
    dask_imported = False

# TODO: Using non-square shapes of the inputs has not been tested.
def forecast(R, V, num_timesteps, num_ens_members, num_cascade_levels, R_thr, 
             extrap_method, decomp_method, bandpass_filter_method, perturbation_method, 
             pixelsperkm, timestep, ar_order=2, vp_par=(10.88,0.23,-7.68), 
             vp_perp=(5.76,0.31,-2.72), conditional=True, use_precip_mask=False, 
             use_probmatching=True, exporter=None, extrap_kwargs={}, 
             filter_kwargs={}):
    """Generate a nowcast ensemble by using the STEPS method described in 
    Bowler et al. 2006: STEPS: A probabilistic precipitation forecasting scheme 
    which merges an extrapolation nowcast with downscaled NWP.
    
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
    num_timesteps : int
      Number of time steps to forecast.
    num_ens_members : int
      The number of ensemble members to generate.
    num_cascade_levels : int
      The number of cascade levels to use.
    R_thr : float
      Specifies the threshold value to use. Applicable if conditional is True.
    extrap_method : str
      Name of the extrapolation method to use. See the documentation of 
      pysteps.advection for the available choices.
    decomp_method : str
      Name of the cascade decomposition method to use. See the documentation 
      of pysteps.cascade.decomposition.
    bandpass_filter_method : str
      Name of the bandpass filter method to use with the cascade decomposition. 
      See the documentation of pysteps.cascade.bandpass_filters.
    perturbation_method : str
      Name of the noise generator to use for the perturbations of the precipitation
      field. See the documentation of pysteps.noise.fftgenerators.
    pixelsperkm : float
      Spatial resolution of the motion field (pixels/kilometer).
    timestep : float
      Time step of the motion vectors (minutes).
    ar_order : int
      The order of the autoregressive model to use.
    vp_par : tuple
      Optional three-element tuple containing the parameters for the standard 
      deviation of the perturbations in the direction parallel to the motion 
      vectors. See noise.motion.initialize_bps. The default values are taken 
      from Bowler et al. 2006.
    vp_perp : tuple
      Optional three-element tuple containing the parameters for the standard 
      deviation of the perturbations in the direction perpendicular to the motion 
      vectors. See noise.motion.initialize_bps. The default values are taken 
      from Bowler et al. 2006.
    conditional : bool
      If set to True, compute the statistics of the precipitation field 
      conditionally by excluding the areas where the values are below the 
      threshold R_thr.
    use_precip_mask : bool
      If True, set pixels outside precipitation areas to the minimum value of 
      the observed field.
    use_probmatching : bool
      If True, apply probability matching to the forecast field in order to 
      preserve the distribution of the most recently observed precipitation 
      field.
    extrap_kwargs : dict
      Optional dictionary that is supplied as keyword arguments to the 
      extrapolation method.
    filter_kwargs : dict
      Optional dictionary that is supplied as keyword arguments to the 
      filter method.
    
    Returns
    -------
    out : ndarray
      Four-dimensional array of shape (num_ens_members,num_timesteps,m,n) 
      containing a time series of forecast precipitation fields for each ensemble 
      member.
    """
    _check_inputs(R, V, ar_order)
    
    if np.any(~np.isfinite(R)):
        raise ValueError("R contains non-finite values")
    
    if np.any(~np.isfinite(V)):
        raise ValueError("V contains non-finite values")
    
    L = R.shape[1]
    extrap_method = advection.get_method(extrap_method)
    R = R[-(ar_order + 1):, :, :].copy()
    
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
        R = np.stack(list(dask.compute(*res)) + [R[-1, :, :]])
    
    if conditional:
        MASK_thr = np.logical_and.reduce([R[i, :, :] >= R_thr for i in range(R.shape[0])])
    else:
        MASK_thr = None
    
    # initialize the band-pass filter
    filter_method = cascade.get_method(bandpass_filter_method)
    filter = filter_method(L, num_cascade_levels, **filter_kwargs)
    
    # compute the cascade decompositions of the input precipitation fields
    decomp_method = cascade.get_method(decomp_method)
    R_d = []
    for i in range(ar_order+1):
        R_ = decomp_method(R[i, :, :], filter, MASK=MASK_thr)
        R_d.append(R_)
    
    # normalize the cascades and rearrange them into a four-dimensional array 
    # of shape (num_cascade_levels,ar_order+1,L,L) for the autoregressive model
    R_c,mu,sigma = _stack_cascades(R_d, num_cascade_levels)
    
    # compute lag-l temporal autocorrelation coefficients for each cascade level
    GAMMA = np.empty((num_cascade_levels, ar_order))
    for i in range(num_cascade_levels):
        R_c_ = np.stack([R_c[i, j, :, :] for j in range(ar_order+1)])
        GAMMA[i, :] = correlation.temporal_autocorrelation(R_c_, MASK=MASK_thr)
    
    _print_corrcoefs(GAMMA)
    
    if ar_order == 2:
        # adjust the lag-2 correlation coefficient to ensure that the AR(p) 
        # process is stationary
        for i in range(num_cascade_levels):
            GAMMA[i, 1] = autoregression.adjust_lag2_corrcoef(GAMMA[i, 0], GAMMA[i, 1])
    
    # estimate the parameters of the AR(p) model from the autocorrelation 
    # coefficients
    PHI = np.empty((num_cascade_levels, ar_order+1))
    for i in range(num_cascade_levels):
        PHI[i, :] = autoregression.estimate_ar_params_yw(GAMMA[i, :])
    
    _print_ar_params(PHI, False)
    
    # discard all except the two last cascades because they are not needed for 
    # the AR(p) model
    R_c = R_c[:, 1:, :, :]
    
    # stack the cascades into a five-dimensional array containing all ensemble 
    # members
    R_c = np.stack([R_c.copy() for i in range(num_ens_members)])
    
    if perturbation_method is not None:
    
        # get methods for perturbations
        init_noise, generate_noise = noise.get_method(perturbation_method)
        
        # initialize the perturbation generator for the precipitation field
        pp = init_noise(R[-1, :, :])
    
    if vp_par is not None:
    
        # initialize the perturbation generators for the motion field
        vps = []
        for j in range(num_ens_members):
            vp_ = noise.motion.initialize_bps(V, vp_par, vp_perp, pixelsperkm, 
                                              timestep)
            vps.append(vp_)
    
    D = [None for j in range(num_ens_members)]
    R_f = [[] for j in range(num_ens_members)]
    
    if use_precip_mask:
        # compute the wet area ratio and the precipitation mask
        war = 1.0*np.sum(R[-1, :, :] >= R_thr) / (R.shape[1]*R.shape[2])
        R_min = np.min(R)
        R_m = R_c.copy()
    
    # iterate each time step
    for t in range(num_timesteps):
        print("Computing nowcasts for time step %d..." % (t+1), end="")
        sys.stdout.flush()
        starttime = time.time()
        
        # iterate each ensemble member
        res = []
        def worker(j):
            if perturbation_method is not None:
                # generate noise field
                EPS = generate_noise(pp)
                # decompose the noise field into a cascade
                EPS = decomp_method(EPS, filter)
            else:
                EPS = None
            
            # iterate the AR(p) model for each cascade level
            for i in range(num_cascade_levels):
            
                # normalize the noise cascade
                if EPS is not None:
                    EPS_ = (EPS["cascade_levels"][i, :, :] - EPS["means"][i]) / EPS["stds"][i]
                else:
                    EPS_ = None
                # apply AR(p) process to cascade level
                R_c[j, i, :, :, :] = \
                    autoregression.iterate_ar_model(R_c[j, i, :, :, :], PHI[i, :], EPS=EPS_)
                # use a separate AR(p) model for the non-perturbed forecast, 
                # from which the mask is obtained
                if use_precip_mask:
                    R_m[j, i, :, :, :] = \
                        autoregression.iterate_ar_model(R_m[j, i, :, :, :], PHI[i, :])
            
            # compute the recomposed precipitation field(s) from the cascades 
            # obtained from the AR(p) model(s)
            R_r = _recompose_cascade(R_c[j, :, :, :], mu, sigma)
            if use_precip_mask:
                R_m_ = _recompose_cascade(R_m[j, :, :, :], mu, sigma)
            
            if use_precip_mask:
                # obtain the precipitation mask from the non-perturbed 
                # forecast that is scale-filtered by the AR(p) model
                
                # compute the threshold value R_mask_thr corresponding to the 
                # same fraction of precipitation pixels (values above R_min) 
                # as in the most recently observed precipitation field
                R_s = R_m_.flatten()
                R_s.sort(kind="quicksort")
                x = 1.0*np.arange(1, len(R_s)+1)[::-1] / len(R_s)
                i = np.argmin(abs(x - war))
                R_mask_thr = R_s[i]
                
                # apply the mask
                MASK_p = R_m_ < R_mask_thr
                R_r[MASK_p] = R_min
            
            if use_probmatching:
                # adjust the empirical probability distribution of the forecast 
                # to match the most recently measured precipitation field
                R_r = probmatching.nonparam_match_empirical_cdf(R_r, R[-1, :, :])
            
            # compute the perturbed motion field
            if vp_par is not None:
                V_ = V + noise.motion.generate_bps(vps[j], t*timestep)
            else:
                V_ = V
            
            # advect the recomposed precipitation field to obtain the forecast 
            # for time step t
            extrap_kwargs.update({"D_prev":D[j], "return_displacement":True})
            R_f_,D_ = extrap_method(R_r, V_, 1, **extrap_kwargs)
            D[j] = D_
            R_f_ = R_f_[0]
            
            return R_f_
        
        for j in range(num_ens_members):
            if not dask_imported or num_ens_members == 1:
                res.append(worker(j))
            else:
                res.append(dask.delayed(worker)(j))
        
        R_f_ = dask.compute(*res) if dask_imported and num_ens_members > 1 else res
        for j in range(num_ens_members):
            R_f[j].append(R_f_[j])
        
        print("done in %.2f seconds." % (time.time() - starttime))
    
    if num_ens_members == 1:
        return np.stack(R_f[0])
    else:
        return np.stack([np.stack(R_f[j]) for j in range(num_ens_members)])

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
    
    m = PHI.shape[0]
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

def _stack_cascades(R_d, num_levels):
  R_c   = []
  mu    = np.empty(num_levels)
  sigma = np.empty(num_levels)
  
  for i in range(num_levels):
      R_ = []
      for j in range(len(R_d)):
          mu_    = R_d[j]["means"][i]
          sigma_ = R_d[j]["stds"][i]
          if j == 2:
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
