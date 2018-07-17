"""Implementation of the STEPS method."""

import numpy as np
from .. import advection
from .. import cascade
from .. import noise
from ..postprocessing import probmatching
from ..timeseries import autoregression, correlation
try:
    import dask
    dask_imported = True
except ImportError:
    dask_imported = False

# TODO: Add options for choosing the perturbation methods.
def forecast(R, V, num_timesteps, num_ens_members, num_cascade_levels, R_thr, 
             extrap_method, decomp_method, bandpass_filter_method, perturbation_method,
             pixelsperkm, timestep, ar_order=2, vp_par=(10.88,0.23,-7.68), 
             vp_perp=(5.76,0.31,-2.72), conditional=True, use_precip_mask=False, 
             use_probmatching=True, extrap_kwargs={}, filter_kwargs={}):
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
      inputs.
    num_timesteps : int
      Number of time steps to forecast.
    num_ens_members : int
      The number of ensemble members to generate.
    num_cascade_levels : int
      The number of cascade levels to use.
    R_thr : float
      Specifies the threshold value to use if conditional is True.
    extrap_method : str
      Name of the extrapolation method to use. See the documentation of the 
      advection module for the available choices.
    decomp_method : str
      Name of the cascade decomposition method to use, see the documentation 
      of the cascade.decomposition module.
    bandpass_filter_method : str
      Name of the bandpass filter method to use with the cascade decomposition, 
      see the documentation of the cascade.bandpass_filters module.
    perturbation_method : str
      Name of the noise generator to use for the perturbations of the precipitation
      field, see the documentation of the noise.fftgenerators module.
    pixelsperkm : float
      Spatial resolution of the motion field (pixels/kilometer).
    timestep : float
      Time step for the motion vectors (minutes).
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
      If set to True, compute the correlation coefficients conditionally by 
      excluding the areas where the values are below the threshold R_thr.
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
    
    L = R.shape[1]
    extrap_method = advection.get_method(extrap_method)
    R = R[-(ar_order + 1):, :, :].copy()
    
    # Advect the previous precipitation fields to the same position with the 
    # most recent one (i.e. transform them into the Lagrangian coordinates).
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
        MASK = np.logical_and.reduce([R[i, :, :] >= R_thr for i in range(R.shape[0])])
    else:
        MASK = None
    
    # Initialize the band-pass filter.
    filter_method = cascade.get_method(bandpass_filter_method)
    filter = filter_method(L, num_cascade_levels, **filter_kwargs)
    
    # Compute the cascade decompositions of the input precipitation fields.
    decomp_method = cascade.get_method(decomp_method)
    R_d = []
    for i in range(ar_order+1):
        R_ = decomp_method(R[i, :, :], filter, MASK=MASK)
        R_d.append(R_)
    
    # Normalize the cascades and rearrange them into a four-dimensional array 
    # of shape (num_cascade_levels,ar_order+1,L,L) for the autoregressive model.
    R_c,mu,sigma = _stack_cascades(R_d, num_cascade_levels)
    
    # Compute lag-l temporal autocorrelation coefficients for each cascade level.
    GAMMA = np.empty((num_cascade_levels, ar_order))
    for i in range(num_cascade_levels):
        R_c_ = np.stack([R_c[i, j, :, :] for j in range(ar_order+1)])
        GAMMA[i, :] = correlation.temporal_autocorrelation(R_c_, MASK=MASK)
    
    _print_corrcoefs(GAMMA)
    
    if ar_order == 2:
        # Adjust the lag-2 correlation coefficient to ensure that the AR(p) 
        # process is stationary.
        for i in range(num_cascade_levels):
            GAMMA[i, 1] = autoregression.adjust_lag2_corrcoef(GAMMA[i, 0], GAMMA[i, 1])
    
    # Estimate the parameters of the AR(p) model from the autocorrelation 
    # coefficients.
    PHI = np.empty((num_cascade_levels, ar_order+1))
    for i in range(num_cascade_levels):
        PHI[i, :] = autoregression.estimate_ar_params_yw(GAMMA[i, :])
    
    _print_ar_params(PHI, False)
    
    # Discard all except the two last cascades because they are not needed for 
    # the AR(p) model.
    R_c = R_c[:, 1:, :, :]
    
    # Stack the cascades into a five-dimensional array containing all ensemble 
    # members.
    R_c = np.stack([R_c.copy() for i in range(num_ens_members)])
    
    if perturbation_method is not None:
    
        # Get methods for perturbations
        init_noise, generate_noise = noise.get_method(perturbation_method)
        
        # Initialize the perturbation generator for the precipitation field.
        pp = init_noise(R[-1, :, :])
    
    if vp_par is not None:
    
        # Initialize the perturbation generators for the motion field.
        vps = []
        for j in range(num_ens_members):
            vp_ = noise.motion.initialize_bps(V, vp_par, vp_perp, pixelsperkm, 
                                              timestep)
            vps.append(vp_)
    
    D = [None for j in range(num_ens_members)]
    R_f = [[] for j in range(num_ens_members)]
    
    if use_precip_mask:
        # Compute the wet area ratio and precipitation mask.
        war = 1.0*np.sum(R[-1, :, :] >= R_thr) / (R.shape[1]*R.shape[2])
        # TODO: The commented code implements a different version of the masking. 
        # Allow the user to choose the method.
        # MASK_p = (R[-1, :, :] >= R_thr).astype(float)
        R_min = np.min(R)
        R_m = R_c.copy()
        
    # Iterate each time step
    for t in range(num_timesteps):
    
        # Iterate each ensemble member
        for j in range(num_ens_members):
        
            # If perturbations are needed
            if perturbation_method is not None:
                # Generate noise field
                EPS = generate_noise(pp)
                # Decompose the noise field
                EPS = decomp_method(EPS, filter)
            else:
                EPS = None
                
            # Iterate the AR(p) model for each cascade level.
            for i in range(num_cascade_levels):
            
                # Normalize the noise cascade
                if EPS is not None:
                    EPS_ = (EPS["cascade_levels"][i, :, :] - EPS["means"][i]) / EPS["stds"][i]
                else:
                    EPS_ = None
                # Apply AR(p) process to cascade level
                R_c[j, i, :, :, :] = \
                    autoregression.iterate_ar_model(R_c[j, i, :, :, :], PHI[i, :], EPS=EPS_)
                # Use a separate AR(p) model for the non-perturbed forecast, 
                # from which the mask is obtained.
                if use_precip_mask:
                    R_m[j, i, :, :, :] = \
                        autoregression.iterate_ar_model(R_m[j, i, :, :], PHI[i, :])
        
            # Compute the recomposed precipitation field from the cascade obtained 
            # from the AR(p) model.
            R_r = [(R_c[j, i, -1, :, :] * sigma[i]) + mu[i] for i in range(num_cascade_levels)]
            R_r = np.sum(np.stack(R_r), axis=0)
            
            if use_precip_mask:
                R_m_ = [(R_m[j, i, -1, :, :] * sigma[i]) + mu[i] for i in range(num_cascade_levels)]
                R_m_ = np.sum(np.stack(R_m_), axis=0)
            
            if vp_par is not None:
                V_ = V + noise.motion.generate_bps(vps[j], t*timestep)
            else:
                V_ = V
            
            if use_precip_mask:
                # Compute the threshold value R_mask_thr corresponding to the 
                # same fraction of precipitation pixels (values above R_min) 
                # as in the most recently observed precipitation field.
                R_s = R_m_.flatten()
                R_s.sort(kind="quicksort")
                x = 1.0*np.arange(1, len(R_s)+1)[::-1] / len(R_s)
                i = np.argmin(abs(x - war))
                R_mask_thr = R_s[i]
                
                MASK_p = R_m_ < R_mask_thr
                R_r[MASK_p] = R_min
            
            if use_probmatching:
                # Adjust the empirical probability distribution of the forecast 
                # to match the most recently measured precipitation field.
                R_r = probmatching.nonparam_match_empirical_cdf(R_r, R[-1, :, :])
            
            # Advect the recomposed precipitation field to obtain the forecast 
            # for time step t.
            extrap_kwargs.update({"D_prev":D[j], "return_displacement":True})
            R_f_,D_ = extrap_method(R_r, V_, 1, **extrap_kwargs)
            D[j] = D_
            R_f_ = R_f_[0]
            
            #if use_precip_mask:
                # Advect the precipitation mask and apply it to the output.
                #extrap_kwargs.update({"D_prev":D[j], "return_displacement":False})
                #MASK_p_ = extrap_method(MASK_p, V, 1, **extrap_kwargs))[0]
                #R_f_[MASK_p_ < 0.5] = R_min
            
            R_f[j].append(R_f_)
    
    if num_ens_members == 1:
        return np.stack(R_f[0])
    else:
        return np.stack([np.stack(R_f[j]) for j in range(num_ens_members)])

def _check_inputs(R, V, ar_order):
    if len(R.shape) != 3:
        raise ValueError("R must be a three-dimensional array")
    if R.shape[0] < ar_order + 1:
        raise ValueError("R.shape[0] < ar_order+1")
    if R.shape[1] != R.shape[2]:
        raise ValueError("the dimensions of the input fields are %dx%d, but square shape expected" % \
                        (R.shape[1], R.shape[2]))
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
