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
             extrap_method, decomp_method, bandpass_filter_method, 
             noise_method, pixelsperkm, timestep, ar_order=2, 
             vel_pert_method=None, conditional=False, use_precip_mask=True, 
             use_probmatching=True, callback=None, return_output=True, 
             extrap_kwargs={}, filter_kwargs={}, noise_kwargs={}, 
             vel_pert_kwargs={}, seed=None):
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
      Specifies the threshold value for minimum observable precipitation 
      intensity. Applicable if use_probmatching is True or conditional is True.
    extrap_method : str
      Name of the extrapolation method to use. See the documentation of 
      pysteps.advection for the available choices.
    decomp_method : str
      Name of the cascade decomposition method to use. See the documentation 
      of pysteps.cascade.decomposition.
    bandpass_filter_method : str
      Name of the bandpass filter method to use with the cascade decomposition. 
      See the documentation of pysteps.cascade.bandpass_filters.
    noise_method : str
      Name of the noise generator to use for perturbating the precipitation 
      field. See the documentation of pysteps.noise.interface.
    pixelsperkm : float
      Spatial resolution of the motion field (pixels/kilometer).
    timestep : float
      Time step of the motion vectors (minutes).
    vel_pert_method : str
      Name of the noise generator to use for perturbing the velocity field. See 
      the documentation of pysteps.noise.interface.
    ar_order : int
      The order of the autoregressive model to use.
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
      field. In this case, use_precip_mask is also set to True.
    callback : function
      Optional function that is called after computation of each time step of 
      the nowcast. The function takes one argument: a three-dimensional array 
      of shape (num_ens_members,h,w), where h and w are the height and width 
      of the input field R, respectively. This can be used, for instance, 
      writing the outputs into files.
    return_output : bool
      Set to False to disable returning the outputs as numpy arrays. This can 
      save memory if the intermediate results are written to output files using 
      the callback function.
    extrap_kwargs : dict
      Optional dictionary that is supplied as keyword arguments to the 
      extrapolation method.
    filter_kwargs : dict
      Optional dictionary that is supplied as keyword arguments to the 
      filter method.
    noise_kwargs : dict
      Optional dictionary that is supplied as keyword arguments to the 
      initializer of the noise generator. See the documentation of 
      pysteps.noise.fftgenerators.
    vel_pert_kwargs : dict
      Optional dictionary that is supplied as keyword arguments to the 
      initializer of the velocity perturbator. See the documentation of 
      pysteps.noise.motion.
    seed : int
      Optional seed number for the random generators.
    
    Returns
    -------
    out : ndarray
      If return_output is True, a four-dimensional array of shape 
      (num_ens_members,num_timesteps,m,n) containing a time series of forecast 
      precipitation fields for each ensemble member. Otherwise, a None value 
      is returned.
    """
    _check_inputs(R, V, ar_order)
    
    if np.any(~np.isfinite(R)):
        raise ValueError("R contains non-finite values")
    
    if np.any(~np.isfinite(V)):
        raise ValueError("V contains non-finite values")
    
    if use_probmatching:
        use_precip_mask = True
    
    print("Computing STEPS nowcast:")
    print("------------------------")
    print("")
    
    print("Inputs:")
    print("-------")
    print("input dimensions: %dx%d" % (R.shape[1], R.shape[2]))
    print("pixels/km:        %g" % pixelsperkm)
    print("time step:        %d minutes" % timestep)
    print("")
    
    print("Methods:")
    print("--------")
    print("extrapolation:        %s" % extrap_method)
    print("bandpass filter:      %s" % bandpass_filter_method)
    print("decomposition:        %s" % decomp_method)
    print("noise generator:      %s" % noise_method)
    print("velocity perturbator: %s" % vel_pert_method)
    print("precipitation mask:   %s" % "yes" if use_precip_mask  else "no")
    print("probability matching: %s" % "yes" if use_probmatching else "no")
    print("")
    
    print("Parameters:")
    print("-----------")
    print("number of time steps:     %d" % num_timesteps)
    print("ensemble size:            %d" % num_ens_members)
    print("number of cascade levels: %d" % num_cascade_levels)
    print("order of the AR(p) model: %d" % ar_order)
    if vel_pert_method is not None:
        vp_par  = vel_pert_kwargs["p_pert_par"]
        vp_perp = vel_pert_kwargs["p_pert_perp"]
        print("velocity perturbations, parallel:      %g,%g,%g" % \
            (vp_par[0],  vp_par[1],  vp_par[2]))
        print("velocity perturbations, perpendicular: %g,%g,%g" % \
            (vp_perp[0], vp_perp[1], vp_perp[2]))
    
    if conditional:
        print("conditional precip. intensity threshold: %g" % R_thr)
    
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
    R_d = None
    
    # compute lag-l temporal autocorrelation coefficients for each cascade level
    GAMMA = np.empty((num_cascade_levels, ar_order))
    for i in range(num_cascade_levels):
        R_c_ = np.stack([R_c[i, j, :, :] for j in range(ar_order+1)])
        GAMMA[i, :] = correlation.temporal_autocorrelation(R_c_, MASK=MASK_thr)
    R_c_ = None
    
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
    
    # discard all except the p-1 last cascades because they are not needed for 
    # the AR(p) model
    R_c = R_c[:, -ar_order:, :, :]
    
    # stack the cascades into a five-dimensional array containing all ensemble 
    # members
    R_c = np.stack([R_c.copy() for i in range(num_ens_members)])
    
    # initialize the random generators
    if noise_method is not None:
        randgen_prec   = []
        randgen_motion = []
        np.random.seed(seed)
        for j in range(num_ens_members):
            rs = np.random.RandomState(seed)
            randgen_prec.append(rs)
            seed = rs.randint(0, high=1e9)
            rs = np.random.RandomState(seed)
            randgen_motion.append(rs)
            seed = rs.randint(0, high=1e9)
    
    if noise_method is not None:
        # get methods for perturbations
        init_noise, generate_noise = noise.get_method(noise_method)
        
        # initialize the perturbation generator for the precipitation field
        pp = init_noise(R[-1, :, :], **noise_kwargs)
    
    if vel_pert_method is not None:
        init_vel_noise, generate_vel_noise = noise.get_method(vel_pert_method)
        
        # initialize the perturbation generators for the motion field
        vps = []
        for j in range(num_ens_members):
            kwargs = {"randstate":randgen_motion[j], 
                      "p_pert_par":vp_par, 
                      "p_pert_perp":vp_perp}
            vp_ = init_vel_noise(V, pixelsperkm, timestep, **kwargs)
            vps.append(vp_)
    
    D = [None for j in range(num_ens_members)]
    R_f = [[] for j in range(num_ens_members)]
    
    if use_precip_mask or use_probmatching:
        MASK_thr = R[-1, :, :] >= R_thr
    
    if use_precip_mask:
        # compute the wet area ratio and the precipitation mask
        war = 1.0*np.sum(MASK_thr) / (R.shape[1]*R.shape[2])
        R_min = np.min(R)
        #R_m = R_c.copy()
    
    if use_probmatching:
        pmm_bin_edges = np.linspace(R_thr, 60, 200)
        hist = np.histogram(R[-1, :, :][MASK_thr], bins=pmm_bin_edges)[0]
        R0_cdf = probmatching.compute_empirical_cdf(pmm_bin_edges, hist)
    
    R = R[-1, :, :]
    
    print("Starting nowcast computation.")
    
    # iterate each time step
    for t in range(num_timesteps):
        print("Computing nowcast for time step %d... " % (t+1), end="")
        sys.stdout.flush()
        starttime = time.time()
        
        # iterate each ensemble member
        def worker(j):
            if noise_method is not None:
                # generate noise field
                EPS = generate_noise(pp, randstate=randgen_prec[j])
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
                #if use_precip_mask:
                #    R_m[j, i, :, :, :] = \
                #        autoregression.iterate_ar_model(R_m[j, i, :, :, :], PHI[i, :])
            
            EPS  = None
            EPS_ = None
            
            # compute the recomposed precipitation field(s) from the cascades 
            # obtained from the AR(p) model(s)
            R_r = _recompose_cascade(R_c[j, :, :, :], mu, sigma)
            if use_precip_mask:
                # obtain the precipitation mask from the non-perturbed 
                # forecast that is scale-filtered by the AR(p) model
                #R_m_ = _recompose_cascade(R_m[j, :, :, :], mu, sigma)
                
                #R_s = R_m_.flatten()
                
                # compute the threshold value R_mask_thr corresponding to the 
                # same fraction of precipitation pixels (forecast values above 
                # R_min) as in the most recently observed precipitation field
                R_s = R_r.flatten()
                R_s.sort(kind="quicksort")
                x = 1.0*np.arange(1, len(R_s)+1)[::-1] / len(R_s)
                i = np.argmin(abs(x - war))
                # handle ties
                if R_s[i] == R_s[i + 1]:
                    i = np.where(R_s == R_s[i])[0][-1] + 1
                R_mask_thr = R_s[i]
                
                # apply the mask and adjust the intensity values to preserve 
                # the wet-area ratio
                MASK_p = R_r < R_mask_thr
                R_r[~MASK_p] = R_r[~MASK_p] + (R_thr - R_mask_thr)
                R_r[MASK_p] = R_min
                
                R_s  = None
                #R_m_ = None
            
            if use_probmatching:
                # adjust the conditional CDF of the forecast to match the most 
                # recently measured precipitation field
                R_out_hist = np.histogram(R_r[~MASK_p], bins=pmm_bin_edges)[0]
                R_out_cdf = probmatching.compute_empirical_cdf(pmm_bin_edges, R_out_hist)
                pmm = probmatching.pmm_init(pmm_bin_edges, R_out_cdf, pmm_bin_edges, R0_cdf)
                R_r[~MASK_p] = probmatching.pmm_compute(pmm, R_r[~MASK_p])
                
                # TODO: this is needed because the probability matching 
                # introduces nan values for some reason. Take a closer look 
                # on this.
                R_r[~np.isfinite(R_r)] = R_thr
                
                R_out_hist = None
                R_out_cdf = None
                pmm = None
                # the old version is currently commented out
                #R_r = probmatching.nonparam_match_empirical_cdf(R_r, R)
            
            # compute the perturbed motion field
            if vel_pert_method is not None:
                V_ = V + generate_vel_noise(vps[j], t*timestep)
            else:
                V_ = V
            
            # advect the recomposed precipitation field to obtain the forecast 
            # for time step t
            extrap_kwargs.update({"D_prev":D[j], "return_displacement":True})
            R_f_,D_ = extrap_method(R_r, V_, 1, **extrap_kwargs)
            D[j] = D_
            R_f_ = R_f_[0]
            
            return R_f_
        
        res = []
        for j in range(num_ens_members):
            if not dask_imported or num_ens_members == 1:
                res.append(worker(j))
            else:
                res.append(dask.delayed(worker)(j))
        
        R_f_ = dask.compute(*res) if dask_imported and num_ens_members > 1 else res
        res = None
        
        print("%.2f seconds." % (time.time() - starttime))
        
        if callback is not None:
            callback(np.stack(R_f_))
            R_f_ = None
        
        if return_output:
            for j in range(num_ens_members):
                R_f[j].append(R_f_[j])
    
    if return_output:
        if num_ens_members == 1:
            return np.stack(R_f[0])
        else:
            return np.stack([np.stack(R_f[j]) for j in range(num_ens_members)])
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
  
  num_inputs = len(R_d)
  
  for i in range(num_levels):
      R_ = []
      for j in range(num_inputs):
          mu_    = R_d[j]["means"][i]
          sigma_ = R_d[j]["stds"][i]
          if j == num_inputs - 1:
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
