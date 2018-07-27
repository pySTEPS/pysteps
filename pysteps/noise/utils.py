"""Miscellaneous utility functions related to generating stochastic perturbations."""

import numpy as np

def compute_noise_stddev_adjs(R, R_thr_1, R_thr_2, F, decomp_method, num_iter, 
                              conditional=True):
    """Simulate the effect of applying a precipitation mask to a Gaussian noise 
    field obtained by the nonparametric filter method. The idea is to decompose 
    the masked noise field into a cascade and compare the standard deviations of 
    each level into those of the observed precipitation intensity field. This 
    gives correction factors for the standard deviations (Bowler et al. 2006). 
    The calculations are done for n realizations of the noise field, and the 
    correction factors are calculated from the average values of the standard 
    deviations.
    
    Parameters
    ----------
    R : array_like
        The input precipitation field, assumed to be in logarithmic units 
        (dBR or reflectivity).
    R_thr_1 : float
        Intensity threshold for precipitation/no precipitation.
    R_thr_2 : float
        Intensity values below R_thr_1 are set to this value.
    F : dict
        A bandpass filter dictionary returned by a method defined in 
        pysteps.cascade.bandpass_filters. This defines the filter to use and 
        the number of cascade levels.
    decomp_method : function
        A function defined in pysteps.cascade.decomposition. Specifies the 
        method to use for decomposing the observed precipitation field and 
        noise field into different spatial scales.
    num_iter : int
        The number of noise fields to generate.
    conditional : bool
        If set to True, compute the statistics conditionally by excluding areas 
        of no precipitation.
    
    Returns
    -------
    out : list
        A list containing the standard deviation adjustment factor for each 
        cascade level.
    """
    if R.shape[0] != R.shape[1]:
        raise ValueError("the dimensions of the input field are %dx%d, but square shape expected" % (R.shape[0], R.shape[1]))
    
    MASK = R >= R_thr_1
    
    R = R.copy()
    R[~np.isfinite(R)] = R_thr_2
    R[~MASK] = R_thr_2
    if not conditional:
        mu,sigma = np.mean(R),np.std(R)
    else:
        mu,sigma = np.mean(R[MASK]),np.std(R[MASK])
    R -= mu
    
    MASK_ = MASK if conditional else None
    decomp_R = decomp_method(R, F, MASK=MASK_)
    
    N_stds = []
    
    for k in range(num_iter):
        # generate Gaussian white noise field, multiply it with the standard 
        # deviation of the observed field and apply the precipitation mask
        N = np.random.randn(R.shape[0], R.shape[1])
        N = np.real(np.fft.ifft2(np.fft.fft2(N) * abs(np.fft.fft2(R))))
        N = (N - np.mean(N)) / np.std(N) * sigma
        N[~MASK] = R_thr_2 - mu
        
        # subtract the mean and decompose the masked noise field into a cascade
        N -= mu
        decomp_N = decomp_method(N, F, MASK=MASK_)
        
        N_stds.append(decomp_N["stds"])
    
    # for each cascade level, compare the standard deviations between the 
    # observed field and the masked noise field, which gives the correction 
    # factors
    return decomp_R["stds"] / np.mean(np.vstack(N_stds), axis=0)
