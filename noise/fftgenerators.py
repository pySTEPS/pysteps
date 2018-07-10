"""Methods for noise generators based on FFT filtering of white noise."""

import numpy as np

# Use the SciPy FFT implementation by default. If SciPy is not installed, fall 
# back to the numpy implementation.
try:
    import scipy.fftpack as fft
except ImportError:
    from numpy import fft

def initialize_param_2d_fft_filter(X, **kwargs):
    """Takes a 2d input field and produces a fourier filter by using the Fast 
    Fourier Transform (FFT).
    
    Parameters
    ----------
    X : array-like
      Two-dimensional square array containing the input field. All values are 
      required to be finite.
      
    Optional kwargs:
    ----------
    win_type : string
       Optional tapering function to be applied to X.
       Default : flat-hanning
    model : string
        The parametric model to be used to fit the power spectrum of X.
        Default : power-law
    weighted : bool
        Whether or not to apply the sqrt(power) as weight in the polyfit() function.
        Default : True
        
    Returns
    -------
    F : array-like
      A two-dimensional array containing the parametric filter.
      It can be passed to generate_noise_2d_fft_filter().
    """
    
    if len(X.shape) != 2:
        raise ValueError("the input is not two-dimensional array")
    if np.any(~np.isfinite(X)):
      raise ValueError("X contains non-finite values")
    if X.shape[0] != X.shape[1]:
        raise ValueError("a square array expected, but the shape of X is (%d,%d)" % \
                         (X.shape[0], X.shape[1]))
       
    # defaults
    win_type = kwargs.get('win_type', 'flat-hanning')
    model    = kwargs.get('model', 'power-law')
    weighted = kwargs.get('weighted', True)
        
    L = X.shape[0]
    
    X = X.copy()
    if win_type is not None:
        X -= X.min()
        tapering = build_2D_tapering_function((L, L), win_type)
    else:
        tapering = np.ones_like(X)
    
    if model.lower() == 'power-law':
       
        # compute radially averaged PSD
        psd = _rapsd(X*tapering)
        
        # wavenumbers
        if L % 2 == 0:
            wn = np.arange(0, int(L/2)+1)
        else:
            wn = np.arange(0, int(L/2))
        
        # compute spectral slope Beta
        if weighted:
            p0 = np.polyfit(np.log(wn[1:]), np.log(psd[1:]), 1, w=np.sqrt(psd[1:]))
        else:
            p0 = np.polyfit(np.log(wn[1:]), np.log(psd[1:]), 1)
        beta = -p0[0]
        
        # compute 2d filter
        if L % 2 == 1:
            XC,YC = np.ogrid[-int(L/2):int(L/2)+1, -int(L/2):int(L/2)+1]
        else:
            XC,YC = np.ogrid[-int(L/2):int(L/2), -int(L/2):int(L/2)]
        R = np.sqrt(XC*XC + YC*YC)
        R = fft.fftshift(R)
        F = R**(-beta)
        F[~np.isfinite(F)] = 1
    
    else:
        raise ValueError("unknown parametric model %s" % model)
    

    return F
 
def initialize_nonparam_2d_fft_filter(X, **kwargs):
    """Takes a 2d input field and produces a fourier filter by using the Fast 
    Fourier Transform (FFT).
    
    Parameters
    ----------
    X : array-like
      Two-dimensional array containing the input field. All values are required 
      to be finite.
      
    Optional kwargs:
    ----------
    win_type : string
       Optional tapering function to be applied to X.
       Default : flat-hanning
    donorm : bool
       Option to normalize the real and imaginary parts.
       Default : False
    
    Returns
    -------
    F : array-like
      A two-dimensional array containing the non-parametric filter.
      It can be passed to generate_noise_2d_fft_filter().
    """
    if len(X.shape) != 2:
        raise ValueError("the input is not two-dimensional array")
    if np.any(~np.isfinite(X)):
      raise ValueError("X contains non-finite values")
      
    # defaults
    win_type = kwargs.get('win_type', 'flat-hanning')
    donorm   = kwargs.get('donorm', False)
      
    X = X.copy()
    if win_type is not None:
        X -= X.min()
        tapering = build_2D_tapering_function(X.shape, win_type)
    else:
        tapering = np.ones_like(X)
    F = fft.fft2(X*tapering)
    
    # normalize the real and imaginary parts
    if donorm:
        F.imag = (F.imag - np.mean(F.imag))/np.std(F.imag)
        F.real = (F.real - np.mean(F.real))/np.std(F.real)
    
    return np.abs(F)
 
def generate_noise_2d_fft_filter(F, seed=None):
    """Produces a field of correlated noise using global Fourier filering.
    
    Parameters
    ----------
    F : array-like
        Two-dimensional array containing the input filter. 
        It can be computed by related methods.
        All values are required to be finite.
    seed : int
        Value to set a seed for the generator. None will not set the seed.
    
    Returns
    -------
    N : array-like
        A two-dimensional numpy array of stationary correlated noise.
    """
    
    if len(F.shape) != 2:
        raise ValueError("the input is not two-dimensional array")
    if np.any(~np.isfinite(F)):
      raise ValueError("F contains non-finite values")
      
    # set the seed
    np.random.seed(seed)
    
    # produce fields of white noise
    N = np.random.randn(F.shape[0], F.shape[1])
    
    # apply the global Fourier filter to impose a correlation structure
    fN = fft.fft2(N)
    fN *= F
    N = np.array(fft.ifft2(fN).real)
    N = (N - N.mean())/N.std()
            
    return N
       
def initialize_nonparam_2d_ssft_filter(X, **kwargs):
    """Function to compute the local Fourier filters using the Short-Space Fourier
    filtering approach.
    
    Reference
    ---------
    Nerini et al. (2017), "A non-stationary stochastic ensemble generator for radar 
    rainfall fields based on the short-space Fourier transform", 
    https://doi.org/10.5194/hess-21-2777-2017.


    Parameters
    ----------
    X : array-like
        Two-dimensional array containing the input field. All values are required 
        to be finite.

    Optional kwargs:
    ----------
    win_size : int or two-element tuple of ints
        Size-length of the window to compute the SSFT.
        Default : (128, 128)
    win_type : string ['hanning', 'flat-hanning'] 
        Type of window used for localization.
        Default : flat-hanning
    overlap : float [0,1[ 
        The proportion of overlap to be applied between successive windows.
        Default : 0.3
    war_thr : float [0,1]
        Threshold for the minimum fraction of rain needed for computing the FFT.
        Default : 0.1

    Returns
    -------
    F : array-like
        Four-dimensional array containing the 2d fourier filters distributed over
        a 2d spatial grid.
    """
    
    if len(X.shape) != 2:
        raise ValueError("X must be a two-dimensional array")
    if np.any(np.isnan(X)):
        raise ValueError("X must not contain NaNs")
        
    # defaults
    win_size = kwargs.get('win_size', (128,128))
    if type(win_size) == int:
        win_size = (win_size, win_size)
    win_type = kwargs.get('win_type', 'flat-hanning')
    overlap  = kwargs.get('overlap', 0.3)
    war_thr  = kwargs.get('war_thr', 0.1)
    
    # make sure non-rainy pixels are set to zero
    min_value = np.min(X)
    X = X.copy()
    X -= min_value
    
    # 
    dim = X.shape
    dim_x = dim[1]
    dim_y = dim[0]
       
    # SSFT algorithm 
    
    # prepare indices
    idxi = np.zeros((2, 1), dtype=int)
    idxj = np.zeros((2, 1), dtype=int)
    
    # number of windows
    num_windows_y = np.ceil( float(dim_y) / win_size[0] ).astype(int)
    num_windows_x = np.ceil( float(dim_x) / win_size[1] ).astype(int)
    
    # domain fourier filter
    F0 = initialize_nonparam_2d_fft_filter(X, win_type, True)
    # and allocate it to the final grid
    F = np.zeros((num_windows_y, num_windows_x, F0.shape[0], F0.shape[1]))
    F += F0[np.newaxis, np.newaxis, :, :]

    # loop rows
    for i in range(F.shape[0]):
        # loop columns
        for j in range(F.shape[1]):
        
            # compute indices of local window
            idxi[0] = np.max( (i*win_size[0] - overlap*win_size[0], 0) ).astype(int)
            idxi[1] = np.min( (idxi[0] + win_size[0]  + overlap*win_size[0], dim_y) ).astype(int)
            idxj[0] = np.max( (j*win_size[1] - overlap*win_size[1], 0) ).astype(int)
            idxj[1] = np.min( (idxj[0] + win_size[1]  + overlap*win_size[1], dim_x) ).astype(int)
       
            # build localization mask
            # TODO: the 0.01 rain threshold must be improved
            mask = _get_mask(dim, idxi, idxj, win_type)
            war = float(np.sum((X*mask) > 0.01)) / ((idxi[1]-idxi[0])*(idxj[1]-idxj[0]))
            
            if war > war_thr:
                # the new filter 
                F[i, j, : ,:] = initialize_nonparam_2d_fft_filter(X*mask, None, True)
                
    return F            
 
def initialize_nonparam_2d_nested_filter(X, gridres=1.0, **kwargs):
    """Function to compute the local Fourier filters using a nested approach.

    Parameters
    ----------
    X : array-like
        Two-dimensional array containing the input field. All values are required 
        to be finite and the domain must be square.
    gridres : float
        Grid resolution in km.
        
    Optional kwargs:
    ----------
    max_level : int 
        Localization parameter. 0: global noise, >0: increasing degree of localization.
        Default : 3
    win_type : string ['hanning', 'flat-hanning'] 
        Type of window used for localization.
        Default : flat-hanning
    war_thr : float [0;1]
        Threshold for the minimum fraction of rain needed for computing the FFT.
        Default : 0.1

    Returns
    -------
    F : array-like
        Four-dimensional array containing the 2d fourier filters distributed over
        a 2d spatial grid.
    """
    
    if len(X.shape) != 2:
        raise ValueError("X must be a two-dimensional array")
    if X.shape[0] != X.shape[1]:
        raise ValueError("a square array expected, but the shape of X is (%d,%d)" % \
                         (X.shape[0], X.shape[1]))
    if np.any(np.isnan(X)):
        raise ValueError("X must not contain NaNs")
        
    # defaults
    max_level = kwargs.get('max_level', 3)
    win_type  = kwargs.get('win_type', 'flat-hanning')
    war_thr   = kwargs.get('war_thr', 0.1)
    
    # make sure non-rainy pixels are set to zero
    min_value = np.min(X)
    X = X.copy()
    X -= min_value
    
    # 
    dim = X.shape
    dim_x = dim[1]
    dim_y = dim[0]
       
    # Nested algorithm 
    
    # prepare indices
    Idxi = np.array([[0, dim_y]])
    Idxj = np.array([[0, dim_x]])
    Idxipsd = np.array([[0, 2**max_level]])
    Idxjpsd = np.array([[0, 2**max_level]])
    
    # generate the FFT sample frequencies
    freq = fft.fftfreq(dim_y, gridres)
    fx,fy = np.meshgrid(freq, freq)
    freq_grid = np.sqrt(fx**2 + fy**2)
    
    # domain fourier filter
    F0 = initialize_nonparam_2d_fft_filter(X, win_type, True)
    # and allocate it to the final grid
    F = np.zeros((2**max_level, 2**max_level, F0.shape[0], F0.shape[1]))
    F += F0[np.newaxis, np.newaxis, :, :]
    
    # now loop levels and build composite spectra
    level=0 
    while level < max_level:

        for m in range(len(Idxi)):
        
            # the indices of rainfall field
            Idxinext, Idxjnext = _split_field(Idxi[m, :], Idxj[m, :], 2)
            # the indices of the field of fourier filters
            Idxipsdnext, Idxjpsdnext = _split_field(Idxipsd[m, :], Idxjpsd[m, :], 2)
            
            for n in range(len(Idxinext)):
            
                mask = _get_mask(dim, Idxinext[n, :], Idxjnext[n, :], win_type)
                war = np.sum((X*mask) > 0.01)/float((Idxinext[n, 1] - Idxinext[n, 0])**2)
                
                if war > war_thr:
                    # the new filter 
                    newfilter = initialize_nonparam_2d_fft_filter(X*mask, None, True)
                    
                    # compute logistic function to define weights as function of frequency
                    # k controls the shape of the weighting function
                    # TODO: optimize parameters
                    k = 0.05
                    x0 = (Idxinext[n, 1] - Idxinext[n, 0])/2.
                    merge_weights = 1/(1 + np.exp(-k*(1/freq_grid - x0)))
                    newfilter *= (1 - merge_weights)
                    
                    # perform the weighted average of previous and new fourier filters
                    F[Idxipsdnext[n, 0]:Idxipsdnext[n, 1], Idxjpsdnext[n,0]:Idxjpsdnext[n, 1], :, :] *= merge_weights[np.newaxis, np.newaxis, :, :]
                    F[Idxipsdnext[n, 0]:Idxipsdnext[n, 1],Idxjpsdnext[n, 0]:Idxjpsdnext[n, 1], :, :] += newfilter[np.newaxis, np.newaxis, :, :] 
            
        # update indices
        level += 1
        Idxi, Idxj = _split_field((0, dim[0]), (0, dim[1]), 2**level)
        Idxipsd, Idxjpsd = _split_field((0, 2**max_level), (0, 2**max_level), 2**level)
        
    return F
 
def generate_noise_2d_ssft_filter(F, seed=None, **kwargs):
    """Function to compute the locally correlated noise using a nested approach.

    Parameters
    ----------
    F : array-like
        Four-dimensional array containing the 2d fourier filters distributed over
        a 2d spatial grid.
    seed : int
        Value to set a seed for the generator. None will not set the seed.
        
    Optional kwargs:
    ----------
    overlap : float 
        Percentage overlap [0-1] between successive windows.
        Default : 0.2
    win_type : string ['hanning', 'flat-hanning'] 
        Type of window used for localization.
        Default : flat-hanning

    Returns
    -------
    N : array-like
        A two-dimensional numpy array of non-stationary correlated noise.

    """
    
    if len(F.shape) != 4:
        raise ValueError("the input is not four-dimensional array")
    if np.any(~np.isfinite(F)):
      raise ValueError("F contains non-finite values")
      
    # defaults
    overlap  = kwargs.get('overlap', 0.2)
    win_type = kwargs.get('win_type', 'flat-hanning')
    
    # set the seed
    np.random.seed(seed)
    
    dim_y = F.shape[2]
    dim_x = F.shape[3]
    dim = (dim_y, dim_x)
    
    # produce fields of white noise
    N = np.random.randn(dim_y, dim_x)
    fN = fft.fft2(N)
    
    # initialize variables
    cN = np.zeros(dim)
    sM = np.zeros(dim)
    
    idxi = np.zeros((2, 1), dtype=int)
    idxj = np.zeros((2, 1), dtype=int)
    
    # get the window size
    win_size = ( float(dim_y)/F.shape[0], float(dim_x)/F.shape[1] )
    
    # loop the windows and build composite image of correlated noise

    # loop rows
    for i in range(F.shape[0]):
        # loop columns
        for j in range(F.shape[1]):
        
            # apply fourier filtering with local filter
            lF = F[i,j,:,:]
            flN = fN * lF
            flN = np.array(np.fft.ifft2(flN).real)
            
            # compute indices of local window
            idxi[0] = np.max( (i*win_size[0] - overlap*win_size[0], 0) ).astype(int)
            idxi[1] = np.min( (idxi[0] + win_size[0]  + overlap*win_size[0], dim_y) ).astype(int)
            idxj[0] = np.max( (j*win_size[1] - overlap*win_size[1], 0) ).astype(int)
            idxj[1] = np.min( (idxj[0] + win_size[1]  + overlap*win_size[1], dim_x) ).astype(int)
            
            # build mask and add local noise field to the composite image
            M = _get_mask(dim, idxi, idxj, win_type)
            cN += flN*M
            sM += M 

    # normalize the field
    cN[sM > 0] /= sM[sM > 0]         
    cN = (cN - cN.mean())/cN.std()
            
    return cN
        
def build_2D_tapering_function(win_size, win_type='flat-hanning'):
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
    
def _rapsd(X):
    """Compute radially averaged PSD of input field X.
    """
    
    if X.shape[0] != X.shape[1]:
        raise ValueError("a square array expected, but the shape of X is (%d,%d)" % \
                         (X.shape[0], X.shape[1]))
    
    L = X.shape[0]
    
    if L % 2 == 1:
        XC,YC = np.ogrid[-int(L/2):int(L/2)+1, -int(L/2):int(L/2)+1]
    else:
        XC,YC = np.ogrid[-int(L/2):int(L/2), -int(L/2):int(L/2)]
    
    R = np.sqrt(XC*XC + YC*YC).astype(int)
    
    F = fft.fftshift(np.fft.fft2(X))
    F = abs(F)**2
    
    if L % 2 == 0:
        r_range = np.arange(0, int(L/2)+1)
    else:
        r_range = np.arange(0, int(L/2))
    
    result = []
    for r in r_range:
        MASK = R == r
        F_vals = F[MASK]
        result.append(np.mean(F_vals))
    
    return np.array(result)

def _split_field(idxi, idxj, Segments):
    """ Split domain field into a number of equally sapced segments.
    """

    sizei = (idxi[1] - idxi[0]) 
    sizej = (idxj[1] - idxj[0]) 
    
    winsizei = int(sizei/Segments)
    winsizej = int(sizej/Segments)
    
    Idxi = np.zeros((Segments**2,2))
    Idxj = np.zeros((Segments**2,2))
    
    count=-1
    for i in range(Segments):
        for j in range(Segments):
            count+=1
            Idxi[count,0] = idxi[0] + i*winsizei
            Idxi[count,1] = np.min( (Idxi[count, 0] + winsizei, idxi[1]) )
            Idxj[count,0] = idxj[0] + j*winsizej
            Idxj[count,1] = np.min( (Idxj[count, 0] + winsizej, idxj[1]) )

    Idxi = np.array(Idxi).astype(int)
    Idxj = np.array(Idxj).astype(int)  
    
    return Idxi, Idxj
    
def _get_mask(Size, idxi, idxj, win_type):
    """Compute a mask of zeros with a window at a given position. 
    """

    idxi = np.array(idxi).astype(int) 
    idxj =  np.array(idxj).astype(int)
    
    win_size = (idxi[1] - idxi[0] , idxj[1] - idxj[0])
    wind = build_2D_tapering_function(win_size, win_type)
    
    mask = np.zeros(Size) 
    mask[idxi.item(0):idxi.item(1), idxj.item(0):idxj.item(1)] = wind
    
    return mask
    
