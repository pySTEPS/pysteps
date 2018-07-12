''' Functions to manipualte array dimensions.'''

import numpy as np

def aggregate_fields_time(R, timestamps, time_window_min, method="sum"):
    """Aggregate fields in time.

    Parameters
    ----------
    R : array-like
        Array of shape (t,m,n) containing the input fields.
        They must be evenly spaced in time.
    timestamps : list
        List of datetime.datetime objects related to time dimension of R.
    time_window_min : float
        The length in minutes of the time window that is used to aggregate the fields.
        The total length of R must be a multiple of time_window_min.
    method : string
        Optional argument that specifies the operation to use to aggregate the values within the time
        window.
        
    Returns
    -------
    outputarray : array-like
        The new array of aggregated precipitation fields of shape (k,m,n), where 
        k = int(t*delta/time_window_min)
    """
    
    if R.shape[0] != len(timestamps):
        raise ValueError("The list of timestamps has length %i, but R contains %i frames" 
                         % (len(timestamps), R.shape[0]))
                         
    R = R.copy()
    timestamps = timestamps.copy()

    # assumes that frames are evenly spaced 
    delta = (timestamps[1] - timestamps[0]).seconds/60
    if (R.shape[0]*delta) % time_window_min:
        raise ValueError('time_window_size does not equally split R')
    
    nframes = int(time_window_min/delta)
    
    R = aggregate_fields(R, nframes, axis=0, method=method)
    
    timestamps = timestamps[nframes-1::nframes]
    
    return R, timestamps

def aggregate_fields(R, window_size, axis=0, method="sum"):
    """Aggregate fields. 
    It attemps to aggregate the given R axis in an integer number of sections of
    length = window_size.  If such a aggregation is not possible, an error is raised.

    Parameters
    ----------
    R : array-like
        Array of shape (t,m,n) containing the input fields.
    window_size : int
        The length of the window that is used to aggregate the fields.
    axis : int
        
    method : string
        Optional argument that specifies the operation to use to aggregate the values within the time
        window.
        
    Returns
    -------
    outputarray : array-like
        The new aggregated array of shape (k,m,n), where k = t/time_window
    """
    
    N = R.shape[axis]
    if N % window_size:
        raise ValueError('window_size does not equally split R')
    
    R = R.copy().swapaxes(axis, 0)    
    R_ = R.reshape((N, -1))
    
    if   method.lower() == "sum":
        R__ = R_.reshape(int(N/window_size), window_size, R_.shape[1]).sum(axis=1)
    elif method.lower() == "mean":
        R__ = R_.reshape(int(N/window_size), window_size, R_.shape[1]).mean(axis=1)
    elif method.lower() == "nansum":
        R__ = np.nansum(R_.reshape(int(N/window_size), window_size, R_.shape[1]), axis=1)
    elif method.lower() == "nanmean":
        R__ = np.nanmean(R_.reshape(int(N/window_size), window_size, R_.shape[1]), axis=1)
    else:
        raise ValueError("unknown method %s" % method)
    
    R = R__.reshape((int(N/window_size), R.shape[1], R.shape[2])).swapaxes(axis, 0)
        
    return R

def square_domain(Xin, method='crop'):
    '''Either pad or crop the data to get a square domain.
    '''
    
    X = Xin.copy()
    if len(Xin.shape) < 2:
        raise ValueError("The number of dimension must be > 2")
    if len(Xin.shape) == 2:
        X = X[None,None,:,:]
    if len(Xin.shape) == 3:
        X = X[None,:,:]
    if len(Xin.shape) > 4:
        raise ValueError("The number of dimension must be <= 4")
        
    if X.shape[2] == X.shape[3]:
        return X.squeeze()
        
    orig_dim = (X.shape)
    orig_dim_n = orig_dim[0]
    orig_dim_t = orig_dim[1]
    orig_dim_y = orig_dim[2]
    orig_dim_x = orig_dim[3]
    
    if method=='pad':
    
        new_dim = np.max(orig_dim[2:]) 
        X_ = np.ones((orig_dim_n, orig_dim_t, new_dim, new_dim))*X.min()
        if(orig_dim_x < new_dim):
            idx_buffer = int((new_dim - orig_dim_x)/2.)
            X_[:, :, :, idx_buffer:(idx_buffer + orig_dim_x)] = X
        elif(orig_dim_y < new_dim):
            idx_buffer = int((new_dim - orig_dim_y)/2.)
            X_[:, :, idx_buffer:(idx_buffer + orig_dim_y), :] = X
        
    elif method=='crop':
    
        new_dim = np.min(orig_dim[2:]) 
        X_ = np.zeros((orig_dim_n, orig_dim_t, new_dim, new_dim))
        if(orig_dim_x > new_dim):
            idx_buffer = int((orig_dim_x - new_dim)/2.)
            X_ = X[:, :, :, idx_buffer:(idx_buffer + new_dim)]
        elif(orig_dim_y > new_dim):
            idx_buffer = int((orig_dim_y - new_dim)/2.)
            X_ = X[:, :, idx_buffer:(idx_buffer + new_dim), :]
            
    else:
        raise ValueError("Unknown type")
            
    return X_.squeeze()
