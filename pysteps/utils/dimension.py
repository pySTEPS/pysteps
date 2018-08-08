''' Functions to manipulate array dimensions.'''

import numpy as np

# TODO: If time_window_min can be set to None, it should be a keyword argument.
def aggregate_fields_time(R, metadata, time_window_min, method="mean"):
    """Aggregate fields in time.
    
    Parameters
    ----------
    R : array-like
        Array of shape (t,m,n) or (l,t,m,n) containing a time series of (ensemble)
        input fields.
        They must be evenly spaced in time.
    metadata : dict
        The metadata dictionary contains all data-related information. It requires 
        the key "timestamps".
    time_window_min : float or None
        The length in minutes of the time window that is used to aggregate the fields.
        The time spanned by the t dimension of R must be a multiple of time_window_min.
        If set to None, it returns a copy of the original R and metadata.
    method : string
        Optional argument that specifies the operation to use to aggregate the values within the time
        window. Default to mean operator.
    
    Returns
    -------
    outputarray : array-like
        The new array of aggregated fields of shape (k,m,n) or (l,k,m,n), where 
        k = t*delta/time_window_min and delta is the time interval between two
        successive timestamps.
    metadata : dict 
        The metadata with updated attributes.
        
    """
    
    R = R.copy()
    metadata = metadata.copy()
    
    if time_window_min is None:
        return R, metadata
    
    timestamps = metadata["timestamps"]
    if "leadtimes" in metadata:
        leadtimes = metadata["leadtimes"]
        
    if len(R.shape) < 3:
        raise ValueError("The number of dimension must be > 2")
    if len(R.shape) == 3:
        axis = 0
    if len(R.shape) == 4:
        axis = 1
    if len(R.shape) > 4:
        raise ValueError("The number of dimension must be <= 4")
    
    if R.shape[axis] != len(timestamps):
        raise ValueError("The list of timestamps has length %i, but R contains %i frames" 
                         % (len(timestamps), R.shape[axis]))
                        
    # assumes that frames are evenly spaced 
    delta = (timestamps[1] - timestamps[0]).seconds/60
    if delta == time_window_min:
        return R, metadata
    if (R.shape[axis]*delta) % time_window_min:
        raise ValueError('time_window_size does not equally split R')
    
    nframes = int(time_window_min/delta)
    
    R = aggregate_fields(R, nframes, axis=axis, method=method)
    
    metadata["timestamps"] = timestamps[nframes-1::nframes]
    if "leadtimes" in metadata:
        metadata["leadtimes"] = leadtimes[nframes-1::nframes]
        
    return R, metadata

def aggregate_fields(R, window_size, axis=0, method="mean"):
    """Aggregate fields. 
    It attemps to aggregate the given R axis in an integer number of sections of
    length = window_size.  If such a aggregation is not possible, an error is raised.

    Parameters
    ----------
    R : array-like
        Array of any shape containing the input fields.
    window_size : int
        The length of the window that is used to aggregate the fields.
    axis : int
        The axis where to perform the aggregation.
    method : string
        Optional argument that specifies the operation to use to aggregate the values within the
        window. Default to mean operator.
        
    Returns
    -------
    outputarray : array-like
        The new aggregated array with shape[axis] = k, where k = R.shape[axis]/window_size
    
    """
    
    N = R.shape[axis]
    if N % window_size:
        raise ValueError('window_size %i does not equally split R.shape[axis] %i' % (window_size, N))
        
    R = R.copy().swapaxes(axis, 0)    
    shape = list(R.shape)
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
    
    shape[0] = int(N/window_size)
    R = R__.reshape(shape).swapaxes(axis, 0)
        
    return R

def square_domain(R, metadata, method="pad", inverse=False):
    """Either pad or crop the data to get a square domain.
    
    Parameters
    ----------
    R : array-like
        Array of shape (m,n) or (t,m,n) containing the input fields.
    metadata : dict
        The metadata dictionary contains all data-related information.
    method : string
        Either pad or crop. 
        If pad, an equal number of zeros is added to both ends of its shortest
        side in order to produce a square domain.
        If crop, an equal number of pixels is removed to both ends of its longest
        side in order to produce a square domain. 
        Note that the crop method is irreversible, while the pad method can be
        reversed with the unsquare_domain() method.
    shape : 2-element tuple
        Necessary for the inverse method only, it is the original shape of the domain.
    inverse : bool
        Perform the inverse method, possible only with the "pad" method
        
    Returns
    -------
    R : array-like
        the reshape dataset
    metadata : dict 
        the metadata with updated attributes.
    
    """
    
    R = R.copy()
    metadata = metadata.copy()
    
    if not inverse:
    
        if len(R.shape) < 2:
            raise ValueError("The number of dimension must be > 1")
        if len(R.shape) == 2:
            R = R[None, None, :]
        if len(R.shape) == 3:
            R = R[None, :]
        if len(R.shape) > 4:
            raise ValueError("The number of dimension must be <= 4")
            
        if R.shape[2] == R.shape[3]:
            return R.squeeze()
            
        orig_dim = (R.shape)
        orig_dim_n = orig_dim[0]
        orig_dim_t = orig_dim[1]
        orig_dim_y = orig_dim[2]
        orig_dim_x = orig_dim[3]
        
        if method == "pad":
        
            new_dim = np.max(orig_dim[2:]) 
            R_ = np.ones((orig_dim_n, orig_dim_t, new_dim, new_dim))*R.min()
            if(orig_dim_x < new_dim):
                idx_buffer = int((new_dim - orig_dim_x)/2.)
                R_[:, :, :, idx_buffer:(idx_buffer + orig_dim_x)] = R
            elif(orig_dim_y < new_dim):
                idx_buffer = int((new_dim - orig_dim_y)/2.)
                R_[:, :, idx_buffer:(idx_buffer + orig_dim_y), :] = R
            
        elif method == "crop":
        
            new_dim = np.min(orig_dim[2:]) 
            R_ = np.zeros((orig_dim_n, orig_dim_t, new_dim, new_dim))
            if(orig_dim_x > new_dim):
                idx_buffer = int((orig_dim_x - new_dim)/2.)
                R_ = R[:, :, :, idx_buffer:(idx_buffer + new_dim)]
            elif(orig_dim_y > new_dim):
                idx_buffer = int((orig_dim_y - new_dim)/2.)
                R_ = R[:, :, idx_buffer:(idx_buffer + new_dim), :]
                
        else:
            raise ValueError("Unknown type")
                
        metadata["orig_domain"] = (orig_dim_y, orig_dim_x)    
        metadata["square_method"] = method 
                
        return R_.squeeze(),metadata
        
    elif inverse:
    
        if len(R.shape) < 2:
            raise ValueError("The number of dimension must be > 2")
        if len(R.shape) == 2:
            R = R[None, None, :]
        if len(R.shape) == 3:
            R = R[None, :]
        if len(R.shape) > 4:
            raise ValueError("The number of dimension must be <= 4")
                   
        method = metadata.pop("square_method")
        if method is not "pad":
            raise ValueError("Inverse method only applicable to padded fields")
        shape = metadata.pop("orig_domain")
        
        if R.shape[2] == shape[0] and R.shape[3] == shape[1]:
            return R.squeeze()
            
        if R.shape[2] == shape[0]:
            idx_buffer = int((R.shape[3] - shape[1])/2.)
            R = R[:, :, :, idx_buffer:(idx_buffer + shape[1])]
        elif R.shape[3] == shape[1]:    
            idx_buffer = int((R.shape[2] - shape[0])/2.)
            R = R[:, :, idx_buffer:(idx_buffer + shape[0]), :]
            
        return R.squeeze(),metadata
