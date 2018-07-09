''' Functions to manipualte array dimensions.'''

import numpy as np

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
