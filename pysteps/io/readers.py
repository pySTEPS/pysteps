"""Methods for reading files.
"""

import numpy as np

def read_timeseries(inputfns, importer, **kwargs):
    """Read a list of input files using io tools and stack them into a 3d array.
  
    Parameters
    ----------
    inputfns : list
        List of input files returned by any function implemented in archive.
    importer : function
        Any function implemented in importers.
    kwargs : dict
        Optional keyword arguments for the importer.
    
    Returns
    -------
    out : tuple
        A three-element tuple containing the precipitation fields read, the quality field,
        and associated metadata.
    """
    R = []
    Q = []
    for ifn in inputfns[0]:
        R_, Q_, metadata = importer(ifn, **kwargs)
        R.append(R_)
        Q.append(Q_)
        
    R = np.concatenate([R_[None, :, :] for R_ in R])      
    #TODO: Q should be organized as R, but this is not trivial as Q_ can be also None or a scalar

    return R, Q, metadata
