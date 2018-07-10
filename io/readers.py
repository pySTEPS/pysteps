"""Miscellaneous utility methods."""

import numpy as np

def read_timeseries(inputfns, importer, **kwargs):
    """Read a list of input files using iotools and stack them into a 3d array.
  
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
        A three-element tuple containing the precipitation fields read and 
        the associated georeferencing data and metadata.
    """
    R = []
    for ifn in inputfns[0]:
        R_, geodata, metadata = importer(ifn, **kwargs)
        R.append(R_)

    return np.concatenate([R_[None, :, :] for R_ in R]), geodata, metadata
