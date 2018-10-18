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
        A three-element tuple containing the precipitation fields read, the quality fields,
        and associated metadata.

    """

    # check for missing data
    if all(ifn is None for ifn in inputfns):
        return None, None, None
    else:
        for ifn in inputfns[0]:
            if ifn is not None:
                Rref, Qref, metadata = importer(ifn, **kwargs)
                break

    R = []
    Q = []
    timestamps = []
    for i,ifn in enumerate(inputfns[0]):
        if ifn is not None:
            R_, Q_, _ = importer(ifn, **kwargs)
            R.append(R_)
            Q.append(Q_)
            timestamps.append(inputfns[1][i])
        else:
            R.append(Rref*np.nan)
            if Qref is not None:
                Q.append(Qref*np.nan)
            else:
                Q.append(None)
            timestamps.append(inputfns[1][i])

    R = np.concatenate([R_[None, :, :] for R_ in R])
    #TODO: Q should be organized as R, but this is not trivial as Q_ can be also None or a scalar
    metadata["timestamps"] = np.array(timestamps)

    return R, Q, metadata
