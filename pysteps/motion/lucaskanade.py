"""
pysteps.motion.lucaskanade
==========================

OpenCV implementation of the Lucas-Kanade method with interpolated motion
vectors for areas with no precipitation.

.. autosummary::
    :toctree: ../generated/

    dense_lucaskanade

"""

import numpy as np
from numpy.ma.core import MaskedArray
from pysteps.exceptions import MissingOptionalDependency

try:
    import cv2
    cv2_imported = True
except ImportError:
    cv2_imported = False
import scipy.spatial
import time

def dense_lucaskanade(R, **kwargs):
    """
    .. _opencv: https://opencv.org/

    .. _`Lucas-Kanade`: https://docs.opencv.org/3.4/dc/d6b/\
    group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323

    OpenCV_ implementation of the local `Lucas-Kanade`_ method with
    interpolation of the sparse motion vectors to fill the whole grid.

    .. _MaskedArray: https://docs.scipy.org/doc/numpy/reference/\
        maskedarray.baseclass.html#numpy.ma.MaskedArray

    .. _ndarray:\
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
    
    .. _Shi-Tomasi: https://docs.opencv.org/3.4.1/dd/d1a/group__\
        imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
        
    Parameters
    ----------
    R : ndarray_ or MaskedArray_
        Array of shape (T,m,n) containing a sequence of T two-dimensional input
        images of shape (m,n).

        In case of an ndarray_, invalid values (Nans or infs) are masked.
        The mask in the MaskedArray_ defines a region where velocity vectors are
        not computed.

    Other Parameters
    ----------------
    buffer_mask : int, optional
        A mask buffer width in pixels. This extends the input mask (if any)
        to help avoiding the erroneous interpretation of velocities near the
        maximum range of the radars (0 by default).

    max_corners_ST : int, optional
        The maxCorners parameter in the `Shi-Tomasi`_ corner detection method.
        It represents the maximum number of points to be tracked (corners),
        by default this is 500. If set to zero, all detected corners are used.

    quality_level_ST : float, optional
        The qualityLevel parameter in the `Shi-Tomasi`_ corner detection method.
        It represents the minimal accepted quality for the points to be tracked
        (corners), by default this is set to 0.1. Higher quality thresholds can
        lead to no detection at all.

    min_distance_ST : int, optional
        The minDistance parameter in the `Shi-Tomasi`_ corner detection method.
        It represents minimum possible Euclidean distance in pixels
        between corners, by default this is set to 3 pixels.

    block_size_ST : int, optional
        The blockSize parameter in the `Shi-Tomasi`_ corner detection method.
        It represents the window size in pixels used for computing a derivative
        covariation matrix over each pixel neighborhood, by default this is set
        to 15 pixels.

    winsize_LK : tuple of int, optional
        The winSize parameter in the `Lucas-Kanade`_ optical flow method.
        It represents the size of the search window that it is used at each
        pyramid level, by default this is set to (50, 50) pixels.

    nr_levels_LK : int, optional
        The maxLevel parameter in the `Lucas-Kanade`_ optical flow method.
        It represents the 0-based maximal pyramid level number, by default this
        is set to 3.

    nr_IQR_outlier : int, optional
        Maximum acceptable deviation from the median velocity value in terms of
        number of inter quantile ranges (IQR). Any velocity that is larger than
        this value is flagged as outlier and excldued from the interpolation.
        By default this is set to 3.

    size_opening : int, optional
        The size of the structuring element kernel in pixels. This is used to
        perform a binary morphological opening on the input fields in order to
        filter isolated echoes due to clutter. By default this is set to 3.
        If set to zero, the fitlering is not perfomed.

    decl_grid : int, optional
        The cell size in pixels of the declustering grid that is used to filter
        out outliers in a sparse motion field and get more representative data
        points before the interpolation. This simply computes new sparse vectors
        over a coarser grid by taking the median of all vectors within one cell.
        By default this is set to 20 pixels. If set to less than 2 pixels, the
        declustering is not perfomed.

    min_nr_samples : int, optional
        The minimum number of samples necessary for computing the median vector
        within given declustering cell, otherwise all sparse vectors in that
        cell are discarded. By default this is set to 2.

    rbfunction : string, optional
        The name of the radial basis function used for the interpolation of the
        sparse vectors. This is based on the Euclidian norm d. By default this
        is set to "inverse" and the available names are "nearest", "inverse",
        "gaussian".

    k : int, optional
        The number of nearest neighbors used for fast interpolation, by default
        this is set to 20. If set equal to zero, it employs all the neighbors.

    epsilon : float, optional
        The adjustable constant used in the gaussian and inverse radial basis
        functions. by default this is computed as the median distance between
        the sparse vectors.

    nchunks : int, optional
        Split the grid points in n chunks to limit the memory usage during the
        interpolation. By default this is set to 5, if set to 1 the interpolation
        is computed with the whole grid.

    extra_vectors : ndarray_, optional
        Additional sparse motion vectors as 2d array (columns: x,y,u,v; rows:
        nbr. of vectors) to be integrated with the sparse vectors from the
        Lucas-Kanade local tracking.
        x and y must be in pixel coordinates, with (0,0) being the upper-left
        corner of the field R. u and v must be in pixel units. By default this
        is set to None.

    verbose : bool, optional
        If set to True, it prints information about the program (True by default).

    Returns
    -------
    out : ndarray_
        Three-dimensional array (2,m,n) containing the dense x- and y-components
        of the motion field.
        Return an empty array when no motion vectors are found.
        
    References
    ----------
    
    Bouguet,  J.-Y.:  Pyramidal  implementation  of  the  affine  Lucas Kanade 
    feature tracker description of the algorithm, Intel Corp., 5, 4, 
    https://doi.org/10.1109/HPDC.2004.1323531, 2001

    Lucas, B. D. and Kanade, T.: An iterative image registration technique with 
    an application to stereo vision, in: Proceedings of the 1981 DARPA Imaging 
    Understanding Workshop, pp. 121–130, 1981.
    
    """

    if len(R.shape) != 3:
        raise ValueError("R has %i dimensions, but a three-dimensional array is expected" % len(R.shape))

    if R.shape[0] < 2:
        raise ValueError("R has %i frame, but at least two frames are expected" % R.shape[0])

    # defaults
    max_corners_ST = kwargs.get("max_corners_ST", 500)
    quality_level_ST = kwargs.get("quality_level_ST", 0.1)
    min_distance_ST = kwargs.get("min_distance_ST", 3)
    block_size_ST = kwargs.get("block_size_ST", 15)
    winsize_LK = kwargs.get("winsize_LK5", (50, 50))
    nr_levels_LK = kwargs.get("nr_levels_LK", 3)
    nr_IQR_outlier = kwargs.get("nr_IQR_outlier", 3)
    size_opening = kwargs.get("size_opening", 3)
    decl_grid = kwargs.get("decl_grid", 20)
    min_nr_samples = kwargs.get("min_nr_samples", 2)
    rbfunction = kwargs.get("rbfunction", "inverse")
    k = kwargs.get("k", 20)
    epsilon = kwargs.get("epsilon", None)
    nchunks = kwargs.get("nchunks", 5)
    extra_vectors = kwargs.get("extra_vectors", None)
    if extra_vectors is not None:
        if len(extra_vectors.shape) != 2:
            raise ValueError("extra_vectors has %i dimensions, but 2 dimensions are expected"
                            % len(extra_vectors.shape))
        if extra_vectors.shape[1] != 4:
            raise ValueError("extra_vectors has %i columns, but 4 columns are expected"
                               % extra_vectors.shape[1])
    verbose = kwargs.get("verbose", True)
    buffer_mask = kwargs.get("buffer_mask", 0)

    if verbose:
        print("Computing the motion field with the Lucas-Kanade method.")
        t0 = time.time()

    # Get mask
    if isinstance(R, MaskedArray):
        mask = np.ma.getmaskarray(R).copy()
    else:
        R = np.ma.masked_invalid(R)
        mask = np.ma.getmaskarray(R).copy()
    R[mask] = np.nanmin(R)  # Remove any Nan from the raw data

    nr_fields = R.shape[0]
    domain_size = (R.shape[1], R.shape[2])
    y0Stack = []
    x0Stack = []
    uStack = []
    vStack = []
    for n in range(nr_fields - 1):

        # extract consecutive images
        prvs = R[n, :, :].copy()
        next = R[n + 1, :, :].copy()

        # skip loop if no precip
        if ~np.any(prvs > prvs.min()) or ~np.any(next > next.min()):
            continue

        # scale between 0 and 255
        prvs = (prvs - prvs.min())/(prvs.max() - prvs.min())*255
        next = (next - next.min())/(next.max() - next.min())*255

        # convert to 8-bit
        prvs = np.ndarray.astype(prvs, "uint8")
        next = np.ndarray.astype(next, "uint8")
        mask_ = np.ndarray.astype(mask[n, :, :], "uint8")

        # buffer the quality mask to ensure that no vectors are computed nearby
        # the edges of the radar mask
        if buffer_mask > 0:
            mask_ = cv2.morphologyEx(mask_, cv2.MORPH_DILATE,
                                     np.ones((int(buffer_mask), int(buffer_mask)),
                                     np.uint8))

        # remove small noise with a morphological operator (opening)
        if size_opening > 0:
            prvs = _clean_image(prvs, n=size_opening)
            next = _clean_image(next, n=size_opening)

        # Shi-Tomasi good features to track
        # TODO: implement different feature detection algorithms (e.g. Harris)
        mask_ = (-1*mask_ + 1).astype('uint8')
        p0 = _ShiTomasi_features_to_track(prvs, max_corners_ST, quality_level_ST,
                                          min_distance_ST, block_size_ST, mask_)

        # skip loop if no features to track
        if p0 is None:
            continue

        # get sparse u, v vectors with Lucas-Kanade tracking
        x0, y0, u, v = _LucasKanade_features_tracking(prvs, next, p0, winsize_LK,
                                                     nr_levels_LK)
        # skip loop if no vectors
        if x0 is None:
            continue

        # exclude outlier vectors
        vel = np.sqrt(u**2 + v**2) # [px/timesteps]
        q1, q2 = np.percentile(vel, [25, 75])
        min_speed_thr = np.max((0, q1 - nr_IQR_outlier*(q2 - q1)))
        max_speed_thr = q2 + nr_IQR_outlier*(q2 - q1)
        keep = np.logical_and(vel < max_speed_thr, vel > min_speed_thr)

        u = u[keep][:, None]
        v = v[keep][:, None]
        y0 = y0[keep][:, None]
        x0 = x0[keep][:, None]

        # stack vectors within time window
        x0Stack.append(x0)
        y0Stack.append(y0)
        uStack.append(u)
        vStack.append(v)

    # return zero motion field is no sparse vectors are found
    if len(x0Stack) == 0:
        return np.zeros((2, domain_size[0], domain_size[1]))

    # convert lists of arrays into single arrays
    x = np.vstack(x0Stack)
    y = np.vstack(y0Stack)
    u = np.vstack(uStack)
    v = np.vstack(vStack)

    if verbose:
        print("--- LK found %i sparse vectors ---" % x0.size)

    # decluster sparse motion vectors
    if decl_grid > 1:
        x, y, u, v = _declustering(x, y, u, v, decl_grid, min_nr_samples)

    # append extra vectors if provided
    if extra_vectors is not None:
        x = np.concatenate((x, extra_vectors[:, 0]))
        y = np.concatenate((y, extra_vectors[:, 1]))
        u = np.concatenate((u, extra_vectors[:, 2]))
        v = np.concatenate((v, extra_vectors[:, 3]))

    # return zero motion field is no sparse vectors are left for interpolation
    if x.size == 0:
        return np.zeros((2, domain_size[0], domain_size[1]))

    if verbose:
        print("--- %i sparse vectors left for interpolation ---" % x.size)

    # kernel interpolation
    _, _, UV = _interpolate_sparse_vectors(x, y, u, v, domain_size, rbfunction=rbfunction,
                                           k=k, epsilon=epsilon, nchunks=nchunks)


    if verbose:
        print("--- %.2f seconds ---" % (time.time() - t0))

    return UV


def _ShiTomasi_features_to_track(R, max_corners_ST, quality_level_ST,
                                 min_distance_ST, block_size_ST, mask):
    """Call the Shi-Tomasi corner detection algorithm.

    Parameters
    ----------
    R : array-like
        Array of shape (m,n) containing the input precipitation field passed as
        8-bit image.

    max_corners_ST : int
        Maximum number of corners to return. If there are more corners than are
        found, the strongest of them is returned.

    quality_level_ST : float
        Parameter characterizing the minimal accepted quality of image corners.
        See original documentation for more details (https://docs.opencv.org).

    min_distance_ST : int
        Minimum possible Euclidean distance between the returned corners [px].

    block_size_ST : int
        Size of an average block for computing a derivative covariation matrix
        over each pixel neighborhood.

    mask : ndarray_
        Array of shape (m,n). It specifies the region in which the corners are
        detected.

    Returns
    -------
    p0 : list
        Output vector of detected corners.

    """
    if not cv2_imported:
        raise MissingOptionalDependency(
            "opencv package is required for the Lucas-Kanade "
            "optical flow method but it is not installed")

    if len(R.shape) != 2:
        raise ValueError("R must be a two-dimensional array")
    if R.dtype != "uint8":
        raise ValueError("R must be passed as 8-bit image")

    # ShiTomasi corner detection parameters
    ShiTomasi_params = dict(maxCorners=max_corners_ST, qualityLevel=quality_level_ST,
                            minDistance=min_distance_ST, blockSize=block_size_ST)

    # detect corners
    p0 = cv2.goodFeaturesToTrack(R, mask=mask, **ShiTomasi_params)

    return p0


def _LucasKanade_features_tracking(prvs, next, p0, winsize_LK, nr_levels_LK):
    """Call the Lucas-Kanade features tracking algorithm.

    Parameters
    ----------
    prvs : array-like
        Array of shape (m,n) containing the first 8-bit input image.
    next : array-like
        Array of shape (m,n) containing the successive 8-bit input image.
    p0 : list
        Vector of 2D points for which the flow needs to be found.
        Point coordinates must be single-precision floating-point numbers.
    winsize_LK : tuple
        Size of the search window at each pyramid level.
        Small windows (e.g. 10) lead to unrealistic motion.
    nr_levels_LK : int
        0-based maximal pyramid level number.
        Not very sensitive parameter.

    Returns
    -------
    x0 : array-like
        Output vector of x-coordinates of detected point motions.
    y0 : array-like
        Output vector of y-coordinates of detected point motions.
    u : array-like
        Output vector of u-components of detected point motions.
    v : array-like
        Output vector of v-components of detected point motions.

    """
    if not cv2_imported:
        raise MissingOptionalDependency(
            "opencv package is required for the Lucas-Kanade method "
            "optical flow method but it is not installed")

    # LK parameters
    lk_params = dict(winSize=winsize_LK, maxLevel=nr_levels_LK,
                     criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 10, 0))

    # Lucas-Kande
    p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, next, p0, None, **lk_params)

    # keep only features that have been found
    st = st[:, 0] == 1
    if np.any(st):
        p1 = p1[st, :, :]
        p0 = p0[st, :, :]
        err = err[st, :]

        # extract vectors
        x0 = p0[:, :, 0]
        y0 = p0[:, :, 1]
        u = np.array((p1 - p0)[:, :, 0])
        v = np.array((p1 - p0)[:, :, 1])
    else:
        x0 = y0 = u = v = None

    return x0, y0, u, v


def _clean_image(R, n=3, thr=0):
    """Apply a binary morphological opening to filter small isolated echoes.

    Parameters
    ----------
    R : array-like
        Array of shape (m,n) containing the input precipitation field.
    n : int
        The structuring element size [px].
    thr : float
        The rain/no-rain threshold to convert the image into a binary image.

    Returns
    -------
    R : array
        Array of shape (m,n) containing the cleaned precipitation field.

    """
    if not cv2_imported:
        raise MissingOptionalDependency(
            "opencv package is required for the Lucas-Kanade method "
            "optical flow method but it is not installed")

    # convert to binary image (rain/no rain)
    field_bin = np.ndarray.astype(R > thr,"uint8")

    # build a structuring element of size (nx)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))

    # apply morphological opening (i.e. erosion then dilation)
    field_bin_out = cv2.morphologyEx(field_bin, cv2.MORPH_OPEN, kernel)

    # build mask to be applied on the original image
    mask = (field_bin - field_bin_out) > 0

    # filter out small isolated echoes based on mask
    R[mask] = np.nanmin(R)

    return R


def _declustering(x, y, u, v, decl_grid, min_nr_samples):
    """Filter out outliers in a sparse motion field and get more representative
    data points. The method assigns data points to a (RxR) declustering grid
    and then take the median of all values within one cell.

    Parameters
    ----------
    x : array_like
        X-coordinates of the origins of the velocity vectors.
    y : array_like
        Y-coordinates of the origins of the velocity vectors.
    u : array_like
        X-components of the velocities.
    v : array_like
        Y-components of the velocities.
    decl_grid : int
        Size of the declustering grid [px].
    min_nr_samples : int
        The minimum number of samples for computing the median within given
        declustering cell.

    Returns
    -------
    A four-element tuple (x,y,u,v) containing the x- and y-coordinates and
    velocity components of the declustered motion vectors.

    """

    # make sure these are all numpy vertical arrays
    x = np.atleast_1d(np.array(x).squeeze())[:, None]
    y = np.atleast_1d(np.array(y).squeeze())[:, None]
    u = np.atleast_1d(np.array(u).squeeze())[:, None]
    v = np.atleast_1d(np.array(v).squeeze())[:, None]

    # return empty arrays if the number of sparse vectors is < min_nr_samples
    if x.size < min_nr_samples:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # discretize coordinates into declustering grid
    xT = x/float(decl_grid)
    yT = y/float(decl_grid)

    # round coordinates to low integer
    xT = np.floor(xT)
    yT = np.floor(yT)

    # keep only unique combinations of coordinates
    xy = np.concatenate((xT,yT), axis=1)
    xyb = np.ascontiguousarray(xy).view(np.dtype((np.void, xy.dtype.itemsize*xy.shape[1])))
    _,idx = np.unique(xyb, return_index=True)
    uxy = xy[idx]

    # now loop through these unique values and average vectors which belong to
    # the same declustering grid cell
    xN = []; yN = []; uN = []; vN = []
    for i in range(uxy.shape[0]):
        idx = np.logical_and(xT == uxy[i, 0], yT == uxy[i, 1])
        npoints = np.sum(idx)
        if npoints >= min_nr_samples:
            xN.append(np.median(x[idx]))
            yN.append(np.median(y[idx]))
            uN.append(np.median(u[idx]))
            vN.append(np.median(v[idx]))

    # convert to numpy arrays
    x = np.array(xN)
    y = np.array(yN)
    u = np.array(uN)
    v = np.array(vN)

    return x, y, u, v


def _interpolate_sparse_vectors(x, y, u, v, domain_size, rbfunction="inverse",
                               k=20, epsilon=None, nchunks=5):

    """Interpolation of sparse motion vectors to produce a dense field of motion
    vectors.

    Parameters
    ----------
    x : array-like
        x coordinates of the sparse motion vectors
    y : array-like
        y coordinates of the sparse motion vectors
    u : array_like
        u components of the sparse motion vectors
    v : array_like
        v components of the sparse motion vectors
    domain_size : tuple
        size of the domain of the dense motion field [px]
    rbfunction : string
        the radial basis rbfunction, based on the Euclidian norm, d.
        default : inverse
        available : nearest, inverse, gaussian
    k : int or "all"
        the number of nearest neighbors used to speed-up the interpolation
        If set equal to "all", it employs all the sparse vectors
        default : 20
    epsilon : float
        adjustable constant for gaussian or inverse functions
        default : median distance between sparse vectors
    nchunks : int
        split the grid points in n chunks to limit the memory usage during the
        interpolation
        default : 5

    Returns
    -------
    X : array-like
        grid
    Y : array-like
        grid
    UV : array-like
        Three-dimensional array (2,domain_size[0],domain_size[1])
        containing the dense U, V motion fields.

    """

    # make sure these are vertical arrays
    x = np.atleast_1d(np.array(x).squeeze())[:, None]
    y = np.atleast_1d(np.array(y).squeeze())[:, None]
    u = np.atleast_1d(np.array(u).squeeze())[:, None]
    v = np.atleast_1d(np.array(v).squeeze())[:, None]
    points  = np.concatenate((x, y), axis=1)
    npoints = points.shape[0]

    if len(domain_size)==1:
        domain_size = (domain_size, domain_size)

    # generate the grid
    xgrid = np.arange(domain_size[1])
    ygrid = np.arange(domain_size[0])
    X, Y = np.meshgrid(xgrid, ygrid)
    grid = np.column_stack((X.ravel(), Y.ravel()))

    U = np.zeros(grid.shape[0])
    V = np.zeros(grid.shape[0])

    # create cKDTree object to represent source grid
    if k > 0:
        k = np.min((k, npoints))
        tree = scipy.spatial.cKDTree(points)

    # split grid points in n chunks
    if nchunks > 1:
        subgrids = np.array_split(grid, nchunks, 0)
        subgrids = [x for x in subgrids if x.size > 0]
    else:
        subgrids = [grid]

    # loop subgrids
    i0=0
    for i,subgrid in enumerate(subgrids):

        idelta = subgrid.shape[0]

        if rbfunction.lower() == "nearest":
            # find indices of the nearest neighbors
            _, inds = tree.query(subgrid, k=1)

            U[i0:(i0 + idelta)] = u.ravel()[inds]
            V[i0:(i0 + idelta)] = v.ravel()[inds]

        else:
            if k <= 0:
                d = scipy.spatial.distance.cdist(points, subgrid, "euclidean").transpose()
                inds = np.arange(u.size)[None, :]*np.ones((subgrid.shape[0],u.size)).astype(int)

            else:
                # find indices of the k-nearest neighbors
                d, inds = tree.query(subgrid, k=k)

            if inds.ndim == 1:
                inds = inds[:, None]
                d    = d[:, None]

            # the bandwidth
            if epsilon is None:
                epsilon = 1
                if npoints > 1:
                    dpoints = scipy.spatial.distance.pdist(points, "euclidean")
                    epsilon = np.median(dpoints)

            # the interpolation weights
            if rbfunction.lower() == "inverse":
                w = 1.0/np.sqrt((d/epsilon)**2 + 1)
            elif rbfunction.lower() == "gaussian":
                w = np.exp(-0.5*(d/epsilon)**2)
            else:
                raise ValueError("unknown radial fucntion %s" % rbfunction)

            if not np.all(np.sum(w, axis=1)):
                w[np.sum(w, axis=1) == 0, :] = 1.

            U[i0:(i0 + idelta)] = np.sum(w*u.ravel()[inds], axis=1)/np.sum(w, axis=1)
            V[i0:(i0 + idelta)] = np.sum(w*v.ravel()[inds], axis=1)/np.sum(w, axis=1)

        i0 += idelta

    # reshape back to original size
    U = U.reshape(domain_size[0], domain_size[1])
    V = V.reshape(domain_size[0], domain_size[1])
    UV = np.stack([U, V])

    return X, Y, UV
