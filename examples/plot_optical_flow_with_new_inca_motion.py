"""
Optical flow
============

This tutorial offers a short overview of the optical flow routines available in 
pysteps and it will cover how to compute and plot the motion field from a 
sequence of radar images.
"""

from datetime import datetime
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from datetime import timedelta

from pysteps import io, motion, rcparams, nowcasts
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field, quiver

import pdb

from numba import jit

################################################################################
# The INCA module functions to test agains the four (one removed) OF methods
# ---------------------------

@jit(nopython=True)
def compute_motion_numba(image1, image2, NI, NJ, nthin,
                                         IS_tmp, JS_tmp):
    nsh = 20
    nqu = 45
    nsq = nsh + nqu
    di = 1
    PAREAMIN = 1.0
    rr_dpa_min = 0.03
    nn = (2 * nqu / di + 1) ** 2.

    for i in range(0, NI, nthin):
        if (i >= nsq) and (i < NI - nsq):
            for j in range(0, NJ, nthin):
                if (j >= nsq) and (j < NJ - nsq):
                    ii1 = max(i - nqu, 0)
                    ii2 = min(i + nqu, NI - 1)
                    jj1 = max(j - nqu, 0)
                    jj2 = min(j + nqu, NJ - 1)

                    sy = 0.
                    sy2 = 0.

                    for ii in range(ii1, ii2+1, di):
                        for jj in range(jj1, jj2+1, di):
                            sy = sy + image1[jj, ii]
                            sy2 = sy2 + image1[jj, ii] ** 2.

                    sigy = sy2 - sy ** 2. / nn
                    isho = -99
                    jsho = -99

                    if (sigy > 0.) and (sy > PAREAMIN):
                        corqx = 0.1
                        for ish in range(-nsh, nsh + 1):
                            for jsh in range(-nsh, nsh + 1):
                                if ( math.sqrt((ish)**2. + (jsh)**2.) > nsh ): continue
                                sx = 0.
                                sx2 = 0.
                                sxy = 0.
                                for ii in range(ii1, ii2+1, di): 
                                    for jj in range(jj1, jj2+1, di): 
                                        ind_x = min(max(0, jj + jsh), NJ - 1)
                                        ind_y = min(max(0, ii + ish), NI - 1)
                                        sx += image2[ind_x, ind_y]
                                        sx2 += image2[ind_x, ind_y] ** 2.
                                        sxy += image2[ind_x, ind_y] * image1[jj, ii]

                                sigx = sx2 - sx ** 2. / nn
                                if sigx > 0.:
                                    corq = (sxy -sx * sy / nn) ** 2. / (sigx * sigy)
                                    if corq > corqx:
                                        corqx = corq
                                        isho = ish
                                        jsho = jsh

                        if (isho != -99) and (corqx > 0.3) and (corqx * sy > 1.) and (sigy / sy >= 0.08):
                            if (abs(isho) < nsh) and (abs(jsho) < nsh):
                                IS_tmp[int(j/nthin),int(i/nthin)] = -isho
                                JS_tmp[int(j/nthin),int(i/nthin)] = -jsho
                        else:
                            IS_tmp[int(j/nthin),int(i/nthin)] = 0
                            JS_tmp[int(j/nthin),int(i/nthin)] = 0

    return IS_tmp, JS_tmp

def within_interval(now_dd, err_dd, do_it):
    """
    Check whether a direction value lies within a certain interval or not.

    Parameters:
    - now_dd (float): The direction value to be checked.
    - err_dd (float): The reference direction value.
    - do_it (int): A flag to control printing.

    Returns:
    - int: 0 if now_dd is within the interval, 1 otherwise.
    """

    version = 0
    window = 45.0

    if (now_dd + window) > 360.0 or (now_dd - window) < 0.0:
        version = 1
    if (err_dd + window) > 360.0 or (err_dd - window) < 0.0:
        version = 1

    if do_it == 1:
        print(f"now_dd: {now_dd:7.2f}, err_dd: {err_dd:7.2f}")
    if version == 1:
        now_dd = np.fmod(now_dd + 360.0, 360.0)
        err_dd = np.fmod(err_dd + 360.0, 360.0)
        if do_it == 1:
            print(f"now_dd: {now_dd:7.2f}, err_dd: {err_dd:7.2f} (version 1)")

    return 0 if (now_dd > (err_dd - window) and now_dd < (err_dd + window)) else 1

def calculate_corrected_motion_vector(NI, NJ, nthin, IS_NOW, JS_NOW, Uerr_, Verr_, IS_ANA, JS_ANA, IS, JS, Uerr_av, Verr_av):
    """
    Calculate a corrected motion vector based on input data and error vectors.

    Parameters:
        - NI (int): Number of elements in the x-direction.
        - NJ (int): Number of elements in the y-direction.
        - nthin (int): Step size for iteration.
        - IS_NOW ((NJ/nthin, NI/nthin,) ndarray): Current error motion vector in the x-direction.
        - JS_NOW ((NJ/nthin, NI/nthin,) ndarray): Current error motion vector in the y-direction.
        - Uerr_ ((NJ/nthin, NI/nthin,) ndarray): Climatological error motion vector in the x-direction.
        - Verr_ ((NJ/nthin, NI/nthin,) ndarray): Climatological error motion vector in the y-direction.
        - IS_ANA ((NJ/nthin, NI/nthin,) ndarray): Analysis motion vector in the x-direction.
        - JS_ANA ((NJ/nthin, NI/nthin,) ndarray): Analysis motion vector in the y-direction.
        - IS ((NJ/nthin, NI/nthin,) ndarray): Output corrected motion vector in the x-direction.
        - JS ((NJ/nthin, NI/nthin,) ndarray): Output corrected motion vector in the y-direction.
        - Uerr_av ((NJ/nthin, NI/nthin,) ndarray): Averaged error motion vector in the x-direction.
        - Verr_av ((NJ/nthin, NI/nthin,) ndarray): Averaged error motion vector in the y-direction.

    Returns:
        - Tuple[ndarray, ndarray, ndarray, ndarray]:
            - The corrected motion vectors in the x-direction (IS).
            - The corrected motion vectors in the y-direction (JS).
            - The averaged error motion vectors in the x-direction (Uerr_av).
            - The averaged error motion vectors in the y-direction (Verr_av).
    """
    print("Now in the works: corrected motion vector.")
    # This is the original approach. Do we need this?
    # for i in range(0, NJ - 1, nthin):
        # for j in range(0, NI - 1, nthin):
    for i in range(0, int(NJ  / nthin) + 1):
        for j in range(0, int(NI / nthin) + 1):
            has_averaged = 0
            # why do we need this?
            # do_it = 1 if i == 480 and j == 90 else 0
            do_it = 0

            if IS_NOW[i, j] != -99 and Uerr_[i, j] > -98.:
                NOW_dd = (int)(270. - np.arctan2(JS_NOW[i, j], IS_NOW[i, j]) * 180. / np.pi) % 360
                NOW_ff = np.sqrt(np.square(IS_NOW[i, j]) + np.square(JS_NOW[i, j]))
                err_dd = (int)(270. - np.arctan2(Verr_[i, j], Uerr_[i, j]) * 180. / np.pi) % 360
                err_ff = np.sqrt(np.square(Uerr_[i, j]) + np.square(Verr_[i, j]))

                if within_interval(float(NOW_dd), float(err_dd), do_it) == 0 and NOW_ff > err_ff * 0.5 and NOW_ff < err_ff * 1.5:
                    Uerr_av[i, j] = float(IS_NOW[i, j] - IS_ANA[i, j])
                    Verr_av[i, j] = float(JS_NOW[i, j] - JS_ANA[i, j])
                else:
                    Uerr_av[i, j] = float(IS_NOW[i, j] - IS_ANA[i, j]) * 0.25 + Uerr_[i, j] * 0.75
                    Verr_av[i, j] = float(JS_NOW[i, j] - JS_ANA[i, j]) * 0.25 + Verr_[i, j] * 0.75
                    has_averaged = 1

            else:
                Uerr_av[i, j] = float(IS_NOW[i, j] - IS_ANA[i, j])
                Verr_av[i, j] = float(JS_NOW[i, j] - JS_ANA[i, j])

            if Uerr_av[i, j] > -98.:
                if IS_ANA[i, j] == -99:
                    IS[i, j] = -99
                    JS[i, j] = -99
                else:
                    IS[i, j] = math.fmod(IS_ANA[i, j],99) + math.fmod(int(Uerr_av[i, j]),99)
                    JS[i, j] = math.fmod(JS_ANA[i, j],99) + math.fmod(int(Verr_av[i, j]),99)

            else:
                IS[i, j] = IS_ANA[i, j]
                JS[i, j] = JS_ANA[i, j]

            if has_averaged == 0:
                if Uerr_av[i, j] < -98.:
                    w_emv_now = 0.
                else:
                    w_emv_now = 0.25

                if Uerr_[i, j] < -98.:
                    w_emv_avg = 0.
                else:
                    w_emv_avg = 0.75

                if w_emv_avg + w_emv_now >= 0.999:
                    Uerr_av[i, j] = int(Uerr_av[i, j] * w_emv_now + Uerr_[i, j] * (1. - w_emv_now))
                    Verr_av[i, j] = int(Verr_av[i, j] * w_emv_now + Verr_[i, j] * (1. - w_emv_now))

    return IS, JS, Uerr_av, Verr_av

@jit(nopython=True)
def interpolate_rmv(ni1, ni2, nj1, nj2, kmax, nsteph, xr, yr, sta, uu, vv, u, v):
    """
    Perform distance-weighted interpolation of values at grid points.

    Parameters:
    - ni1 (int): Lower bound of the first dimension of the grid.
    - ni2 (int): Upper bound of the first dimension of the grid.
    - nj1 (int): Lower bound of the second dimension of the grid.
    - nj2 (int): Upper bound of the second dimension of the grid.
    - wgts (ndarray): Weights vector for each interpolation point.
    - kmax (int): Maximum value of k for looping.
    - nsteph (float): A scaling factor for the interpolated values.
    - xr (ndarray): Array containing x-coordinates of the interpolation points.
    - yr (ndarray): Array containing y-coordinates of the interpolation points.
    - sta (ndarray): Array indicating whether the interpolation point is valid.
    - uu (ndarray): Array containing values to be interpolated (component u).
    - vv (ndarray): Array containing values to be interpolated (component v).
    - u (ndarray): Array to store the interpolated values (component u).
    - v (ndarray): Array to store the interpolated values (component v).

    Returns:
    - u (ndarray): Array containing the interpolated values (component u).
    - v (ndarray): Array containing the interpolated values (component v).
    """
    for j in range(nj1, nj2):
        for i in range(ni1, ni2):
            wgtsum = 0.0

            u[i-ni1, j-nj1] = 0.0
            v[i-ni1, j-nj1] = 0.0

            for k in range(0, kmax):
                if sta[k]:
                    rsq = np.sqrt((xr[k]-j)**2 + (yr[k]-i)**2)
                    if rsq < 1.0:
                        wgts = 1000000.0
                    else:
                        wgts = 1.0 / rsq**3
                    wgtsum += wgts

            for k in range(0, kmax):
                if sta[k]:
                    rsq = np.sqrt((xr[k]-j)**2 + (yr[k]-i)**2)
                    if rsq < 1.0:
                        wgts = 1000000.0
                    else:
                        wgts = 1.0 / rsq**3
                    wgts = wgts / wgtsum
                    u[i-ni1, j-nj1] += wgts * uu[k]
                    v[i-ni1, j-nj1] += wgts * vv[k]

            u[i-ni1, j-nj1] *= nsteph / 3.6
            v[i-ni1, j-nj1] *= nsteph / 3.6

    return u, v

def inca_motion_vectors_simplified(PREV, timestep,settings={}):
# def inca_motion_vectors_simplified(ANA, PREV, NWPUAWIND, Uerr_, Verr_,
#     initTime, inca_domain, timestep, settings,adv_files):
    """This function calculates motion vectors for precipitation nowcasting.
    following the same approach as the operation INCA C code. It computes
    the motion vector in lower resolution (nthin) for two radar fields and
    for the error of the previous nowcasting (+15 min). Then it performs a
    correction based on a climatology error to keep trend recent performance.
    Then it is cross-checked with the upper air wind field (500hPa and 700hPa) 
    for consistency before doing the interpolation to the original resolution.


    Args:
        ANA ((N,M,) ndarray): the current precip analysis
        PREV ((N,M,O,2,) ndarray): the previous analyses and nowcasts
        NWPUAWIND (dict): the upper air wind (U,V) field (500hPa, 600hP and 700hPa)
        Uerr_ ((N/nthin, M/nthin,) ndarray): Climatological error motion vector in the x-direction.
        Verr_ ((N/nthin, M/nthin,) ndarray): Climatological error motion vector in the y-direction.
        initTime (datetime): the initialization of the INCA+ run
        inca_domain (domain class): all there is to know about the domain
        timestep (int): the accumulation time for precipitation [seconds]
        settings (dict): the nowcasting settings

    Returns:
        (N,M,2,) ndarray: the motion vectors
        (N,M,2,) ndarray: the error motion vectors
    """

    print('Creating INCA motion vectors')

    start_nwc = time.monotonic()

    #Taking the needed parameters from the nowcasting settings dict
    nback = settings.get('nback', 3)
    PMIN = settings.get('PMIN', 0.05)
    nthin = settings.get('nthin', 15)
    delmax = settings.get('delmax', 5.)
    
    # Create factor and apply
    factor = timestep / 3600.
    nsteph = 1/factor

    NI = int((metadata['x2']-metadata['x1'])/metadata['xpixelsize'])
    NJ = int((metadata['y2']-metadata['y1'])/metadata['ypixelsize'])

    NX = int(np.ceil(NI / nthin))
    NY = int(np.ceil(NJ / nthin))

    Uana = np.full((NJ, NI),-99.)[::nthin, ::nthin]
    Vana = np.full((NJ, NI),-99.)[::nthin, ::nthin]

    Uana, Vana = compute_motion_numba(
            PREV[1,...], PREV[0,...],
            NI, NJ, nthin, Uana, Vana)

    boole_ = ((Uana > -99) & (Vana > -99))

    X_orig = metadata['xpixelsize']*np.arange(NI)+metadata['x1']
    Y_orig = metadata['ypixelsize']*np.arange(NJ)+metadata['y1']
    X, Y = X_orig[::nthin], Y_orig[::nthin]
    XX, YY =np.meshgrid(X,Y)

    UU = np.zeros([NJ, NI])
    VV = np.zeros([NJ, NI])
    UU, VV = interpolate_rmv(int(metadata['y1']/metadata['xpixelsize']), int(metadata['y2']/metadata['xpixelsize']), 
        int(metadata['x1']/metadata['xpixelsize']), int(metadata['x2']/metadata['xpixelsize']), 
        len(Uana.flatten()), nsteph, XX.flatten()/metadata['xpixelsize'], YY.flatten()/metadata['xpixelsize'], 
        boole_.flatten(), Uana.flatten(), Vana.flatten(), UU, VV)

    extrapolate = nowcasts.get_method("extrapolation")
    R[~np.isfinite(R)] = metadata["zerovalue"]
    NWC_field = extrapolate(PREV[1,...], np.append(UU[np.newaxis,...],VV[np.newaxis,...],axis=0), 12)

    #Initialize all the required matrices for following code
    Uana = np.full((NJ, NI),-99.)[::nthin, ::nthin]
    Vana = np.full((NJ, NI),-99.)[::nthin, ::nthin]
    Unow = np.full((NJ, NI),-99.)[::nthin, ::nthin]
    Vnow = np.full((NJ, NI),-99.)[::nthin, ::nthin]
    IS = np.full((NJ, NI),-99.)[::nthin,::nthin]
    JS = np.full((NJ, NI),-99.)[::nthin,::nthin]
    Uerr_av = np.full((NJ, NI),-99.)[::nthin,::nthin]
    Verr_av = np.full((NJ, NI),-99.)[::nthin,::nthin]
    Uerr_ = np.full((NJ, NI),0.)[::nthin,::nthin]
    Verr_ = np.full((NJ, NI),0.)[::nthin,::nthin]

    Uana, Vana = compute_motion_numba(
            PREV[2,...], PREV[1,...],
            NI, NJ, nthin, Uana, Vana)

    Unow, Vnow = compute_motion_numba(
            PREV[2,...], NWC_field[0,...],
            NI, NJ, nthin, Unow, Vnow)

    IS, JS, Uerr_av, Verr_av = calculate_corrected_motion_vector(NI, NJ, 
                nthin, Unow, Vnow, Uerr_, Verr_, Uana, Vana, IS, JS, Uerr_av, Verr_av)

    boole_ = ((IS > -99) & (JS > -99))

    UU = np.zeros([NJ, NI])
    VV = np.zeros([NJ, NI])
    UU, VV = interpolate_rmv(int(metadata['y1']/metadata['xpixelsize']), int(metadata['y2']/metadata['xpixelsize']), 
        int(metadata['x1']/metadata['xpixelsize']), int(metadata['x2']/metadata['xpixelsize']), 
        len(Uana.flatten()), nsteph, XX.flatten()/metadata['xpixelsize'], YY.flatten()/metadata['xpixelsize'], 
        boole_.flatten(), IS.flatten(), JS.flatten(), UU, VV)
    
    OFZ = np.append(UU[np.newaxis,...],VV[np.newaxis,...],axis=0)

    print("Elapsed time of INCA motion vector: %s [h:mm:ss].",
        timedelta(seconds=int(time.monotonic() - start_nwc)))

    return OFZ



################################################################################
# Read the radar input images
# ---------------------------
#
# First, we will import the sequence of radar composites.
# You need the pysteps-data archive downloaded and the pystepsrc file
# configured with the data_source paths pointing to data folders.

# Selected case
date = datetime.strptime("201505151630", "%Y%m%d%H%M")
data_source = rcparams.data_sources["mch"]

###############################################################################
# Load the data from the archive
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

root_path = data_source["root_path"]
path_fmt = data_source["path_fmt"]
fn_pattern = data_source["fn_pattern"]
fn_ext = data_source["fn_ext"]
importer_name = data_source["importer"]
importer_kwargs = data_source["importer_kwargs"]
timestep = data_source["timestep"]

# Find the input files from the archive
fns = io.archive.find_by_date(
    date, root_path, path_fmt, fn_pattern, fn_ext, timestep=5, num_prev_files=9
)

# Read the radar composites
importer = io.get_method(importer_name, "importer")
R, quality, metadata = io.read_timeseries(fns, importer, **importer_kwargs)

del quality  # Not used

###############################################################################
# Preprocess the data
# ~~~~~~~~~~~~~~~~~~~

# Convert to mm/h
R, metadata = conversion.to_rainrate(R, metadata)

# Store the reference frame
R_ = R[-1, :, :].copy()
R_here = R.copy()

# Log-transform the data [dBR]
R, metadata = transformation.dB_transform(R, metadata, threshold=0.1, zerovalue=-15.0)

# Nicely print the metadata
pprint(metadata)

################################################################################
# INCA TREC method with error motion correction (Haiden et al 2011)
# ---------------------------------------------------
#
# This module implements the anisotropic diffusion method presented in Proesmans
# et al. (1994), a robust optical flow technique which employs the notion of
# inconsitency during the solution of the optical flow equations.

# oflow_method = motion.get_method("proesmans")
R_here[~np.isfinite(R)] = 0
V4 = inca_motion_vectors_simplified(R_here[-3:, :, :],timestep*60)

# Plot the motion field
plot_precip_field(R_, geodata=metadata, title="INCA motion")
quiver(V4, geodata=metadata, step=25)
plt.show()


################################################################################
# Lucas-Kanade (LK)
# -----------------
#
# The Lucas-Kanade optical flow method implemented in pysteps is a local
# tracking approach that relies on the OpenCV package.
# Local features are tracked in a sequence of two or more radar images. The
# scheme includes a final interpolation step in order to produce a smooth
# field of motion vectors.

oflow_method = motion.get_method("LK")
V1 = oflow_method(R[-3:, :, :])

# Plot the motion field on top of the reference frame
plot_precip_field(R_, geodata=metadata, title="LK")
quiver(V1, geodata=metadata, step=25)
plt.show()

################################################################################
# Variational echo tracking (VET)
# -------------------------------
#
# This module implements the VET algorithm presented
# by Laroche and Zawadzki (1995) and used in the McGill Algorithm for
# Prediction by Lagrangian Extrapolation (MAPLE) described in
# Germann and Zawadzki (2002).
# The approach essentially consists of a global optimization routine that seeks
# at minimizing a cost function between the displaced and the reference image.

oflow_method = motion.get_method("VET")
V2 = oflow_method(R[-3:, :, :])

# Plot the motion field
plot_precip_field(R_, geodata=metadata, title="VET")
quiver(V2, geodata=metadata, step=25)
plt.show()

################################################################################
# Dynamic and adaptive radar tracking of storms (DARTS)
# -----------------------------------------------------
#
# DARTS uses a spectral approach to optical flow that is based on the discrete
# Fourier transform (DFT) of a temporal sequence of radar fields.
# The level of truncation of the DFT coefficients controls the degree of
# smoothness of the estimated motion field, allowing for an efficient
# motion estimation. DARTS requires a longer sequence of radar fields for
# estimating the motion, here we are going to use all the available 10 fields.

oflow_method = motion.get_method("DARTS")
R[~np.isfinite(R)] = metadata["zerovalue"]
V3 = oflow_method(R)  # needs longer training sequence

# Plot the motion field
plot_precip_field(R_, geodata=metadata, title="DARTS")
quiver(V3, geodata=metadata, step=25)
plt.show()

################################################################################
# Anisotropic diffusion method (Proesmans et al 1994)
# ---------------------------------------------------
#
# This module implements the anisotropic diffusion method presented in Proesmans
# et al. (1994), a robust optical flow technique which employs the notion of
# inconsitency during the solution of the optical flow equations.

# oflow_method = motion.get_method("proesmans")
# R[~np.isfinite(R)] = metadata["zerovalue"]
# V4 = oflow_method(R[-2:, :, :])

# # Plot the motion field
# plot_precip_field(R_, geodata=metadata, title="Proesmans")
# quiver(V4, geodata=metadata, step=25)
# plt.show()

# sphinx_gallery_thumbnail_number = 1
