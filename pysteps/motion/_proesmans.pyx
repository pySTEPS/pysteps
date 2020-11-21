# -*- coding: utf-8 -*-

"""
Cython module for the Proesmans optical flow algorithm
"""

#from cython.parallel import parallel, prange
import numpy as np
from scipy.ndimage import convolve

cimport cython
cimport numpy as np

ctypedef np.float64_t float64
ctypedef np.intp_t intp

from libc.math cimport floor, sqrt

cdef float64 _INTENSITY_SCALE = 1.0 / 255.0

def _compute_advection_field(float64 [:, :, :] R, lam, intp num_iter,
                             intp n_levels):
    R_p = [_construct_image_pyramid(R[0, :, :], n_levels),
           _construct_image_pyramid(R[1, :, :], n_levels)]

    cdef intp m = R_p[0][-1].shape[0]
    cdef intp n = R_p[0][-1].shape[1]

    cdef np.ndarray[float64, ndim=4] V_cur = np.zeros((2, 2, m, n))
    cdef np.ndarray[float64, ndim=4] V_next

    cdef np.ndarray[float64, ndim=3] GAMMA = np.empty((2, R.shape[1], R.shape[2]))

    for i in range(n_levels-1, -1, -1):
        _proesmans(np.stack([R_p[0][i], R_p[1][i]]), V_cur, num_iter, lam)

        m = R_p[0][i-1].shape[0]
        n = R_p[0][i-1].shape[1]

        V_next = np.zeros((2, 2, m, n))

        if i > 0:
            _initialize_next_level(V_cur, V_next)
            V_cur = V_next

    _compute_consistency_maps(V_cur, GAMMA)

    return V_cur, GAMMA

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef _compute_next_pyramid_level(float64 [:, :] I_src,
                                 float64 [:, :] I_dest):
    cdef intp dh = I_dest.shape[0]
    cdef intp dw = I_dest.shape[1]
    cdef intp x, y

    for y in range(dh):
        for x in range(dw):
            I_dest[y, x] = (I_src[2*y, 2*x] + I_src[2*y, 2*x+1] + \
                            I_src[2*y+1, 2*x] + I_src[2*y+1, 2*x+1]) / 4.0

cdef _construct_image_pyramid(float64 [:, :] R, intp n_levels):
    cdef intp m = R.shape[0]
    cdef intp n = R.shape[1]
    cdef np.ndarray[float64, ndim=2] R_next

    R_out = [R]
    cdef float64 [:, :] R_cur = R
    for i in range(1, n_levels):
        R_next = np.zeros((int(m/2), int(n/2)))
        _compute_next_pyramid_level(R_cur, R_next)
        R_cur = R_next

        R_out.append(R_cur)
        m = int(m / 2)
        n = int(n / 2)

    return R_out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef _proesmans(float64 [:, :, :] R, float64 [:, :, :, :] V, intp num_iter,
                float64 lam):
    cdef intp x, y
    cdef intp i, j
    cdef float64 xd, yd
    cdef float64 It
    cdef float64 ic
    cdef float64 gx, gy

    cdef intp m = R.shape[1]
    cdef intp n = R.shape[2]

    cdef np.ndarray[float64, ndim=4] G = np.zeros((2, 2, R.shape[1], R.shape[2]))

    G[0, :, :, :] = _compute_gradients(R[0, :, :])
    G[1, :, :, :] = _compute_gradients(R[1, :, :])

    cdef np.ndarray[float64, ndim=3] GAMMA = np.zeros((2, R.shape[1], R.shape[2]))
    cdef float64 v_avg_1, v_avg_2
    cdef float64 v_next_1, v_next_2

    cdef float64 [:, :] R_j_1
    cdef float64 [:, :] R_j_2
    cdef float64 [:, :] G_j_1
    cdef float64 [:, :] G_j_2
    cdef float64 [:, :, :] V_j
    cdef float64 [:, :] GAMMA_j

    for i in range(num_iter):
        _compute_consistency_maps(V, GAMMA)

        for j in range(2):
            R_j_1 = R[j, :, :]
            R_j_2 = R[1-j, :, :]
            G_j_1 = G[j, 0, :, :]
            G_j_2 = G[j, 1, :, :]
            V_j = V[j, :, :, :]
            GAMMA_j = GAMMA[j, :, :]

            for y in range(1, m-1):
            #for y in prange(1, m - 1, schedule='static', nogil=True):
                for x in range(1, n-1):
                    v_avg_1 = _compute_laplacian(GAMMA_j, V_j, x, y, 0)
                    v_avg_2 = _compute_laplacian(GAMMA_j, V_j, x, y, 1)

                    xd = x + v_avg_1
                    yd = y + v_avg_2
                    if xd >= 0 and xd < n - 1 and yd >= 0 and yd < m - 1:
                        It = (_linear_interpolate(R_j_2, xd, yd) - \
                            R_j_1[y, x]) * _INTENSITY_SCALE
                        gx = G_j_1[y, x]
                        gy = G_j_2[y, x]
                        ic = lam * It / (1.0 + lam * (gx * gx + gy * gy))
                        v_next_1 = v_avg_1 - gx * ic
                        v_next_2 = v_avg_2 - gy * ic
                    else:
                        # use consistency-weighted average as the next value
                        # if (xd,yd) is outside the image
                        v_next_1 = v_avg_1
                        v_next_2 = v_avg_2

                    V_j[0, y, x] = v_next_1
                    V_j[1, y, x] = v_next_2

            _fill_edges(V[j, :, :, :])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef float64 _compute_laplacian(float64 [:, :] gi, float64 [:, :, :] Vi, intp x,
                                intp y, intp j): #nogil:
    cdef float64 v
    cdef float64 sumWeights = (gi[y-1, x] + gi[y, x-1] + \
                              gi[y, x+1] + gi[y+1, x]) / 6.0 + \
                              (gi[y-1, x-1] + gi[y-1, x+1] + \
                              gi[y+1, x-1] + gi[y+1, x+1]) / 12.0

    if sumWeights > 1e-8:
        v = (gi[y-1, x] * Vi[j, y-1, x] + gi[y, x-1] * Vi[j, y, x-1] + \
                gi[y, x+1] * Vi[j, y, x+1] + gi[y+1, x] * Vi[j, y+1, x]) / 6.0 + \
                (gi[y-1, x-1] * Vi[j, y-1, x-1] + gi[y-1, x+1] * Vi[j, y-1, x+1] + \
                gi[y+1, x-1] * Vi[j, y+1, x-1] + gi[y+1, x+1] * Vi[j, y+1, x+1]) / 12.0

        return v / sumWeights
    else:
        return 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _compute_consistency_maps(float64 [:, :, :, :] V,
                                    float64 [:, :, :] GAMMA):
    cdef intp x, y
    cdef intp i
    cdef intp m, n
    cdef float64 xd, yd
    cdef float64 ub, vb
    cdef float64 uDiff, vDiff
    cdef float64 c
    cdef float64 c_sum
    cdef intp c_count
    cdef float64 K
    cdef float64 g

    cdef float64 [:, :] V11, V12, V21, V22

    m = V.shape[2]
    n = V.shape[3]

    for i in range(2):
        c_sum = 0.0
        c_count = 0

        V11 = V[i, 0, :, :]
        V12 = V[i, 1, :, :]
        V21 = V[1-i, 0, :, :]
        V22 = V[1-i, 1, :, :]

        #for y in prange(m, schedule='guided', nogil=True):
        for y in range(m):
            for x in range(n):
                xd = x + V[i, 0, y, x]
                yd = y + V[i, 1, y, x]

                if xd >= 0 and yd >= 0 and xd < n and yd < m:
                    ub = _linear_interpolate(V21, xd, yd)
                    vb = _linear_interpolate(V22, xd, yd)

                    uDiff = V11[y, x] + ub
                    vDiff = V12[y, x] + vb

                    c = sqrt(uDiff * uDiff + vDiff * vDiff)

                    GAMMA[i, y, x] = c
                    c_sum += c
                    c_count += 1
                else:
                    GAMMA[i, y, x] = -1.0

        if c_count > 0:
            K = 0.9 * c_sum / c_count
        else:
            K = 0.0

        #for y in prange(m, schedule='guided', nogil=True):
        for y in range(m):
            for x in range(n):
                if K > 1e-8:
                    if GAMMA[i, y, x] >= 0.0:
                        g = GAMMA[i, y, x]
                        GAMMA[i, y, x] = 1.0 / (1.0 + (g / K) * (g / K))
                    else:
                        GAMMA[i, y, x] = 1.0
                else:
                    GAMMA[i, y, x] = 1.0

cdef np.ndarray[float64, ndim=3] _compute_gradients(float64 [:, :]  I):
    # use 3x3 Sobel kernels for computing partial derivatives
    cdef np.ndarray[float64, ndim=2] Kx = np.zeros((3, 3))
    cdef np.ndarray[float64, ndim=2] Ky = np.zeros((3, 3))

    Kx[0, 0] = 1.0 / 8.0 * _INTENSITY_SCALE
    Kx[0, 1] = 0.0
    Kx[0, 2] = -1.0 / 8.0 * _INTENSITY_SCALE
    Kx[1, 0] = 2.0 / 8.0 * _INTENSITY_SCALE
    Kx[1, 1] = 0.0
    Kx[1, 2] = -2.0 / 8.0 * _INTENSITY_SCALE
    Kx[2, 0] = 1.0 / 8.0 * _INTENSITY_SCALE
    Kx[2, 1] = 0.0
    Kx[2, 2] = -1.0 / 8.0 * _INTENSITY_SCALE

    Ky[0, 0] = 1.0 / 8.0 * _INTENSITY_SCALE
    Ky[0, 1] = 2.0 / 8.0 * _INTENSITY_SCALE
    Ky[0, 2] = 1.0 / 8.0 * _INTENSITY_SCALE
    Ky[1, 0] = 0.0
    Ky[1, 1] = 0.0
    Ky[1, 2] = 0.0
    Ky[2, 0] = -1.0 / 8.0 * _INTENSITY_SCALE
    Ky[2, 1] = -2.0 / 8.0 * _INTENSITY_SCALE
    Ky[2, 2] = -1.0 / 8.0 * _INTENSITY_SCALE

    cdef np.ndarray[float64, ndim=3] G = np.zeros((2, I.shape[0], I.shape[1]))

    G[0, :, :] = convolve(I, Kx, mode="constant", cval=0.0)
    G[1, :, :] = convolve(I, Ky, mode="constant", cval=0.0)

    return G

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _fill_edges(float64 [:, :, :] V): #nogil:
    cdef intp x, y
    cdef intp i

    cdef intp m = V.shape[1]
    cdef intp n = V.shape[2]

    for i in range(2):
        # top and bottom edges
        for x in range(1, n-1):
            V[i, 0, x] = V[i, 1, x]
            V[i, m - 1, x] = V[i, m - 2, x]

        # left and right edges
        for y in range(1, m-1):
            V[i, y, 0] = V[i, y, 1]
            V[i, y, n-1] = V[i, y, n-2]

        # corners
        V[i, 0, 0] = V[i, 1, 1]
        V[i, 0, n - 1] = V[i, 1, n - 2]
        V[i, m - 1, 0] = V[i, m - 2, 1]
        V[i, m - 1, n - 1] = V[i, m - 2, n - 2]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef _initialize_next_level(float64 [:, :, :, :] V_prev,
                            float64 [:, :, :, :] V_next):
    cdef intp m_prev = V_prev.shape[2]
    cdef intp n_prev = V_prev.shape[3]

    cdef intp m_next = V_next.shape[2]
    cdef intp n_next = V_next.shape[3]

    cdef float64 vxf, vyf
    cdef float64 vxb, vyb
    cdef float64 xc, yc
    cdef intp xci, yci
    cdef intp xn, yn

    cdef float64 [:, :] V_prev_1 = V_prev[0, 0, :, :]
    cdef float64 [:, :] V_prev_2 = V_prev[0, 1, :, :]
    cdef float64 [:, :] V_prev_3 = V_prev[1, 0, :, :]
    cdef float64 [:, :] V_prev_4 = V_prev[1, 1, :, :]

    for yn in range(m_next):
        yc = yn / 2.0
        yci = yn / 2
        for xn in range(n_next):
            xc = xn / 2.0
            xci = xn / 2

            if xn % 2 != 0 or yn % 2 != 0:
                vxf = _linear_interpolate(V_prev_1, xc, yc)
                vyf = _linear_interpolate(V_prev_2, xc, yc)
                vxb = _linear_interpolate(V_prev_3, xc, yc)
                vyb = _linear_interpolate(V_prev_4, xc, yc)
            else:
                if xci > n_prev - 1:
                    xci = n_prev - 1
                if yci > m_prev - 1:
                    yci = m_prev - 1
                vxf = V_prev[0, 0, yci, xci]
                vyf = V_prev[0, 1, yci, xci]
                vxb = V_prev[1, 0, yci, xci]
                vyb = V_prev[1, 1, yci, xci]

            V_next[0, 0, yn, xn] = 2.0 * vxf
            V_next[0, 1, yn, xn] = 2.0 * vyf
            V_next[1, 0, yn, xn] = 2.0 * vxb
            V_next[1, 1, yn, xn] = 2.0 * vyb

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef float64 _linear_interpolate(float64 [:, :] I, float64 x, float64 y): #nogil:
    cdef intp x0 = int(x)
    cdef intp x1 = x0 + 1
    cdef intp y0 = int(y)
    cdef intp y1 = y0 + 1

    if x0 < 0:
        x0 = 0
    if x0 > I.shape[1] - 1:
        x0 = I.shape[1]-1
    if x1 < 0:
        x1 = 0
    if x1 > I.shape[1] - 1:
        x1 = I.shape[1]-1
    if y0 < 0:
        y0 = 0
    if y0 > I.shape[0] - 1:
        y0 = I.shape[0]-1
    if y1 < 0:
        y1 = 0
    if y1 > I.shape[0] - 1:
        y1 = I.shape[0]-1

    cdef float64 I_a = I[y0, x0]
    cdef float64 I_b = I[y1, x0]
    cdef float64 I_c = I[y0, x1]
    cdef float64 I_d = I[y1, x1]

    cdef float64 w_a = (x1-x) * (y1-y)
    cdef float64 w_b = (x1-x) * (y-y0)
    cdef float64 w_c = (x-x0) * (y1-y)
    cdef float64 w_d = (x-x0) * (y-y0)

    return w_a*I_a + w_b*I_b + w_c*I_c + w_d*I_d
