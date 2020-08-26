import warnings

import numpy as np
from scipy.ndimage import convolve


def radial_average(s, k, k0):
    dk = k[0,1]-k[0,0]
    on_rad = (k >= k0-dk*0.5) & (k < k0+dk*0.5)
    return abs(s[on_rad]).mean()


def log_slope(log_k, log_power_spectrum):
    lk_min = log_k.min()
    lk_max = log_k.max()
    lk_range = lk_max-lk_min
    lk_min += (1/6)*lk_range
    lk_max -= (1/6)*lk_range
    dk = lk_range / 32
 
    selected = (lk_min <= log_k) & (log_k <= lk_max)
    num_selected = np.count_nonzero(selected)
    lk_sel = log_k[selected]
    ps_sel = log_power_spectrum[selected]
    (alpha,_) = np.polyfit(lk_sel,ps_sel,1)
    alpha = -alpha

    return alpha


def balanced_spatial_average(x,k):
    ones = np.ones_like(x)
    return convolve(x,k)/convolve(ones,k)


def get_alpha_seq(P):
    ki = np.fft.fftfreq(P.shape[1])
    kj = np.fft.fftfreq(P.shape[2])
    k_sqr = ki[:,None]**2 + kj[None,:]**2
    k = np.sqrt(k_sqr)
    k = np.stack([k]*P.shape[0])

    fp = np.stack([np.fft.fft2(p) for p in P])
    fp_abs = abs(fp)
    log_power_spectrum = np.log(fp_abs**2)
    valid = (k!=0) & np.isfinite(log_power_spectrum)
    return log_slope(np.log(k[valid]), log_power_spectrum[valid])


def rainfarm_downscale(P, alpha=None, ds_factor=16, threshold=None):
    ki = np.fft.fftfreq(P.shape[0])
    kj = np.fft.fftfreq(P.shape[1])
    k_sqr = ki[:,None]**2 + kj[None,:]**2
    k = np.sqrt(k_sqr)

    ki_ds = np.fft.fftfreq(P.shape[0]*ds_factor, d=1/ds_factor)
    kj_ds = np.fft.fftfreq(P.shape[1]*ds_factor, d=1/ds_factor)
    k_ds_sqr = ki_ds[:,None]**2 + kj_ds[None,:]**2
    k_ds = np.sqrt(k_ds_sqr)

    if alpha is None:
        fp = np.fft.fft2(P)
        fp_abs = abs(fp)
        log_power_spectrum = np.log(fp_abs**2)
        valid = (k!=0) & np.isfinite(log_power_spectrum)
        alpha = log_slope(np.log(k[valid]), log_power_spectrum[valid])

    fg = np.exp(complex(0,1)*2*np.pi*np.random.rand(*k_ds.shape))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fg *= np.sqrt(k_ds_sqr ** (-alpha/2))
    fg[0,0] = 0
    g = np.fft.ifft2(fg).real
    g /= g.std()
    r = np.exp(g)

    P_u = np.repeat(np.repeat(P, ds_factor, axis=0), ds_factor, axis=1)
    rad = int(round(ds_factor/np.sqrt(np.pi)))
    (mx,my) = np.mgrid[-rad:rad+0.01,-rad:rad+0.01]
    tophat = ((mx**2+my**2) <= rad**2).astype(float)
    tophat /= tophat.sum()

    P_agg = balanced_spatial_average(P_u,tophat)
    r_agg = balanced_spatial_average(r,tophat)
    r *= P_agg / r_agg

    if threshold is not None:
        r[r<threshold] = 0

    return r
