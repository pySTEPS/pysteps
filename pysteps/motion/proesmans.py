
import numpy as np
from scipy.ndimage import gaussian_filter
from pysteps.motion._proesmans import _compute_advection_field

def proesmans(input_images, lam=50.0, num_iter=100, num_levels=6, filter_std=0.0):
    """The anisotropic diffusion method by Proesmans et al.
    
    Parameters
    ----------
    input_images : array_like
        Array of shape (2,m,n) containing the two input images.
    lam : float
        Multiplier of the smoothness term. Smaller values give smoother motion
        field.
    num_iter : float
        Number of iterations to use.
    num_levels : int
        Number of image pyramid levels to use.
    filter_std : float
        Standard deviation of optional Gaussian filter that is applied before
        computing the optical flow.
    """
    im1 = input_images[-2, :, :].copy()
    im2 = input_images[-1, :, :].copy()

    im = np.stack([im1, im2])
    im_min = np.min(im)
    im_max = np.max(im)
    im = (im - im_min) / (im_max - im_min) * 255.0

    if filter_std > 0.0:
        im[0, :, :] = gaussian_filter(im[0, :, :], filter_std)
        im[1, :, :] = gaussian_filter(im[1, :, :], filter_std)

    return _compute_advection_field(im, lam, num_iter, num_levels)
