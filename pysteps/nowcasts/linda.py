"""
pysteps.nowcasts.linda
======================

This module implements the Lagrangian INtegro-Difference equation model with
Autoregression (LINDA). The model combines elements from extrapolation,
S-PROG, STEPS and ANVIL, cell tracking and integro-difference equation (IDE)
models with the aim of producing improved nowcasts for convective events. It
consists of the following components:

1. feature detection to identify rain cells
2. advection-based nowcast
3. autoregressive integrated (ARI) process to predict growth and decay
4. convolution to account for loss of predictability
5. stochastic perturbations to simulate forecast errors

Focusing on convective cells, LINDA uses a sparse representation of the input
data. The cells are identified using a blob detection method :cite:`?`. The
advection field is determined using based on the detected blobs, and the
remaining components of the model are applied in the Lagrangian coordinates.
Using the ARI process is adapted from ANVIL :cite:`PCLH2020`, and the
convolution is adapted from the integro-difference equation (IDE)
methodology developed in :cite:`?`. Combination of these two approaches
essentially replaces the cascade decomposition used in S-PROG and STEPS.
Using the convolution has several advantages such as the ability to handle
anisotropic structure, domain boundaries and missing data.

All components of the model are localized at the identified rain cells. The
advection field is determined using the sparse Lucas-Kanade method and
interpolated to cover the whole domain. The convolution is implemented by using
a spatially variable kernel that can describe anisotropic structure. Based on
the marginal distribution and covariance structure of forecast errors, localized
perturbations are generated by adapting the SSFT methodology developed in
:cite:`NBSG2017`.

.. autosummary::
    :toctree: ../generated/

    forecast
"""

import dask
import numpy as np
from scipy.optimize import LinearConstraint, minimize, minimize_scalar
from scipy.signal import convolve


def forecast(
    precip_fields_fields,
    advection_field,
    ari_order=1,
    add_perturbations=True,
    n_ens_members=24,
):
    """Generate a nowcast ensemble by using the Lagrangian INtegro-Difference
    equation model with Autoregression (LINDA).

    Parameters
    ----------
    precip_fields : array_like
        Array of shape (ari_order+2,m,n) containing the input precipitation
        fields ordered by timestamp from oldest to newest. The time steps
        between the inputs are assumed to be regular.
    advection_field : array_like
        Array of shape (2,m,n) containing the x- and y-components of the
        advection field. The velocities are assumed to represent one time step
        between the inputs.
    ari_order : int
        Order of the ARI(p,1) model.
    add_perturbations : bool
        Set to False to disable perturbations and generate a deterministic
        nowcast.
    n_ens_members : int
        Number of ensemble members.

    Returns
    -------
    out : numpy.ndarray
        A four-dimensional array of shape (n_ens_members,n_timesteps,m,n)
        containing a time series of forecast precipitation fields for each
        ensemble member. The time series starts from t0+timestep, where
        timestep is taken from the input fields.
    """
    pass


def _compute_convolution_kernel(params, cutoff=6.0):
    phi, sigma1, sigma2 = params[:3]

    sigma1 = abs(sigma1)
    sigma2 = abs(sigma2)

    phi_r = phi / 180.0 * np.pi
    R_inv = np.array([[np.cos(phi_r), np.sin(phi_r)], [-np.sin(phi_r), np.cos(phi_r)]])

    bb_y1, bb_x1, bb_y2, bb_x2 = _compute_ellipse_bbox(phi, sigma1, sigma2, cutoff)

    x = np.arange(int(bb_x1), int(bb_x2) + 1).astype(float)
    if len(x) % 2 == 0:
        x = np.arange(int(bb_x1) - 1, int(bb_x2) + 1).astype(float)
    y = np.arange(int(bb_y1), int(bb_y2) + 1).astype(float)
    if len(y) % 2 == 0:
        y = np.arange(int(bb_y1) - 1, int(bb_y2) + 1).astype(float)

    X, Y = np.meshgrid(x, y)
    XY = np.vstack([X.flatten(), Y.flatten()])
    XY = np.dot(R_inv, XY)

    x2 = XY[0, :] * XY[0, :]
    y2 = XY[1, :] * XY[1, :]
    result = np.exp(-((x2 / sigma1 + y2 / sigma2) ** params[3]))
    result /= np.sum(result)

    return np.reshape(result, X.shape)


def _compute_ellipse_bbox(phi, sigma1, sigma2, cutoff):
    r1 = cutoff * sigma1
    r2 = cutoff * sigma2
    phi_r = phi / 180.0 * np.pi

    if np.abs(phi_r - np.pi / 2) > 1e-6 and np.abs(phi_r - 3 * np.pi / 2) > 1e-6:
        alpha = np.arctan(-r2 * np.sin(phi_r) / (r1 * np.cos(phi_r)))
        w = r1 * np.cos(alpha) * np.cos(phi_r) - r2 * np.sin(alpha) * np.sin(phi_r)

        alpha = np.arctan(r2 * np.cos(phi_r) / (r1 * np.sin(phi_r)))
        h = r1 * np.cos(alpha) * np.sin(phi_r) + r2 * np.sin(alpha) * np.cos(phi_r)
    else:
        w = sigma2 * cutoff
        h = sigma1 * cutoff

    return -abs(h), -abs(w), abs(h), abs(w)


# Get anisotropic convolution kernel parameters from the given parameter vector.
def _get_anisotropic_kernel_params(p):
    theta = np.arctan2(p[1], p[0])
    sigma1 = np.sqrt(p[0] * p[0] + p[1] * p[1])
    sigma2 = sigma1 * p[2]

    return theta, sigma1, sigma2, p[3]


# Compute a 2d convolution by ignoring non-finite values.
def _masked_convolution(field, kernel):
    mask = np.isfinite(field)

    field = field.copy()
    field[~mask] = 0.0

    field_c = np.ones(field.shape) * np.nan
    field_c[mask] = convolve(field, kernel, mode="same")[mask]
    field_c[mask] /= convolve(mask.astype(float), kernel, mode="same")[mask]

    return field_c


# Constrained optimization of AR(1) parameters
def optimize_ar1_params(field_src, field_dst, weights, num_workers=1):
    def worker(i):
        def objf(p, *args):
            field_ar = p * field_src[0]
            return np.nansum(weights[i] * (field_dst - field_ar) ** 2.0)

        bounds = (-0.98, 0.98)
        p_opt = minimize_scalar(objf, method="bounded", bounds=bounds)

        return i, p_opt.x

    res = []
    for i in range(weights.shape[0]):
        res.append(dask.delayed(worker)(i))

    res = dask.compute(*res, num_workers=num_workers, scheduler="multiprocessing")

    psi = np.empty((weights.shape[0], 1))
    for r in res:
        psi[r[0]] = r[1]

    return psi


# Constrained optimization of AR(2) parameters
def _optimize_ar2_params(field_src, field_dst, weights, num_workers=1):
    def worker(i):
        def objf(p, *args):
            field_ar = p[0] * field_src[1] + p[1] * field_src[0]
            return np.nansum(weights[i] * (field_dst - field_ar) ** 2.0)

        bounds = [(-1.98, 1.98), (-0.98, 0.98)]
        constraints = [
            LinearConstraint(
                np.array([(1, 1), (-1, 1)]),
                (-np.inf, -np.inf),
                (0.98, 0.98),
                keep_feasible=True,
            )
        ]
        p_opt = minimize(
            objf,
            (0.8, 0.0),
            method="trust-constr",
            bounds=bounds,
            constraints=constraints,
        )

        return i, p_opt.x

    res = []
    for i in range(weights.shape[0]):
        res.append(dask.delayed(worker)(i))

    res = dask.compute(*res, num_workers=num_workers, scheduler="multiprocessing")

    psi = np.empty((weights.shape[0], 2))
    for r in res:
        psi[r[0], :] = r[1]

    return psi
