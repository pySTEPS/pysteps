#!/bin/env python
"""
Generation of stochastic noise
==============================

This example script shows how to run the stochastic noise field generators
included in pysteps.

These noise fields are used as perturbation terms during an extrapolation
nowcast in order to represent the uncertainty in the evolution of the rainfall
field.
"""

from matplotlib import cm, pyplot as plt
import numpy as np
import os
from pysteps import io, rcparams
from pysteps.noise.fftgenerators import initialize_param_2d_fft_filter
from pysteps.noise.fftgenerators import initialize_nonparam_2d_fft_filter
from pysteps.noise.fftgenerators import generate_noise_2d_fft_filter
from pysteps.utils import rapsd
from pysteps.visualization import plot_precip_field, plot_spectrum1d

###############################################################################
# Read precipitation field
# ------------------------
#
# First thing,  the radar composite is imported and transformed in units
# of dB.
# This image will be used to train the Fourier filters that are necessary to
# produce the fields of spatially correlated noise.

# Import the example radar composite
root_path = rcparams.data_sources["mch"]["root_path"]
filename = os.path.join(root_path, "20160711", "AQC161932100V_00005.801.gif")
precip = io.import_mch_gif(filename, product="AQC", unit="mm", accutime=5.0)

# Convert to mm/h
precip = precip.pysteps.to_rainrate()

# Nicely print the metadata
print(precip)

# Plot the rainfall field
plot_precip_field(precip)
plt.show()

# Log-transform the data
precip = precip.pysteps.db_transform()

# Assign a fill value to all the Nans
precip = precip.fillna(precip.min())

###############################################################################
# Parametric filter
# -----------------
#
# In the parametric approach, a power-law model is used to approximate the power
# spectral density (PSD) of a given rainfall field.
#
# The parametric model uses  a  piece-wise  linear  function  with  two  spectral
# slopes (beta1 and beta2) and one breaking point

# Fit the parametric PSD to the observation
filter_parametric = initialize_param_2d_fft_filter(precip.values)

# Compute the observed and fitted 1D PSD
size = np.max(filter_parametric["input_shape"])
if size % 2 == 1:
    wn = np.arange(0, int(size / 2) + 1)
else:
    wn = np.arange(0, int(size / 2))
spectrum_1d, freq = rapsd(precip, fft_method=np.fft, return_freq=True)
spectrum_1d_param = np.exp(
    filter_parametric["model"](np.log(wn), *filter_parametric["pars"])
)

# Extract the scaling break in km, beta1 and beta2
scaling_break = size / np.exp(filter_parametric["pars"][0])
beta_1 = filter_parametric["pars"][2]
beta_2 = filter_parametric["pars"][3]

# Plot the observed power spectrum and the model
fig, ax = plt.subplots()
plot_scales = [512, 256, 128, 64, 32, 16, 8, 4]
plot_spectrum1d(
    freq,
    spectrum_1d,
    x_units="km",
    y_units="dBR",
    color="k",
    ax=ax,
    label="Observed",
    wavelength_ticks=plot_scales,
)
plot_spectrum1d(
    freq,
    spectrum_1d_param,
    x_units="km",
    y_units="dBR",
    color="r",
    ax=ax,
    label="Fit",
    wavelength_ticks=plot_scales,
)
plt.legend()
ax.set_title(
    "Radially averaged log-power spectrum of precip\n"
    f"scale break={scaling_break:.0f} km, beta_1={beta_1:.1f}, beta_2={beta_2:.1f}"
)
plt.show()

###############################################################################
# Nonparametric filter
# --------------------
#
# In the nonparametric approach,  the Fourier filter is obtained directly
# from the power spectrum of the observed precipitation field precip.

filter_nonparametric = initialize_nonparam_2d_fft_filter(precip.values)

###############################################################################
# Noise generator
# ---------------
#
# The parametric and nonparametric filters obtained above can now be used
# to produce N realizations of random fields of prescribed power spectrum,
# hence with the same correlation structure as the initial rainfall field.

seed = 42
num_realizations = 3

# Generate noise = []
samples_param = []
samples_nonparam = []
Nnp = []
for k in range(num_realizations):
    samples_param.append(generate_noise_2d_fft_filter(filter_parametric, seed=seed + k))
    samples_nonparam.append(
        generate_noise_2d_fft_filter(filter_nonparametric, seed=seed + k)
    )

# Plot the generated noise fields

fig, ax = plt.subplots(nrows=2, ncols=3)

# parametric noise
ax[0, 0].imshow(samples_param[0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[0, 1].imshow(samples_param[1], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[0, 2].imshow(samples_param[2], cmap=cm.RdBu_r, vmin=-3, vmax=3)

# nonparametric noise
ax[1, 0].imshow(samples_nonparam[0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[1, 1].imshow(samples_nonparam[1], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[1, 2].imshow(samples_nonparam[2], cmap=cm.RdBu_r, vmin=-3, vmax=3)

for i in range(2):
    for j in range(3):
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
plt.tight_layout()
plt.show()

###############################################################################
# The above figure highlights the main limitation of the parametric approach
# (top row), that is, the assumption of an isotropic power law scaling
# relationship, meaning that anisotropic structures such as rainfall bands
# cannot be represented.
#
# Instead, the nonparametric approach (bottom row) allows generating
# perturbation fields with anisotropic  structures, but it also requires a
# larger sample size and is sensitive to the quality of the input data, e.g.
# the presence of residual clutter in the radar image.
#
# In addition, both techniques assume spatial stationarity of the covariance
# structure of the field.

# sphinx_gallery_thumbnail_number = 3
