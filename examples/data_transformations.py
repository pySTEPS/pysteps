# -*- coding: utf-8 -*-
"""
Data transformations
====================

The statistics of intermittent precipitation rates are particularly non-Gaussian
and display an asymmetric distribution bounded at zero.
Such properties restrict the usage of well-established statistical methods that
assume symmetric or Gaussian data.

A common workaround is to introduce a suitable data transformation to approximate
a normal distribution.

In this example, we test the data transformation methods available in pysteps
in order to obtain a more symmetric distribution of the precipitation data
(excluding the zeros).
The currently available transformations include the Box-Cox, dB, square-root and
normal quantile transforms.

"""

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pysteps import io, rcparams
from scipy.stats import skew

###############################################################################
# Read the radar input images
# ---------------------------
#
# First, we will import the sequence of radar composites.
# You need the pysteps-data archive downloaded and the pystepsrc file
# configured with the data_source paths pointing to data folders.

# Selected case
date = datetime.strptime("201609281600", "%Y%m%d%H%M")
data_source = rcparams.data_sources["fmi"]


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

# Get 1 hour of observations in the data archive
fns = io.archive.find_by_date(
    date, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_next_files=11
)

# Read the radar composites
importer = io.get_method(importer_name, "importer")
precip = io.read_timeseries(fns, importer, **importer_kwargs)

# Convert to rain rate
precip = precip.pysteps.to_rainrate()

# Keep only positive rainfall values
precip = precip.where(precip > 0)

###############################################################################
# Test data transformations
# -------------------------

# Define method to visualize the data distribution with boxplots and plot the
# corresponding skewness
def plot_distribution(data, labels, skw):

    # Extract and flatten the arrays
    data = [da.values.flatten() for da in data]
    # Keep finite values only
    data = [da[np.isfinite(da)] for da in data]

    n_points = len(data)
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax2.plot(np.arange(n_points + 2), np.zeros(n_points + 2), ":r")
    ax1.boxplot(data, labels=labels, sym="", medianprops={"color": "k"})

    ymax = []
    for i in range(n_points):
        y = skw[i]
        x = i + 1
        ax2.plot(x, y, "*r", ms=10, markeredgecolor="k")
        ymax.append(np.max(data[i]))

    # ylims
    ylims = np.percentile(ymax, 50)
    ax1.set_ylim((-1 * ylims, ylims))
    ylims = np.max(np.abs(skw))
    ax2.set_ylim((-1.1 * ylims, 1.1 * ylims))

    # labels
    ax1.set_ylabel(r"Standardized values [$\sigma$]")
    ax2.set_ylabel(r"Skewness []", color="r")
    ax2.tick_params(axis="y", labelcolor="r")


###############################################################################
# Box-Cox transform
# ~~~~~~~~~~~~~~~~~
# The Box-Cox transform is a well-known power transformation introduced by
# `Box and Cox (1964)`_. In its one-parameter version, the Box-Cox transform
# takes the form T(x) = ln(x) for lambda = 0, or T(x) = (x**lambda - 1)/lambda
# otherwise.
#
# To find a suitable lambda, we will experiment with a range of values
# and select the one that produces the most symmetric distribution, i.e., the
# lambda associated with a value of skewness closest to zero.
# To visually compare the results, the transformed data are standardized.
#
# .. _`Box and Cox (1964)`: https://doi.org/10.1111/j.2517-6161.1964.tb00553.x

data = []
labels = []
skw = []

# Test a range of values for the transformation parameter Lambda
lambdas = np.linspace(-0.4, 0.4, 11)
for i, this_lambda in enumerate(lambdas):
    precip_ = precip.pysteps.boxcox_transform(boxcox_lambda=this_lambda)
    precip_ = (precip_ - np.mean(precip_)) / np.std(precip_)
    data.append(precip_)
    labels.append("{0:.2f}".format(this_lambda))
    skw.append(skew(precip_, axis=None, nan_policy="omit"))  # skewness

# Plot the transformed data distribution as a function of lambda
plot_distribution(data, labels, skw)
plt.title("Box-Cox transform")
plt.tight_layout()
plt.show()

# Best lambda
idx_best = np.argmin(np.abs(skw))
best_lambda = lambdas[idx_best]

print("Best parameter lambda: %.2f\n(skewness = %.2f)" % (best_lambda, skw[idx_best]))

###############################################################################
# Compare data transformations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data = []
labels = []
skw = []

###############################################################################
# Rain rates
# ~~~~~~~~~~
# First, let's have a look at the original rain rate values.

data.append((precip - np.mean(precip)) / np.std(precip))
labels.append("precip")
skw.append(skew(precip, axis=None, nan_policy="omit"))

###############################################################################
# dB transform
# ~~~~~~~~~~~~
# We transform the rainfall data into dB units: 10*log(precip)

precip_ = precip.pysteps.db_transform()
data.append((precip_ - np.mean(precip_)) / np.std(precip_))
labels.append("dB")
skw.append(skew(precip_, axis=None, nan_policy="omit"))

###############################################################################
# Square-root transform
# ~~~~~~~~~~~~~~~~~~~~~
# Transform the data using the square-root: sqrt(precip)

precip_ = precip.pysteps.sqrt_transform()
data.append((precip_ - np.mean(precip_)) / np.std(precip_))
labels.append("sqrt")
skw.append(skew(precip_, axis=None, nan_policy="omit"))

###############################################################################
# Box-Cox transform
# ~~~~~~~~~~~~~~~~~
# We now apply the Box-Cox transform using the best parameter lambda found above.

precip_ = precip.pysteps.boxcox_transform(best_lambda)
data.append((precip_ - np.mean(precip_)) / np.std(precip_))
labels.append("Box-Cox\n($\lambda=$%.2f)" % best_lambda)
skw.append(skew(precip_, axis=None, nan_policy="omit"))

###############################################################################
# Normal quantile transform
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# At last, we apply the empirical normal quantile (NQ) transform as described in
# `Bogner et al (2012)`_.
#
# .. _`Bogner et al (2012)`: http://dx.doi.org/10.5194/hess-16-1085-2012

precip_ = precip.pysteps.nq_transform()
data.append((precip_ - np.mean(precip_)) / np.std(precip_))
labels.append("NQ")
skw.append(skew(precip_, axis=None, nan_policy="omit"))

###############################################################################
# By plotting all the results, we can notice first of all the strongly asymmetric
# distribution of the original data (precip) and that all transformations manage to
# reduce its skewness. Among these, the Box-Cox transform (using the best parameter
# lambda) and the normal quantile (NQ) transform provide the best correction.
# Despite not producing a perfectly symmetric distribution, the square-root (sqrt)
# transform has the strong advantage of being defined for zeros, too, while all
# other transformations need an arbitrary rule for non-positive values.

plot_distribution(data, labels, skw)
plt.title("Data transforms")
plt.tight_layout()
plt.show()
