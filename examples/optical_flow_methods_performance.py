# coding: utf-8

"""
Optical flow methods performance
================================

In this example we test the performance of the optical flow methods available in
pySTEPS using idealized motion fields.

To test the performance, using an example precipitation field we will:
- Read precipitation field from a file
- Morph the precipitation field using a given motion field (linear or rotor) to
  generate a sequence of moving precipitation patterns.
- Using the available optical flow methods, retrieve the motion field from the
  precipitation time sequence.

Let's first load the libraries that we will use.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from scipy.ndimage import uniform_filter

import pysteps as stp
from pysteps import motion

"""
Let's load the precipitation field from one of the examples files.
"""
file_name = "sample_mch_radar_composite_00.gif"
reference_field, quality, metadata = stp.io.import_mch_gif(file_name,
                                                           product="AQC",
                                                           unit="mm",
                                                           accutime=5.)

del quality  # Not used



## convert to mm/h
reference_field, metadata = stp.utils.to_rainrate(reference_field, metadata)

## threshold the data
reference_field[reference_field < 0.1] = 0.0

## transform to dBR
reference_field, _ = stp.utils.dB_transform(reference_field)

reference_field = np.ma.masked_invalid(reference_field)

# reference_field dimensions is (lat, lon), let's swap them to (lon, lat) to
# apply the motion fields (u, v)
reference_field = reference_field.swapaxes(0, 1)

# Lets create an imaginary grid on the image and create a motion field to be
# applied to the image.

# Set the grid values (x,y) between -1 and 1
x_pos = np.arange(reference_field.shape[0])
y_pos = np.arange(reference_field.shape[1])
x, y = np.meshgrid(x_pos, y_pos, indexing='ij')


def create_motion_field(motion_type):
    """
    Create idealized motion fields to be applied to the reference image.

    The supported motion fields are:
        - linear_x: (u=2, v=0)
        - linear_y: (u=0, v=2)
        - rotor: rotor field
    """

    ideal_motion = np.zeros((2,) + reference_field.shape)

    if motion_type == "linear_x":
        ideal_motion[0, :] = 2  # Motion along x
    elif motion_type == "linear_y":
        ideal_motion[1, :] = 2  # Motion along y
    elif motion_type == "rotor":
        ideal_motion[0, :] = y
        ideal_motion[1, :] = -x
    else:
        raise ValueError("motion_type not supported.")
    return ideal_motion


expected_motion = create_motion_field('linear_y')

### Set figures default properties
import matplotlib

step = 30
arrows_scale = 20
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rcParams['ytick.labelsize'] = 18
matplotlib.rcParams['axes.titlesize'] = 24

# Now, we apply morph the reference image by applying the displacement field:

morphed_field, mask = motion.vet.morph(reference_field, expected_motion)
mask = np.array(mask, dtype=bool)

input_fields = np.ma.MaskedArray(morphed_field, mask=mask)[np.newaxis, :]

for t in range(1, 9):
    morphed_field, mask = motion.vet.morph(input_fields[t - 1], expected_motion)
    mask = np.array(mask, dtype=bool)

    morphed_field = np.ma.MaskedArray(morphed_field[np.newaxis, :],
                                      mask=mask[np.newaxis, :])

    input_fields = np.ma.concatenate([input_fields, morphed_field],
                                     axis=0)

input_fields = np.ma.masked_invalid(input_fields)

# pysteps.plt.animate(input_fields, nloops=1)

input_fields.data[np.ma.getmaskarray(input_fields)] = 0

# We need to swap the axes because the optical flow methods expect (lat, lon) or
# (y,x) indexing convention.
input_fields = input_fields.swapaxes(2, 1)
expected_motion = expected_motion.swapaxes(1, 2)
x = x.swapaxes(0, 1)
y = y.swapaxes(0, 1)

mask = input_fields.mask.any(axis=0)

precip_data, _ = stp.utils.dB_transform(input_fields.max(axis=0),
                                        inverse=True)
precip_data[precip_data.mask] = 0

precip_mask = ((uniform_filter(precip_data, size=20) > 0.1)
               & ~input_fields.mask.any(axis=0))

precip_mask_float = np.array(precip_mask, dtype='float32')

cmap = get_cmap('jet')
cmap.set_under('grey', alpha=0.25)
cmap.set_over('none')

for method_name in ["LucasKanade", "VET", "DARTS"]:
    oflow_method = motion.get_method(method_name)

    if method_name != "DARTS":
        _input = input_fields[-3:]
    else:
        _input = input_fields

    computed_motion = oflow_method(_input, verbose=False)

    # Compare retrieved displacement field with the applied one
    plt.close()
    plt.figure(figsize=(13, 13))

    plt.subplot(121, aspect='equal')
    step = 31
    plt.title("Expected motion")
    plt.quiver(x[::step, ::step], y[::step, ::step],
               expected_motion[0, ::step, ::step],
               expected_motion[1, ::step, ::step],
               scale=arrows_scale)
    plt.pcolormesh(x, y, precip_mask_float, vmin=0.1, vmax=0.5, cmap=cmap)

    plt.subplot(122, aspect='equal')
    plt.title(f"{method_name} motion")
    plt.quiver(x[::step, ::step], y[::step, ::step],
               computed_motion[0, ::step, ::step],
               computed_motion[1, ::step, ::step], scale=arrows_scale)
    plt.pcolormesh(x, y, precip_mask_float, vmin=0.1, vmax=0.5, cmap=cmap)

    plt.show()

    # To evaluate the accuracy of the computed_motion vectors, we will use
    # a relative RMSE measure.
    # Relative MSE = < (expected_motion - computed_motion)^2 > / <expected_motion^2 >
    # Relative RMSE = sqrt(Relative MSE)
    mse = ((expected_motion - computed_motion)[:, precip_mask] ** 2).mean()

    rel_mse = mse / (expected_motion[:, precip_mask] ** 2).mean()
    print(f"{method_name} Relative RMSE: {np.sqrt(rel_mse) * 100:.2f}%")
