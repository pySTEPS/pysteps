"""
Parametric nowcast from three rain-rate fields
==============================================

This self-contained example builds three synthetic observations, then lets the
parametric engine estimate motion and parameters and generate an ensemble.
It needs no radar archive, database, or labelled-array dependency.
"""

import matplotlib.pyplot as plt
import numpy as np

from pysteps import nowcasts
from pysteps.noise.parametric import generate_stochastic_field

# Construct an illustrative rain-rate field and a translating observation
# history. In an application, pass three measured rain-rate arrays instead.
transformed = generate_stochastic_field(
    (128, 128),
    dict(beta_1=-2.4, beta_2=-3.1, rain_fraction=0.2, mean_db=6.0, stdev_db=3.0),
    kmperpixel=1.0,
    scale_break=20.0,
    threshold=0.0,
    seed=9,
)
rain = 10 ** (transformed / 10)
rain[rain < 1.0] = 0
observations = np.stack(
    [
        np.roll(rain, -2, axis=1),
        np.roll(rain, -1, axis=1),
        rain,
    ]
)

###############################################################################
# Forecast directly in mm/h. Motion is estimated internally from the latest
# two fields; no transformed input or stored cascade/parameter state is needed.
forecast = nowcasts.get_method("parametric")
ensemble, diagnostics = forecast(
    observations,
    timesteps=6,
    n_ens_members=4,
    kmperpixel=1.0,
    timestep=5.0,
    seed=42,
    return_diagnostics=True,
)
print("Output shape:", ensemble.shape)
print("Selected generation parameters:", diagnostics["parameters"])
print("Moment source:", diagnostics["safeguards"]["moment_source"])

###############################################################################
# Compare the newest observation with a member and the ensemble mean at 30 min.
fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
for ax, field, title in zip(
    axes,
    [observations[-1], ensemble[0, -1], ensemble[:, -1].mean(axis=0)],
    ["Newest observation", "Member 1, +30 min", "Ensemble mean, +30 min"],
):
    image = ax.imshow(field, origin="lower", vmin=0, vmax=20, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
fig.colorbar(image, ax=axes, label="Rain rate (mm/h)", shrink=0.75)
plt.show()
