Parametric rainfall estimators
==============================

The parametric estimators provide NumPy-based spatial statistics and temporal
model utilities. They support the three-observation forecast interface described in :doc:`parametric_nowcast`
and can also be used independently.

Spatial estimation
------------------

Use a finite rainfall field and retain the threshold and dry-value convention
when transforming it. Occurrence and marginal fitting use explicitly separate
cutoffs: one identifies rainy pixels, and the other selects the intensity
samples used for fitting. This example uses illustrative settings::

    import numpy as np
    import xarray as xr
    from pysteps.utils.transformation import dB_transform
    from pysteps.utils.statistics import compute_field_statistics
    from pysteps.utils.spectral import rapsd_logarithmic, fit_power_law_spectrum

    rng = np.random.default_rng(42)
    rain = rng.lognormal(1.0, 0.8, size=(256, 256))
    rain[rng.uniform(size=rain.shape) > 0.3] = 0.0
    dataset = xr.Dataset(
        {"rain": (("y", "x"), rain, {"units": "mm/h"})},
        attrs={"precip_var": "rain"},
    )
    transformed_dataset = dB_transform(dataset, threshold=1.0, zerovalue=-0.1)
    transformed = transformed_dataset["rain"].values
    metadata = transformed_dataset["rain"].attrs
    statistics = compute_field_statistics(
        transformed, metadata["threshold"], lower_truncation=1.25
    )
    frequency, power = rapsd_logarithmic(transformed)
    pixel_size_km = 1.0
    scale_break_km = 15.0
    spectrum_fit = fit_power_law_spectrum(
        frequency, power, scale_break=scale_break_km / pixel_size_km
    )

``ordinary_mean`` and ``ordinary_stdev`` describe pixels at or above the
occurrence threshold. ``latent_mean`` and ``latent_stdev`` estimate the parent
Normal from samples at or above the fitting cutoff. They are different estimates;
the fitting cutoff does not change the wet fraction. ``nonzero_fit`` reports
sample count and fitting status, including finite bounded solutions whose
parameters should not be interpreted as an unrestricted interior fit.

The radial estimator subtracts the field mean, divides by its standard
deviation, and averages unscaled FFT power in logarithmic frequency bins. It differs from
:func:`pysteps.utils.spectral.rapsd`, which uses different binning and
normalization. Its returned arrays contain linear frequencies in cycles per
pixel and mean linear power per Fourier mode, not annulus-integrated energy.
The fit uses signed slopes, an optional prescribed break, and a continuous
piecewise-linear model in 10*log10 units. The two initial bins and frequencies
at or above 10**(-0.4) cycles per pixel are excluded by default. The fixed-break
model is linear in its coefficients and is solved by unweighted least squares.

Insufficient variance produces empty spectrum arrays. Insufficient or
unidentified fits produce explicit statuses and unavailable coefficients, not
climatological substitutions. Field statistics and spectra reject missing input
pixels; callers must apply a deliberate missing-data policy before using these
complete-grid estimators. Real inputs are converted to float64 for estimation
without being modified. Neither function clips fitted parameters.

Temporal estimation and modelling
---------------------------------

:func:`pysteps.timeseries.parametric.lagrangian_autocorrelation` accepts prepared
cascade histories with shape ``(time, level, y, x)`` and a fixed velocity array
with shape ``(2, y, x)`` in pixels per timestep. It uses successive unit-step
advection. Its default ``boundary="zero"`` fills values outside the grid with
zero after each step and correlates the whole grid. Zero is the mean anomaly
of a normalized cascade. The ``boundary="mask"`` alternative excludes pixels
that are outside the valid support of any aligned history. These two boundary
choices produce different correlation estimates.

:func:`pysteps.timeseries.parametric.ar2_correlation_time` converts observed lag
pairs into the stationary AR(2) one-sided discrete ACF sum including lag zero.
Output time units equal the supplied timestep units. Invalid and nonstationary
pairs remain unavailable. This quantity is not a forecast-skill horizon.

Reference-time estimation rescales the selected level times to one reference
wavelength and takes their geometric mean. By default it uses the first three
levels; every contributing time must be positive and finite. Use ``levels=(0,)``
to estimate from the largest level alone, as the forecast does by default.
Supply the scaling exponent separately. Use nominal wavelengths from the
cascade filter::

    from pysteps.cascade.bandpass_filters import filter_gaussian
    from pysteps.timeseries.parametric import (
        estimate_reference_correlation_time, model_cascade_correlations
    )

    bandpass = filter_gaussian((256, 256), 6, d=1 / pixel_size_km)
    wavelengths_km = 1 / bandpass["central_freqs"]
    observed_times_minutes = np.array([180.0, 70.0, 40.0, 20.0, 10.0, 5.0])
    exponent = 0.8  # illustrative temporal scaling exponent
    reference_minutes = estimate_reference_correlation_time(
        observed_times_minutes, wavelengths_km, exponent
    )
    correlations, diagnostics = model_cascade_correlations(
        reference_minutes, wavelengths_km, exponent,
        alpha=2.72, timestep=5.0, return_diagnostics=True
    )

The first band's nominal wavelength is a label for a broad low-pass response,
not an energy-weighted effective wavelength. The temporal model applies the
nonperiodic lag-2 adjustment when inverting the target timescale. Requested times
at or below one timestep use white-noise lag pairs, achieving one timestep.
Diagnostics retain requested and achieved times and identify sub-timestep limits.

The estimators leave regional parameter selection and bounds to the caller.
See :doc:`parametric_nowcast` for their use within a complete forecast and
:doc:`calibrated_generator` for generating fields from the estimated parameters.
