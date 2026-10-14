Parametric rainfall estimators
==============================

The parametric estimators provide NumPy-based spatial statistics and temporal
model utilities extracted from ``pysteps_param``. They are the first components
of a proposed parametric nowcast engine; they do not yet implement the public
three-observation forecast endpoint.

Spatial estimation
------------------

Use a finite rainfall field and retain the threshold and dry-value convention
when transforming it. Occurrence and marginal fitting use explicitly separate
cutoffs. This example uses illustrative settings, not a complete reference
forecast configuration::

    import numpy as np
    from pysteps.utils.transformation import dB_transform
    from pysteps.utils.statistics import compute_field_statistics
    from pysteps.utils.spectral import rapsd_logarithmic, fit_power_law_spectrum

    rng = np.random.default_rng(42)
    rain = rng.lognormal(1.0, 0.8, size=(256, 256))
    rain[rng.uniform(size=rain.shape) > 0.3] = 0.0
    transformed, metadata = dB_transform(rain, threshold=1.0, zerovalue=-0.1)
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

The radial estimator preserves the source method's logarithmic bin edges,
standardization, and unscaled FFT power convention. It differs from
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
advection. Its default ``boundary="zero"`` reproduces production zero filling
after each step. The explicit ``boundary="mask"`` alternative uses common
finite support across the aligned histories and changes the estimates; it is
not a silent replacement of the evaluated convention.

:func:`pysteps.timeseries.parametric.ar2_correlation_time` converts observed lag
pairs into the stationary AR(2) one-sided discrete ACF sum including lag zero.
Output time units equal the supplied timestep units. Invalid and nonstationary
pairs remain unavailable. This quantity is not a forecast-skill horizon.

The selected paper reference-time calculation uses the geometric mean of
scale-adjusted times from the first three levels. Every contributing time must
be positive and finite. Explicit ``levels=(0,)`` reproduces the production
largest-level reference convention. The scaling exponent is supplied separately;
the helper does not recalibrate it. Use nominal wavelengths from the actual
cascade filter::

    from pysteps.cascade.bandpass_filters import filter_gaussian
    from pysteps.timeseries.parametric import (
        estimate_reference_correlation_time, model_cascade_correlations
    )

    bandpass = filter_gaussian((256, 256), 6, d=1 / pixel_size_km)
    wavelengths_km = 1 / bandpass["central_freqs"]
    observed_times_minutes = np.array([180.0, 70.0, 40.0, 20.0, 10.0, 5.0])
    exponent = 0.7718612877449645
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

Validation and provenance
-------------------------

The estimator tests compare against frozen numerical references evaluated from
the original source functions, together with independent full-FFT spectra,
known truncated-Normal samples, and explicit AR recurrence sums. Reference
metadata records source revisions and SHA-256 hashes. Tests require neither
the external repositories nor a database. The reference-generation script in
``pysteps/tests/data/generate_parametric_reference.py`` documents how to refresh
the fixture using the source project's environment.

The defaults used to configure the eventual engine will be associated with the
evaluated paper and case-study configuration. These utilities leave regional
calibration settings and operational parameter-repair policies explicit. The
scope and extraction record are maintained under ``doc/proposals/``.
