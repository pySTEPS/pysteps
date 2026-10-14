Calibrated stochastic rainfall fields
=====================================

The calibrated generator creates a NumPy rainfall-marginal field from an explicit
statistical parameter mapping. It implements the ``inverse_calibrated_v2`` method
extracted from ``pysteps_param``. It is a component of the parametric
nowcast engine; ensemble state evolution and observation preparation are separate
stages. See :ref:`sphx_glr_auto_examples_plot_calibrated_noise_generator.py`
for a radar-data example using the new estimators, explicit moment selection,
calibration diagnostics, rainfall distributions, spectra, and normalized
cascade innovations. It uses the same Swiss composite as the original
stochastic-noise tutorial on a complete subdomain.

Generating a field
------------------

Supply signed spectral slopes, rain fraction, and caller-selected Normal location
and scale in transformed rainfall units. The example values illustrate the
interface; the parametric engine estimates and selects these parameters from
the observations::

    import numpy as np
    from pysteps.noise.parametric import generate_calibrated_field

    parameters = {
        "beta_1": -2.4,
        "beta_2": -3.1,
        "rain_fraction": 0.08,
        "mean_db": 6.0,
        "stdev_db": 3.5,
    }
    field, diagnostics = generate_calibrated_field(
        (256, 256), parameters,
        kmperpixel=1.0, scale_break=15.0, threshold=0.0,
        seed=0, return_diagnostics=True,
    )
    assert field.shape == (256, 256) and np.isfinite(field).all()

``shape`` is in ``(y, x)`` order; pixel spacing and break wavelength are in km.
The threshold is in the field's transformed units, so 1 mm/h rainfall occurrence
corresponds to a 0 dBR threshold. Location and scale are typically latent Normal
parameters, but this helper consumes the active moments supplied by the caller
and performs no latent-versus-ordinary moment selection or regional slope repair.
The input mapping is not modified.

The default result is the field alone. Optional diagnostics report estimated
spectral slopes, distance, number of attempts, selected attempt, tolerance status,
target and calibrated parameters, settings, and all candidate estimates and
distances. A finite returned field does not imply that the requested spectral
tolerance was met.

Preserved reference behaviour
-----------------------------

Independent ``U(0,1)`` samples are FFT-filtered using half the signed power slopes
as amplitude exponents. Both supplied slopes receive the default additive
correction -0.5; the location and scale receive no inverse correction.

The histogram marginal mapper uses 500 bins and clips probabilities to
``[0.001, 0.999]``. It selects a wet-area cutoff using a bin's upper edge and maps
the selected sample to Normal quantiles using bin-centre interpolation. The
returned dry anomalies remain continuous. Mapping can change the realised wet
fraction and spectrum, and the clipped tails constrain extremes; occurrence and
marginal matching are approximate.

Spectra are assessed on a copy with values at or below the transformed rain
threshold replaced by the dry value. The distance is
``sqrt(sum((estimated_beta-target_beta)**2/beta_variances))`` with reference
variances ``(0.067738, 0.076405)``. These are empirical weights. Fractions at or
below 0.1 permit at most five attempts; higher fractions use one. The first
candidate with distance at or below 1 is accepted. Otherwise the closest finite
candidate is selected, retaining the first exact tie. If all spectral assessments
are unavailable, the last field is returned with an explicit failure status.

The correction, weights, acceptance threshold, retry settings, bin count, and
quantile clip can all be overridden. No reference defaults were retuned during
this port.

Random streams and dry cases
---------------------------

``seed`` creates a local NumPy random stream without changing global random state.
Alternatively, pass a caller-owned ``rng``; it advances in place and must be
independent for each ensemble member. Supply ``rng`` or ``seed``. Retry counts
affect stream consumption, so reproducibility requires matching parameters,
settings, and numerical implementation as well as the initial seed.

An explicit zero rain fraction produces a constant dry field with zero attempts
and no spectral-success claim. Positive-fraction mappings with too few distinct
wet pixels raise an error. Missing/nonfinite input and invalid parameters also
raise errors; they are not classified as dry rainfall.

Cascade innovations and marginal adjustment
------------------------------------------

The raw field retains its rainfall marginal. The parametric engine generates it
first, then obtains normalized cascade innovations through decomposition::

    from pysteps.cascade.bandpass_filters import filter_gaussian
    from pysteps.cascade.decomposition import decomposition_fft

    bandpass = filter_gaussian(field.shape, 6)
    innovations = decomposition_fft(
        field, bandpass, compute_stats=True, normalize=True
    )
    assert np.isfinite(innovations["cascade_levels"]).all()

The raw field is not guaranteed to have global mean zero and variance one.
Existing native STEPS noise-method registrations retain their perturbation
interface; the calibrated parametric engine consumes this helper directly.

The shared :func:`pysteps.postprocessing.probmatching.match_rainfall_distribution`
also supports the post-AR marginal adjustment. Its default suppresses no positive
rain fraction. The source post-AR dry policy is reproduced by explicitly passing
``min_rain_fraction=0.025``; that policy is distinct from stochastic generation.

Validation
----------

Frozen complete fields and diagnostics compare the port with original functions
on seeded square/rectangular configurations. Tests also exercise filter exponents,
physical units, histogram cutoff semantics, random-stream independence, all
candidate-selection outcomes, and the source post-AR mapping. Fixtures include
source revisions and hashes and are included in wheel package data.

Saved paper fields provide additional comparisons at two seeds. Identical target
parameters reproduce the original fields exactly in the recorded run. When the
ported estimators supply slightly different least-squares slopes, the complete
estimator-to-generator chain shows small numerical differences from histogram
interpolation. The revision-specific report under ``doc/proposals/`` records
those differences separately from identical-target generator parity. These checks
do not constitute a complete nowcast or forecast-skill evaluation.
