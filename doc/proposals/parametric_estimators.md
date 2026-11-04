# Step 2: parameter estimators and temporal-model utilities

Implemented on `parametric-nowcasting`, 14 October 2026, under the
[agreed contribution scope](parametric_nowcasting.md). The public forecast engine
will assemble these helpers internally; this stage provides the independently
callable numerical components, not the three-field forecast endpoint.

## Public helpers

| Module | Function | Purpose |
| --- | --- | --- |
| `pysteps.utils.statistics` | `compute_field_statistics` | Wet fraction, ordinary wet-pixel moments, and optional latent Normal fitting |
| `pysteps.utils.statistics` | `fit_lower_truncated_normal` | Profile-likelihood fit of the parent Normal from samples above a cutoff |
| `pysteps.utils.spectral` | `rapsd_logarithmic` | Source-compatible radial mean power and frequency arrays in logarithmic bins |
| `pysteps.utils.spectral` | `fit_power_law_spectrum` | Signed single-slope or continuous fixed-break spectral fit |
| `pysteps.timeseries.parametric` | `lagrangian_autocorrelation` | Lag-1/lag-2 estimates from supplied cascade histories and velocity |
| `pysteps.timeseries.parametric` | `ar2_correlation_time` | Stationary AR(2) discrete ACF sum including lag zero |
| `pysteps.timeseries.parametric` | `estimate_reference_correlation_time` | Geometric mean of selected scale-adjusted level times |
| `pysteps.timeseries.parametric` | `fit_ar2_correlations` | Invert a target time into an adjusted nonperiodic lag pair |
| `pysteps.timeseries.parametric` | `model_cascade_correlations` | Apply temporal scaling to nominal wavelengths, retaining resolution diagnostics |

Interfaces use NumPy arrays and ordinary Python containers. NumPy and SciPy are
the only numerical dependencies added by these helpers, and both are already
pySTEPS dependencies. The extraction has no database, xarray, pandas, or statsmodels
requirement. Fixture regeneration and the comparison against external paper fields
use the source project's environment, including xarray, to evaluate its original
functions; these are developer checks, not dependencies of the port or CI tests.

## Scientific conventions and intentional interface changes

- Field statistics and radial spectra take finite, complete fields. Invalid
  input is rejected. Dry and failed estimates remain explicit; estimates are
  not clipped or replaced with regional climatology at this stage.
- Estimation converts real inputs to float64 and does not modify them. Source
  parity is assessed on the same float64 fields; input-precision changes should
  not be confused with a change to the fitting method.
- Rain occurrence includes samples equal to the occurrence threshold. The
  denominator is the complete grid. Ordinary moments use population standard
  deviations. Latent fitting has a separate cutoff and distinct result fields.
- The latent fit preserves the source profile likelihood and bounded standardized
  cutoff interval `[-12, 12]`. Diagnostics identify insufficient/constant samples,
  fitting failure, and boundary-limited finite solutions, without changing the
  returned source-compatible location and scale.
- The spectrum is the radial mean of unscaled FFT power from a standardized
  field. Real-FFT multiplicities reproduce the full-plane mean. Historical
  logarithmic edges, including their endpoint rounding, are retained. This is
  distinct from the existing `rapsd` binning/normalization convention.
- Spectrum arrays contain linear frequency and power. Conversion to the source
  function's `10*log10` representation is explicit. The fit excludes two initial
  bins and frequencies at or above `10**(-0.4)` cycles/pixel by default. The
  breakpoint is a wavelength in units consistent with the supplied frequencies.
- A prescribed-break model is linear in its coefficients. The port solves the
  same unweighted least-squares objective with `numpy.linalg.lstsq`, instead of
  the source's nonlinear `curve_fit` call. Small optimizer/roundoff differences
  are allowed in coefficient comparisons; the spectrum itself is unchanged.
  Insufficient samples and rank-deficient models return explicit statuses.
- Lagrangian correlations retain successive unit-step advection and the
  production `boundary="zero"` convention. An explicit `boundary="mask"`
  alternative uses common finite support; it changes the estimates and is not
  the reference default. Missing input observations are still rejected.
- AR correlation times use caller-supplied time units and the full lag-zero
  contribution. Invalid/nonstationary pairs remain NaN. No AR(1) substitution is
  applied. These are persistence summaries, not skill horizons.
- Reference-time estimation defaults to levels `(0, 1, 2)`, following the selected
  paper method. Every selected time must be valid. Explicit `levels=(0,)`
  reproduces the production CLI's largest-level reference. The source research
  workflow selected the three-level method without changing the production CLI;
  the port exposes the distinction rather than silently conflating them.
- Wavelengths come from the actual cascade filter's `central_freqs`. They remain
  nominal filter labels. Scaling exponents and regional calibration settings are
  supplied explicitly; this stage does not fit or retune them.
- Inverting a timescale uses the adjusted lag pair throughout. Requested cascade
  times at or below one timestep resolve to white noise, achieving one timestep.
  Diagnostics retain requested and achieved values and sub-timestep limit flags.

## Source provenance

Extraction was checked against:

- `pysteps_param` revision `94bd4c1571637a8f246ac391470e1a820d6376a6`:
  `src/pysteps_param/rain_stats_compute.py` and `shared_utils.py`.
- `psteps_param_paper` revision `335f7660775480d865d9771820a81923c461832a`:
  `workflows/reference_timescale_experiment.py`, with the selected method described
  in `docs/selected_correlation_scaling.md`.

The frozen [numerical reference fixture](../../pysteps/tests/data/parametric_estimators_reference.json)
contains file SHA-256 hashes and numerical-library versions. Its expectations
were evaluated from the original numerical function definitions on deterministic
synthetic fields, rather than calculated by the port. Loading only those
definitions avoids package-level imports of operational/database modules.

To regenerate in an environment containing the original project's dependencies,
with this pySTEPS checkout available on `PYTHONPATH`:

```bash
python pysteps/tests/data/generate_parametric_reference.py \
    /path/to/pysteps_param /path/to/psteps_param_paper
```

Fixture updates should retain the source revision and explain any intended
numerical change. Normal test runs use the committed JSON and need neither source
repository nor a database.

## Validation

Recorded result: **90 targeted regression tests passed** under Python 3.12,
NumPy 2.5.1, and SciPy 1.18.0. Both documentation examples executed successfully.
The test run emitted nine warnings from existing transformation and
autoregression tests. This is a targeted suite, not a full pySTEPS test run or a
complete Sphinx documentation build.

The targeted regression suite covers the new helpers and existing spectral,
transformation, and autoregression functions. It uses the normal pySTEPS FMI test
inputs for existing AR tests:

```bash
PYSTEPSRC=/path/to/test/pystepsrc python -m pytest -q \
    pysteps/tests/test_parametric_estimators.py \
    pysteps/tests/test_utils_spectral.py \
    pysteps/tests/test_utils_transformation.py \
    pysteps/tests/test_timeseries_autoregression.py
```

Independent checks include full-complex-FFT radial means for even/odd square and
rectangular directional fields, orientation invariance, analytic piecewise power
laws, known truncated-Normal quantiles, explicit AR recurrence sums, reference
wavelength invariance, boundary treatments, and dry/invalid inputs. Source-parity
checks cover spectra, marginal estimates, advected correlations, timescale
inversion, and the selected paper reference calculation.

The saved paper inputs `auckland_flood`, `isolated_convection`, and `maximum_war`
were compared at 1 km resolution, a 15 km spectral break, a 1 mm/h occurrence
threshold, and a 1.25 dBR latent fitting cutoff. Spectra matched exactly in this
run; marginal estimates matched to the asserted numerical precision. Maximum
spectral coefficient differences were below `3e-7`, arising from the equivalent
linear least-squares solution. The
[comparison report](parametric_estimators_validation.json) records input hashes,
source identity, individual differences, and fit statuses.

The retained comparison script reproduces this check without database access:

```bash
python pysteps/tests/data/compare_parametric_paper_fields.py \
    /path/to/pysteps_param /path/to/psteps_param_paper \
    --output /tmp/parametric_estimators_validation.json
```

These are estimator and temporal-model checks. They do not claim that the complete
forecast engine has been ported or that forecast skill has been re-evaluated.

## Documentation and next stage

API docstrings and reference entries are supplied, along with the executable
examples in `doc/source/user_guide/parametric_estimators.rst`. The next stage is
the calibrated stochastic field generator, which will consume these estimators
and return its calibration diagnostics. Forecast-state preparation, reference
configuration assembly, generation-time parameter safeguards, and the public
three-observation forecast call remain work for the engine stage.
