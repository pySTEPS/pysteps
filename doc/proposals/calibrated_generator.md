# Step 3: calibrated stochastic field generator

Implemented on `parametric-nowcasting` under the
[agreed contribution scope](parametric_nowcasting.md), 14 October 2026.

## Public components

- `pysteps.noise.parametric.power_law_filter`: full-FFT amplitude filter with
  signed power slopes, continuous prescribed break, and explicit physical units.
- `pysteps.noise.parametric.generate_stochastic_field`: source-compatible filtered
  uniform-noise field and histogram marginal mapping, without inverse correction.
- `pysteps.noise.parametric.generate_calibrated_field`: inverse spectral correction,
  candidate assessment and selection, and optional calibration diagnostics.
- `pysteps.postprocessing.probmatching.match_rainfall_distribution`: common
  NumPy marginal mapper for generation and the later post-AR adjustment.

The generator accepts shape `(y, x)`, a mapping containing the five active scalar
parameters (`beta_1`, `beta_2`, `rain_fraction`, `mean_db`, `stdev_db`), physical
pixel spacing and break wavelength in km, and a transformed rain threshold. It
returns a float64 NumPy field, or `(field, diagnostics)` when requested. Supplied
parameters are copied and remain unchanged. No database object, xarray container,
or new runtime dependency is required.

The generator consumes caller-selected active moments; it does not automatically
choose latent or ordinary moments or apply regional generation QA. These policies
belong to the forthcoming engine. The raw field retains its rainfall marginal;
the engine obtains normalized innovations by decomposing it with `normalize=True`.
Existing native STEPS noise registrations retain their existing interface.

## Reference algorithm and defaults

The source is `pysteps_param/src/pysteps_param/stochastic_generator.py` at revision
`94bd4c1571637a8f246ac391470e1a820d6376a6`. Its SHA-256 is
`94c491e4fd90da0c733ce65b48227b6b73dc31a8fc6a23a44e211ad36b3f9343`.
The source method identifier `inverse_calibrated_v2` is retained.

| Setting | Preserved default |
| --- | --- |
| White-noise input | Independent `U(0,1)` samples |
| Additive spectral correction | `-0.5` on both power slopes |
| Marginal inverse correction | None on active Normal location/scale |
| Histogram bins | `500` |
| Normal quantile probability clip | `[0.001, 0.999]` |
| Default dry value for assessment | `threshold - 0.1` |
| Low-fraction retry boundary | `rain_fraction <= 0.1` |
| Maximum attempts at low fraction | `5` |
| Attempts at higher fraction | `1` |
| Spectral error variances | `(0.067738, 0.076405)` |
| Accepted weighted distance | `<= 1.0` |

No calibration constants were retuned. The spectral bias, error weights, tolerance,
retry boundary/count, histogram bins, and quantile clip are configurable.

The mapper preserves the source histogram cutoff's **upper edge**, followed by
strictly-above-threshold sample selection and bin-centre interpolation to clipped
Normal quantiles. Wet mapped values may cross back below the rain threshold, and
the returned dry anomalies remain continuous. Neither exact wet fraction nor
global standardization is claimed.

Assess slopes on a copy whose values `<= threshold` are replaced by the configured
dry value. Select the first candidate meeting the distance tolerance; otherwise
select the smallest finite distance, preserving the first tie. If all assessments
fail, select the last candidate with an explicit `unavailable_spectrum` status,
infinite distance, and no success claim. The implementation retains only the
best/current candidate fields rather than all attempted grids.

The shared marginal helper defaults to no suppression of positive fractions.
The source post-AR policy, which returns dry output for target fraction `< 0.025`,
is available through the explicit `min_rain_fraction` setting. The helper preserves
valid source mappings and rejects nonfinite/degenerate input instead of inventing
a dry field after an invalid calculation.

An explicitly confirmed zero target is a documented interface extension: return
the dry value, consume no random draws, and report zero attempts without spectral
success. Invalid parameters and failed positive-fraction mappings remain errors.

## Reproducibility and validation

Randomness uses a local `default_rng(seed)` or a caller-owned NumPy Generator or
RandomState. Global NumPy random state is unchanged. Caller streams advance in
place; the engine must give each member an independent stream. Retry policy and
numerical implementation are part of reproducibility.

The frozen [reference metadata](../../pysteps/tests/data/calibrated_generator_reference.json)
and [complete fields](../../pysteps/tests/data/calibrated_generator_reference.npz)
were generated from the original numerical functions, result class, and constants.
A minimal five-scalar dataclass supplies the attributes those functions require,
without operational metadata or QA. No database is accessed. The source spectral
estimator is used when creating expectations; the port uses the new estimators.

Tests cover complete seeded fields, source diagnostics and selection decisions,
the post-AR mapping, independently calculated filter responses, physical-unit
invariance, marginal cutoff/quantile semantics, dry/error cases, and random streams.
The fixtures are explicitly included in the package manifest so installed tests
can find both these and the step-2 reference data.

To regenerate using the source project's environment and this checkout available
on `PYTHONPATH`:

```bash
python pysteps/tests/data/generate_calibrated_reference.py /path/to/pysteps_param
```

The saved paper cases `auckland_flood`, `isolated_convection`, and `maximum_war`
were compared at seeds `0` and `20260908`, with a 15 km break, 1 km pixels,
1 mm/h occurrence threshold, and a 1.25 dBR latent fitting cutoff. These checks
use **raw estimated targets**, without regional moment selection or spectral
repair, and therefore evaluate the primitives rather than production engine QA.

For identical target parameters, all six generated fields matched the source
exactly in this run. With the complete ported estimator-to-generator chain, the
largest field difference was below `4e-6` dBR. This arises from small differences
between equivalent spectral regression solvers amplified by histogram
interpolation. Candidate counts, selected attempts, tolerance outcomes, and
rain-threshold classifications agreed in all six comparisons. The comparison
uses an explicit `1e-4` dBR absolute bound for this coupled check, separately from
the tight identical-target check. The
[validation report](calibrated_generator_validation.json) records source/input
identity and individual results.

```bash
python pysteps/tests/data/compare_calibrated_paper_fields.py \
    /path/to/pysteps_param /path/to/psteps_param_paper \
    --output /tmp/calibrated_generator_validation.json
```

These are generator and estimator-chain checks, not temporal nowcast acceptance
or a forecast-skill comparison. Full ensemble evolution and generation-time
parameter safeguards remain work for the engine stage.

Validation used Python 3.12, NumPy 2.5.1, and SciPy 1.18.0. The targeted suite
passed 161 tests covering the new components and existing probability matching,
noise, spectral, transformation, and autoregression behavior (11 warnings from
existing modules). A built wheel included all reference fixtures; its 100
estimator/generator tests passed when run from the extracted wheel outside the
checkout. Both user-guide examples executed successfully. Python formatting and
`git diff --check` passed. This was not a full repository test run, and a full
Sphinx documentation build was not run because Sphinx was unavailable.

## Documentation and next stage

API docstrings and noise-module reference entries are provided, with executable
examples in `doc/source/user_guide/calibrated_generator.rst`. The next stage is
the three-observation parametric nowcast engine, assembling transformation, motion,
estimation, reference settings and parameter safeguards, calibrated innovations,
advection/AR updates, and marginal feedback within one forecast interface.
