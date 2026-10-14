# Step 4: three-observation parametric nowcast engine

Implemented on `parametric-nowcasting`, 15 October 2026, under the
[agreed scope](parametric_nowcasting.md), using the accepted
[estimators](parametric_estimators.md) and
[calibrated stochastic generator](calibrated_generator.md).

## Endpoint

```python
from pysteps import nowcasts

ensemble, diagnostics = nowcasts.get_method("parametric")(
    rain,                         # floating ndarray (3, y, x), mm/h
    timesteps=12,
    n_ens_members=20,
    kmperpixel=1.0,
    timestep=5.0,                 # minutes, fractional values supported
    seed=42,
    return_diagnostics=True,
)
```

The result is a float64 NumPy array `(member, lead, y, x)` in mm/h, starting
one observation interval after the newest field. Supplied motion is optional;
otherwise it is estimated from the newest two transformed observations.
All preparation and estimation happen internally. Public registration is
`pysteps.nowcasts.get_method("parametric")`; it is separate from native STEPS's
existing `noise_method="parametric"` option.

The [engine](../../pysteps/nowcasts/parametric.py),
[user guide](../source/user_guide/parametric_nowcast.rst), and
[runnable gallery example](../../examples/plot_parametric_nowcast.py) describe
the settings and diagnostics. NumPy, SciPy, and existing pySTEPS components are
sufficient; internal Lucas-Kanade motion uses the existing OpenCV dependency.
No new runtime packages are required.

## Preserved scientific behavior

The engine uses dBR observations, normalized Gaussian cascades, source-compatible
Lagrangian correlation estimates, the discrete AR(2) persistence definition,
and the accepted spectral/marginal estimators. It selects ordinary or fitted
latent generation moments, applies the regional low-coverage triplet fallback,
then enforces joint spectral and persistence bounds. Raw estimates, selections,
bounds, and effective settings remain separate in diagnostics.

Each forecast advances Eulerian cascade histories by their respective one/two
timestep backward trajectories, adds normalized calibrated innovations, applies
the source global normalization across levels and space, recomposes with
innovation moments, and performs the reference marginal mapping. The resulting
rainfall is returned before boundary tapering. The transformed field is then
tapered and decomposed again for the next state. This feedback, and stochastic
inflow following zero-anomaly advection, are part of the method.

Independent member streams are spawned from a local SeedSequence. Every member
starts from observed initial states; kernels do not mutate shared observations.
Leading members agree when ensemble size increases. The old source member-state
aliasing bug is absent in the identified source revision and is explicitly
covered by the port's single-member/prefix regression test.

Lead-wise callbacks receive all members in mm/h. They can modify their copy
without affecting state. `return_output=False` avoids storing the full horizon.
Callback exceptions propagate. Diagnostics record member seeds and per-lead
generator decisions without persisting anything.

## Reference configuration and deliberate interface changes

The numerical source revision is `pysteps_param`
`94bd4c1571637a8f246ac391470e1a820d6376a6`. The paper configuration and exported
inputs come from `psteps_param_paper`
`335f7660775480d865d9771820a81923c461832a`.

Defaults are traced to `config/db/config.json` in the paper repository and
`config/db/nowcast_v2_acceptance.json` in the cases repository: six levels,
1 mm/h occurrence threshold, 20 km break, exponent 0.81, lag-pair exponent 2.72,
and the largest-level persistence reference. The marginal fitting cutoff is
1.25 dBR. Optical flow uses the source `dense_lucaskanade_v1` settings on the
latest pair. Source `StepsParameters` supplies moment selection (distance 0.5),
joint slope constraints, and reference-time bounds of 60–720 minutes.

The low-coverage fallback applies below fraction 0.05, replacing the slopes and
reference time together with `-1.45`, `-3.253958088759979`, and
`330.61752959848894` minutes. The last two are regional medians frozen in the
source `config/low_war_fallback.json`, with its original archive provenance.
All these controls are configurable in `estimation_kwargs`; generator controls
are configurable in `noise_kwargs`. This extraction does not retune them.

The later selected paper analysis uses the top three levels and exponent
0.7718612877. The standalone reference estimator supports it, and the engine
can select it explicitly through `reference_levels` and `temporal_exponent`.
It is not silently combined with the earlier evaluated forecast defaults.

Interface changes are explicit:

- Three complete chronological rain-rate arrays replace database lookups.
- Confirmed newest dry/subthreshold rainfall produces a zero ensemble; older
  wet observations do not override the newest dry condition.
- Earlier constant fields produce zero normalized anomalies, avoiding division
  by zero or normalized roundoff artifacts. Unavailable temporal estimates
  outside the low-coverage policy raise; an investigated `reference_time` prior
  can be supplied. Unavailable positive-rain marginal moments also raise.
- The post-AR fraction cutoff of 0.025 is preserved and configurable. The
  operational issue-time eligibility gate at 0.03 is left to applications.
- Results are raw float64 rain rates before NetCDF packing. The operational
  200 mm/h rejection cap is left to applications; finite output is required.
- Integer lead counts, AR(2), and fixed parameters/motion are implemented.
  Storage, NWP blending, irregular lead times, and velocity perturbations remain
  outside this first endpoint.

## Source parity and limits

[Frozen complete forecast arrays](../../pysteps/tests/data/parametric_nowcast_reference.npz)
and [metadata](../../pysteps/tests/data/parametric_nowcast_reference.json) cover
two deterministic synthetic histories, including positive low coverage and a
7.5-minute cadence. Expectations were generated using original transformation,
estimators, generation QA, calibrated generation, and `update_field` definitions.
Only operational imports are removed; numerical statements are unchanged.
The source CLI's feedback/member loop is assembled explicitly outside storage.
Tests do not need either external repository or a database.

Regenerate in the source environment, with this checkout on `PYTHONPATH`:

```bash
python pysteps/tests/data/generate_nowcast_reference.py /path/to/pysteps_param
```

The [paper-history validation report](parametric_nowcast_validation.json)
records six comparisons: three archived 256×256 histories, two seeds, two
members, and twelve five-minute leads. Two histories produce positive-rain
forecasts; the low-coverage history exercises the explicit post-AR dry policy.
The archive layout is `[current, previous, future1, ...]`; indices `[1, 0, 2]`
provide three actual chronological observations. The forecast reference time
is therefore five minutes later than the old experiment cycle. Motion is
re-estimated for this latest pair rather than reusing older archived motion.

With identical source-selected parameters and lag pairs, **all six complete
one-hour forecasts matched exactly** in this run. This isolates state evolution,
advection, calibrated innovations, marginal feedback, and member initialization.
The public call also runs both motion and parameter estimation internally. Its
selected parameters and lag pairs agree at the asserted numerical precision,
but the equivalent source nonlinear spectral regression and ported linear
least-squares solver differ in their final digits.

Histogram interpolation, wet-pixel membership, and repeated feedback amplify
those perturbations. Across the six coupled runs:

| Quantity | Largest observed difference |
| --- | ---: |
| Whole forecast RMSE | 0.004162 mm/h |
| Individual pixel rain rate | 1.406 mm/h |
| Mean field rain rate | 0.000401 mm/h |
| Wet fraction per field | 0.0001374 |
| Conditional dBR mean | 0.006208 dBR |
| Conditional dBR standard deviation | 0.004302 dBR |
| Estimated spectral slope | 0.000688 |
| Realized lag-1/lag-2 cascade correlation | 0.000229 |

The largest run had 25 changed rain classifications among 1,572,864 output
pixels; calibration candidate selections agreed in every comparison. These
results support preservation of the evaluated method on the retained cases,
while documenting why the complete estimator-to-forecast chain is not claimed
to be pixel-identical. They are port checks, not new skill scores, a replay of
all published experiment cycles, or evidence of universal regional performance.
The prior scientific evaluation remains associated with its original revision.

Reproduce without database access:

```bash
python pysteps/tests/data/compare_parametric_nowcast_paper.py \
    /path/to/pysteps_param /path/to/psteps_param_paper \
    --output /tmp/parametric_nowcast_validation.json
```

The targeted regression suite passed **226 tests** (9 excluded existing STEPS
skill/export cases, 18 warnings from existing modules). This includes 54 new
engine tests, accepted components, existing noise/probability matching/spectral/
transformation/AR tests, and existing STEPS nowcast behavior. Tests cover full
source forecasts, exact cached-advection parity for zero/constant/varying flow,
fractional cadence, independent streams and input purity, streaming equivalence,
dry/partly dry/sparse histories, safeguard choices, and invalid input/settings.
The gallery example executed successfully with internal motion and estimation.
The additional [Swiss radar tutorial](../../examples/plot_parametric_nowcast_swiss.py)
also ran with the actual MeteoSwiss example observations at 11:50, 11:55, and
12:00 UTC on 31 January 2017. A complete 300 km LV03 subdomain, aggregated to
2 km, produced a finite `(20, 6, 150, 150)` ensemble and four rendered figures:
observation, ensemble mean, four members, and 1 mm/h exceedance probabilities.
Domain metadata and probability calculations were checked. This demonstrates
the interface on Swiss data; it does not retune or validate a Swiss calibration.
The wheel built successfully; its engine, trajectory helper, and forecast
fixtures matched the checkout by SHA-256. All 154 packaged engine, estimator,
and generator tests passed from the extracted wheel outside the checkout.
Python formatting and `git diff --check` passed. These checks used Python
3.12.13, NumPy 2.5.1, and SciPy 1.18.0; the source/reference records also capture
hashes of the existing pySTEPS kernels used by both implementations.
This is a targeted test run; a full repository test suite and Sphinx build are
not claimed.

## Non-parametric contribution: assessment, not implementation

Decision, 15 October 2026: defer the non-parametric engine and discussion with
upstream maintainers until the parametric contribution is in pySTEPS. The
assessment below is retained as background; it is not current delivery work.

There is merit in a subsequent contribution, chiefly the **Eulerian stochastic
inflow treatment** evaluated in the source comparisons. Native upstream STEPS
already supports non-parametric spectral noise, observed cascade correlations,
and empirical CDF matching. Porting its entire operational wrapper would
duplicate existing functionality and bring database concerns into this scope.
See the [upstream STEPS forecast reference](https://pysteps.readthedocs.io/en/stable/generated/pysteps.nowcasts.steps.forecast.html).

The source `pysteps_inflow.py` instead moves normalized histories before AR
updates, adds innovations throughout the Eulerian domain, and avoids a second
output-advection pass. That can sustain statistically generated rain at inflow
and enable the comparison used in the paper. It also changes interpolation and
boundary evolution; it should be evaluated and exposed as an explicit method
choice. Unlike the parametric engine, it retains native STEPS estimation and AR
state, with empirical output matching outside that state.

The design implication is modest:

1. Keep fixed-flow trajectory caching and zero-anomaly sampling independent of
   rainfall parameter estimation and marginal choice. This engine does so in
   `pysteps/nowcasts/_cascade_advection.py`, with parity tests of its numerical
   behavior. The new helper remains private until a public use is justified.
2. Preserve separate state-update policies. Swapping parametric innovations for
   non-parametric noise in this engine would retain the parametric marginal
   feedback and would not reproduce the paper's non-parametric counterpart.
3. For a later upstream implementation, assess an explicit boundary/evolution
   option within native STEPS that reuses its preparation and forecast kernels.
   Avoid copying the source adapter's version-locked private method overrides;
   it was validated specifically against pySTEPS 1.21.2, whereas this checkout
   is 1.21.5.
4. A separate three-rain-field convenience interface could share validation,
   transformation metadata, motion preparation, and streaming conventions if
   useful to upstream. It need not change this parametric public interface or
   force a general framework before either method is accepted.

The non-parametric engine and native STEPS evolution are unchanged by this
contribution. Its useful boundary extension is on hold and can be raised as a
separate scientific contribution after the parametric engine is upstream.
