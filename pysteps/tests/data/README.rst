Parametric regression data
==========================

The parametric tests combine independent numerical checks with frozen results
from the original WRNZ implementation. Normal test runs read the committed JSON
and NPZ files; they need neither the external repositories nor a database.
The simulation recovery reference described below records a separate pySTEPS
end-to-end regression case.

The source-comparison JSON ``provenance`` entries identify the original revisions,
file SHA-256 hashes, and numerical-library versions. Case entries record parameters,
settings, and seeds. The NPZ files contain complete expected fields and histories.
Retain these identities when updating fixtures and explain intended numerical
changes. For comparisons with the original implementation, expected values must
continue to come from that implementation rather than being replaced with
pySTEPS results.

Fixture generation
------------------

Run these commands from the pySTEPS repository root in an environment containing
the original project's dependencies, with this checkout on ``PYTHONPATH``::

    python pysteps/tests/data/generate_parametric_reference.py \
        /path/to/pysteps_param /path/to/psteps_param_paper
    python pysteps/tests/data/generate_calibrated_reference.py \
        /path/to/pysteps_param
    python pysteps/tests/data/generate_nowcast_reference.py \
        /path/to/pysteps_param

These commands overwrite the corresponding ``*_reference.json`` and
``*_reference.npz`` files beside the scripts. They extract original numerical
definitions without importing the operational database components. Original
function names and repository paths are intentional here: they establish where
the expected results came from.

The committed references identify ``pysteps_param`` revision
``94bd4c1571637a8f246ac391470e1a820d6376a6`` and, where used,
``psteps_param_paper`` revision ``335f7660775480d865d9771820a81923c461832a``.
The generator and forecast metadata also identify settings and kernels used in
the comparisons. Compatibility with later source revisions is not assumed.

Comparisons with saved rainfall cases
-------------------------------------

The comparison scripts require the external paper repository's saved inputs.
They read those inputs and write a JSON report to the requested output path::

    python pysteps/tests/data/compare_parametric_paper_fields.py \
        /path/to/pysteps_param /path/to/psteps_param_paper \
        --output /tmp/parametric_estimators_validation.json
    python pysteps/tests/data/compare_calibrated_paper_fields.py \
        /path/to/pysteps_param /path/to/psteps_param_paper \
        --output /tmp/calibrated_generator_validation.json
    python pysteps/tests/data/compare_parametric_nowcast_paper.py \
        /path/to/pysteps_param /path/to/psteps_param_paper \
        --leads 3 --output /tmp/parametric_nowcast_validation.json

The estimator and generator scripts use ``figs/individual_field_fits/*.npz``.
The forecast script uses the saved case histories under ``figs/extreme_parameter_experiment``;
its module docstring explains the chronological selection of observations.
Reports include source identity, input hashes, settings, and numerical
differences. An existing report at the output path is overwritten.

Compare identical-parameter evolution separately from the complete chain of
estimation and generation. Equivalent spectral regression solvers can produce
small coefficient differences that histogram interpolation and forecast
feedback amplify. Test comments and assertions explain the comparison
tolerances. These checks establish numerical agreement for the specified cases;
they do not measure forecast skill or establish calibration in another region.

The current pySTEPS API and independent numerical tests define the maintained
behaviour. A deliberate method change may differ from the WRNZ references, but
must explain the difference and its validation.

Simulation parameter recovery
-----------------------------

``parametric_simulation_reference.json`` records the deterministic example in
``examples/plot_parametric_simulation.py``: a 128 by 128 grid at 1 km, a generated
initial field with seed 9, two translated copies, uniform motion of one pixel
per interval, and one member with seed 42 over twelve five-minute steps.
The reference time is 400 minutes at 256 km. The JSON records all inputs, the
source revision and example hash, numerical-library versions, and expected
recovered statistics. This reference was generated from the pySTEPS example;
it is not an additional comparison with WRNZ.

Run the complete case with::

    python -m pytest -q pysteps/tests/test_simulation_parametric.py::test_simulation_example_end_to_end_parameter_recovery

The test generates the history, calls the public ``simulate`` function, checks
the output coordinates and units, and estimates occurrence, fitted Normal
parameters, spectral slopes, and Lagrangian correlations from the returned
rainfall. All twelve fields and all ten overlapping three-field histories are
included. It does not run plotting or the example's 32-run Monte Carlo check.

The translated initialization creates a temporal transient. Expected implied
correlation times therefore describe this fixed setup, including its bias;
the recovered reference time is approximately 1001.54 minutes. The test does
not require recovery of the prescribed 400 minutes and does not define a
universal acceptable bias or replace the paper's scientific evaluation.

Regression tolerances allow two pixels in occurrence, small numerical changes
in fitted moments and slopes, and 1e-6 absolute error in mean lag correlations.
Implied times have a relative tolerance of 0.1%, since the conversion is sensitive
near unit correlation. These are fixed-seed numerical tolerances, not Monte Carlo
acceptance bounds. The test gives the exact tolerances for each quantity.

A deliberate change to the generation or estimation method should compare
its results with this baseline before updating the expected statistics.
Reproduce the generation and recovery steps documented in the example with
the stored case settings; review changes, then update the JSON and its
provenance explicitly. Normal tests only read the reference file.
