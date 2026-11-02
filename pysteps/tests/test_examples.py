"""
Regression tests for the sphinx-gallery examples under ``examples/``.

Each example script is executed as-is and every matplotlib figure it
produces is compared, pixel by pixel, against a reference PNG stored under
``image_comparison_files/examples/<example_name>/``.

To (re)generate the baseline images after an intentional change to an
example's plots, run:

    PYSTEPS_UPDATE_BASELINE_IMAGES=1 pytest pysteps/tests/test_examples.py

and commit the updated PNGs.
"""

import os
import runpy
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.testing.compare import compare_images

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
BASELINE_DIR = Path(__file__).resolve().parent / "image_comparison_files" / "examples"

# RMS pixel value tolerance (0-255 scale) passed to
# matplotlib.testing.compare.compare_images. 0 requires an exact pixel match.
IMAGE_COMPARISON_TOLERANCE = 0

UPDATE_BASELINE_IMAGES = bool(os.environ.get("PYSTEPS_UPDATE_BASELINE_IMAGES"))

EXAMPLE_SCRIPTS = [
    "advection_correction.py",
    "anvil_nowcast.py",
    "data_transformations.py",
    "ens_kalman_filter_blended_forecast.py",
    "linda_nowcasts.py",
    "LK_buffer_mask.py",
    "optical_flow_methods_convergence.py",
    "plot_cascade_decomposition.py",
    "plot_custom_precipitation_range.py",
    "plot_ensemble_verification.py",
    "plot_extrapolation_nowcast.py",
    "plot_linear_blending.py",
    "plot_noise_generators.py",
    "plot_optical_flow.py",
    "plot_steps_nowcast.py",
    "probability_forecast.py",
    "rainfarm_downscale.py",
    "steps_blended_forecast.py",
    "thunderstorm_detection_and_tracking.py",
]


def _run_example(script_name):
    """Execute an example script and return the list of figures it created."""
    plt.close("all")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        runpy.run_path(str(EXAMPLES_DIR / script_name), run_name="__main__")
    return [plt.figure(num) for num in plt.get_fignums()]


@pytest.mark.parametrize("script_name", EXAMPLE_SCRIPTS)
def test_example_plots(script_name, tmp_path):
    example_name = Path(script_name).stem
    baseline_dir = BASELINE_DIR / example_name

    try:
        figures = _run_example(script_name)
        assert figures, f"{script_name} did not produce any matplotlib figures"

        if UPDATE_BASELINE_IMAGES:
            baseline_dir.mkdir(parents=True, exist_ok=True)
            for old_baseline in baseline_dir.glob("fig_*.png"):
                old_baseline.unlink()
            for i, fig in enumerate(figures):
                fig.savefig(baseline_dir / f"fig_{i:02d}.png")
            return

        baseline_images = sorted(baseline_dir.glob("fig_*.png"))
        assert baseline_images, (
            f"No baseline images found for {script_name} in {baseline_dir}. "
            "Generate them with PYSTEPS_UPDATE_BASELINE_IMAGES=1 pytest ..."
        )
        assert len(figures) == len(baseline_images), (
            f"{script_name} produced {len(figures)} figure(s) but "
            f"{len(baseline_images)} baseline image(s) exist in {baseline_dir}"
        )

        for i, fig in enumerate(figures):
            actual_path = tmp_path / f"fig_{i:02d}.png"
            fig.savefig(actual_path)
            baseline_path = baseline_dir / f"fig_{i:02d}.png"
            result = compare_images(
                str(baseline_path), str(actual_path), tol=IMAGE_COMPARISON_TOLERANCE
            )
            assert result is None, result
    finally:
        plt.close("all")
