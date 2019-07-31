# -*- coding: utf-8 -*-

import datetime
import pytest
import numpy as np
from pysteps import io, motion, nowcasts, rcparams, utils, verification


def _import_mch_gif(prv, nxt):

    date = datetime.datetime.strptime("201505151630", "%Y%m%d%H%M")
    data_source = rcparams.data_sources["mch"]

    # Load data source config
    root_path = data_source["root_path"]
    path_fmt = data_source["path_fmt"]
    fn_pattern = data_source["fn_pattern"]
    fn_ext = data_source["fn_ext"]
    importer_name = data_source["importer"]
    importer_kwargs = data_source["importer_kwargs"]
    timestep = data_source["timestep"]

    # Find the input files from the archive
    fns = io.archive.find_by_date(
        date,
        root_path,
        path_fmt,
        fn_pattern,
        fn_ext,
        timestep=timestep,
        num_prev_files=prv,
        num_next_files=nxt,
    )

    # Read the radar composites
    importer = io.get_method(importer_name, "importer")
    R, _, metadata = io.read_timeseries(fns, importer, **importer_kwargs)

    # Convert to rain rate
    R, metadata = utils.conversion.to_rainrate(R, metadata)

    # Upscale data to 2 km
    R, metadata = utils.dimension.aggregate_fields_space(R, metadata, 2000)

    # Log-transform the data to unit of dBR, set the threshold to 0.1 mm/h,
    # set the fill value to -15 dBR
    R, metadata = utils.transformation.dB_transform(
        R, metadata, threshold=0.1, zerovalue=-15.0
    )

    # Set missing values with the fill value
    R[~np.isfinite(R)] = -15.0

    return R, metadata


steps_arg_names = (
    "n_ens_members",
    "n_cascade_levels",
    "ar_order",
    "mask_method",
    "probmatching_method",
    "max_crps",
)

steps_arg_values = [
    (5, 6, 2, None, None, 1.51),
    (5, 6, 2, "incremental", None, 6.38),
    (5, 6, 2, "sprog", None, 7.35),
    (5, 6, 2, "obs", None, 7.36),
    (5, 6, 2, None, "cdf", 0.66),
    (5, 6, 2, None, "mean", 1.55),
]


@pytest.mark.parametrize(steps_arg_names, steps_arg_values)
def test_steps(
    n_ens_members,
    n_cascade_levels,
    ar_order,
    mask_method,
    probmatching_method,
    max_crps,
):
    """Tests STEPS nowcast."""
    # inputs
    R, metadata = _import_mch_gif(2, 0)
    R_o = _import_mch_gif(0, 3)[0][1:, :, :]
    # optical flow
    of_method = motion.get_method("LK")
    V = of_method(R)
    # nowcast
    nowcast_method = nowcasts.get_method("steps")
    num_timesteps = 1
    R_f = nowcast_method(
        R,
        V,
        n_timesteps=3,
        R_thr=metadata["threshold"],
        kmperpixel=2.0,
        timestep=metadata["accutime"],
        seed=42,
        n_ens_members=n_ens_members,
        n_cascade_levels=n_cascade_levels,
        ar_order=ar_order,
        mask_method=mask_method,
        probmatching_method=probmatching_method,
    )
    # result
    result = verification.probscores.CRPS(R_f[-1], R_o[-1])
    assert result < max_crps
