# -*- coding: utf-8 -*-
"""Analyze uncertainty of motion field with increasing lead time. The analyses
are done by comparing initial motion fields to those estimated in the future.
For a description of the method, see :cite:`BPS2006`."""

import argparse
from datetime import datetime, timedelta
import pickle
import numpy as np
from scipy import linalg as la
from pysteps import io, motion
from pysteps import rcparams
from pysteps.utils import transformation

# TODO: Don't hard-code these.
num_prev_files = 9
use_precip_mask = False
R_min = 0.1

argparser = argparse.ArgumentParser(
    description="Estimate motion perturbation parameters for STEPS."
)
argparser.add_argument("startdate", type=str, help="start date (YYYYmmDDHHMM)")
argparser.add_argument("enddate", type=str, help="end date (YYYYmmDDHHMM)")
argparser.add_argument("datasource", type=str, help="data source to use")
argparser.add_argument(
    "oflow", type=str, help="optical flow method to use (darts, lucaskanade or vet)"
)
argparser.add_argument(
    "maxleadtime", type=int, help="maximum lead time for the analyses (minutes)"
)
argparser.add_argument("outfile", type=str, help="output file name")
argparser.add_argument(
    "--accum",
    nargs="?",
    type=str,
    metavar="filename",
    help="accumulate statistics to previously computed file <filename>",
)
args = argparser.parse_args()

datasource = rcparams["data_sources"][args.datasource]

startdate = datetime.strptime(args.startdate, "%Y%m%d%H%M")
enddate = datetime.strptime(args.enddate, "%Y%m%d%H%M")

importer = io.get_method(datasource["importer"], "importer")

motionfields = {}

oflow = motion.get_method(args.oflow)

# compute motion fields
# ---------------------

# TODO: This keeps all motion fields in memory during the analysis period, which
# can take a lot of memory.

curdate = startdate
while curdate <= enddate:
    try:
        fns = io.archive.find_by_date(
            curdate,
            datasource["root_path"],
            datasource["path_fmt"],
            datasource["fn_pattern"],
            datasource["fn_ext"],
            datasource["timestep"],
            num_prev_files=9,
        )
    except IOError:
        curdate += timedelta(minutes=datasource["timestep"])
        continue

    if any([fn[0] is None for fn in fns]):
        curdate += timedelta(minutes=datasource["timestep"])
        continue

    R, _, metadata = io.readers.read_timeseries(
        fns, importer, **datasource["importer_kwargs"]
    )

    # TODO: Here we assume that metadata["xpixelsize"] = metadata["ypixelsize"]
    vsf = 60.0 / datasource["timestep"] * metadata["xpixelsize"] / 1000.0

    missing_data = False
    for i in range(R.shape[0]):
        if not np.any(np.isfinite(R[i, :, :])):
            missing_data = True
            break

    if missing_data:
        curdate += timedelta(minutes=datasource["timestep"])
        continue

    R[~np.isfinite(R)] = metadata["zerovalue"]
    if use_precip_mask:
        MASK = np.any(R < R_min, axis=0)
    R = transformation.dB_transform(R)[0]

    if args.oflow == "vet":
        R_ = R[-2:, :, :]
    else:
        R_ = R

    # TODO: Allow the user to supply parameters for the optical flow.
    V = oflow(R_) * vsf
    # discard the motion field if the mean velocity is abnormally large
    if np.nanmean(np.linalg.norm(V, axis=0)) > 0.5 * R.shape[1]:
        curdate += timedelta(minutes=datasource["timestep"])
        continue

    if use_precip_mask:
        V[0, :, :][MASK] = np.nan
        V[1, :, :][MASK] = np.nan
    motionfields[curdate] = V.astype(np.float32)

    curdate += timedelta(minutes=datasource["timestep"])

# compare initial and future motion fields
# ----------------------------------------

dates = sorted(motionfields.keys())
if args.accum is None:
    results = {}
else:
    with open(args.accum, "rb") as f:
        results = pickle.load(f)

for i, date1 in enumerate(dates):
    V1 = motionfields[date1].astype(float)
    if not use_precip_mask:
        N = la.norm(V1, axis=0)
    else:
        N = np.ones(V1.shape[1:]) * np.nan
        MASK = np.isfinite(V1[0, :, :])
        N[MASK] = la.norm(V1[:, MASK], axis=0)
    V1_par = V1 / N
    V1_perp = np.stack([-V1_par[1, :, :], V1_par[0, :, :]])

    if date1 + timedelta(minutes=args.maxleadtime) > enddate:
        continue

    for date2 in dates[i + 1 :]:
        lt = (date2 - date1).total_seconds() / 60
        if lt > args.maxleadtime:
            continue

        V2 = motionfields[date2].astype(float)

        DV = V2 - V1

        DP_par = DV[0, :, :] * V1_par[0, :, :] + DV[1, :, :] * V1_par[1, :, :]
        DP_perp = DV[0, :, :] * V1_perp[0, :, :] + DV[1, :, :] * V1_perp[1, :, :]

        if not lt in results.keys():
            results[lt] = {}
            results[lt]["dp_par_sum"] = 0.0
            results[lt]["dp_par_sq_sum"] = 0.0
            results[lt]["dp_perp_sum"] = 0.0
            results[lt]["dp_perp_sq_sum"] = 0.0
            results[lt]["n_samples"] = 0

        if use_precip_mask:
            MASK = np.logical_and(np.isfinite(V1[0, :, :]), np.isfinite(V2[0, :, :]))
            DP_par = DP_par[MASK]
            DP_perp = DP_perp[MASK]
            n_samples = np.sum(MASK)
        else:
            n_samples = DP_par.size

        results[lt]["dp_par_sum"] += np.sum(DP_par)
        results[lt]["dp_par_sq_sum"] += np.sum(DP_par**2)
        results[lt]["dp_perp_sum"] += np.sum(DP_perp)
        results[lt]["dp_perp_sq_sum"] += np.sum(DP_perp**2)
        results[lt]["n_samples"] += n_samples

with open("%s" % args.outfile, "wb") as f:
    pickle.dump(results, f)
