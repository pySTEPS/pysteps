# -*- coding: utf-8 -*-
"""Fit STEPS motion perturbation parameters to the output of run_vel_pert_analysis.py
and optionally plots the results. For a description of the method, see
:cite:`BPS2006`."""

import argparse
import pickle
from matplotlib import pyplot
import numpy as np
from scipy.optimize import curve_fit

description = (
    "Fit STEPS motion perturbation parameters to the results produced"
    " by run_vel_pert_analysis.py and optionally plot the results."
)

argparser = argparse.ArgumentParser(description=description)
argparser.add_argument("inputfile", type=str, help="name of the input file")
argparser.add_argument(
    "--plot",
    nargs="?",
    type=str,
    metavar="filename",
    help="plot the results and save the figure to <filename>",
)
args = argparser.parse_args()

with open(args.inputfile, "rb") as f:
    results = pickle.load(f)

f = lambda t, a, b, c: a * pow(t, b) + c

leadtimes = sorted(results.keys())

std_par = []
std_perp = []

for lt in leadtimes:
    dp_par_sum = results[lt]["dp_par_sum"]
    dp_par_sq_sum = results[lt]["dp_par_sq_sum"]
    dp_par_n = results[lt]["n_samples"]
    mu = dp_par_sum / dp_par_n

    std_par.append(
        np.sqrt((dp_par_sq_sum - 2 * mu * dp_par_sum + dp_par_n * mu ** 2) / dp_par_n)
    )

    dp_perp_sum = results[lt]["dp_perp_sum"]
    dp_perp_sq_sum = results[lt]["dp_perp_sq_sum"]
    dp_perp_n = results[lt]["n_samples"]
    mu = dp_perp_sum / dp_perp_n

    std_perp.append(
        np.sqrt(
            (dp_perp_sq_sum - 2 * mu * dp_perp_sum + dp_perp_n * mu ** 2) / dp_perp_n
        )
    )

try:
    p_par = curve_fit(f, leadtimes, std_par)[0]
    p_perp = curve_fit(f, leadtimes, std_perp)[0]
    fit_succeeded = True
    print("p_par  = %s" % str(p_par))
    print("p_perp = %s" % str(p_perp))
except RuntimeError:
    fit_succeeded = False
    print("Parameter fitting failed.")

if args.plot is not None:
    pyplot.figure()

    pyplot.scatter(leadtimes, std_par, c="r")
    t = np.linspace(0.5 * leadtimes[0], 1.025 * leadtimes[-1], 200)
    pyplot.scatter(leadtimes, std_perp, c="g")
    if fit_succeeded:
        (l1,) = pyplot.plot(t, f(t, *p_par), "r-")
        (l2,) = pyplot.plot(t, f(t, *p_perp), "g-")

    p_str_1 = lambda p: "%.2f\cdot t^{%.2f}+%.2f" % (p[0], p[1], p[2])
    p_str_2 = lambda p: "%.2f\cdot t^{%.2f}%.2f" % (p[0], p[1], p[2])
    if fit_succeeded:
        lbl = lambda p: p_str_1(p) if p[2] > 0.0 else p_str_2(p)
        pyplot.legend(
            [l1, l2],
            [
                "Parallel: $f(t)=%s$" % lbl(p_par),
                "Perpendicular: $f(t)=%s$" % lbl(p_perp),
            ],
            fontsize=12,
        )
    pyplot.xlim(0.5 * leadtimes[0], 1.025 * leadtimes[-1])
    pyplot.xlabel("Lead time (minutes)", fontsize=12)
    pyplot.ylabel("Standard deviation of differences (km/h)", fontsize=12)
    pyplot.grid(True)

    pyplot.savefig(args.plot, bbox_inches="tight")
