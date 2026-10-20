# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 5. Minimization fit of DESI DR1 P1D with `forest_mpg_gp`
#
# Fit the DESI DR1 (QMLE3) P1D with the CM2026 baseline using the ForestFlow
# Gaussian-process emulator `forest_mpg_gp`. Everything goes through the
# `lya_interface` Cobaya graph (CAMB, cosmology snapshot, IGM histories,
# ForestFlow, cup1d likelihood); cup1d is only the backend underneath, and no
# native `Analysis` is built.
#
# The configuration mirrors `cup1d/configs/desi_dr1_tests/cm2026_base_forestmpggp.yaml`:
# rebinning factor 16 and `As`/`ns` priors spanning the training mean +- 2 sigma.
# It needs the calibrated covariance `ForestFlow/data/covariance/l1O_cov_forest_mpg_gp.npy`,
# which is never silently disabled. Native DESI blinding is preserved for the
# star-parameter diagnostics; do not unblind fitted results here.
#
# This notebook is the P1D-only starting point for the later DR1 P1D + DR2 BAO
# combination: that notebook swaps the YAML for `examples/joint_desi_dr1_dr2_bao.yaml`
# and keeps the same minimization workflow. Start with a fresh kernel and run in order.

# %%
# %load_ext autoreload
# %autoreload 2


import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pathlib import Path
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from lya_interface.fit import Fit
from lya_interface.initial_conditions import global_ic_point
from lya_interface.diagnostics import (
    evaluate_point, format_goodness_of_fit,
    goodness_of_fit, plot_point_with_residuals,
)

ROOT = next(
    (p for p in [Path.cwd(), *Path.cwd().parents]
     if (p / "examples/cup1d_forestmpggp_evaluate.yaml").is_file()),
    None,
)
if ROOT is None:
    import lya_interface
    ROOT = Path(lya_interface.__file__).resolve().parents[2]
CONFIG = ROOT / "examples/cup1d_forestmpggp_evaluate.yaml"
# DESI DR1 P1D has been unblinded: set True to show star parameters without the
# native blinding offsets. Keep False for blinded data sets.
UNBLIND = False
print("Configuration:", CONFIG)

# %% [markdown]
# ## Load the configuration and build the model
#
# The YAML owns fixed values, sampled-parameter priors, projection settings and
# data selection. The starting point is the native fiducial reference of every
# sampled parameter.

# %%
# `Fit` owns the configuration and the Cobaya model, like cup1d's `Analysis`. It
# drops the legacy derived columns (star parameters are computed explicitly below)
# and requires the calibrated covariance.
t0 = perf_counter()
fit = Fit(CONFIG)
info, model, reference = fit.info, fit.model, fit.reference
print(f"Model ready in {perf_counter() - t0:.1f} s; {len(reference)} sampled parameters")

# %% [markdown]
# ## Initial point
#
# Start from the global initial conditions fitted with `forest_mpg_gp`
# (`cup1d/data/ics/DESIY1_QMLE3_forest_mpg_gp_ic_global.npy`), interpolated at the
# redshift nodes of every coefficient, as native cup1d does. `As` and `ns` are not
# stored in that file and keep their fiducial values. Set `USE_IC = False` to start
# from the fiducial references instead.
#
# The first evaluation includes CAMB, the GP emulator and projection. The chi2 is
# the P1D data term with its correlated covariance, without prior penalties.

# %%
USE_IC = True
initial_point = global_ic_point(info, reference) if USE_IC else dict(reference)
outside = {
    name: value for name, value in initial_point.items()
    if not info["params"][name]["prior"]["min"] <= value <= info["params"][name]["prior"]["max"]
}
if outside:
    print("WARNING: initial values outside the prior bounds:", outside)
t0 = perf_counter()
result, _ = evaluate_point(model, initial_point)
print(f"First evaluation: {perf_counter() - t0:.1f} s")
display({"chi2_P1D": result.chi2_data, "ndata": result.ndata})
fig, axes = plot_point_with_residuals(model, initial_point, title="DR1 P1D + forest_mpg_gp: initial point")

# %%
t0 = perf_counter()
nn = 50
for ii in range(nn):
    result, _ = evaluate_point(model, initial_point)
print(f"1 evaluation: {(perf_counter() - t0)/nn:.2f} s")

# %% [markdown]
# ## Minimization
#
# Derivative-free Nelder-Mead in the unit cube of the sampled parameters, within
# the YAML prior bounds. No chains are written. A budget-limited or
# non-converged result is not a best fit. To skip the fit and work on a saved
# one, set `RUN_MINIMIZATION = False` and run the restore cell near the end.

# %%
RUN_MINIMIZATION = True
MAX_EVALS = 3000
best_fit = fit.best_fit
if RUN_MINIMIZATION:
    best_fit = fit.minimize(
        initial_point, max_evals=MAX_EVALS, verbose=True, report_every=100
    )
    display({
        "success": best_fit.success,
        "message": best_fit.message,
        "evaluations": best_fit.evaluations,
        "chi2_P1D": best_fit.chi2_data,
        "minus2_loglike_total": best_fit.minus2_loglike_total,
    })
    if not best_fit.success:
        print("WARNING: minimization did not converge; do not call this a best fit. "
              "Rerun with initial_point = dict(best_fit.point) to continue.")

# %% [markdown]
# ## Best-fit model and goodness of fit
#
# Recomputed at the fitted point, not taken from the optimizer's last trial.
# Degrees of freedom are `N_data - N_sampled`.

# %%
if best_fit is None:
    print("Set RUN_MINIMIZATION = True and run the minimization cell first.")
else:
    stats = goodness_of_fit(model, best_fit.point)
    print(format_goodness_of_fit(stats, multiline=True))
    fig, axes = plot_point_with_residuals(model, best_fit.point, title="DR1 P1D + forest_mpg_gp: best fit")
    plt.show()

# %% [markdown]
# ## Local errors and star parameters
#
# Newton curvature errors at the minimum (local Gaussian approximation, including
# priors; not a chain). Unless `UNBLIND = True`, `Delta2_star` and `n_star` carry
# the native DESI blinding offsets.

# %%
RUN_LOCAL_ERRORS = True
local_errors = fit.local_errors
if best_fit is not None and RUN_LOCAL_ERRORS:
    local_errors = fit.estimate_errors(method="gauss_newton")
    display(local_errors.errors)

# %%
if best_fit is not None:
    suffix = "" if UNBLIND else " (blinded)"
    display(dict(zip(("Delta2_star" + suffix, "n_star" + suffix), fit.star_values(unblind=UNBLIND))))

# %% [markdown]
# ## Error ellipse in the (n_star, Delta2_star) plane
#
# The local parameter covariance is propagated to the compressed parameters
# through finite-difference derivatives, keeping all correlations. The cross is
# the minimum; the ellipses are the joint 68% and 95% regions of the local
# Gaussian approximation. Unless `UNBLIND = True`, values include the native DESI
# blinding offsets and the legend and axis labels say "(blinded)". The
# local errors are computed here if the cell above was not run. To overlay several
# fits, pass the same axes: `fit_b.plot_star_ellipse(ax=ax, label="b")`.

# %%
if best_fit is None:
    print("Run or restore a best fit first.")
else:
    fig, ax, star_summary = fit.plot_star_ellipse(unblind=UNBLIND)
    plt.show()
    display({k: v for k, v in star_summary.items() if k != "propagated"})
    local_errors = fit.local_errors

# %% [markdown]
# ## Save the minimization results
#
# Writes one small `.npy` next to the notebook, only when a fit was run here: the
# configuration path and digests of it and its assets, the initial and best-fit
# points with the fit summary, and the local errors if computed. The best-fit
# point is also the warm start for the joint P1D + BAO notebook
# (`Fit.from_results(RESULTS_FILE).best_fit.point`).

# %%
RESULTS_FILE = "minimizer_results_forestmpggp.npy"
if best_fit is not None and RUN_MINIMIZATION:
    fit.save(RESULTS_FILE)
    print(f"Saved {RESULTS_FILE}")

# %% [markdown]
# ## Restore a saved fit
#
# The counterpart of cup1d's `Analysis.from_results`: rebuild the model from the
# recorded configuration and restore the best fit and local errors, without
# refitting. It warns if the configuration had to be found elsewhere, if it or its
# assets changed since the fit, or if the recomputed chi2 differs from the saved
# one. The current model is closed first, so only one model is in memory. After
# restoring, the plotting, star-parameter and ellipse cells above can be rerun, or
# directly `fit.plot_star_ellipse()`.

# %%
RESTORE = True
UNBLIND = True
if RESTORE and Path(RESULTS_FILE).is_file():
    fit.close()
    fit = Fit.from_results(RESULTS_FILE)
    info, model, reference = fit.info, fit.model, fit.reference
    best_fit, local_errors = fit.best_fit, fit.local_errors
    print(format_goodness_of_fit(best_fit.statistics, multiline=True))
    display({"success": best_fit.success, "message": best_fit.message,
             "local_errors": None if local_errors is None else local_errors.errors})
    fig, ax, star_summary = fit.plot_star_ellipse(unblind=UNBLIND)
    plt.show()
elif RESTORE:
    print(f"No results file {RESULTS_FILE}; run the minimization and save it first.")

# %%
fit.close()
