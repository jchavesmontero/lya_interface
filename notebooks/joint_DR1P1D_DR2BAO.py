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
# # Joint DESI DR1 P1D (forest_mpg_gp) and DESI DR2 Lyα BAO (Vega)
#
# One Cobaya graph with a shared cosmology (CAMB), IGM histories and ForestFlow
# `forest_mpg_gp` emulator feeds two data likelihoods:
#
# - **DR1 P1D**: the CM2026 baseline of `05_dr1_p1d_forestmpggp_minimization`
#   (cup1d is only the backend).
# - **DR2 BAO**: Vega on the DR2 correlations from `data_bao/dr2`. With
#   `LYA_ONLY = True` (default), only Lyα×Lyα with the Lyα×Lyα block of the
#   smoothed global covariance; otherwise all four (Lyα×Lyα, Lyα×Lyβ, Lyα×QSO,
#   Lyβ×QSO), which needs much more memory. ForestFlow
#   supplies the Lyα bias, β and small-scale nonlinear correction at the BAO
#   effective redshift, so `bias_LYA` and `beta_LYA` are not free.
#
# The BAO nuisances are those sampled in the DR2 baseline that enter the chosen
# correlations, with its priors (`examples/joint_dr1p1d_forestmpggp_dr2bao[_lya].yaml`). The P1D–BAO
# cross-covariance is assumed zero. Start with a fresh kernel and run in order.

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
    format_goodness_of_fit, goodness_of_fit, plot_point_with_residuals, vega_component,
)

# True: only the Lya x Lya correlation, with the Lya x Lya block of the DR2
# covariance (fits in a laptop). False: all four DR2 correlations.
LYA_ONLY = True
NAME = "joint_dr1p1d_forestmpggp_dr2bao" + ("_lya" if LYA_ONLY else "")
ROOT = next(
    (p for p in [Path.cwd(), *Path.cwd().parents]
     if (p / f"examples/{NAME}.yaml").is_file()),
    None,
)
if ROOT is None:
    import lya_interface
    ROOT = Path(lya_interface.__file__).resolve().parents[2]
CONFIG = ROOT / f"examples/{NAME}.yaml"
RESULTS_FILE = f"minimizer_results_{NAME.removeprefix('joint_')}.npy"
# DESI DR1 P1D has been unblinded: set True to show star parameters without the
# native blinding offsets.
UNBLIND = False
print("Configuration:", CONFIG)

# %% [markdown]
# ## Build the joint model
#
# Loading Vega reads the DR2 correlations and distortion matrices and the global
# covariance (~1.8 GB for all four correlations, 50 MB for Lyα×Lyα only).

# %%
t0 = perf_counter()
fit = Fit(CONFIG)
info, model, reference = fit.info, fit.model, fit.reference
vega = vega_component(model)
print(f"Model ready in {perf_counter() - t0:.1f} s; {len(reference)} sampled parameters "
      f"({len(vega.mapping)} BAO)")


def plot_model(point, title):
    """Plot the P1D model at every redshift and the BAO wedges of every correlation."""
    plot_point_with_residuals(model, point, title=f"DR1 P1D: {title}")
    plt.show()
    fit.plot_vega(point, title=f"DR2 BAO: {title}")
    plt.show()


# %% [markdown]
# ## Initial point
#
# IGM and P1D nuisances start from the `forest_mpg_gp` global initial conditions,
# as in notebook 05; `As` and `ns` keep their fiducial values. The BAO
# nuisances start from their priors: the Gaussian prior means, or the DR2
# baseline values for flat priors.

# %%
initial_point = global_ic_point(info, reference)
display({name: initial_point[name] for name in vega.mapping})
print(format_goodness_of_fit(goodness_of_fit(model, initial_point), multiline=True))
plot_model(initial_point, "initial point")

# %% [markdown]
# ## Joint minimization
#
# Nelder-Mead on the sum of both data terms and the priors, within the prior
# bounds (the Gaussian BAO priors are truncated to Vega's ranges, so every bound
# is finite). It stops when `-2logL_total` improves by less than 0.1 in 200
# evaluations, or at `MAX_EVALS`.

# %%
RUN_MINIMIZATION = True
MAX_EVALS = 3000
best_fit = fit.best_fit
if RUN_MINIMIZATION:
    best_fit = fit.minimize(initial_point, max_evals=MAX_EVALS, verbose=True, report_every=100)
    display({
        "success": best_fit.success,
        "message": best_fit.message,
        "evaluations": best_fit.evaluations,
        "minus2_loglike_total": best_fit.minus2_loglike_total,
    })
    if not best_fit.success:
        print("WARNING: minimization did not converge; do not call this a best fit. "
              "Rerun with initial_point = dict(best_fit.point) to continue.")

# %% [markdown]
# ## Best-fitting model

# %%
if best_fit is None:
    print("Run the minimization cell first.")
else:
    print(format_goodness_of_fit(goodness_of_fit(model, best_fit.point), multiline=True))
    plot_model(best_fit.point, "best fit")

# %% [markdown]
# ## Local errors and the (Δ²_star, n_star, α_∥, α_⊥) triangle
#
# Gauss-Newton curvature of both data terms plus the priors at the minimum (a
# local Gaussian approximation, not a chain), propagated to the star parameters
# and shown with the BAO dilations `ap` (α_∥) and `at` (α_⊥). Unless
# `UNBLIND = True`, the star parameters carry the native DESI blinding offsets
# and are labelled "(blinded)".

# %%
RUN_LOCAL_ERRORS = True
if best_fit is not None and RUN_LOCAL_ERRORS:
    fit.estimate_errors(method="gauss_newton")
    display(fit.local_errors.errors)

# %%
TRIANGLE_PARAMETERS = ("Delta2star", "nstar", "ap", "at")
if best_fit is None:
    print("Run the minimization cell first.")
else:
    fig, axes, triangle = fit.plot_triangle(TRIANGLE_PARAMETERS, unblind=UNBLIND)
    plt.show()
    display({"values": triangle["values"], "errors": triangle["errors"]})

# %% [markdown]
# ## Save the minimization results
#
# Writes the configuration path and digests, the initial and best-fit points and
# the local errors (if computed). Only after a fit run here.

# %%
if best_fit is not None and RUN_MINIMIZATION:
    fit.save(RESULTS_FILE)
    print(f"Saved {RESULTS_FILE}")

# %% [markdown]
# ## Restore a saved fit
#
# Rebuilds the joint model from the recorded configuration and restores the best
# fit and local errors without refitting (warning if the configuration or its
# assets changed). The current model is closed first.

# %%
RESTORE = True
if RESTORE and Path(RESULTS_FILE).is_file():
    fit.close()
    fit = Fit.from_results(RESULTS_FILE)
    info, model, reference = fit.info, fit.model, fit.reference
    vega = vega_component(model)
    best_fit = fit.best_fit
    print(format_goodness_of_fit(best_fit.statistics, multiline=True))
    display({"success": best_fit.success, "message": best_fit.message})
elif RESTORE:
    print(f"No results file {RESULTS_FILE}; run the minimization and save it first.")

# %% [markdown]
# ## Restored best-fitting model

# %%
if fit.best_fit is not None:
    plot_model(fit.best_fit.point, "restored best fit")

# %% [markdown]
# ## Restored contours

# %%
if fit.best_fit is not None:
    fig, axes, triangle = fit.plot_triangle(TRIANGLE_PARAMETERS, unblind=UNBLIND)
    plt.show()
    display({"values": triangle["values"], "errors": triangle["errors"]})

# %%
fit.close()
