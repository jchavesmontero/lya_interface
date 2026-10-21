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
# # Joint DESI DR1 P1D (forest_mpg_gp) and DESI DR2 Lyα×Lyα BAO (Vega)
#
# One Cobaya graph with a shared cosmology (CAMB), IGM histories and ForestFlow
# `forest_mpg_gp` emulator feeds two data likelihoods:
#
# - **DR1 P1D**: the CM2026 baseline with `forest_mpg_gp` (cup1d is only the backend).
# - **DR2 BAO**: Vega on the DR2 Lyα×Lyα correlation from `data_bao/dr2`, with the
#   Lyα×Lyα block of the smoothed global covariance, fitted from `RMIN` = 30 Mpc/h
#   (DR2 baseline) or 10 Mpc/h (Kaiser metals only) to 180 Mpc/h. ForestFlow
#   supplies the Lyα bias, β and small-scale nonlinear correction at the BAO
#   effective redshift, so `bias_LYA` and `beta_LYA` are not free.
#
# **Metals** (`METALS` below):
#
# - `"kaiser_fixed_beta"` (default): the line-of-sight Kaiser silicon model of
#   `06_dr1_p1d_forestmpggp_kaiser_minimization` in both likelihoods. The silicon
#   bias and decorrelation histories, and the SiII×SiII amplitude factor and
#   decorrelation, are single `metal_` parameters used by both likelihoods; Vega
#   takes their values at z_eff. The silicon betas are fixed at 0.5, as in notebook 06.
# - `"kaiser_free_beta"`: the same, with `beta_SiIII` and `beta_SiII` free (one
#   node each), shared by P1D and BAO.
# - `"templates"`: CM2026 silicon templates in P1D and four independent Vega
#   silicon biases (not a consistent metal model).
#
# With Kaiser metals the fit starts from the Kaiser global initial conditions
# (`cup1d/data/ics/DESIY1_QMLE3_forest_mpg_gp_kaiser_ic_global.npy`, the best fit of
# notebook 06). The same file sets Vega's silicon amplitude exponents
# `alpha_<line> = d ln|b| / d ln(1+z)` of the cup1d bias histories at z_eff
# (`metal_alpha_reference`). They are frozen when Vega is built.
#
# With `START_FROM`, the fit instead starts from the best fit saved by a previous
# run (`fit.save`), and that best fit also sets the exponents. A variation that has
# already been saved resumes from its own fit. Otherwise, `RMIN = 10` starts from the
# saved `RMIN = 30` fit with the same metals, so the 10 Mpc/h fit continues from the
# 30 Mpc/h one.
#
# `FIXED_COSMO = True` (`"kaiser_fixed_beta"` only) fixes the cosmology at
# Planck18_noBAO (`As`, `ns`, and `ap` = `at` = 1, since it is the cosmology of the
# Vega template), the filtering scale at the `mpg_central` fiducial
# (`igm_kF_kms_*` = 1) and `p1d_R_coeff_*` at the `RMIN = 10` best fit
# (`examples/*_fixcosmo.yaml`). Until it has been saved, it starts from the
# free-cosmology fit with the same `RMIN`, and the triangle shows BAO nuisances instead of (Δ²_star, n_star, α_∥, α_⊥).
#
# The BAO nuisances (`xi3d_*`) are those sampled in the DR2 baseline that enter
# Lyα×Lyα, with its priors (`examples/joint_dr1p1d_forestmpggp_dr2bao_lya[_kaiser_*].yaml`).
# Prefixes: `igm_` IGM (shared), `metal_` Kaiser silicon (shared), `p1d_` P1D-only
# nuisances, `xi3d_` ξ3D-only nuisances; `ap`, `at` and the cosmology keep their names.
# The IGM coefficients have flat priors spanning the `mpg` simulations (as in cup1d).
# After each minimization, the parameters close to a prior bound are listed.
# The P1D–BAO cross-covariance is assumed zero. Start with a fresh kernel and run in order.

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
from lya_interface.initial_conditions import default_ic_path, global_ic_point
from lya_interface.names import translate_legacy
from lya_interface.diagnostics import (
    format_goodness_of_fit, goodness_of_fit, plot_point_with_residuals, vega_component,
)

# Metal model shared by P1D and Lya x Lya: "kaiser_fixed_beta", "kaiser_free_beta"
# or "templates" (CM2026 templates in P1D, free Vega silicon biases).
METALS = "kaiser_fixed_beta"
# Minimum separation of the Lya x Lya fit in Mpc/h: 30 (DR2 baseline) or 10 (Kaiser metals).
RMIN = 10
BASE_NAME = {
    "kaiser_fixed_beta": "joint_dr1p1d_forestmpggp_dr2bao_lya_kaiser_fixed_beta",
    "kaiser_free_beta": "joint_dr1p1d_forestmpggp_dr2bao_lya_kaiser_free_beta",
    "templates": "joint_dr1p1d_forestmpggp_dr2bao_lya",
}[METALS]
if RMIN != 30 and METALS == "templates":
    raise ValueError("RMIN != 30 is only configured for the Kaiser metal models")
NAME = BASE_NAME if RMIN == 30 else f"{BASE_NAME}_rmin{RMIN}"
# Saved fit (fit.save) whose best fit is the starting point and sets Vega's silicon
# alpha_<line>; None: global initial conditions. Default for RMIN != 30: the RMIN = 30
# fit with the same metals.
START_FROM = None if RMIN == 30 else f"minimizer_results_{BASE_NAME.removeprefix('joint_')}.npy"
# True: fixed Planck18_noBAO cosmology, igm_kF_kms_* and p1d_R_coeff_* (kaiser_fixed_beta only).
FIXED_COSMO = True
if FIXED_COSMO:
    if METALS != "kaiser_fixed_beta":
        raise ValueError("FIXED_COSMO is only configured for kaiser_fixed_beta")
    # Start from the free-cosmology fit with the same metals and RMIN.
    START_FROM = f"minimizer_results_{NAME.removeprefix('joint_')}.npy"
    NAME = f"{NAME}_fixcosmo"
# Global initial conditions: the Kaiser ones (best fit of notebook 06) for Kaiser metals.
IC_VARIATION = None if METALS == "templates" else "kaiser"
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
# A variation that has already been saved resumes from its own best fit.
if Path(RESULTS_FILE).is_file():
    START_FROM = RESULTS_FILE
# DESI DR1 P1D has been unblinded: set True to show star parameters without the
# native blinding offsets.
UNBLIND = False
print("Configuration:", CONFIG)
print("Start from:", START_FROM)

# %% [markdown]
# ## Build the joint model
#
# Loading Vega reads the DR2 Lyα×Lyα correlation and distortion matrix and the
# Lyα×Lyα block of the global covariance (~50 MB). With Kaiser metals, the silicon
# `alpha_<line>` exponents passed to Vega are printed: those of the best fit of
# `START_FROM`, or of the global initial conditions without it.

# %%
t0 = perf_counter()
fit = Fit(CONFIG, metal_alpha_reference=START_FROM)
info, model, reference = fit.info, fit.model, fit.reference
vega = vega_component(model)
print(f"Model ready in {perf_counter() - t0:.1f} s; {len(reference)} sampled parameters "
      f"({len(vega.mapping)} BAO, {len(vega.metal_parameters)} shared metal)")
if vega.metal_alpha:
    display({"Vega alpha_<line>": vega.metal_alpha})


def plot_model(point, title):
    """Plot the P1D model at every redshift and the BAO wedges of every correlation."""
    plot_point_with_residuals(model, point, title=f"DR1 P1D: {title}")
    plt.show()
    fit.plot_vega(point, title=f"DR2 BAO: {title}")
    plt.show()


# %% [markdown]
# ## Initial point
#
# IGM and P1D nuisances (including the shared silicon parameters) start from the
# global initial conditions: the Kaiser ones of notebook 06 for Kaiser metals, the
# `forest_mpg_gp` ones of notebook 05 for templates. Values are interpolated at the
# redshift nodes of every coefficient. With `"kaiser_free_beta"` the silicon betas
# are not in the IC file and start at their references (0.5). `As` and `ns` are
# taken from the file when it stores them. The BAO nuisances start from their
# priors: the Gaussian prior means, or the DR2 baseline values for flat priors.
#
# With `START_FROM`, every parameter stored in that fit starts from its best fit
# (only the point is read; the fit's model is not rebuilt). Parameters it lacks
# keep the values above.

# %%
ic_file = default_ic_path(info, IC_VARIATION)
print("Initial conditions:", ic_file)
initial_point = global_ic_point(info, reference, ic_file)
if START_FROM is not None:
    # Fits saved before the metal_/xi3d_ prefixes use the old names.
    start = translate_legacy(np.load(START_FROM, allow_pickle=True).item()["best_fit"]["point"])
    initial_point.update({name: start[name] for name in initial_point if name in start})
    print("Start: best fit of", START_FROM)
    missing = sorted(set(initial_point) - set(start))
    if missing:
        print("Not in that fit, kept from the initial conditions:", missing)
# Sampled Vega and shared metal parameters (fixed ones, e.g. ap/at with FIXED_COSMO, are not in the point).
display({name: initial_point[name] for name in vega.public_parameters if name in initial_point})
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
    fit.near_bounds()
    if vega.metal_alpha:
        # Frozen at START_FROM: once saved, rerunning this variation updates them; repeat until they agree.
        display({"Vega alpha_<line> (used)": vega.metal_alpha,
                 "alpha_<line> of this fit": vega.metal_alpha_at(best_fit.point)})

# %%
point = fit.best_fit.point if fit.best_fit is not None else initial_point
for name, (low, high) in zip(model.parameterization.sampled_params(), model.prior.bounds()):
    print(f"{name:26s} {point[name]: .6g}   [{low:.6g}, {high:.6g}]")


# %%
initial_point =  best_fit.point

# %%
plot_model(initial_point, "initial point")

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
TRIANGLE_PARAMETERS = (("xi3d_bias_hcd", "xi3d_beta_hcd", "xi3d_L0_hcd", "xi3d_bias_CIV_eff") if FIXED_COSMO
                       else ("Delta2star", "nstar", "ap", "at"))
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
    fit.near_bounds()
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
