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
# # 7. DESI DR1 P1D with `lace_mpg`: Cobaya minimization vs native cup1d
#
# Test that the Cobaya graph reproduces the native cup1d minimization of the
# CM2026 baseline (LaCE Gaussian-process emulator `lace_mpg`):
#
# * the cosmology comes from Cobaya (CAMB through `LyaCosmology`), the emulator
#   inputs from the same cup1d mapping as native cup1d
#   (`cup1d.theory.emulator_inputs`), and the P1D from cup1d's LaCE emulator;
# * the YAML `examples/dr1_cm2026_base_evaluate.yaml` is generated from
#   `cup1d/configs/cm2026/cm2026_base.yaml`
#   (`python scripts/config_from_cup1d.py ../cup1d/configs/cm2026/cm2026_base.yaml examples/dr1_cm2026_base_evaluate.yaml`).
#
# Checks: (1) chi2 at the cup1d best fit; (2) a Cobaya minimization from the
# same initial conditions as cup1d reaches the same minimum (chi2 and star
# parameters); (3) Gauss-Newton errors agree (both from
# `cup1d.inference.curvature`, flat priors included).
#
# The cup1d reference is a `minimizer_results.npy` written by
# `cup1d/scripts/data/data_sampler.py None --stage minimize`. DESI DR1 P1D is
# unblinded: `UNBLIND = True` shows true star parameters (the blinded ones differ
# by the same offsets in both codes).

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

from cup1d.inference.analysis import Analysis
from lya_interface.fit import Fit
from lya_interface.initial_conditions import global_ic_point
from lya_interface.posterior import native_parameter_name

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "examples/dr1_cm2026_base_evaluate.yaml").is_file())
CONFIG = ROOT / "examples/dr1_cm2026_base_evaluate.yaml"
CUP1D_RESULT = Path("~/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_QMLE3/global_opt/lace_mpg/chain_1/minimizer_results.npy").expanduser()
UNBLIND = True
print("Cobaya configuration:", CONFIG)
print("cup1d minimizer result:", CUP1D_RESULT)

# %% [markdown]
# ## Native cup1d best fit

# %%
native = Analysis.from_results(CUP1D_RESULT).fitter
native_point = native.parameters_from_sampling_point(native.mle_cube)
native_star = dict(native.mle_cosmo)  # blinded
if UNBLIND:
    native_star = native.like.blind.unblind(native_star)
native.estimate_mle_errors(method="gauss_newton")  # flat priors as Gaussians (default)
native_star_errors = np.sqrt(np.diag(native.mle_cosmo_covariance))[:2]
print(f"cup1d: chi2 = {native.mle_chi2:.3f}, Delta2_star = {native_star['Delta2_star']:.4f} "
      f"+- {native_star_errors[0]:.4f}, n_star = {native_star['n_star']:.4f} +- {native_star_errors[1]:.4f}")

# %% [markdown]
# ## Cobaya model and the cup1d best fit in it
#
# Parameters map one to one (`igm_tau_eff_0` -> `tau_eff_0`, ...). The chi2 must
# agree to numerical precision (673.5442 vs 673.5443 when this notebook was written).

# %%
t0 = perf_counter()
fit = Fit(CONFIG)
names = list(fit.reference)
to_cobaya = {native_parameter_name(name): name for name in names}
cup1d_point = {to_cobaya[name]: float(value) for name, value in native_point.items()}
print(f"Model ready in {perf_counter() - t0:.1f} s; {len(names)} sampled parameters")

posterior = fit.model.logposterior(cup1d_point)
chi2_cobaya = -2 * float(np.sum(posterior.loglikes))
print(f"chi2 at the cup1d best fit: cobaya {chi2_cobaya:.4f}, cup1d {native.mle_chi2:.4f}, "
      f"difference {chi2_cobaya - native.mle_chi2:+.2e}")

t0 = perf_counter()
for _ in range(10):
    fit.model.logposterior(cup1d_point)
print(f"one evaluation: {(perf_counter() - t0) / 10:.2f} s")

# %% [markdown]
# ## Cobaya minimization
#
# From the same global initial conditions as cup1d (`USE_IC = True`) or from the
# cup1d best fit (`False`: should stay there). Both minimizers minimize the
# posterior (flat priors: the likelihood). Takes a while: about 50 parameters,
# a few thousand evaluations.

# %%
RUN_MINIMIZATION = True
USE_IC = True
MAX_EVALS = 6000
initial_point = global_ic_point(fit.info, fit.reference) if USE_IC else dict(cup1d_point)
if RUN_MINIMIZATION:
    best = fit.minimize(initial_point, max_evals=MAX_EVALS, verbose=True, report_every=200)
    print(f"success={best.success}, evaluations={best.evaluations}, chi2={best.chi2_data:.3f}: {best.message}")

# %% [markdown]
# ## Comparison

# %%
cobaya_star = dict(zip(("Delta2_star", "n_star"), fit.star_values(unblind=UNBLIND)))
fit.estimate_errors()  # Gauss-Newton, flat priors as Gaussians (default)
_, _, summary = fit.plot_star_ellipse(label="Cobaya", unblind=UNBLIND)
plt.close()
cobaya_star_errors = np.array([summary["sigma_Delta2_star"], summary["sigma_n_star"]])

print(f"{'':14s}{'cup1d':>12s}{'Cobaya':>12s}{'diff / sigma':>14s}")
print(f"{'chi2':14s}{native.mle_chi2:12.3f}{fit.best_fit.chi2_data:12.3f}")
for i, name in enumerate(("Delta2_star", "n_star")):
    diff = (cobaya_star[name] - native_star[name]) / native_star_errors[i]
    print(f"{name:14s}{native_star[name]:12.4f}{cobaya_star[name]:12.4f}{diff:+14.2f}")
    print(f"{'  sigma':14s}{native_star_errors[i]:12.4f}{cobaya_star_errors[i]:12.4f}")

# every parameter: shift between the two minima in units of the cup1d error
errors = dict(zip(native.like.free_param_names, np.sqrt(np.diag(native.mle_covariance))))
shift = {name: (fit.best_fit.point[to_cobaya[name]] - native_point[name]) / errors[name]
         for name in native_point}
worst = sorted(shift, key=lambda n: -abs(shift[n]))[:8]
print("\nlargest parameter shifts (Cobaya - cup1d) / sigma:")
for name in worst:
    print(f"  {name:20s} {shift[name]:+.2f}")

# %% [markdown]
# ## Star parameters
#
# Both best fits with their Gauss-Newton 1 and 2 sigma ellipses (contours whose
# projections are the 1D intervals).

# %%
from cup1d.postprocessing.minimizer.star import plot_star_simulations, simulation_star_params

simulations = simulation_star_params(native.like.theory.emulator)
cobaya_cov = summary["propagated"].covariance
fig = plot_star_simulations(simulations, [
    dict(label=f"cup1d ($\\chi^2$ = {native.mle_chi2:.1f})", star=native_star,
         covariance=native.mle_cosmo_covariance, color="C1"),
    dict(label=f"Cobaya ($\\chi^2$ = {fit.best_fit.chi2_data:.1f})", star=cobaya_star,
         covariance=cobaya_cov, color="C0", marker="P"),
])
plt.show()

# %%
fit.close()
