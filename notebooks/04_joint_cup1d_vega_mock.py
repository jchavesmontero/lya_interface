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
# # 4. Joint cup1d DESI DR1 and Vega Y3-mock evaluation
#
# This notebook is an interactive integration check, not a consistent joint
# cosmological analysis: it combines DESI DR1 P1D with Vega's Y3 mock, which
# need not prefer the same parameters.  One Cobaya graph supplies the shared
# cosmology, interface-owned IGM histories, and ForestFlow Arinyo coefficients.
# cup1d receives projected P1D; Vega receives only the bias-free nonlinear
# correction on its fixed template at `z_eff=2.3`.
#
# The exploratory fit varies `As`, `ns`, four mean-flux coefficients,
# `ap`, `at`, one cup1d metal nuisance, one cup1d HCD nuisance, and Vega's
# HCD parameters. All other required cup1d coefficients are frozen at their
# native references. The Vega template is fixed and P1D--Vega cross-covariance
# is assumed zero. No chains, emulator assets, covariances, or mock data are
# written by this notebook.

# %%
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from copy import deepcopy
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from cobaya.model import get_model

from lya_interface.configuration import load_configuration
from lya_interface.diagnostics import evaluate_components, evaluate_point, minimize_point, plot_point

ROOT = next(
    (path for path in [Path.cwd(), *Path.cwd().parents]
     if (path / "examples/joint_desi_dr1_vega_mock.yaml").is_file()),
    None,
)
if ROOT is None:
    import lya_interface
    ROOT = Path(lya_interface.__file__).resolve().parents[2]
CONFIG = ROOT / "examples/joint_desi_dr1_vega_mock.yaml"
print("Configuration:", CONFIG)

# %% [markdown]
# ## Select a compact, inspectable parameter subspace
#
# `cup1d_evaluate.yaml` retains the full CM2026 registry because both native
# branches require all their coefficients. This cell freezes the unrelated
# entries at their configured references *before* creating Cobaya. The four
# `igm_tau_eff_*` entries are the requested interpolation nodes.

# %%
FREE_PARAMETERS = {
    "As", "ns",
    "igm_tau_eff_0", "igm_tau_eff_1", "igm_tau_eff_2", "igm_tau_eff_3",
    "p1d_f_Lya_SiIII_0", "p1d_HCD_damp1_0",
    "ap", "at", "beta_hcd", "L0_hcd",
}

def reference_value(definition):
    """Return an explicit native reference without inventing a prior centre."""
    if not isinstance(definition, dict) or "ref" not in definition:
        raise ValueError(f"No explicit reference for parameter definition: {definition!r}")
    value = definition["ref"]
    return value["dist"] if isinstance(value, dict) and "dist" in value else value

def compact_joint_configuration(path=CONFIG):
    """Load the full registry and freeze every parameter outside this demo."""
    info = load_configuration(path)
    for name, definition in tuple(info["params"].items()):
        if name not in FREE_PARAMETERS and isinstance(definition, dict) and "prior" in definition:
            info["params"][name] = reference_value(definition)
    # Interface components keep an explicit registry. Give each one this
    # resolved compact configuration so fixed coefficients remain routable.
    for section in ("theory", "likelihood"):
        for name, options in info[section].items():
            if name.startswith("lya_interface.") and "parameter_definitions" in options:
                options["parameter_definitions"] = deepcopy(info["params"])
    return info

info = compact_joint_configuration()
initial_point = {
    name: reference_value(definition)
    for name, definition in info["params"].items()
    if isinstance(definition, dict) and "prior" in definition
}
assert set(initial_point) == FREE_PARAMETERS
display(initial_point)

# %% [markdown]
# ## Initialize once and inspect shared Arinyo outputs
#
# This requires the local CAMB, ForestFlow covariance/emulator assets, DESI DR1
# P1D inputs and the Vega Y3 mock already referenced by the configuration. It
# does not run a sampler. If your CAMB build uses MPI, launch Jupyter in an
# environment where CAMB can initialize normally.

# %%
t0 = perf_counter()
model = get_model(info)
posterior = model.logposterior(initial_point)
if not np.isfinite(posterior.logpost):
    model.close()
    raise RuntimeError("Initial point is outside the configured prior or model domain.")

P1D_NAME = next(name for name in model.likelihood if name.endswith("Cup1DLikelihood"))
VEGA_NAME = next(name for name in model.likelihood if name.endswith("VegaLikelihood"))
p1d = model.likelihood[P1D_NAME]
vega = model.likelihood[VEGA_NAME]

def point_summary(point):
    """Evaluate once and retain distinct data terms and shared diagnostics."""
    components = evaluate_components(model, point)
    result, _ = evaluate_point(model, point)
    diagnostics = model.provider.get_result("forestflow_diagnostics")
    arinyo = {
        z: {name: values["arinyo"][name] for name in
            ("bias", "bias_eta", "q1", "q2", "kvav", "av", "bv", "kp")}
        for z, values in diagnostics.items()
    }
    return components, result, arinyo

initial_components, initial_p1d, initial_arinyo = point_summary(initial_point)
display({
    "initialization_seconds": perf_counter() - t0,
    "chi2_P1D_data": initial_p1d.chi2_data,
    "minus2_loglike_by_component": initial_components.minus2_loglike,
    "minus2_loglike_total": initial_components.total_minus2_loglike,
    "arinyo": initial_arinyo,
})

# %% [markdown]
# ## Plot cup1d against DESI DR1 and Vega against the Y3 mock
#
# The P1D plot is cup1d's maintained panel renderer. Vega uses its maintained
# wedge plotter after the public adapter builds the exact same native parameter
# dictionary used for its data-only likelihood. Vega's `peak` bookkeeping may
# mutate the temporary dictionary, never the sampled point.

# %%
def vega_native_values(point):
    """Route current public BAO values and inject shared ForestFlow products."""
    model.logposterior(point)
    return vega.parameters_for_evaluation({name: point[name] for name in vega.mapping})

def plot_vega_point(point, title):
    """Render each configured Vega correlation against its input mock data."""
    values = vega_native_values(point)
    correlations = vega.backend.compute_model(values, run_init=False)
    for name, correlation in correlations.items():
        vega.backend.plots.plot_4wedges(
            models=[correlation], corr_name=name, title=title,
            mu_bin_labels=True, no_font=True,
        )
        plt.show()

fig, axes = plot_point(model, initial_point, title="DESI DR1 cup1d: initial point")
plt.show()
plot_vega_point(initial_point, "Vega Y3 mock: initial point")

# %% [markdown]
# ## Bounded joint minimization
#
# This minimizes the sum of the two data objectives in the declared independent
# likelihood approximation. It is a smoke fit, not a convergence claim: leave
# it disabled for ordinary plotting, and do not call a budget-limited result a
# best fit. Priors bound the search but are not silently duplicated inside Vega.

# %%
RUN_MINIMIZATION = False
MAX_EVALS = 350
best_fit = None
if RUN_MINIMIZATION:
    best_fit = minimize_point(model, initial_point, max_evals=MAX_EVALS,
                              verbose=True, report_every=50)
    display({
        "success": best_fit.success,
        "message": best_fit.message,
        "evaluations": best_fit.evaluations,
        "chi2_P1D_data": best_fit.chi2_data,
        "minus2_loglike_total": best_fit.minus2_loglike_total,
    })
    if not best_fit.success:
        raise RuntimeError("Joint minimization did not converge; inspect the bounded result only.")

# %% [markdown]
# ## Re-evaluate and plot the actual best point
#
# The diagnostics and both visualizations below are recomputed from
# `best_fit.point`; they are not copied from the optimizer's final trial.

# %%
if best_fit is None:
    print("Set RUN_MINIMIZATION = True to produce the bounded joint best-point plots.")
else:
    best_components, best_p1d, best_arinyo = point_summary(best_fit.point)
    display({
        "chi2_P1D_data": best_p1d.chi2_data,
        "minus2_loglike_by_component": best_components.minus2_loglike,
        "minus2_loglike_total": best_components.total_minus2_loglike,
        "arinyo": best_arinyo,
    })
    fig, axes = plot_point(model, best_fit.point, title="DESI DR1 cup1d: bounded joint best point")
    plt.show()
    plot_vega_point(best_fit.point, "Vega Y3 mock: bounded joint best point")

# %%
model.close()
