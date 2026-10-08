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
from lya_interface.derived import star_parameters
from lya_interface.diagnostics import evaluate_components, evaluate_point, minimize_point, plot_point
from lya_interface.parameters import route

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
# entries at their explicit native references before creating Cobaya. The four
# `igm_tau_eff_*` entries are the requested interpolation nodes. `drp_QSO` is
# held internally by Vega at its native mean and is not a notebook parameter.

# %%
FREE_PARAMETERS = {
    # Shared cosmology and interface-owned IGM histories.
    "general": (
        "As", "ns",
        "igm_tau_eff_0", "igm_tau_eff_1", "igm_tau_eff_2", "igm_tau_eff_3",
    ),
    # cup1d-only contaminants/systematics.
    "P1D": (
        "p1d_f_Lya_SiIII_0", "p1d_HCD_damp1_0",
    ),
    # Vega-only dilation/HCD parameters. Ly-alpha bias parameters are derived
    # from ForestFlow and must never appear here as independently varied terms.
    "BAO": (
        "ap", "at", "beta_hcd", "L0_hcd",
    ),
}
SAMPLED_PARAMETERS = frozenset().union(*FREE_PARAMETERS.values())
# This is the complete Vega-facing namespace, not merely the entries that
# happen to be varied in a particular notebook run.
VEGA_PARAMETERS = frozenset(("ap", "at", "beta_hcd", "L0_hcd"))

def reference_value(definition):
    """Return an explicit native reference without inventing a prior centre."""
    if not isinstance(definition, dict) or "ref" not in definition:
        raise ValueError(f"No explicit reference for parameter definition: {definition!r}")
    value = definition["ref"]
    return value["dist"] if isinstance(value, dict) and "dist" in value else value

def compact_joint_configuration(path=CONFIG):
    """Load the full registry and freeze unrelated entries at native references."""
    info = load_configuration(path)
    # The inherited CM2026 file declares legacy scalar derived columns.  They
    # are not inputs to this joint evaluation and their providers are not part
    # of this compact graph, so do not request them from Cobaya.
    for name, definition in tuple(info["params"].items()):
        if isinstance(definition, dict) and definition.get("derived") is True:
            info["params"].pop(name)
    for name, definition in tuple(info["params"].items()):
        if name not in SAMPLED_PARAMETERS and isinstance(definition, dict) and "prior" in definition:
            info["params"][name] = reference_value(definition)
    # Preserve ownership in the component registries. cup1d and ForestFlow
    # have no BAO aliases, while Vega receives the full public configuration
    # so it can verify migration of native Gaussian priors.
    p1d_definitions = {
        name: definition for name, definition in info["params"].items()
        if name not in VEGA_PARAMETERS
    }
    for section in ("theory", "likelihood"):
        for name, options in info[section].items():
            if name.startswith("lya_interface.") and "parameter_definitions" in options:
                options["parameter_definitions"] = deepcopy(
                    info["params"] if name.endswith("VegaLikelihood") else p1d_definitions
                )
    return info

info = compact_joint_configuration()
initial_point = {
    name: reference_value(definition)
    for name, definition in info["params"].items()
    if isinstance(definition, dict) and "prior" in definition
}
assert set(initial_point) == SAMPLED_PARAMETERS
display({section: {name: initial_point[name] for name in names}
         for section, names in FREE_PARAMETERS.items()})

# %% [markdown]
# ## Compute and inspect the shared ForestFlow state before either data likelihood
#
# This requires the local CAMB, ForestFlow covariance/emulator assets, DESI DR1
# P1D inputs and the Vega Y3 mock already referenced by the configuration. It
# does not run a sampler. The small helper below executes only Cobaya's theory
# components (cosmology and ForestFlow), stopping before cup1d or Vega. This
# lets us inspect their common prediction before evaluating either dataset.
# If your CAMB build uses MPI, launch Jupyter in an environment where CAMB can
# initialize normally.

# %%
def calculate_shared_theory(model, point):
    """Populate Cobaya's shared theory products without evaluating likelihoods."""
    sampled = model.parameterization.check_sampled(point)
    inputs = model.parameterization.to_input(model._to_sampled_array(sampled))
    model.provider.set_current_input_params(inputs)
    for (component, likelihood_index), dependencies in zip(
        model._component_order.items(), model._params_of_dependencies
    ):
        if likelihood_index is not None:
            break
        success = component.check_cache_and_compute(
            {name: inputs[name] for name in component.input_params},
            want_derived=False,
            dependency_params=[inputs[name] for name in dependencies],
            cached=False,
        )
        if not success:
            raise RuntimeError(f"Shared theory component failed: {component}")
    return inputs


def plot_shared_coefficients(diagnostics):
    """Plot ForestFlow bias and Arinyo coefficients before data evaluation."""
    redshifts = np.array(sorted(diagnostics))
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].plot(redshifts, [diagnostics[z]["bias"] for z in redshifts], "o-", label=r"$b_\delta$")
    axes[0].plot(redshifts, [diagnostics[z]["bias_eta"] for z in redshifts], "s-", label=r"$b_\eta$")
    axes[0].set(xlabel="redshift", ylabel="coefficient", title="ForestFlow bias coefficients")
    axes[0].grid(alpha=.25)
    axes[0].legend()
    for coefficient in ("q1", "q2", "kvav", "av", "bv", "kp"):
        axes[1].plot(
            redshifts,
            [diagnostics[z]["arinyo"][coefficient] for z in redshifts],
            "o-", label=coefficient,
        )
    axes[1].set(xlabel="redshift", ylabel="coefficient", title="ForestFlow Arinyo coefficients")
    axes[1].grid(alpha=.25)
    axes[1].legend(ncol=2)
    return figure


def likelihood_preflight(p1d, vega, inputs):
    """Evaluate native backends directly to retain their individual failure modes."""
    prediction = p1d.provider.get_result("forestflow_p1d")
    report = {"forestflow_p1d_valid": prediction["valid"]}
    try:
        nuisance = route({name: inputs[name] for name in p1d.mapping}, p1d.mapping)
        result = p1d.backend.evaluate_from_p1d(
            prediction["powers"], prediction["context"], nuisance
        )
        report["cup1d"] = {"loglike": float(result.loglike), "chi2_data": float(result.chi2_data)}
    except Exception as error:
        report["cup1d"] = {"exception": f"{type(error).__name__}: {error}"}
    try:
        values = vega.parameters_for_evaluation({name: inputs[name] for name in vega.mapping})
        chi2 = vega.backend.chi2(values, include_priors=False)
        report["vega"] = {"chi2_data": float(chi2), "loglike": float(-.5 * chi2)}
    except Exception as error:
        report["vega"] = {"exception": f"{type(error).__name__}: {error}"}
    return report


t0 = perf_counter()
model = get_model(info)
P1D_NAME = next(name for name in model.likelihood if name.endswith("Cup1DLikelihood"))
VEGA_NAME = next(name for name in model.likelihood if name.endswith("VegaLikelihood"))
p1d = model.likelihood[P1D_NAME]
vega = model.likelihood[VEGA_NAME]
forestflow = next(
    component for component in model.theory.values()
    if component.__class__.__name__ == "ForestFlowTheory"
)

input_values = calculate_shared_theory(model, initial_point)
# ``forestflow_diagnostics`` is an inspection product, not a likelihood
# requirement, so retrieve it from the just-evaluated theory state directly.
forestflow_diagnostics = forestflow.get_result("forestflow_diagnostics")
display(plot_shared_coefficients(forestflow_diagnostics))
display({
    "shared_theory_seconds": perf_counter() - t0,
    "forestflow_inputs": model.provider.get_result("forestflow_p1d")["inputs"],
    "likelihood_preflight": likelihood_preflight(p1d, vega, input_values),
})

# This notebook consumes the raw shared products itself; do not ask Cobaya to
# materialize unrelated legacy derived columns from the inherited CM2026
# registry during this likelihood-only check.
posterior = model.logposterior(initial_point, return_derived=False)
if not np.isfinite(posterior.logpost):
    prediction = model.provider.get_result("forestflow_p1d") or {}
    diagnostic = {
        "logprior_terms": list(map(float, posterior.logpriors)),
        "loglike_terms": dict(zip(model.likelihood, map(float, posterior.loglikes))),
        "forestflow_valid": prediction.get("valid"),
        "forestflow_reason": prediction.get("reason"),
        "likelihood_preflight": likelihood_preflight(p1d, vega, input_values),
    }
    model.close()
    raise RuntimeError(f"Initial point is invalid. Exact diagnostic: {diagnostic}")

def point_summary(point):
    """Evaluate once and retain distinct data terms and shared diagnostics."""
    components = evaluate_components(model, point)
    result, _ = evaluate_point(model, point)
    diagnostics = forestflow.get_result("forestflow_diagnostics")
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
    model.logposterior(point, return_derived=False)
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
# RUN_MINIMIZATION = False
RUN_MINIMIZATION = True
MAX_EVALS = 350
# The Gaussian priors below remain unbounded in Cobaya. These are only broad,
# finite simplex search boxes (five standard deviations); they do not change
# the posterior or truncate either prior.
OPTIMIZATION_BOUNDS = {
    "beta_hcd": (0.05, 0.95),
    "L0_hcd": (-5.0, 15.0),
}
best_fit = None
if RUN_MINIMIZATION:
    best_fit = minimize_point(
        model, initial_point, max_evals=MAX_EVALS, verbose=True, report_every=50,
        optimization_bounds=OPTIMIZATION_BOUNDS,
    )
    display(
        {
            "success": best_fit.success,
            "message": best_fit.message,
            "evaluations": best_fit.evaluations,
            "chi2_P1D_data": best_fit.chi2_data,
            "minus2_loglike_total": best_fit.minus2_loglike_total,
        }
    )
    if not best_fit.success:
        raise RuntimeError(
            "Joint minimization did not converge; inspect the bounded result only."
        )

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

# %% [markdown]
# ## Final compact parameter and mean-flux summary
#
# The summary uses the same final point just plotted: the bounded joint result
# when available, otherwise the initial reference point. The mean-flux curve
# is evaluated at the union of the P1D redshifts and Vega's BAO effective
# redshift, with their locations marked separately.

# %%
def report_final_point(point):
    """Print star/BAO parameters and plot mean flux on all data redshifts."""
    model.logposterior(point, return_derived=False)
    cosmology = next(
        component for component in model.theory.values()
        if component.__class__.__name__ == "LyaCosmology"
    ).get_lya_cosmology()
    stars = star_parameters(cosmology)
    diagnostics = forestflow.get_result("forestflow_diagnostics")
    redshifts = np.array(sorted(diagnostics))
    mean_flux = np.array([diagnostics[z]["mean_flux"] for z in redshifts])
    p1d_redshifts = np.asarray(p1d.backend.get_prediction_request().redshifts)
    bao_redshifts = np.asarray([vega.zeff])
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(redshifts, mean_flux, "o-", color="C0", label="shared mean-flux history")
    axis.scatter(p1d_redshifts,
                 [diagnostics[float(z)]["mean_flux"] for z in p1d_redshifts],
                 color="C1", marker="s", zorder=3, label="P1D redshifts")
    axis.scatter(bao_redshifts,
                 [diagnostics[float(z)]["mean_flux"] for z in bao_redshifts],
                 color="C3", marker="D", zorder=4, label="Vega BAO effective redshift")
    axis.set(xlabel="redshift", ylabel=r"mean flux $\langle F\rangle$",
             title="Final-point mean-flux history")
    axis.grid(alpha=.25)
    axis.legend()
    display({
        "Delta2star": stars["Delta2star"],
        "nstar": stars["nstar"],
        "ap": point["ap"],
        "at": point["at"],
        "mean_flux": dict(zip(map(float, redshifts), map(float, mean_flux))),
    })
    return figure


final_point = initial_point if best_fit is None else best_fit.point
display(report_final_point(final_point))

# %%
model.close()

# %%
