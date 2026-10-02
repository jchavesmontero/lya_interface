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
# # 1. A first ForestFlow–cup1d likelihood evaluation
#
# Follow the actual chain:
#
# CAMB → immutable cosmology snapshot → native IGM histories → ForestFlow Arinyo coefficients → uncontaminated P1D → cup1d observation model and rebinning → correlated data likelihood.
#
# This example inherits cup1d's CM2026 baseline, changing only its emulator alias from `lace_mpg` to `forest_mpg`. It uses DESI QMLE3, four-node native IGM histories, the baseline contaminants/resolution, rebinning factor 8 and calibrated emulator covariance. The graph still uses CAMB and ForestFlow directly, not a hidden native `Analysis`.
#
# Existing ForestFlow weights, IGM histories, DESI data and `ForestFlow/data/covariance/l1O_cov_forest_mpg_fix.npy` are required. The covariance is not silently disabled if missing. No cell trains or regenerates assets. Native DESI blinding is preserved for exported star diagnostics; do not unblind fitted results here. Numerical convergence of the 1000-realization estimator is still a separate production accuracy question; see [validation](../docs/validation.md). Start with a fresh kernel and run cells in order.
#
# Optional check: set `USE_DESI_DR2_LYA_BAO = True` below after installing Cobaya's BAO data. P1D and BAO share one CAMB provider; separate component log-likelihoods are displayed. This assumes independence (no P1D–BAO cross-covariance). The CM2026 background is fixed, so this BAO contribution is constant with respect to the baseline sampled parameters. See the README for a BAO-only distance check that varies H0 without requiring ForestFlow covariance.

# %%
# Select the kernel from the environment containing the editable sibling installs.
# These limits are most effective in a fresh kernel, before NumPy/CAMB imports.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pathlib import Path
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from cobaya.model import get_model
from lya_interface.configuration import load_configuration
from lya_interface.diagnostics import cup1d_component

# Works when Jupyter starts in the repository or its notebooks directory.
candidates = [Path.cwd(), *Path.cwd().parents]
ROOT = next((p for p in candidates if (p / "examples/cup1d_evaluate.yaml").is_file()), None)
if ROOT is None:
    import lya_interface
    ROOT = Path(lya_interface.__file__).resolve().parents[2]
if not (ROOT / "examples/cup1d_evaluate.yaml").is_file():
    raise FileNotFoundError("Open this notebook from the lya_interface checkout.")
USE_DESI_DR2_LYA_BAO = False  # Optional independence-approximation check.
BAO_PACKAGES_PATH = None  # Set to your Cobaya data directory, or use its configured default.
CONFIG = ROOT / "examples" / (
    "cup1d_desidr2_lya_bao.yaml" if USE_DESI_DR2_LYA_BAO else "cup1d_evaluate.yaml"
)
print("Configuration:", CONFIG)


# %% [markdown]
# ## Load the configuration
#
# The YAML owns fixed values, sampled-parameter priors, projection settings and data selection. References are the native fiducial physical values; this example does not initialize from a previously saved LaCE fit/IC file. `load_configuration` resolves asset paths, so there is no need to change directories or modify `sys.path`.

# %%
info = load_configuration(CONFIG)
if USE_DESI_DR2_LYA_BAO and BAO_PACKAGES_PATH is not None:
    info["packages_path"] = str(Path(BAO_PACKAGES_PATH).expanduser().resolve())
likelihood_options = info["likelihood"]["lya_interface.likelihoods.cup1d.Cup1DLikelihood"]
covariance_path = Path(likelihood_options["covariance_asset"])
if not covariance_path.is_file():
    raise FileNotFoundError(
        f"CM2026 requires the calibrated ForestFlow covariance: {covariance_path}. "
        "Supply this product before evaluation; do not disable it to mimic the baseline."
    )
# Only sampled parameters belong in the point; fixed parameters stay in the YAML.
reference = {
    name: definition["ref"]
    for name, definition in info["params"].items()
    if isinstance(definition, dict) and "prior" in definition
}
display(reference)


# %% [markdown]
# ## Build the Cobaya graph and evaluate a point
#
# The first evaluation includes CAMB, emulation and projection; an identical repeat is cached. Keep the model open while inspecting its provider. The final cell closes it.

# %%
t0 = perf_counter()
model = get_model(info)
t1 = perf_counter()
posterior = model.logposterior(reference)
t2 = perf_counter()
repeat = model.logposterior(reference)
t3 = perf_counter()

prediction = model.provider.get_result("forestflow_p1d")
if not prediction["valid"]:
    model.close()
    raise ValueError(prediction["reason"])
likelihood = cup1d_component(model)
snapshot = model.provider.get_result("lya_cosmology")
display({
    "initialization_seconds": t1 - t0,
    "cold_evaluation_seconds": t2 - t1,
    "cached_evaluation_seconds": t3 - t2,
    "logposterior": float(posterior.logpost),
    "derived": dict(zip(model.parameterization.derived_params(), posterior.derived)),
})
np.testing.assert_array_equal(posterior.derived, repeat.derived)


# %% [markdown]
# ## Separate the data likelihood from the posterior
#
# Cobaya adds the sampled-parameter priors. The cup1d backend returns the data likelihood only. Nuisance aliases are explicitly routed to their native names. With this configuration, `loglike = -chi2_data / 2`: the fixed covariance log-determinant and Gaussian `ndata * log(2π)` normalization are omitted. The residual uses the correlated covariance, not independent error bars.

# %%
from lya_interface.parameters import route

public_nuisance = {
    name: reference[name] if name in reference else info["params"][name]
    for name in likelihood.mapping
}
native_nuisance = route(public_nuisance, likelihood.mapping)
result = likelihood.backend.evaluate_from_p1d(
    prediction["powers"], prediction["context"], native_nuisance
)
display({
    "chi2_data": result.chi2_data,
    "loglike_data": result.loglike,
    "logdet_cov_included": result.logdet_cov,
    "ndata": result.ndata,
    "minus2_logposterior": -2 * posterior.logpost,
})
assert result.valid
component_loglikes = dict(zip(model.likelihood, map(float, posterior.loglikes)))
display({"component_loglikes": component_loglikes,
         "loglike_total": float(np.sum(posterior.loglikes))})
p1d_name = next(name for name, component in model.likelihood.items() if component is likelihood)
np.testing.assert_allclose(result.loglike, component_loglikes[p1d_name])


# %% [markdown]
# ## Inspect the intermediate physical quantities
#
# The prediction request specifies redshifts and fine velocity-space k grids. These are **before contaminants and rebinning**, so do not compare the raw arrays directly to the observed bins. `M(z) = H(z)/(1+z)` converts velocity and comoving coordinates; the same cosmology snapshot feeds every stage.

# %%
request = likelihood.backend.get_prediction_request()
display({
    "units": request.units,
    "stage": request.stage,
    "groups": [
        {"dataset": g.identifier, "redshifts": g.redshifts,
         "fine_grid_lengths": [len(k) for k in g.k_ikms]}
        for g in request.groups
    ],
})
display({
    "first_redshift": prediction["context"].redshifts[0],
    "first_emulator_input": dict(prediction["inputs"][0]),
    "mean_flux": prediction["context"].mean_flux[0],
    "M_kms_per_Mpc": prediction["context"].dkms_diMpc[0],
})
# Arinyo coefficients are a diagnostic, not a likelihood requirement.
# Read them from the theory component; the provider only routes requested products.
forestflow = next(c for c in model.theory.values() if c.__class__.__name__ == "ForestFlowTheory")
arinyo = forestflow.get_result("forestflow_arinyo")
display({"Arinyo_at_first_redshift": dict(arinyo[request.redshifts[0]])})


# %% [markdown]
# ## Evaluate a parameter point directly
#
# Start from the sampled fiducial values and edit `parameter_values` to check
# another point. Fixed parameters remain in the YAML. This returns the P1D
# data chi², without parameter-prior penalties; priors and physical-domain
# support are still checked. If BAO is enabled, its contribution is reported
# separately through the total data log-likelihood. No minimization is run.

# %%
from lya_interface.diagnostics import evaluate_point

parameter_values = dict(reference)
# Example: parameter_values["igm_tau_eff_0"] = 0.01
display(parameter_values)

point_result, _ = evaluate_point(model, parameter_values)
point_posterior = model.logposterior(parameter_values)
display({
    "chi2_P1D": point_result.chi2_data,
    "loglike_P1D": point_result.loglike,
    "ndata_P1D": point_result.ndata,
    "minus2_loglike_total": -2 * float(np.sum(point_posterior.loglikes)),
})

# %%
import time

start = time.perf_counter()
for ii in range(10):
    point_result, _ = evaluate_point(model, parameter_values)
print(f"10 likelihood evaluations: {time.perf_counter() - start:.3f} s")

# %% [markdown]
# ## Plot data and the initial model at every redshift
#
# These panels reuse cup1d's maintained P1D renderer. Models include contaminants/resolution and rebinning onto the data bins. Error bars show marginal diagonal errors; the figure title reports the full correlated joint chi². Per-redshift diagnostic chi² values need not sum to it.

# %%
from lya_interface.diagnostics import plot_point, minimize_point

initial_point = dict(reference)
fig, axes = plot_point(model, initial_point, title="CM2026 + ForestFlow: initial point")
plt.show()


# %% [markdown]
# ## Inspect provenance
#
# The provenance collector reports sibling revisions, resolved requests and asset checksums. This cell only displays a summary; it does not write files. A command-line evaluation can save the full record using `--provenance`.

# %%
from lya_interface.provenance import collect

provenance = collect(info, model)
print("Provenance sections:", list(provenance))

# %% [markdown]
# ## Run a minimization
#
# Set the flag to `True` when you want to fit. This varies **all** sampled baseline parameters, minimizing the data likelihood within the native bounds. It uses derivative-free Nelder-Mead in scaled coordinates, with an explicit initial simplex and adaptive coefficients. It can take longer than the initial evaluation; increase the budget if necessary. No chains or fit files are written. Convergence of one bounded start is not a global-optimum guarantee.

# %%
RUN_MINIMIZATION = False
best_fit_point = None
if RUN_MINIMIZATION:
    fit = minimize_point(
        model, initial_point, max_evals=10000, verbose=True, report_every=100
    )
    display(
        {
            "success": fit.success,
            "message": fit.message,
            "evaluations": fit.evaluations,
            "chi2_P1D": fit.chi2_data,
            "minus2_loglike_total": fit.minus2_loglike_total,
        }
    )
    best_fit_point = fit.point
    if not fit.success:
        raise RuntimeError(
            "Minimization did not converge; do not label its output a best fit."
        )
    best_fit_point = fit.point  # Keep fitted cosmology internal, as in native cup1d.

# %%
best_fit_point = fit.point
point_result, _ = evaluate_point(model, best_fit_point)
point_result

# %% [markdown]
# ## Plot the best-fitting model at every redshift
#
# This reevaluates the actual fitted point, never reuses the initial prediction as a stand-in. Enable and run the previous cell first. Set `residuals=True` for the native data/model-ratio panels. Rerun model initialization if you wish to work further after closing it.

# %%
if best_fit_point is None:
    print("Enable RUN_MINIMIZATION and run the preceding cell to obtain a best-fit plot.")
else:
    fig, axes = plot_point(model, best_fit_point, title="CM2026 + ForestFlow: best fit")
    plt.show()
model.close()

# %%
best_fit_point

# %%
np.save("best_fit_point.npy", best_fit_point)

# %%
