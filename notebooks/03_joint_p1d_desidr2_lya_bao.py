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
# # 3. Combine cup1d P1D and DESI DR2 Lyα BAO
#
# Evaluate the two data terms in a single Cobaya model, inspect their separate
# contributions, plot the initial P1D model, and optionally minimize the total
# likelihood. Both components use the same CAMB provider; no second cosmology
# calculation or native cup1d `Analysis` is constructed.
#
# The configuration inherits the **CM2026 baseline with `forest_mpg`**. It
# retains the baseline IGM histories, contaminants, resolution, rebinning and
# calibrated covariance. The native DESI blinding remains enabled. Existing
# scientific assets are required; no cell trains models or generates covariance.
#
# This combination assumes independence: log L_total = log L_P1D + log L_BAO.
# No measured P1D–BAO cross-covariance is included. Importantly, CM2026 fixes
# the background cosmology, so BAO is constant with respect to the baseline
# sampled parameters. This notebook checks composition, not a joint
# cosmological constraint. Varying the background requires a separately
# validated configuration and an appropriate ForestFlow cosmology domain.

# %% [markdown]
# ## Install BAO data once, outside the notebook
#
# In the environment containing the editable sibling installs, run:
#
# ```bash
# cobaya-install bao.desi_dr2.desi_bao_lya --packages-path /path/to/cobaya_packages --just-data --no-set-global
# ```
#
# Set `BAO_PACKAGES_PATH` below to that directory, or leave it as `None` to use
# Cobaya's configured default. The notebook does not download or install data.
# It also requires `ForestFlow/data/covariance/l1O_cov_forest_mpg_fix.npy`;
# missing calibrated covariance is an error, not a reason to disable it.
# Start with a fresh kernel and run cells in order.

# %%
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
from lya_interface.diagnostics import cup1d_component, evaluate_point, plot_point, minimize_point

ROOT = next(
    (p for p in [Path.cwd(), *Path.cwd().parents]
     if (p / "examples/cup1d_desidr2_lya_bao.yaml").is_file()),
    None,
)
if ROOT is None:
    import lya_interface
    ROOT = Path(lya_interface.__file__).resolve().parents[2]
CONFIG = ROOT / "examples/cup1d_desidr2_lya_bao.yaml"
if not CONFIG.is_file():
    raise FileNotFoundError("Open this notebook from the lya_interface checkout.")
BAO_PACKAGES_PATH = "/home/jchaves/Proyectos/projects/lya/igmhub/cobaya_packages"
BAO_NAME = "bao.desi_dr2.desi_bao_lya"
print("Configuration:", CONFIG)

# %% [markdown]
# ## Load the joint configuration and reference point
#
# The loader resolves the inherited YAML and scientific asset paths. Only
# sampled parameters go into the point; fixed background values remain in the
# configuration. Do not silently replace CM2026 with the validation demo.

# %%
info = load_configuration(CONFIG)
if BAO_PACKAGES_PATH is not None:
    info["packages_path"] = str(Path(BAO_PACKAGES_PATH).expanduser().resolve())
assert BAO_NAME in info["likelihood"]
p1d_options = info["likelihood"]["lya_interface.likelihoods.cup1d.Cup1DLikelihood"]
covariance_path = Path(p1d_options["covariance_asset"])
if not covariance_path.is_file():
    raise FileNotFoundError(
        f"CM2026 requires the calibrated ForestFlow covariance: {covariance_path}. "
        "Supply this product before evaluating the joint baseline."
    )
reference = {
    name: definition["ref"]
    for name, definition in info["params"].items()
    if isinstance(definition, dict) and "prior" in definition
}
display(reference)

# %%
reference = np.load("best_fit_point.npy", allow_pickle=True).item()

# %% [markdown]
# ## Evaluate and separate the likelihood contributions
#
# The posterior includes parameter priors; the summed data log-likelihood
# does not. The P1D term uses the full correlated covariance and omits its
# fixed Gaussian normalization. Report BAO with the convention implemented by
# Cobaya's native component, rather than treating the combined objective as
# a P1D-only chi². Derived/fitted cosmology is not printed here.

# %%
t0 = perf_counter()
model = get_model(info)
try:
    posterior = model.logposterior(reference)
    if not np.isfinite(posterior.logpost):
        raise ValueError("The initial joint point is outside the prior or model domain.")
    p1d_component = cup1d_component(model)
    P1D_NAME = next(name for name, component in model.likelihood.items()
                    if component is p1d_component)
    initial_result, _ = evaluate_point(model, reference)
except Exception:
    model.close()
    raise

def likelihood_summary(point):
    """Display data contributions without exposing fitted cosmology."""
    post = model.logposterior(point)
    if not np.isfinite(post.logpost):
        raise ValueError("Invalid joint point")
    terms = dict(zip(model.likelihood, map(float, post.loglikes)))
    np.testing.assert_allclose(sum(terms.values()), np.sum(post.loglikes))
    return {
        "loglike_P1D": terms[P1D_NAME],
        "loglike_DESI_DR2_lya_BAO": terms[BAO_NAME],
        "loglike_total": float(np.sum(post.loglikes)),
        "minus2_loglike_total": -2 * float(np.sum(post.loglikes)),
        "logprior_total": float(np.sum(post.logpriors)),
        "logposterior": float(post.logpost),
    }

initial_summary = likelihood_summary(reference)
np.testing.assert_allclose(initial_result.loglike, initial_summary["loglike_P1D"])
display({"initialization_and_evaluation_seconds": perf_counter() - t0,
         "chi2_P1D": initial_result.chi2_data,
         "ndata_P1D": initial_result.ndata,
         **initial_summary})

# %% [markdown]
# ## Plot the initial P1D prediction at every redshift
#
# The maintained cup1d renderer includes contaminants, resolution and data
# rebinning. Its title reports **P1D-only** correlated chi², not the combined
# objective above. BAO is a separate distance likelihood, not an extra P1D
# spectrum to overlay on these panels.

# %%
initial_point = dict(reference)
fig, axes = plot_point(model, initial_point, title="CM2026 + ForestFlow + DESI DR2 Lyα BAO: initial")
plt.show()

# %% [markdown]
# ## Optional minimization of the combined data likelihood
#
# Set the flag to `True` to run bounded Nelder-Mead minimization over all sampled
# baseline parameters. The objective includes **both** likelihoods, but not
# parameter-prior penalties within the bounds. With the fixed CM2026
# background, BAO adds a constant and should not move the P1D best fit.
# This is not a cosmological BAO fit or a global-optimum guarantee. No chains
# or fit files are written, and the fitted point stays internal for blinding.

# %%
RUN_MINIMIZATION = True
best_fit_point = None
if RUN_MINIMIZATION:
    fit = minimize_point(model, initial_point, max_evals=500, verbose=True, report_every=100)
    display({"success": fit.success, "message": fit.message,
             "evaluations": fit.evaluations, "chi2_P1D": fit.chi2_data,
             "minus2_loglike_total": fit.minus2_loglike_total})
    if not fit.success:
        raise RuntimeError("Minimization did not converge; do not label its output a best fit.")
    best_fit_point = fit.point

# %% [markdown]
# ## Best-fit components and P1D plot
#
# Reevaluate the actual fitted point and display each data contribution.
# BAO should retain its initial value because the background is fixed.
# Enable the previous cell first. The final cell closes the Cobaya model;
# initialize a new model before continuing work after that.

# %%
if best_fit_point is None:
    print("Enable RUN_MINIMIZATION and run the preceding cell for a best-fit plot.")
else:
    best_summary = likelihood_summary(best_fit_point)
    np.testing.assert_allclose(
        best_summary["loglike_DESI_DR2_lya_BAO"],
        initial_summary["loglike_DESI_DR2_lya_BAO"],
    )
    display(best_summary)
    fig, axes = plot_point(model, best_fit_point, title="CM2026 + ForestFlow + DESI DR2 Lyα BAO: best fit")
    plt.show()
model.close()
