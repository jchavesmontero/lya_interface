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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 2. Parameter changes and cached calculations
#
# This notebook shows why cosmology, IGM/projection and observation-model parameters are separate Cobaya components. A nuisance change reuses the raw prediction; an IGM change reruns ForestFlow/projection but reuses cosmology; a cosmology change recomputes both.
#
# These examples inherit cup1d's CM2026 baseline with the `forest_mpg` alias: DESI QMLE3 data, four-node IGM histories, baseline contaminants/resolution and rebinning factor 8. Existing corrected ForestFlow weights and its calibrated covariance file are required. Nothing trains or generates covariance, and missing assets are not replaced by mocks.
#
# Native DESI diagnostic blinding is preserved. A one-dimensional slice is not an inference result, and the 1000-realization estimator still needs a production convergence budget. See [validation and limitations](../docs/validation.md). Start with a fresh kernel and run cells in order.

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

# Works when Jupyter starts in the repository or its notebooks directory.
candidates = [Path.cwd(), *Path.cwd().parents]
ROOT = next((p for p in candidates if (p / "examples/cup1d_evaluate.yaml").is_file()), None)
if ROOT is None:
    import lya_interface
    ROOT = Path(lya_interface.__file__).resolve().parents[2]
if not (ROOT / "examples/cup1d_evaluate.yaml").is_file():
    raise FileNotFoundError("Open this notebook from the lya_interface checkout.")
CONFIG = ROOT / "examples/cup1d_evaluate.yaml"
print("Configuration:", CONFIG)


# %%
info = load_configuration(CONFIG)
# Only sampled parameters belong in the point; fixed parameters stay in the YAML.
reference = {
    name: definition["ref"]
    for name, definition in info["params"].items()
    if isinstance(definition, dict) and "prior" in definition
}
display(reference)


# %% [markdown]
# ## Set up diagnostics
#
# Counters count actual snapshot/emulation calculations, not calls to `logposterior`. We also compare the returned cached object identities. All changes below stay inside the demonstration priors; a point outside a prior can be rejected before any theory calculation.

# %%
model = get_model(info)
cosmology = next(c for c in model.theory.values() if c.__class__.__name__ == "LyaCosmology")
forestflow = next(c for c in model.theory.values() if c.__class__.__name__ == "ForestFlowTheory")

def evaluate(point):
    start = perf_counter()
    posterior = model.logposterior(point)
    elapsed = perf_counter() - start
    prediction = model.provider.get_result("forestflow_p1d")
    if not prediction["valid"]:
        raise ValueError(prediction["reason"])
    return {
        "loglike": float(np.sum(posterior.loglikes)),
        "stars": dict(zip(model.parameterization.derived_params(), posterior.derived)),
        "seconds": elapsed,
        "snapshot_calls": cosmology.calls,
        "forestflow_calls": forestflow.calls,
        "snapshot": model.provider.get_result("lya_cosmology"),
        "prediction": prediction,
    }

def summary(label, result):
    return {"change": label, **{
        key: result[key] for key in (
            "loglike", "seconds", "snapshot_calls", "forestflow_calls"
        )
    }}



# %% [markdown]
# ## Compare each dependency layer
#
# `p1d_f_Lya_SiIII_0` is a native SiIII contaminant coefficient, not a physical linear metal amplitude. `igm_tau_eff_0` is the native mean-flux-history coefficient, not τ itself. Keep their YAML conventions rather than interpreting all aliases as simple multiplicative scalings.

# %%
baseline = evaluate(reference)
repeat = evaluate(reference)
nuisance = evaluate({**reference, "p1d_f_Lya_SiIII_0": reference["p1d_f_Lya_SiIII_0"] + 0.01})
igm = evaluate({**reference, "igm_tau_eff_0": reference["igm_tau_eff_0"] + 0.002})
changed_cosmology = evaluate({**reference, "ns": reference["ns"] + 0.0005})
restored = evaluate(reference)

rows = [
    summary(label, result) for label, result in [
        ("reference", baseline), ("identical repeat", repeat),
        ("SiIII nuisance only", nuisance), ("IGM only", igm),
        ("ns only", changed_cosmology), ("restore reference", restored)
    ]
]
for row in rows:
    print(row)

assert repeat["prediction"] is baseline["prediction"]
assert nuisance["prediction"] is baseline["prediction"]
assert igm["snapshot"] is baseline["snapshot"]
assert igm["forestflow_calls"] == baseline["forestflow_calls"] + 1
assert changed_cosmology["snapshot_calls"] == baseline["snapshot_calls"] + 1
assert changed_cosmology["forestflow_calls"] == igm["forestflow_calls"] + 1
assert restored["snapshot"] is baseline["snapshot"]
assert restored["prediction"] is baseline["prediction"]
np.testing.assert_allclose(restored["loglike"], baseline["loglike"], rtol=0, atol=0)
print("Dependency reuse and exact cached restoration verified.")


# %% [markdown]
# ## A tiny one-dimensional likelihood scan
#
# Vary `ns`, holding every other sampled and fixed parameter at its reference value. This is a **conditional slice**, not a marginal posterior or a best-fit search. It costs three cosmology/emulator evaluations. Exported star parameters retain native blinding and change with cosmology, not with the contaminants.

# %%
ns_values = np.array([0.962, reference["ns"], 0.971])
scan = [evaluate({**reference, "ns": float(ns)}) for ns in ns_values]
minus2_loglike = np.array([-2 * r["loglike"] for r in scan])

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(ns_values, minus2_loglike - minus2_loglike.min(), "o-")
axes[0].set(xlabel=r"$n_s$", ylabel=r"$\Delta(-2\log \mathcal{L})$",
            title="Conditional data-likelihood slice")
axes[1].plot(ns_values, [r["stars"]["nstar"] for r in scan], "o-")
axes[1].set(xlabel=r"$n_s$", ylabel=r"$n_\star$ (native blinded)", title="Cosmology-only derived quantity")
fig.tight_layout()
plt.show()
model.close()


# %% [markdown]
# ## Next: sampling or minimization
#
# The complete configurations are already provided in `examples/cup1d_mcmc.yaml` and `examples/cup1d_mle.yaml`. From the repository root, run either:
#
# ```bash
# python examples/evaluate_point.py examples/cup1d_mcmc.yaml --output output/my_demo_chain
# python examples/evaluate_point.py examples/cup1d_mle.yaml --output output/my_demo_minimum
# ```
#
# Those commands write outputs; this notebook does not start a sampler. The MCMC example is only a short plumbing check, not a converged posterior, and the minimizer uses the exported native CM2026 bounds. See [parameter mapping](../docs/parameter_mapping.md) before changing which parameters are sampled. When editing public parameter definitions, keep component `parameter_definitions` synchronized (the YAML anchors already do this). Do not add a second likelihood for the same data.
