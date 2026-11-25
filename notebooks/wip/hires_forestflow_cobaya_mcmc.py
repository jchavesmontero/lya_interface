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
# # Cobaya MCMC through `lya_interface`
#
# Sample the 53-parameter `hires_forestflow` posterior (DESI DR1 QMLE3 +
# Karacayli2022, `forest_mpg_gp`) with Cobaya's Metropolis sampler through the
# `lya_interface` dependency graph.
#
# **Configuration.** The interface YAML
# `lya_interface/examples/hires_forestflow_evaluate.yaml` is generated from the
# native cup1d YAML by `lya_interface.from_cup1d`. It points `native_config` at
# the cup1d file, so data, emulator, `k_rebin_factor`, `n_k_emu`, contaminants and
# covariance factors are read from cup1d. Parameter bounds and references, the
# fixed cosmology and the frozen `M(z)` come from a native `Analysis`. There is no
# `projection` block: the P3D to P1D quadrature is ForestFlow's default
# `P1DIntegratorKMu`, the same object native cup1d uses. The numerical settings
# are cup1d's defaults: `k_rebin_factor: 32`, `n_k_emu: 64` and
# `cosmo_prior_nsigma: 2` (flat `As`/`ns` priors over the training mean +- 2 sigma).
#
# **Unit cube.** As in native cup1d, Cobaya samples every parameter in [0, 1]
# (`<name>_cube`, `sample_in_unit_cube: true`), and the physical parameters are
# computed from it and saved in the chain. The map is linear, so the priors stay
# flat in the physical parameters. Without it, the covariance of `As` (~1e-9)
# next to order-one coefficients fails Cobaya's positive-definiteness check.
#
# **Speed blocks.** Cobaya times each component and splits the parameters into
# blocks by what they trigger: `As`/`ns` (snapshot + emulator + likelihood),
# the 16 IGM parameters (emulator + likelihood) and the 35 contaminant and
# resolution parameters (likelihood only). The likelihood itself costs about as
# much as the emulator, so the speed ratios are below 2: dragging and
# oversampling are switched off explicitly. Section 3 measures this.
#
# **Validation.** Before sampling, run `notebooks/tests/hires_forestflow_validation`:
# it (re)generates the interface YAML and checks that native cup1d and the
# interface give the same P1D and chi2.
#
# **Steps.**
# 1. Starting point and proposal covariance, cached on disk:
#    best fit (minimized from the global IC if missing) and its Gauss-Newton
#    covariance; a learned Cobaya covariance or another run's covariance is
#    preferred when available.
# 2. Production YAML for NERSC.
# 3. Test run: a short chain with the production settings, the time per call of
#    each block, and a forecast of the production time.
# 4. Load the chain back from disk and plot it.
#
# Cosmological results are blinded star parameters only.

# %%
# %load_ext autoreload
# %autoreload 2

import os

# leave two cores free; must be set before NumPy/torch are imported
N_THREADS = max(1, (os.cpu_count() or 3) - 2)
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS)

import gc
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from cup1d.utils.utils import get_path_repo

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

ROOT = next(
    (p for p in [Path.cwd(), *Path.cwd().parents]
     if (p / "examples/hires_forestflow_evaluate.yaml").is_file()),
    None,
)
if ROOT is None:
    import lya_interface
    ROOT = Path(lya_interface.__file__).resolve().parents[2]
CUP1D_CONFIG = Path(get_path_repo("cup1d")) / "configs/hires/hires_forestflow.yaml"
INTERFACE_CONFIG = ROOT / "examples/hires_forestflow_evaluate.yaml"

# Everything this notebook writes, in the data folder:
# <OUTPUT_ROOT>/<data_label>/<variation or fit_type>/<emulator_label>
# (out_folder of hires_forestflow.yaml). OUTPUT_ROOT is $CFS/data on NERSC and
# <lya>/data on a laptop.
if "CFS" in os.environ:
    OUTPUT_ROOT = Path(os.environ["CFS"]) / "data"
else:
    OUTPUT_ROOT = Path(get_path_repo("cup1d")).parents[1] / "data"
OUT_DIR = OUTPUT_ROOT / "DESIY1_QMLE3__Karacayli2022/global_opt/forest_mpg_gp/cobaya_mcmc"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# best_fit.npy, gauss_newton_covmat.txt, proposal_covmat.txt, cobaya_mcmc.yaml,
# best_fit.png, chains/ and test/: the same files as scripts/prepare_mcmc.py
from lya_interface import mcmc_setup

paths = mcmc_setup.RunPaths(OUT_DIR)
BESTFIT_FILE = paths.best_fit
PRODUCTION_YAML = paths.production_yaml
TEST_PREFIX = paths.test_prefix

# Proposal covariance from another run (e.g. the full-data run when this is a
# data split): its Cobaya ``<prefix>.covmat``. Parameters are matched by name;
# those missing fall back to the YAML ``proposal`` widths. None: see section 1.
COVMAT_FROM = None

RUN_MINIMIZER = True  # minimize from the IC when no best fit is saved
# Redo the best fit even when one is saved (starting from it), and with it the
# Gauss-Newton errors, both covariances and the production YAML; a covariance
# learned by a previous chain is not reused. Use it after changing the
# configuration (priors, parameterization, data).
FORCE_MINIMIZATION = True
print("threads:", N_THREADS)
print("output folder:", OUT_DIR)
if not INTERFACE_CONFIG.is_file():
    raise FileNotFoundError(f"{INTERFACE_CONFIG} is missing: run notebooks/tests/hires_forestflow_validation")

# %% [markdown]
# ## Interface model
#
# With the saved best fit and local errors when available.

# %%
fit = mcmc_setup.load_fit(INTERFACE_CONFIG, paths)
info, model = fit.info, fit.model
names = list(fit.reference)  # sampled public names, in Cobaya order
print(len(names), "sampled parameters")

# %% [markdown]
# ## 1. Starting point and proposal covariance
#
# Cached on disk, so this is done once per configuration:
#
# * **Best fit** (`best_fit.npy`): minimized from the global IC when missing, then
#   saved together with its Gauss-Newton local errors.
# * **Gauss-Newton covariance** (`gauss_newton_covmat.txt`): regularized (symmetric,
#   positive definite, no standard deviation larger than a quarter of the prior
#   range, which matters for parameters at a bound).
# * **Proposal covariance** (`proposal_covmat.txt`), in order of preference:
#   1. `COVMAT_FROM`, e.g. the learned covariance of the full-data run for a
#      data split;
#   2. the covariance Cobaya learned in a previous production run of this
#      configuration (`chains/mcmc.covmat`);
#   3. the Gauss-Newton covariance.
#
# All are Cobaya covariance files (a `# name ...` header and the matrix), matched
# by parameter name, so a covariance from a run with a different parameter set can
# be reused: the parameters it lacks start from their YAML `proposal` widths.
# Learned covariances are in unit-cube units (`<name>_cube`); they are converted
# with this configuration's prior ranges, so reuse them only between runs with
# the same priors (e.g. data splits). `proposal_covmat.txt` is written in
# unit-cube units for the sampler; the others are physical.

# %%
start_point = mcmc_setup.ensure_best_fit(
    fit, paths, force=FORCE_MINIMIZATION, run_minimizer=RUN_MINIMIZER)

# %% [markdown]
# ### Best-fit check
#
# Goodness of fit recomputed at the best fit (degrees of freedom
# `N_data - N_sampled`), the compressed parameters (with the native DESI blinding
# offsets unless `UNBLIND = True`) and the P1D of each data set with its
# residuals. Error bars and per-panel chi2 use marginal (diagonal) blocks, so the
# panel chi2 need not add up to the correlated total in the title. The figure is
# saved as `best_fit.png` in `OUT_DIR`.

# %%
UNBLIND = False
fig = mcmc_setup.plot_best_fit(
    fit, paths, title="DESI DR1 + Karacayli2022, forest_mpg_gp: best fit", unblind=UNBLIND)
plt.show()

# %%
proposal_cov, proposal_names = mcmc_setup.write_proposal_covmat(
    fit, paths, covmat_from=COVMAT_FROM, force=FORCE_MINIMIZATION)

# %% [markdown]
# ## 2. Production YAML
#
# The production file overlays the generated interface YAML
# (`interface_base`) with the starting point, the sampler and the output. Chains
# start from a small Gaussian ball (0.1 sigma of the proposal) around the
# best fit (or the IC), so the MPI chains are independent.
#
# MCMC settings: the proposal covariance is learned on the fly (`learn_proposal`)
# and written to `chains/mcmc.covmat` at every checkpoint; that is the one to
# reuse for data splits or reruns (`COVMAT_FROM`). No dragging or
# oversampling (speed ratios < 2). The starting point is given in physical
# units and translated to the unit cube.
#
# **Convergence.** Cobaya runs several independent chains (one per MPI rank) and
# compares them (Gelman-Rubin): $R-1$ is, roughly, the spread of the parameter
# means *between* chains divided by the spread of the samples *within* a chain
# (for the worst combination of parameters). Chains that have all forgotten
# their starting point and sampled the same posterior give $R-1 \to 0$; with
# $R-1 = 0.02$ the means are known to ~$0.14\sigma$. The run stops when $R-1$ of
# the means is below `Rminus1_stop` and that of the 95% limits below
# `Rminus1_cl_stop`. Start with 0.02; to tighten to 0.01, change it in the YAML
# and resubmit: the run resumes from its checkpoint.
#
# **Running it** (`scripts/run_mcmc.py`):
#
# ```
# python <lya_interface>/scripts/run_mcmc.py <OUT_DIR>/cobaya_mcmc.yaml --test    # laptop, login node
# sbatch <lya_interface>/scripts/nersc/mcmc_job.sh                                # production, resubmits itself
# python <lya_interface>/scripts/run_mcmc.py <OUT_DIR>/cobaya_mcmc.yaml --status  # convergence so far
# ```
#
# Choose chains x threads per node first with `scripts/thread_scaling.py` on a
# compute node. The covariance and the output in the YAML are relative to
# `OUT_DIR`; the interface configuration is `lya_interface:<path>`, resolved
# against the lya_interface checkout of each machine, so `OUT_DIR` can be copied
# to `$CFS/data/...` on NERSC as it is. On NERSC the same files are produced
# by `sbatch scripts/nersc/prepare_job.sh` (`scripts/prepare_mcmc.py`).

# %%
mcmc_setup.write_production_yaml(
    fit, paths, INTERFACE_CONFIG, start_point, proposal_cov, proposal_names)

# %% [markdown]
# ## 3. Test run and time forecast
#
# **Per-block cost.** With the Cobaya model, time a call after changing only
# parameters of one kind; components whose inputs did not change are cached.
# Every call uses a new point: Cobaya also caches the last few results of each
# component, so repeating points would time cache hits.
#
# **Test chain.** Run it in a terminal with `scripts/run_mcmc.py` (printed below),
# not in this kernel: one implementation for the laptop, the NERSC login node
# and production, and the kernel does not hold Cobaya's output lock or a second
# emulator in memory. It runs the production YAML with only the output
# (`test/`, overwritten) and the length changed, and prints the acceptance
# rate, the time per step and a forecast for `--target-steps` steps per chain.
# It checks the full path (sampling, checkpoint and covariance files); section
# 4 then loads and plots it. It is not a converged posterior.

# %%
def time_block(model, point, block, n_repeat=6):
    """Seconds per log-posterior when only the parameters in ``block`` change."""
    model.logposterior(point)  # cache the starting point
    elapsed = 0.0
    for ii in range(n_repeat):
        moved = dict(point)
        step = (ii + 1) * 2e-4 * (-1) ** ii  # a new point at every call
        for name in block:
            low, high = info["params"][name]["prior"]["min"], info["params"][name]["prior"]["max"]
            moved[name] = float(np.clip(point[name] + step * (high - low), low, high))
        t0 = perf_counter()
        model.logposterior(moved)
        elapsed += perf_counter() - t0
    return elapsed / n_repeat


blocks = {
    "As, ns": [n for n in names if n in ("As", "ns")],
    "IGM": [n for n in names if n.startswith("igm_")],
    "contaminants + resolution": [n for n in names if not n.startswith("igm_") and n not in ("As", "ns")],
}
block_time = {}
for label, block in blocks.items():
    block_time[label] = time_block(model, start_point, block)
    print(f"{label:>26s} ({len(block):2d} params): {block_time[label]:.3f} s per call")

# %%
fit.close()  # free the minimization model and emulator before the test run
gc.collect()
print("run in a terminal, from the lya_interface folder:")
print(f"  python scripts/run_mcmc.py {os.path.relpath(PRODUCTION_YAML, ROOT)} --test --test-samples 1000")

# %% [markdown]
# ## 4. Load the chain from disk and plot
#
# Read back what the run wrote, as the production analysis will: the chain, the
# learned covariance, and the getdist samples. Star parameters are blinded.

# %%
os.path.relpath(PRODUCTION_YAML, ROOT)

# %%
from cobaya import load_samples
from getdist import plots

# written by the terminal test run of section 3
TEST_EXISTS = Path(f"{TEST_PREFIX}.updated.yaml").is_file()
if not TEST_EXISTS:
    print(f"no test chain at {TEST_PREFIX}; run in a terminal, from the lya_interface folder:")
    print(f"  python scripts/run_mcmc.py {os.path.relpath(PRODUCTION_YAML, ROOT)} --test --test-samples 1000")
else:
    collection = load_samples(str(TEST_PREFIX), skip=0, combined=True, to_getdist=False)
    data = collection.data
    print(f"read {len(data)} points; columns: {list(data.columns)[:6]} ...")
    print("learned covariance written:", Path(f"{TEST_PREFIX}.covmat").is_file())

    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(data["minuslogpost"].to_numpy())
    axes[0].set_ylabel(r"$-\log P$")
    # one parameter per speed block: a flat trace means its proposals are all rejected
    for ax, name in zip(axes[1:], ("As", "ns", "igm_tau_eff_0", "p1d_f_Lya_SiIII_0")):
        ax.plot(data[name].to_numpy())
        ax.set_ylabel(name.replace("_", r"\_"))
    axes[-1].set_xlabel("accepted step")
    plt.tight_layout()

# %%
if TEST_EXISTS:
    samples = load_samples(str(TEST_PREFIX), skip=0.3, to_getdist=True)
    labels = {"Delta2star": r"\Delta^2_\star", "nstar": r"n_\star", "alphastar": r"\alpha_\star"}
    for name, label in labels.items():
        if samples.getParamNames().parWithName(name) is not None:
            samples.getParamNames().parWithName(name).label = label

    # getdist drops parameters that never move (e.g. As/ns in a short test whose
    # steps in that block were all rejected), so plot only the ones it kept
    wanted = ["Delta2star", "nstar", "igm_tau_eff_0", "igm_tau_eff_3", "igm_sigT_kms_0"]
    kept = [name for name in wanted if samples.getParamNames().parWithName(name) is not None]
    if set(wanted) - set(kept):
        print("constant in this chain, not plotted:", sorted(set(wanted) - set(kept)))
    g = plots.get_subplot_plotter(width_inch=8)
    g.triangle_plot(samples, kept, filled=True)

# %% [markdown]
# ## Comparison with emcee (to do)
#
# The native emcee chains are the reference. Once a production Cobaya chain exists,
# convert both to getdist `MCSamples` and compare the marginals of the shared
# public parameters (native names map to public ones through
# `lya_interface.from_cup1d.public_names`).

# %% [markdown]
#

# %% [markdown]
#

# %% [markdown]
#
