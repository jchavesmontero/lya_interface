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
# # Diagnostics of a Cobaya MCMC run
#
# For any output prefix written by `scripts/run_mcmc.py` (test or production):
#
# 1. **Run files** (not `.txt`): settings actually used (`.updated.yaml`),
#    convergence history (`.progress`: acceptance rate and $R-1$ at each check),
#    state (`.checkpoint`), and the proposal covariance learned by Cobaya
#    (`.covmat`) against the initial one.
# 2. **Chains**: traces of $-\log P$ and of a few parameters per chain, to see
#    the burn-in; per-parameter Gelman-Rubin $R-1$ and effective sample size.
# 3. **Posterior** (after discarding `BURN_IN`): triangle plot of the blinded
#    star parameters and the IGM, 1D marginals of every parameter within its
#    prior, and parameters whose posterior piles up at a prior bound.
#
# A test run is not converged: use it to check the machinery and the burn-in,
# not the posterior.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# output prefix: <folder>/<name> of the run (e.g. .../cobaya_mcmc/chains/mcmc)
RUN_DIR = Path("/home/jchaves/Proyectos/projects/lya/data/DESIY1_QMLE3__Karacayli2022/global_opt/forest_mpg_gp/cobaya_mcmc")
CHAIN_PREFIX = RUN_DIR / "test/mcmc"
# initial proposal covariance given to the sampler (unit-cube names), to compare
# with the learned one; None to skip
INITIAL_COVMAT = RUN_DIR / "proposal_covmat.txt"
BURN_IN = 0.3  # fraction of each chain discarded for the posterior
TRACE_PARAMS = ["As", "ns", "igm_tau_eff_0", "igm_sigT_kms_0", "p1d_f_Lya_SiIII_0", "p1d_R_coeff_0"]
N_TRACE_CHAINS = 8  # chains drawn in the trace plots

STAR_LABELS = {"Delta2star": r"\Delta^2_\star", "nstar": r"n_\star", "alphastar": r"\alpha_\star"}

# %% [markdown]
# ## 1. Run files

# %%
updated = yaml.safe_load(Path(f"{CHAIN_PREFIX}.updated.yaml").read_text())
mcmc = updated["sampler"]["mcmc"]
checkpoint = yaml.safe_load(Path(f"{CHAIN_PREFIX}.checkpoint").read_text())["sampler"]["mcmc"]
params = updated["params"]
sampled = [name for name, d in params.items() if isinstance(d, dict) and "prior" in d]
physical = [name[:-len("_cube")] for name in sampled if name.endswith("_cube")]

print("sampler settings used:")
for key in ("covmat", "learn_proposal", "learn_every", "proposal_scale", "drag", "oversample_power",
            "Rminus1_stop", "Rminus1_cl_stop", "max_samples", "seed"):
    print(f"  {key:>16s}: {mcmc.get(key)}")
print(f"\n{len(sampled)} sampled parameters"
      + (f", in the unit cube ({len(physical)} physical saved)" if physical else ""))
print("checkpoint:", checkpoint)

# columns: N, timestamp, acceptance rate, R-1 (means), R-1 (95% limits); skip the timestamp
progress = np.atleast_2d(np.loadtxt(f"{CHAIN_PREFIX}.progress", usecols=(0, 2, 3, 4)))
print(f"\n{len(progress)} convergence checks")
print(f"{'N (all chains)':>15s} {'acceptance':>11s} {'R-1 means':>10s} {'R-1 95% CL':>11s}")
for row in progress:
    print(f"{row[0]:15.0f} {row[1]:11.3f} {row[2]:10.3f} {row[3]:11.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
axes[0].plot(progress[:, 0], progress[:, 1], "o-")
axes[0].set_ylabel("acceptance rate")
axes[0].axhspan(0.15, 0.35, color="0.92")
axes[1].semilogy(progress[:, 0], progress[:, 2], "o-", label="means")
if np.any(np.isfinite(progress[:, 3])):
    axes[1].semilogy(progress[:, 0], progress[:, 3], "s-", label="95% limits")
axes[1].axhline(mcmc.get("Rminus1_stop") or 0.02, color="k", ls="--", lw=0.8, label="stop")
axes[1].set_ylabel(r"$R-1$")
axes[1].legend(frameon=False)
for ax in axes:
    ax.set_xlabel("accepted points, all chains")
plt.tight_layout()

# %% [markdown]
# **Learned proposal covariance.** Cobaya starts learning it once $R-1$ is below
# `learn_proposal_Rminus1_max_early` (30), later below
# `learn_proposal_Rminus1_max` (2). Identical to the initial one means it has
# not learned yet (normal for a short test); in production it should change.
# Ratios far from 1 show which directions the Gauss-Newton estimate got wrong.

# %%
def read_covmat(filename):
    """Cobaya covariance file: (matrix, names)."""
    with open(filename) as handle:
        names = handle.readline().lstrip("#").split()
    return np.loadtxt(filename), names


learned, learned_names = read_covmat(f"{CHAIN_PREFIX}.covmat")
print("learned covmat:", learned.shape, "symmetric:", np.allclose(learned, learned.T),
      "positive definite:", bool(np.all(np.linalg.eigvalsh(learned) > 0)))
if INITIAL_COVMAT is not None and Path(INITIAL_COVMAT).is_file():
    initial, initial_names = read_covmat(INITIAL_COVMAT)
    index = [initial_names.index(name) for name in learned_names]
    ratio = np.sqrt(np.diag(learned)) / np.sqrt(np.diag(initial))[index]
    if np.allclose(ratio, 1):
        print("identical to the initial proposal: not learned yet")
    else:
        order = np.argsort(ratio)
        print("sigma learned / initial: median %.2f" % np.median(ratio))
        for i in list(order[:5]) + list(order[-5:]):
            print(f"  {learned_names[i]:>28s}: {ratio[i]:.2f}")

sigma = np.sqrt(np.diag(learned))
fig, ax = plt.subplots(figsize=(6, 5))
image = ax.imshow(learned / np.outer(sigma, sigma), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_title("learned correlation matrix")
ax.set_xlabel("parameter index")
ax.set_ylabel("parameter index")
plt.colorbar(image, ax=ax)
plt.tight_layout()

# %% [markdown]
# ## 2. Chains: burn-in and mixing

# %%
from cobaya import load_samples

chains = load_samples(str(CHAIN_PREFIX), skip=0, combined=False)
chains = chains if isinstance(chains, list) else [chains]
lengths = [len(chain) for chain in chains]
print(f"{len(chains)} chains, accepted points per chain: min {min(lengths)}, max {max(lengths)}")
columns = list(chains[0].data.columns)
missing = [name for name in TRACE_PARAMS if name not in columns]
if missing:
    print("not in the chains (check the names):", missing)
trace_params = [name for name in TRACE_PARAMS if name in columns]

draw = chains[:N_TRACE_CHAINS]
fig, axes = plt.subplots(len(trace_params) + 1, 1, figsize=(10, 1.6 * (len(trace_params) + 1)), sharex=True)
for chain in draw:
    axes[0].plot(chain.data["minuslogpost"].to_numpy(), lw=0.6)
axes[0].set_ylabel(r"$-\log P$")
for ax, name in zip(axes[1:], trace_params):
    for chain in draw:
        ax.plot(chain.data[name].to_numpy(), lw=0.6)
    ax.set_ylabel(name.replace("_", r"\_"), fontsize=9)
for ax in axes:
    ax.axvline(BURN_IN * min(lengths), color="k", ls="--", lw=0.8)
axes[-1].set_xlabel("accepted point")
fig.suptitle(f"first {len(draw)} chains; dashed: burn-in cut ({BURN_IN:.0%})")
plt.tight_layout()

# %% [markdown]
# Per-parameter Gelman-Rubin after the burn-in: $R-1 \approx$ variance of the
# chain means / mean within-chain variance (Cobaya's criterion uses the worst
# combination of parameters, so it is larger). Values $\gg 0.02$: the chains
# have not sampled the same distribution yet.

# %%
def weighted_moments(chain, name, burn_in):
    data = chain.data.iloc[int(burn_in * len(chain)):]
    weights = data["weight"].to_numpy()
    values = data[name].to_numpy()
    mean = np.average(values, weights=weights)
    return mean, np.average((values - mean) ** 2, weights=weights)


rows = []
for name in sampled:
    moments = np.array([weighted_moments(chain, name, BURN_IN) for chain in chains])
    rows.append((name, np.var(moments[:, 0]) / np.mean(moments[:, 1])))
rows.sort(key=lambda row: -row[1])
print("largest per-parameter R-1 after burn-in:")
for name, value in rows[:10]:
    print(f"  {name:>30s}: {value:.3f}")

# %% [markdown]
# ## 3. Posterior

# %%
from getdist import plots

samples = load_samples(str(CHAIN_PREFIX), skip=BURN_IN, combined=True, to_getdist=True)
names = samples.getParamNames()
for name, label in STAR_LABELS.items():
    if names.parWithName(name) is not None:
        names.parWithName(name).label = label
print(f"{samples.numrows} points after burn-in, total weight {samples.norm:.0f}")

ess = {}
for name in physical:
    index = names.numberOfName(name)
    if index >= 0:
        try:
            ess[name] = samples.getEffectiveSamples(index)
        except Exception:  # very short or stuck chains
            pass
if ess:
    worst = sorted(ess.items(), key=lambda item: item[1])[:8]
    print("smallest effective sample sizes:", {k: round(v) for k, v in worst})

# %%
triangle = ["Delta2star", "nstar", "igm_tau_eff_0", "igm_sigT_kms_0", "igm_gamma_0", "igm_kF_kms_0"]
triangle = [name for name in triangle
            if names.parWithName(name) is not None and np.ptp(samples[name]) > 0]
g = plots.get_subplot_plotter(width_inch=9)
g.triangle_plot(samples, triangle, filled=True)
plt.suptitle("blinded star parameters" if {"Delta2star", "nstar"} & set(triangle) else "")

# %% [markdown]
# 1D marginals of every physical parameter over its full prior range (the x axis
# is the prior). Mass piled against an edge means the prior, not the data, sets
# the constraint there.

# %%
import re

NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def prior_range(name):
    """Physical prior range; from the unit-cube map 'lambda x_cube: low + width * x_cube'."""
    if name + "_cube" in params:
        body = params[name]["value"].split(":", 1)[1].split("*")[0]  # "low + width "
        low, width = (float(v) for v in re.findall(NUMBER, body.replace(" ", ""))[:2])
        return low, low + width
    return params[name]["prior"]["min"], params[name]["prior"]["max"]


plot_names = [name for name in physical if names.parWithName(name) is not None]
n_cols = 6
n_rows = int(np.ceil(len(plot_names) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols, 1.6 * n_rows))
near = []
for ax, name in zip(axes.flat, plot_names):
    low, high = prior_range(name)
    values = samples[name]
    ax.hist(values, bins=30, range=(low, high), weights=samples.weights, color="C0", alpha=0.7)
    ax.set_xlim(low, high)
    ax.set_yticks([])
    ax.set_title(name.replace("_", r"\_"), fontsize=8)
    ax.tick_params(labelsize=6)
    edge = 0.02 * (high - low)
    mass = (np.sum(samples.weights[values < low + edge]) + np.sum(samples.weights[values > high - edge])) / samples.norm
    if mass > 0.05:
        near.append((name, mass))
for ax in list(axes.flat)[len(plot_names):]:
    ax.axis("off")
plt.tight_layout()

print("posterior mass within 2% of a prior bound (> 5%):")
for name, mass in sorted(near, key=lambda item: -item[1]):
    print(f"  {name:>30s}: {mass:.0%}")
if not near:
    print("  none")

# %%
