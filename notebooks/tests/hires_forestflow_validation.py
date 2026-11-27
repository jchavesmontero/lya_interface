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
# # Native cup1d vs `lya_interface`: `hires_forestflow`
#
# Validates the interface configuration used by
# `notebooks/wip/hires_forestflow_cobaya_mcmc` before sampling: native cup1d and
# the interface must give the same P1D model and chi2 at the same parameter
# points. Run it again after changing the cup1d YAML, regenerating the interface
# YAML, or updating cup1d, ForestFlow or lya_interface.
#
# **Configuration.** The interface YAML
# `examples/hires_forestflow_evaluate.yaml` is generated from the native cup1d
# YAML by `lya_interface.from_cup1d` (`REGENERATE_CONFIG`). It points
# `native_config` at the cup1d file, so data, emulator, rebinning, `n_k_emu`,
# contaminants, `syst_data_labels` and covariance factors are read from cup1d;
# parameter bounds and references, the fixed cosmology and the frozen `M(z)` come
# from a native `Analysis`. Both use ForestFlow's `P1DIntegratorKMu`.
#
# **Points.** The native starting point (global IC) and points displaced from it
# by 0.01 and 0.03 of each prior range (the region a chain explores).
#
# **Requirement.** The rebinned P1D of the two pipelines must agree to
# $|P_{\rm interface}/P_{\rm native} - 1| < 10^{-4}$ in every bin. The chi2
# difference is reported too, but it is not the criterion: a fixed fractional
# model difference $\epsilon$ changes chi2 by
# $\approx 2\epsilon\,\sum (m-d)^T C^{-1} m$, which grows with the distance of the
# point from the data.
#
# **Known difference.** Both pipelines run the same CAMB with the same
# parameters, but native LaCE also requests CMB outputs (`Want_CMB`), which changes
# CAMB's transfer sampling and the linear power at $k < 1\,{\rm Mpc}^{-1}$ by up to
# $10^{-3}$ (both are within CAMB's default accuracy; see
# `notebooks/tests/linear_power_rescaling`). It shows up as a P1D offset of
# $3$–$7\times10^{-5}$, decreasing with redshift, and $|\Delta\chi^2| \lesssim 0.1$.

# %%
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

# Everything this notebook writes (explicit; output/ is not tracked by git).
OUT_DIR = ROOT / "output/hires_forestflow/validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGENERATE_CONFIG = False  # rewrite the interface YAML from the cup1d YAML
P1D_TOLERANCE = 1e-4  # max |P_interface / P_native - 1| in any bin
SHIFTS = (0.01, 0.03)  # displacements, as a fraction of each prior range
N_PER_SHIFT = 5
print("threads:", N_THREADS)

# %% [markdown]
# ## Native cup1d
#
# The native `Analysis` is freed before the interface model is built (two GP
# emulators at once do not fit comfortably in memory), so its chi2 and rebinned
# P1D are stored first.

# %%
from cup1d import Analysis, Args
from lya_interface.from_cup1d import public_names, write_interface_config

native_args = Args.from_yaml(str(CUP1D_CONFIG), verbose=False)
native_args.out_folder = str(OUT_DIR / "native")  # not the YAML's cwd-relative folder
analysis = Analysis(native_args)
like = analysis.fitter.like
if REGENERATE_CONFIG or not INTERFACE_CONFIG.is_file():
    write_interface_config(CUP1D_CONFIG, INTERFACE_CONFIG, analysis=analysis)
    print("wrote", INTERFACE_CONFIG)

to_public = public_names(CUP1D_CONFIG)
start_native = {name: p["value"] for name, p in like.free_params.items()}
bounds = {name: (p["min_value"], p["max_value"]) for name, p in like.free_params.items()}

rng = np.random.default_rng(0)
test_points = [("start", start_native)]
for shift in SHIFTS:
    for ii in range(N_PER_SHIFT):
        point = {}
        for name, value in start_native.items():
            low, high = bounds[name]
            point[name] = float(np.clip(value + shift * (high - low) * rng.normal(), low, high))
        test_points.append((f"shift {shift} #{ii}", point))

chi2_native, p1d_native = [], []
t0 = perf_counter()
for label, point in test_points:
    chi2_native.append(float(like.get_chi2(point)))
    prediction = like.get_p1d_kms(point)
    prediction = prediction[0] if isinstance(prediction, tuple) else prediction
    p1d_native.append({key: [np.asarray(row) for row in rows] for key, rows in prediction.items()})
time_native = (perf_counter() - t0) / len(test_points)
print(f"native: {time_native:.2f} s per point (chi2 + P1D)")

data_z = {key: np.asarray(dataset.z) for key, dataset in like.data.items()}
data_k = {key: [np.asarray(k) for k in dataset.k_kms] for key, dataset in like.data.items()}
del analysis, like
gc.collect();

# %% [markdown]
# ## Interface

# %%
from cobaya.model import get_model
from lya_interface.configuration import load_configuration
from lya_interface.fit_core import evaluate_point

info = load_configuration(INTERFACE_CONFIG)
model = get_model(info)

chi2_interface, p1d_interface = [], []
t0 = perf_counter()
for label, point in test_points:
    public = {to_public[name]: value for name, value in point.items()}
    try:
        result, powers = evaluate_point(model, public)
        chi2_interface.append(float(result.chi2_data))
        p1d_interface.append({key: [np.asarray(row) for row in rows] for key, rows in powers.items()})
    except Exception as error:  # rejected point: report it instead of stopping
        print(f"{label}: interface rejected the point ({error})")
        chi2_interface.append(np.inf)
        p1d_interface.append(None)
time_interface = (perf_counter() - t0) / len(test_points)
print(f"interface: {time_interface:.2f} s per point")

# %% [markdown]
# ## Comparison
#
# Per point: chi2 of both pipelines and the largest fractional P1D difference
# over all bins of all data sets.

# %%
def max_fraction(native, interface):
    """Signed a/b - 1 with the largest modulus over all data sets and bins."""
    if interface is None:
        return np.inf
    fractions = np.concatenate([interface[key][iz] / native[key][iz] - 1
                                for key in native for iz in range(len(native[key]))])
    return fractions[np.argmax(np.abs(fractions))]


worst_p1d = [max_fraction(a, b) for a, b in zip(p1d_native, p1d_interface)]
print(f"{'point':>16s} {'chi2 native':>12s} {'chi2 interface':>15s} {'delta chi2':>11s} {'max P1D frac':>13s}")
for (label, _), a, b, f in zip(test_points, chi2_native, chi2_interface, worst_p1d):
    print(f"{label:>16s} {a:12.3f} {b:15.3f} {b - a:11.4f} {f:13.2e}")

max_p1d = np.max(np.abs(worst_p1d))
max_delta = np.max(np.abs(np.array(chi2_interface) - np.array(chi2_native)))
print(f"\nmax |P1D fraction| = {max_p1d:.2e}, max |delta chi2| = {max_delta:.4f}")
print("PASS" if max_p1d < P1D_TOLERANCE else "FAIL", f"(tolerance {P1D_TOLERANCE:.0e} on the P1D)")

# %% [markdown]
# Fractional difference per bin at the starting point, one panel per data set and
# one line per redshift. A difference that grows with $k$ points to the
# observation model (resolution, contaminants); a smooth offset that depends on
# $z$ points to the linear power or the background.

# %%
START = 0
keys = list(p1d_native[START])
fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 4), sharey=True, squeeze=False)
for ax, key in zip(axes[0], keys):
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_z[key])))
    for iz, z in enumerate(data_z[key]):
        if p1d_interface[START] is None:
            break
        fraction = p1d_interface[START][key][iz] / p1d_native[START][key][iz] - 1
        ax.plot(data_k[key][iz], 1e5 * fraction, color=colors[iz], label=rf"$z={z:.1f}$")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_title(key.replace("_", r"\_"))
    ax.set_xlabel(r"$k_\parallel\ [{\rm s\,km^{-1}}]$")
axes[0, 0].set_ylabel(r"$P_{\rm interface}/P_{\rm native} - 1$ [$10^{-5}$]")
axes[0, -1].legend(fontsize=8, ncol=2, frameon=False)
plt.tight_layout()

# %%
