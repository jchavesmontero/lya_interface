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
# # Accuracy of the As/ns rescalings of the linear power
#
# Neither pipeline runs a Boltzmann solver at each MCMC step. Instead, each one
# obtains $P_{\rm lin}(k, z)$ for new $(A_s, n_s)$ from a fixed computation:
#
# * **native cup1d**: `lace.cosmo.rescale_cosmology.RescaledCosmology` multiplies
#   the fiducial CAMB power by the ratio of primordial spectra (the call made by
#   `cup1d.theory.Theory.get_cosmology`);
# * **lya_interface**: Cobaya's CAMB theory caches the transfer functions
#   (`CambTransfers`, background only) and calls
#   `power_spectra_from_transfer()` with the new primordial spectrum. The
#   resulting `Pk_interpolator` is copied into a `CobayaCosmologySnapshot`
#   (cubic spline in $\ln k$ on `snapshot_nodes` points) by `LyaCosmology`.
#
# Each rescaling is compared with a **direct CAMB run** with the same settings
# as that pipeline: a new `lace` `Cosmology` for native, and a new Cobaya model
# with $A_s$, $n_s$ fixed for the interface. The direct runs from the two
# pipelines are also compared with each other, which separates the error of the
# rescaling itself from differences in CAMB settings and interpolation.
#
# Tests at $z = 2, 3, 4$. All settings come from the configurations used in
# the hires MCMC: the cup1d YAML (fiducial cosmology) and the generated
# interface YAML (fixed background, $A_s$/$n_s$ priors, CAMB and `LyaCosmology`
# options).

# %%
import os
from copy import deepcopy
from pathlib import Path
from time import perf_counter

N_THREADS = max(1, (os.cpu_count() or 4) - 2)
os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"
rcParams["font.size"] = 12

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "examples/hires_forestflow_evaluate.yaml").is_file())
INTERFACE_CONFIG = ROOT / "examples/hires_forestflow_evaluate.yaml"
CUP1D_CONFIG = (ROOT / "../cup1d/configs/hires/hires_forestflow.yaml").resolve()

ZS = (2.0, 3.0, 4.0)
# k_perp of the P3D -> P1D integral reaches 100 /Mpc; the Arinyo model and the
# emulator inputs live at k <~ 10 /Mpc
K_MPC = np.geomspace(1e-3, 100.0, 400)
K_BANDS = {"k < 10": K_MPC < 10.0, "10 <= k <= 100": K_MPC >= 10.0}
KP_MPC = 0.7  # forest_mpg_gp kp_iMpc: pivot of the emulator inputs Delta2_p, n_p

# %% [markdown]
# ## Settings from the MCMC configurations

# %%
from cup1d import Args
from lace.cosmo.cosmology import Cosmology
from lace.cosmo.rescale_cosmology import RescaledCosmology
from lya_interface.configuration import load_configuration

info = load_configuration(INTERFACE_CONFIG)
native_args = Args.from_yaml(str(CUP1D_CONFIG), verbose=False)
LYA_COSMOLOGY = "lya_interface.theory.lya_cosmology.LyaCosmology"

fixed = {name: value for name, value in info["params"].items() if not isinstance(value, dict)}
prior = {name: info["params"][name]["prior"] for name in ("As", "ns")}
camb_options = deepcopy(info["theory"]["camb"])
lya_cosmology_options = deepcopy(info["theory"][LYA_COSMOLOGY])

fid = Cosmology(cosmo_label=native_args.fid_cosmo_label)
fid_primordial = fid.get_primordial_params()
fid_background = fid.get_background_params()
print("native fiducial:", native_args.fid_cosmo_label)
print("interface fixed parameters:", fixed)
for name, value in fixed.items():
    if name in fid_background and not np.isclose(value, fid_background[name], rtol=1e-10, atol=0):
        print(f"  WARNING {name}: interface {value} vs native {fid_background[name]}")
print("Cobaya CAMB extra_args:", camb_options.get("extra_args"))
print("LyaCosmology:", lya_cosmology_options)

# %% [markdown]
# Test points: the fiducial, the MCMC reference point, the edges of the flat
# $A_s$/$n_s$ priors and two corners (largest rescaling).

# %%
As_fid, ns_fid = fid_primordial["As"], fid_primordial["ns"]
As_lo, As_hi = prior["As"]["min"], prior["As"]["max"]
ns_lo, ns_hi = prior["ns"]["min"], prior["ns"]["max"]
POINTS = {
    "fiducial": (As_fid, ns_fid),
    "MCMC ref": (info["params"]["As"]["ref"], info["params"]["ns"]["ref"]),
    "As min": (As_lo, ns_fid),
    "As max": (As_hi, ns_fid),
    "ns min": (As_fid, ns_lo),
    "ns max": (As_fid, ns_hi),
    "As max, ns min": (As_hi, ns_lo),
    "As min, ns max": (As_lo, ns_hi),
}
for label, (As, ns) in POINTS.items():
    print(f"{label:>15s}  As = {As:.4e}  ns = {ns:.4f}")

# %% [markdown]
# ## Native cup1d: `RescaledCosmology` vs direct CAMB (`lace` `Cosmology`)

# %%
def linear_power(cosmo):
    """P_lin(k, z) on K_MPC and emulator inputs at KP_MPC, for every z in ZS."""
    power = np.array([cosmo.get_linP_Mpc(z, K_MPC) for z in ZS])
    params = [cosmo.get_linP_Mpc_params(z, KP_MPC) for z in ZS]
    return dict(P=power,
                Delta2_p=np.array([p["Delta2_p"] for p in params]),
                n_p=np.array([p["n_p"] for p in params]))


native_rescaled, native_direct = {}, {}
t_rescaled = t_direct = 0.0
for label, (As, ns) in POINTS.items():
    t0 = perf_counter()
    native_rescaled[label] = linear_power(RescaledCosmology(fid, {"As": As, "ns": ns}))
    t_rescaled += perf_counter() - t0
    t0 = perf_counter()
    params = {**fid.input_cosmo_params_dict, "As": As, "ns": ns}
    native_direct[label] = linear_power(Cosmology(cosmo_params_dict=params))
    t_direct += perf_counter() - t0
print(f"native: rescaled {t_rescaled / len(POINTS):.2f} s, direct CAMB {t_direct / len(POINTS):.2f} s per point")

# %% [markdown]
# ## lya_interface: cached Cobaya transfers vs a direct Cobaya CAMB run
#
# A minimal Cobaya model (CAMB + `LyaCosmology` + a dummy likelihood requesting
# the snapshot at `ZS`) with the MCMC's CAMB and snapshot options.
#
# * *rescaled*: one model with $A_s$, $n_s$ sampled; the fiducial is evaluated
#   first, so every other point reuses the cached transfer functions;
# * *direct*: a new model per point with $A_s$, $n_s$ fixed, so CAMB computes
#   everything from scratch.
#
# The component call counts confirm that the transfers are computed once in the
# rescaled model.

# %%
from cobaya.model import get_model


def snapshot_probe(_self=None):
    """Dummy likelihood: only requests the cosmology snapshot."""
    return 0.0


def cobaya_info(As=None, ns=None):
    """Cobaya input with As/ns sampled (None) or fixed to the given values."""
    params = dict(fixed)
    params["As"] = {"prior": prior["As"]} if As is None else As
    params["ns"] = {"prior": prior["ns"]} if ns is None else ns
    return {
        "params": params,
        "theory": {"camb": deepcopy(camb_options), LYA_COSMOLOGY: deepcopy(lya_cosmology_options)},
        "likelihood": {"snapshot_probe": {
            "external": snapshot_probe,
            "requires": {"lya_cosmology": {"redshifts": list(ZS)}},
        }},
        "timing": True,
    }


def snapshot_power(model, point):
    model.logposterior(point)
    return linear_power(model.provider.get_result("lya_cosmology"))


rescaled_model = get_model(cobaya_info())
cobaya_rescaled = {"fiducial": snapshot_power(rescaled_model, {"As": As_fid, "ns": ns_fid})}
t0 = perf_counter()
for label, (As, ns) in POINTS.items():
    if label != "fiducial":
        cobaya_rescaled[label] = snapshot_power(rescaled_model, {"As": As, "ns": ns})
t_rescaled = (perf_counter() - t0) / (len(POINTS) - 1)

cobaya_direct = {}
t0 = perf_counter()
for label, (As, ns) in POINTS.items():
    cobaya_direct[label] = snapshot_power(get_model(cobaya_info(As, ns)), {})
t_direct = (perf_counter() - t0) / len(POINTS)
print(f"Cobaya: rescaled {t_rescaled:.2f} s, direct CAMB {t_direct:.2f} s per point (incl. model setup)")

print("\ncalls per component in the rescaled model (transfers should be 1):")
for component in rescaled_model.components:
    timer = getattr(component, "timer", None)
    print(f"  {component.get_name():>40s}: {getattr(timer, 'n', 'n/a')}")

# %% [markdown]
# ## Results
#
# Fractional differences of $P_{\rm lin}$, max over $k$ in each band, in units
# of $10^{-5}$:
#
# * **native resc.**: native rescaled / native direct $-1$ (native rescaling error);
# * **Cobaya resc.**: Cobaya rescaled / Cobaya direct $-1$ (interface rescaling error);
# * **direct**: Cobaya direct / native direct $-1$ (CAMB settings and
#   interpolation, no rescaling);
# * **pipelines**: Cobaya rescaled / native rescaled $-1$ (what enters the two
#   likelihoods).

# %%
COMPARISONS = {
    "native resc.": (native_rescaled, native_direct),
    "Cobaya resc.": (cobaya_rescaled, cobaya_direct),
    "direct": (cobaya_direct, native_direct),
    "pipelines": (cobaya_rescaled, native_rescaled),
}


def max_frac(a, b, mask):
    """Signed fractional difference a/b - 1 with the largest modulus over mask."""
    frac = (a / b - 1)[mask]
    return frac[np.argmax(np.abs(frac))]


for band, mask in K_BANDS.items():
    print(f"\nmax |P_lin ratio - 1| x 1e5, {band} /Mpc")
    header = f"{'point':>15s} {'z':>4s}" + "".join(f"{name:>14s}" for name in COMPARISONS)
    print(header)
    for label in POINTS:
        for iz, z in enumerate(ZS):
            row = f"{label:>15s} {z:4.1f}"
            for a, b in COMPARISONS.values():
                row += f"{1e5 * max_frac(a[label]['P'][iz], b[label]['P'][iz], mask):14.2f}"
            print(row)

# %% [markdown]
# Emulator inputs at $k_p = 0.7\,{\rm Mpc}^{-1}$, fractional differences in
# units of $10^{-6}$. Native `RescaledCosmology` has its own
# `get_linP_Mpc_params`, so this also tests the analytic shift of
# $\Delta^2_p$, $n_p$ that native cup1d feeds to the emulator.

# %%
for name in ("Delta2_p", "n_p"):
    print(f"\n{name}: (ratio - 1) x 1e6")
    print(f"{'point':>15s} {'z':>4s}" + "".join(f"{c:>14s}" for c in COMPARISONS))
    for label in POINTS:
        for iz, z in enumerate(ZS):
            row = f"{label:>15s} {z:4.1f}"
            for a, b in COMPARISONS.values():
                row += f"{1e6 * (a[label][name][iz] / b[label][name][iz] - 1):14.2f}"
            print(row)

# %% [markdown]
# Ratios as a function of $k$: one column per redshift and one row per test
# point (fiducial, MCMC reference, and the corner with the largest rescaling).

# %%
PLOT_POINTS = ("fiducial", "MCMC ref", "As max, ns min")
STYLES = {"native resc.": "C0", "Cobaya resc.": "C1", "direct": "C2", "pipelines": "k"}

fig, axes = plt.subplots(len(PLOT_POINTS), len(ZS), figsize=(4 * len(ZS), 2.8 * len(PLOT_POINTS)),
                         sharex=True, sharey="row")
for row, label in enumerate(PLOT_POINTS):
    for iz, z in enumerate(ZS):
        ax = axes[row, iz]
        for name, (a, b) in COMPARISONS.items():
            ax.plot(K_MPC, 1e5 * (a[label]["P"][iz] / b[label]["P"][iz] - 1),
                    color=STYLES[name], lw=1.2, ls="--" if name == "pipelines" else "-",
                    label=name)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.axvline(KP_MPC, color="0.6", lw=0.6, ls=":")
        ax.set_xscale("log")
        if row == 0:
            ax.set_title(rf"$z = {z:.0f}$")
        if iz == 0:
            ax.set_ylabel(rf"{label}" "\n" r"$P_{\rm lin}$ ratio $- 1$ [$10^{-5}$]")
        if row == len(PLOT_POINTS) - 1:
            ax.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")
axes[0, 0].legend(fontsize=9, frameon=False)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Why the direct runs differ: CAMB settings
#
# Both pipelines run the same CAMB package. The cell below compares every field
# of the `CAMBparams` each one builds. All cosmological fields agree; only the
# output requests differ:
#
# * native LaCE computes the matter power at 256 redshifts in $[0, 10]$ with
#   `kmax = 200.2` and also requests CMB outputs (`Want_CMB`/`WantCls`);
# * Cobaya computes it only at the requested redshifts, with `kmax = 200` and no
#   CMB outputs.
#
# Requesting CMB outputs changes how CAMB samples $k$ and time for the transfer
# functions, and with it the linear power at $k < 1\,{\rm Mpc}^{-1}$ by up to
# $\sim 10^{-3}$ (largest near the BAO scale). The other settings and both
# interpolation chains contribute $< 10^{-6}$. A run with `AccuracyBoost = 3` shows
# that both default runs are within CAMB's default accuracy, so neither is
# "right"; the difference only matters for bitwise agreement between pipelines.

# %%
import re

import camb


def camb_fields(params):
    """CAMBparams as {field: printed values}, from its text representation."""
    fields = {}
    for line in str(params).splitlines():
        match = re.match(r"\s*(\w+)\s*=\s*(.*)", line)
        if match:
            fields.setdefault(match.group(1), []).append(match.group(2).strip())
    return fields


probe_info = cobaya_info(As_fid, ns_fid)
probe_info["likelihood"]["snapshot_probe"]["requires"]["CAMBdata"] = None
probe_model = get_model(probe_info)
probe_model.logposterior({})
cobaya_params = probe_model.provider.get_CAMBdata().Params
fid._ensure_camb_results(full=True)
native_params = fid.CAMBdata.Params

print("camb:", camb.__version__, camb.__file__)
native_fields, cobaya_fields = camb_fields(native_params), camb_fields(cobaya_params)
print("CAMBparams fields that differ (native | Cobaya):")
for name in sorted(set(native_fields) | set(cobaya_fields)):
    a, b = native_fields.get(name), cobaya_fields.get(name)
    if a != b:
        short = lambda v: v if v is None or len(str(v)) < 80 else f"<{len(v)} entries>"
        print(f"  {name}: {short(a)} | {short(b)}")


# %% [markdown]
# One setting at a time, at the fiducial point and $z = 3$, all relative to plain
# CAMB with Cobaya's settings, plus a high-accuracy reference.

# %%
from scipy.interpolate import CubicSpline

Z_TEST = 3.0
COBAYA_ZS = sorted(set(lya_cosmology_options.get("redshifts", [])) | set(ZS) | {3.0}, reverse=True)
BANDS = [(1e-3, 1e-2), (1e-2, 0.05), (0.05, 0.2), (0.2, 1.0), (1.0, 10.0), (10.0, 100.0)]


def camb_power(want_cmb=False, kmax=200.0, boost=1.0):
    """Linear delta_nonu power at Z_TEST from CAMB's own grid (no z interpolation)."""
    params = native_params.copy()
    params.set_accuracy(AccuracyBoost=boost)
    params.WantCls = params.Want_CMB = want_cmb
    params.set_matter_power(redshifts=COBAYA_ZS, kmax=kmax, nonlinear=False, silent=True)
    k, z, power = camb.get_results(params).get_linear_matter_power_spectrum(
        var1="delta_nonu", var2="delta_nonu", hubble_units=False, k_hunit=False)
    row = power[int(np.argmin(np.abs(np.asarray(z) - Z_TEST)))]
    return np.exp(CubicSpline(np.log(k), np.log(row))(np.log(K_MPC)))


reference = camb_power()
variants = {
    "Want_CMB = True (native)": camb_power(want_cmb=True),
    "kmax = 200.2 (native)": camb_power(kmax=200.2),
    "AccuracyBoost = 3": camb_power(boost=3.0),
    "Want_CMB = True, AccuracyBoost = 3": camb_power(want_cmb=True, boost=3.0),
}
print("signed extreme of P / P_Cobaya-settings - 1 per k band [1/Mpc]")
print(f"{'':36s}" + "".join(f"{f'{a:g}-{b:g}':>12s}" for a, b in BANDS))
for name, power in variants.items():
    fraction = power / reference - 1
    row = ""
    for a, b in BANDS:
        values = fraction[(K_MPC >= a) & (K_MPC < b)]
        row += f"{values[np.argmax(np.abs(values))]:12.1e}"
    print(f"{name:36s}{row}")

# %% [markdown]
# ## Interpretation
#
# * A rescaling is exact in principle (linear theory: $P = T^2(k, z)\,
#   P_{\rm prim}(k)$ with the background fixed). **native resc.** and **Cobaya
#   resc.** should therefore be at the level of interpolation noise and the
#   same at every test point.
# * **direct** carries the pipeline difference. It comes from the fixed CAMB
#   computation (native LaCE also requests CMB outputs, which changes the
#   transfer sampling), not from the rescaling or the interpolation. Making the
#   two pipelines request the same CAMB outputs is the way to remove it.
# * For reference, the P1D of the two pipelines differs by $3$–$6\times10^{-5}$
#   (decreasing with $z$) at every MCMC test point, which gives
#   $|\Delta\chi^2| \lesssim 0.08$.
