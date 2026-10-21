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
# # Kaiser silicon model shared by P1D and ξ3D
#
# One line-of-sight Kaiser model of the silicon contamination is used by the DESI
# DR1 P1D likelihood (cup1d, `metal_model_type: Kaiser`) and by the DESI DR2
# Lyα×Lyα correlation function (Vega). For components a, b (Lyα or a silicon
# transition):
#
# $$P_{ab}(k,\mu) = b_a b_b\,(1+\beta_a\mu^2)(1+\beta_b\mu^2)\,P_L(k)\,D_{ab}\,W_{ab}(k_\parallel)$$
#
# - Lyα×Lyα is ForestFlow's model. Per leg, Lyα carries $\sqrt{D_{\rm NL}}$ and a
#   metal the gas pressure smoothing $d_m=e^{-(k/k_p)^2/2}$ (ForestFlow's Arinyo
#   $k_p$; metals trace the same gas): $D_{\alpha m}=\sqrt{D_{\rm NL}}\,d_m$ and
#   $D_{mn}=d_m d_n$.
# - $W_{\alpha m}(k_\parallel)=\exp(-k_\parallel^2\sigma_m^2/2)$ is a line-of-sight
#   decorrelation between Lyα and metal $m$ (distinct Lyα and metal line profiles,
#   Ma et al. 2026), with one σ for SiIII and one for the SiII lines. It acts only
#   on Lyα×metal: metal autos and crosses between metal lines are not decorrelated.
# - One flux bias per transition and one β and σ per species. As the DR1 metal
#   and HCD parameters, each is a cup1d redshift history: two nodes (z=2.2, 4.2)
#   interpolated linearly, of $\ln|b|$ and $\ln\sigma$ (power-law-like). Vega uses
#   the same histories evaluated at its $z_{\rm eff}=2.33$.
# - All pairs closer than 250 Mpc/h are included (Lyα with SiIII(1207),
#   SiII(1190, 1193, 1260) and all Si×Si pairs); CIV is not in P1D.
#
# **P1D.** The line-of-sight displacement becomes $\cos(k\,\Delta v)$, and since $W$
# depends only on $k_\parallel$ the metal parameters factor out of the transverse
# integral:
#
# $$P^{\rm 1D}_{\alpha m}=b_m W_{\alpha m}\,(A_0+\beta_m A_1),\qquad
# P^{\rm 1D}_{mn}=b_m b_n\,[I_0+(\beta_m+\beta_n)I_1+\beta_m\beta_n I_2]$$
#
# with $A_j=\frac{1}{2\pi}\int dk_\perp k_\perp P_L\sqrt{D_{\rm NL}}\,d_m(b+b_\eta f\mu^2)\mu^{2j}$
# and $I_j=\frac{1}{2\pi}\int dk_\perp k_\perp P_L d_m^2\mu^{2j}$, computed in the same
# quadrature as the Lyα P1D. **ξ3D.** Vega's metal power spectra get
# `small scale nl = forestflow` (√D_NL on Lyα×metal and $d_m$ per metal leg) and
# the Lyα×metal decorrelation from lya_interface; σ in km/s is converted to Mpc/h with the
# fixed fiducial cosmology of the data coordinates.
#
# The bias, β and σ histories are single parameters shared by both likelihoods
# (`examples/joint_dr1p1d_forestmpggp_dr2bao_lya_kaiser.yaml`); Vega also takes
# the growth rate from the sampled cosmology. This notebook (1) validates the
# moment approach against the direct integral of the contaminated P3D and its
# impact on the DESI DR1 χ², (2) checks that Vega uses exactly the same 3D
# kernel, and (3) shows by eye how each metal parameter changes P1D and ξ3D at
# the same time.

# %%
# %load_ext autoreload
# %autoreload 2

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.integrate import simpson
from forestflow.model.arinyo import ArinyoModel
from forestflow.model.linear import LinearTheoryGrid
from vega import utils as vega_utils
from cup1d.models.contaminants.metals.si_kaiser import velocity_offset
from cup1d.likelihood.external import _quadrature, project_fused
from lya_interface.fit import Fit
from lya_interface.initial_conditions import global_ic_point
from lya_interface.parameters import route
from lya_interface.diagnostics import cup1d_component, evaluate_point, vega_component

ROOT = next(
    (p for p in [Path.cwd(), *Path.cwd().parents]
     if (p / "examples/joint_dr1p1d_forestmpggp_dr2bao_lya_kaiser.yaml").is_file()),
    None,
)
if ROOT is None:
    import lya_interface
    ROOT = Path(lya_interface.__file__).resolve().parents[2]
CONFIG = ROOT / "examples/joint_dr1p1d_forestmpggp_dr2bao_lya_kaiser.yaml"
print("Configuration:", CONFIG)

# %% [markdown]
# ## Build the joint model and the reference point
#
# IGM and P1D nuisances from the `forest_mpg_gp` initial conditions; the metal
# parameters (not in that file) and the BAO nuisances at their references.

# %%
fit = Fit(CONFIG)
model = fit.model
vega = vega_component(model)
p1d = cup1d_component(model)
kaiser = p1d.backend.model_cont.metal_models["Si_kaiser"]
reference_point = global_ic_point(fit.info, fit.reference)
METALS = [name for name in fit.reference if name.startswith("metal_")]
display({name: reference_point[name] for name in METALS})
print("km/s -> Mpc/h for Vega:", vega.kms_to_Mpch)


def p1d_state(point):
    """Evaluate the graph at ``point`` and return the pieces of the P1D model."""
    model.logposterior(point, return_derived=False)
    prediction = model.provider.get_result("forestflow_p1d")
    values = model.parameterization.input_params()
    nuisance = route({k: values[k] for k in p1d.mapping}, p1d.mapping)
    return prediction, nuisance

# %% [markdown]
# ## 1. Validate the moments against the direct integral of the contaminated P3D
#
# For every DR1 redshift we build the full contaminated P3D (Lyα + every
# Lyα×metal term with its $\cos(k_\parallel\Delta r)$ + metal autos + metal
# crosses) and integrate it over $k_\perp$ directly, then compare with
# Lyα P1D + moments-based metal terms:
#
# - **(a) identity**: same quadrature as production (algebra check, should agree to
#   machine precision);
# - **(b) convergence**: an independent dense quadrature (6001 Simpson nodes in
#   $\ln k_\perp$ on $[10^{-5}, 190]$ /Mpc), which measures the accuracy of the
#   production quadrature itself.
#
# `METAL_BOOST` multiplies the silicon biases (both nodes) to make the test
# sensitive.

# %%
METAL_BOOST = 3.0
point = dict(reference_point)
for name in METALS:
    if name.startswith("metal_b_Si"):
        point[name] += np.log(METAL_BOOST)  # coefficients are ln|b|
prediction, nuisance = p1d_state(point)
snapshot = model.provider.get_result("lya_cosmology")
forestflow = next(c for c in model.theory.values() if c.__class__.__name__ == "ForestFlowTheory")
arinyo = forestflow.get_result("forestflow_arinyo")
group = p1d.backend.request.groups[0]


def direct_contaminated_p1d(z, k_kms, ln_k_perp=None):
    """k_perp integral of the full contaminated P3D, in km/s.

    With ``ln_k_perp=None`` the production quadrature is used.
    """
    M = float(snapshot.get_dkms_diMpc(np.array([z]))[0])
    arinyo_model = ArinyoModel(fiducial_cosmology=snapshot)
    linear = LinearTheoryGrid(z=np.array([z]), cosmology=snapshot)
    if ln_k_perp is None:
        k3d, mu, integrate = _quadrature(forestflow.integrator, (np.asarray(k_kms) * M)[None, :])
        k3d, mu = k3d[0], mu[0]
        k_par = k3d * mu
    else:
        k_par, k_perp = np.meshgrid(np.asarray(k_kms) * M, np.exp(ln_k_perp), indexing="ij")
        k3d = np.hypot(k_par, k_perp)
    mu2 = (k_par / k3d) ** 2
    lin = arinyo_model.linear.get_linP_Mpc(linear, np.array([z]), k3d[None])[0]
    growth = arinyo_model.linear.get_fz(linear, np.array([z]))[0]
    pars = arinyo_model.default_params | {k: np.asarray(v) for k, v in arinyo[z].items()}
    dnl = arinyo_model.nonlinear_correction(lin, k3d, np.sqrt(mu2), pars)
    lya_kaiser = pars["bias"] + pars["bias_eta"] * growth * mu2
    leg = np.exp(-0.5 * (k3d / pars["kp"]) ** 2)  # pressure smoothing of one metal leg
    par = kaiser.coefficients(z, nuisance)
    metal_kaiser = {l: b * (1 + beta * mu2) * leg for l, (b, beta, s) in par.items()}
    window = {l: np.exp(-0.5 * (k_par / M * s) ** 2) for l, (b, beta, s) in par.items()}
    p3d_lya = lin * dnl * lya_kaiser**2
    p3d = p3d_lya.copy()
    lines = list(par)
    for l in lines:
        p3d += (2 * np.cos(k_par / M * velocity_offset("Lya", l)) * lin * np.sqrt(dnl)
                * lya_kaiser * metal_kaiser[l] * window[l])
    for i, a in enumerate(lines):  # metal x metal: no decorrelation
        p3d += lin * metal_kaiser[a] ** 2
        for b in lines[i + 1:]:
            p3d += (2 * np.cos(k_par / M * velocity_offset(a, b)) * lin
                    * metal_kaiser[a] * metal_kaiser[b])
    if ln_k_perp is None:
        project = lambda p: integrate(p[None])[0]
    else:
        project = lambda p: simpson(p * k_perp**2 / (2 * np.pi), x=ln_k_perp, axis=-1)
    return project(p3d) * M, project(p3d_lya) * M


# The model projects on 64 k per redshift and interpolates (n_k_emu); the
# algebra check uses the projection at every k (its accuracy is in the chi2 table).
full_lya, full_moments = project_fused(
    p1d.backend.request, snapshot, arinyo, kmax_iMpc=forestflow.emulator.kmax_1d_iMpc,
    integrator=forestflow.integrator, moments=True)
dense = np.linspace(np.log(1e-5), np.log(190.0), 6001)
rows = []
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for iz, (z, k) in enumerate(zip(group.redshifts, group.k_ikms)):
    k = np.asarray(k)
    lya = np.asarray(full_lya[group.identifier][iz])
    moments = full_moments[group.identifier][iz]
    metal = kaiser.get_contamination([z], [k], like_params=nuisance, metal_moments=[moments])[0]
    approx = lya + metal
    same_total, same_lya = direct_contaminated_p1d(z, k)
    dense_total, dense_lya = direct_contaminated_p1d(z, k, dense)
    rows.append(dict(
        z=z,
        identity_max_rel=float(np.max(np.abs(approx / same_total - 1))),
        convergence_max_rel=float(np.max(np.abs(approx / dense_total - 1))),
        metal_convergence_max_rel=float(np.max(np.abs(metal - (dense_total - dense_lya)))
                                        / np.max(np.abs(dense_total - dense_lya))),
        max_metal_fraction=float(np.max(np.abs(metal / approx))),
    ))
    axes[0].plot(k, approx / dense_total - 1, label=f"z={z:.1f}")
    axes[1].plot(k, metal / lya, label=f"z={z:.1f}")
axes[0].set(xscale="log", xlabel=r"$k_\parallel$ [s/km]", ylabel="moments / direct (dense) - 1",
            title="Moment approach vs direct integral")
axes[1].set(xscale="log", xlabel=r"$k_\parallel$ [s/km]", ylabel=r"metal terms / $P_{1D}^{\rm Ly\alpha}$",
            title=f"Metal contribution (biases x{METAL_BOOST:g})")
axes[1].legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.show()
display(rows)
assert max(r["identity_max_rel"] for r in rows) < 1e-8, "moment algebra disagrees with the direct integral"
assert max(r["convergence_max_rel"] for r in rows) < 5e-3, "production quadrature not converged"

# %% [markdown]
# ### Impact on the DESI DR1 χ²
#
# The requirement is $|\Delta\chi^2|<0.05$. The production projection uses the
# converged quadrature on 64 log-spaced k per redshift, interpolated to the
# rebinning grid. At the reference point, the DR1 χ² is recomputed projecting
# at every k, and with the Lyα P1D and the metal moments from a dense quadrature
# (split Gauss-Legendre in ln k, 150+150 nodes on $k_\perp\in[10^{-5},190]$ /Mpc,
# projected one redshift at a time). With the metal pressure smoothing every
# integrand is negligible well before $k_\perp=190$ /Mpc.

# %%
from cup1d.likelihood.external import (
    PredictionGroup, PredictionRequest, make_integrator)

prediction, nuisance = p1d_state(reference_point)
arinyo = forestflow.get_result("forestflow_arinyo")
kmax_1d = forestflow.emulator.kmax_1d_iMpc


def project(integrator):
    """Lyα P1D and metal moments, one redshift at a time (bounded memory)."""
    lya, moments = {}, {}
    for g in p1d.backend.request.groups:
        rows_l, rows_m = [], []
        for z, k in zip(g.redshifts, g.k_ikms):
            request = PredictionRequest((PredictionGroup(g.identifier, (z,), (k,)),),
                                        p1d.backend.request.identity)
            row_lya, row_moments = project_fused(request, snapshot, arinyo, kmax_iMpc=kmax_1d,
                                                 integrator=integrator, moments=True)
            rows_l.append(row_lya[g.identifier][0])
            rows_m.append(row_moments[g.identifier][0])
        lya[g.identifier], moments[g.identifier] = tuple(rows_l), tuple(rows_m)
    return lya, moments


def dr1_chi2(lya, moments):
    return p1d.backend.evaluate_from_p1d(lya, prediction["context"], nuisance, moments).chi2_data


dense_lya, dense_moments = project(make_integrator(dict(
    rule="k_mu", k_perp_min_iMpc=1e-5, k_perp_max_iMpc=190.0, n_low=150, n_high=150)))
production = dr1_chi2(prediction["powers"], prediction["metal_moments"])
reference = dr1_chi2(dense_lya, dense_moments)
every_k_lya, every_k_moments = project(forestflow.integrator)
chi2_table = {
    "chi2 (production: quadrature + n_k interpolation)": production,
    "delta chi2: n_k interpolation (vs every k)": production - dr1_chi2(every_k_lya, every_k_moments),
    "delta chi2: Lya P1D quadrature": dr1_chi2(prediction["powers"], dense_moments) - reference,
    "delta chi2: metal-moment quadrature": dr1_chi2(dense_lya, prediction["metal_moments"]) - reference,
    "delta chi2: total quadrature": production - reference,
}
display(chi2_table)

# %% [markdown]
# ## 2. Check that Vega uses the same 3D kernel
#
# For every Lyα×metal and metal×metal pair, Vega's metal power spectrum divided by
# its linear power and binning factor must equal
# $b_a b_b(1+\beta_a\mu^2)(1+\beta_b\mu^2)D_{ab}W_{ab}$ evaluated with the same
# inputs (√D_NL for the Lyα leg, the pressure smoothing for each metal leg). The
# growth rate entering the Lyα β is that of the sampled cosmology in both.

# %%
model.logposterior(reference_point, return_derived=False)
native = vega.parameters_for_evaluation({k: reference_point[k] for k in vega.public_parameters})
local = vega.backend._get_lcl_prms(native)
local["peak"] = False
pk_lin = vega.backend.fiducial["pk_full"]
coefficients = model.provider.get_result("forestflow_coefficients")[vega.zeff]
h = vega.template_h
kernel_checks = {}
for pair, spectrum in vega.backend.models["lyaxlya"].metals.Pk_metal.items():
    vega_kernel = spectrum.compute(pk_lin, local, fast_metals=False)
    vega_kernel = vega_kernel / (pk_lin * (spectrum.pk_Gk if spectrum.use_Gk else 1.0))
    b1, beta1, b2, beta2 = vega_utils.bias_beta(local, *pair)
    mu2 = spectrum.muk_grid**2
    ours = b1 * b2 * (1 + beta1 * mu2) * (1 + beta2 * mu2)
    if "LYA" in pair:
        ours = ours * np.sqrt(ArinyoModel.nonlinear_correction(
            (pk_lin / h**3)[None, :], (h * spectrum.k_grid)[None, :], spectrum.muk_grid, coefficients))
    metal_legs = sum("LY" not in tracer for tracer in pair)
    ours = ours * np.exp(-0.5 * (h * spectrum.k_grid / coefficients["kp"]) ** 2) ** metal_legs
    if "LYA" in pair:  # decorrelation of Lya x metal only
        metal = next(tracer for tracer in pair if tracer != "LYA")
        sigma = native["forestflow_metal_decorrelation"].get(metal, 0.0)
        ours = ours * np.exp(-0.5 * (spectrum.k_par_grid * sigma) ** 2)
    good = np.abs(ours) > 1e-12 * np.abs(ours).max()
    kernel_checks[pair] = float(np.max(np.abs(vega_kernel[good] / ours[good] - 1)))
display(kernel_checks)
assert max(kernel_checks.values()) < 1e-10, "Vega and the P1D model use different 3D kernels"

# %% [markdown]
# ## 3. The metal terms in P1D, one by one
#
# Each additive term divided by the Lyα P1D at three redshifts, at the reference
# point (crosses include their $2\cos(k\Delta v)$ oscillation).

# %%
prediction, nuisance = p1d_state(reference_point)
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, iz in zip(axes, (0, 4, 9)):
    z, k = group.redshifts[iz], np.asarray(group.k_ikms[iz])
    lya = np.asarray(prediction["powers"][group.identifier][iz])
    terms = kaiser.p1d_terms(z, k, prediction["metal_moments"][group.identifier][iz], nuisance)
    for name, term in terms.items():
        if np.max(np.abs(term / lya)) > 1e-4:
            ax.plot(k, term / lya, label=name.replace("_", "×"))
    ax.set(xscale="log", title=f"z={z:.1f}", xlabel=r"$k_\parallel$ [s/km]")
axes[0].set_ylabel(r"term / $P_{1D}^{\rm Ly\alpha}$")
axes[-1].legend(fontsize=7)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Vary each metal parameter in P1D and ξ3D at the same time
#
# For each parameter, the top row shows the full P1D model (on the data bins,
# after contaminants, resolution and rebinning) relative to the reference model at
# three redshifts, and the next figure overlays the Vega Lyα×Lyα wedges for the
# same values on the DR2 data. Edit `SCANS` to try other values.

# %%
# Each scan sets a whole family (both redshift nodes). Bias scans multiply |b|
# (the coefficients are ln|b|); beta and sigma scans set the value directly.
SCANS = {
    "b_SiIII": ("factor", [0.5, 1.0, 2.0]),
    "b_SiIIa": ("factor", [0.5, 1.0, 2.0]),
    "b_SiIIb": ("factor", [0.5, 1.0, 2.0]),
    "b_SiIIc": ("factor", [0.5, 1.0, 2.0]),
    "beta_SiIII": ("value", [0.0, 0.5, 1.5]),
    "beta_SiII": ("value", [0.0, 0.5, 1.5]),
    "sig_SiIII": ("value", [10.0, 100.0, 300.0]),
    "sig_SiII": ("value", [10.0, 100.0, 300.0]),
}
P1D_REDSHIFT_INDICES = (0, 4, 9)


def vega_lyaxlya(point):
    """Vega Lyα×Lyα model at ``point`` (all native parameters, shared ForestFlow)."""
    model.logposterior(point, return_derived=False)
    values = vega.parameters_for_evaluation({k: point[k] for k in vega.public_parameters})
    return vega.backend.compute_model(values, run_init=False)["lyaxlya"]


def family_point(family, kind, value):
    """Reference point with every node of ``family`` changed."""
    point = dict(reference_point)
    for name in fit.reference:
        if name.startswith(f"metal_{family}_"):
            if kind == "factor":
                point[name] = reference_point[name] + np.log(value)
            elif family.startswith("sig_"):
                point[name] = np.log(value)  # coefficients are ln(sigma / km/s)
            else:
                point[name] = value
    return point


def scan(name, kind, values):
    """Show the effect of family ``name`` on P1D and on the Lyα×Lyα wedges."""
    reference_p1d = evaluate_point(model, reference_point)[1]
    data = p1d.backend.data
    key = next(iter(data))
    fig, axes = plt.subplots(1, len(P1D_REDSHIFT_INDICES), figsize=(15, 3.5), sharey=True)
    models, labels = [], []
    for value in values:
        point = family_point(name, kind, value)
        powers = evaluate_point(model, point)[1]
        for ax, iz in zip(axes, P1D_REDSHIFT_INDICES):
            k = np.asarray(data[key].k_kms[iz])
            ax.plot(k, np.asarray(powers[key][iz]) / np.asarray(reference_p1d[key][iz]),
                    label=f"{value:g}")
            ax.set(xscale="log", title=f"P1D, z={data[key].z[iz]:.1f}", xlabel=r"$k_\parallel$ [s/km]")
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
        models.append(vega_lyaxlya(point))
        labels.append(f"{name} {'x' if kind == 'factor' else '='} {value:g}")
    axes[0].set_ylabel("P1D / reference")
    axes[-1].legend(title=name, fontsize=8)
    fig.suptitle(f"{name}: P1D")
    plt.tight_layout()
    plt.show()
    vega.backend.plots.plot_4wedges(models=models, labels=labels, corr_name="lyaxlya",
                                    mu_bin_labels=True, no_font=True)
    vega.backend.plots.fig.suptitle(f"{name}: DR2 Lyα×Lyα")
    plt.show()


for name, (kind, values) in SCANS.items():
    scan(name, kind, values)

# %%
fit.close()
