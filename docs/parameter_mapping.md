# Parameter ownership and priors

Each component receives explicit `parameter_definitions` equal to top-level
Cobaya `params`; YAML anchors avoid divergence. The Python loader rejects
conflicting copies. Registry entries record public/backend names, owner, units,
transformation, status, consumer, prior, reference and proposal. Unknown names and
missing configured coefficients fail initialization. Cobaya additionally rejects unassigned sampled parameters.

| Public name | Backend | Owner/consumer | Meaning |
|---|---|---|---|
| As, ns, H0, ombh2, omch2, mnu, nnu, omk, nrun, tau | same | cosmology/CAMB | ordinary CAMB units/conventions |
| igm_tau_eff_i | tau_eff_i | IGM/P1DEmulatorTheory | native fiducial optical-depth rescaling coefficient |
| igm_gamma_i | gamma_i | IGM/P1DEmulatorTheory | fiducial gamma rescaling coefficient |
| igm_sigT_kms_i | sigT_kms_i | IGM/P1DEmulatorTheory | fiducial thermal-history rescaling coefficient, not km/s itself |
| igm_kF_kms_i | kF_kms_i | IGM/P1DEmulatorTheory | fiducial pressure-history coefficient, not inverse km/s itself |
| metal_<native coefficient> | native coefficient | nuisance/Cup1DLikelihood and VegaLikelihood | Kaiser silicon coefficient (`b_`, `beta_`, `sig_`, `f_SiIIxSiII`, `sig_SiIIxSiII`), one parameter shared by P1D and xi3D |
| p1d_<native coefficient> | native coefficient | nuisance/Cup1DLikelihood | P1D-only metal template, HCD or resolution coefficient |
| xi3d_<name> | Vega native name (`parameter_mapping`) | VegaLikelihood | xi3D-only Vega nuisance; `ap` and `at` keep their names |
| Delta2star | Delta2_star | derived/LyaCosmology | dimensionless linear-power amplitude |
| nstar | n_star | derived/LyaCosmology | log slope |
| alphastar | alpha_star | derived/LyaCosmology | log curvature, not primordial running |

Counts/order come from native Args, not dictionary ordering. Native IGM models
preserve pivots, interpolation/spline options, nodes, reversed polynomial
coefficient ordering, smoothing and fiducial histories. In the demo,
`tau_eff_otype: exp` makes the pivot coefficient a log multiplier of fiducial tau;
the other three `const` coefficients directly multiply their physical fiducial
histories. References are therefore 0 and 1, not physical IGM quantities.

Results saved before the `metal_`/`xi3d_` prefixes (Kaiser metals as `p1d_*`, Vega
nuisances with bare names) are translated by `lya_interface.names.translate_legacy`,
which `Fit.from_results` and `metal_alpha_reference` apply.

All configured coefficients are explicitly fixed or sampled by Cobaya. Terms
with zero native free coefficients retain the selected native fixed values. To
vary one, enable its coefficient count in the native config and add a public
definition. Native coefficient transformations remain owned by cup1d.

The likelihood adds no native statistical priors. Native star/Gaussian IGM or
contaminant priors are rejected until exported explicitly to Cobaya, once each.
Computing stars does not impose a star prior. Scientific model-domain exclusions
are separate from prior densities. `ignore_prior: true` maximizes the likelihood,
but Cobaya still enforces support bounds; the demo minimum is not a production
MLE over a validated wide parameter domain.

The default adapter policy rejects blinded data. The CM2026 examples explicitly
select `blinding_policy: native`: the likelihood requests `lya_blinding` from
the cosmology component, which reads static dataset metadata and applies the
same cup1d seed and offsets to exported Delta2star/nstar/alphastar diagnostics.
Physical snapshots, IGM inputs, projections and likelihoods remain unshifted.
The notebooks neither remove the offsets nor display fitted As/ns. As in native
cup1d, internal sampled coordinates are physical, not newly blinded variables;
this policy does not make raw sampled-coordinate files safe for public release.
The unblinded `validation_demo.yaml` remains a separate regression fixture.

