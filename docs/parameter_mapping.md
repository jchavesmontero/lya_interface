# Parameter ownership and priors

Each component receives explicit `parameter_definitions` equal to top-level
Cobaya `params`; YAML anchors avoid divergence. The Python loader rejects
conflicting copies. Registry entries record public/backend names, owner, units,
transformation, status, consumer, prior, reference and proposal. Unknown names,
missing configured coefficients and ambiguous As/logA definitions fail
initialization. Cobaya additionally rejects unassigned sampled parameters.

| Public name | Backend | Owner/consumer | Meaning |
|---|---|---|---|
| As, ns, H0, ombh2, omch2, mnu, nnu, omk, nrun, tau | same | cosmology/CAMB | ordinary CAMB units/conventions |
| igm_tau_eff_i | tau_eff_i | IGM/ForestFlowTheory | native fiducial optical-depth rescaling coefficient |
| igm_gamma_i | gamma_i | IGM/ForestFlowTheory | fiducial gamma rescaling coefficient |
| igm_sigT_kms_i | sigT_kms_i | IGM/ForestFlowTheory | fiducial thermal-history rescaling coefficient, not km/s itself |
| igm_kF_kms_i | kF_kms_i | IGM/ForestFlowTheory | fiducial pressure-history coefficient, not inverse km/s itself |
| p1d_<native coefficient> | native coefficient | nuisance/Cup1DLikelihood | native metal, HCD or resolution coefficient |
| Delta2star | Delta2_star | derived/LyaCosmology | dimensionless linear-power amplitude |
| nstar | n_star | derived/LyaCosmology | log slope |
| alphastar | alpha_star | derived/LyaCosmology | log curvature, not primordial running |

Counts/order come from native Args, not dictionary ordering. Native IGM models
preserve pivots, interpolation/spline options, nodes, reversed polynomial
coefficient ordering, smoothing and fiducial histories. In the demo,
`tau_eff_otype: exp` makes the pivot coefficient a log multiplier of fiducial tau;
the other three `const` coefficients directly multiply their physical fiducial
histories. References are therefore 0 and 1, not physical IGM quantities.

All configured coefficients are explicitly fixed or sampled by Cobaya. Terms
with zero native free coefficients retain the selected native fixed values. To
vary one, enable its coefficient count in the native config and add a public
definition. Native coefficient transformations remain owned by cup1d.

For a log-amplitude prior, replace sampled As with:

```yaml
logA:
  prior: {min: 2.9, max: 3.2}
  ref: 3.047
  proposal: 0.01
  drop: true
As:
  value: 'lambda logA: 1e-10 * np.exp(logA)'
```

Do not sample both. Uniform logA and uniform As are different prior measures.
The registry accepts this explicit exponential transformation.

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

## Joint structured configuration

Joint configurations may use `parameters.general`, `parameters.P1D`, and
`parameters.BAO` instead of flat `params`. The loader converts that form to
Cobaya's flat namespace and saves explicit public/native/section/transform/
unit/status records in `interface_metadata.parameter_mapping`. Native aliases
must be unique across the three sections, and a derived record cannot have a
sampling prior. General is the sole owner of shared cosmology/IGM parameters;
P1D and BAO nuisances remain independent even if their names resemble one
another.
