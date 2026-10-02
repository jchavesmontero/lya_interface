# Scientific and numerical conventions

Linear power is explicitly **linear baryon+CDM** (`delta_nonu`, LaCE `bc`), in
Mpc³. No total-matter substitution or implicit Mpc/h conversion is permitted.
Growth is CAMB `fsigma8/sigma8`, matching LaCE. The bridge can represent massive
neutrinos, but nonzero masses are **not admitted by the initial ForestFlow demo**;
arbitrary extended cosmologies are not supported merely because CAMB runs them.
Radiation content, hierarchy, primordial pivots and other settings are explicit
in YAML/provenance. The initial admitted domain fixes densities, mass, radiation,
curvature and running; limited H0 variations are tested independently of the
native wrapper's primordial-only setup.

With `M=H(z)/(1+z)` in km/s/Mpc:

```text
k_iMpc = M*k_ikms         P1D_kms = M*P1D_Mpc
Plin_kms = M³*Plin_Mpc3   Cov_kms[i,j] = M[i]*M[j]*Cov_Mpc[i,j]
sigT_Mpc = sigT_kms/M     kF_Mpc = kF_kms*M
mF = exp(-tau_eff)
```

Read bundle labels/pivot/domain, never rely on incidental array ordering.
`alpha_p` is optional; it is absent from the corrected six-input MPG bundle.
Arinyo uses `(bias+bias_eta*f*mu²)²`; do not rename bias_eta to beta. A diagnostic
beta would be `bias_eta*f/bias`.

Projection reuses ForestFlow's `P1DIntegrator`, integrating
`k_perp²*P3D*dln(k_perp)/(2*pi)`. Demo transverse bounds are 0.001–100 /Mpc with
**96 Simpson nodes**. Current velocity conversion is included in the required
maximum hypotenuse. Both interpolation bounds and both local fitting intervals
are validated. CAMB's configured 200 /Mpc range is checked, not assumed sufficient.
Insufficient numerical coverage, malformed grids, missing assets and incompatible
metadata remain errors, not silently rejected sampled points.

Scientific-domain checks use the native MPG training summary assets, cup1d's
pairwise hull with explicit factor 1.0, physical IGM restrictions, bundle zmax
and kmax1d, and explicit background/radiation/neutrino support. The summaries
assume kp=0.7 /Mpc: other pivots require matching domain assets and are rejected.
The initial domain bridge requires the full 30-simulation MPG cube; held-out,
reduced and other suites must provide separately implemented matching domain
assets rather than silently use the full-suite hull.
Hull factor changes extrapolation policy and needs separate justification. The
demo lower redshift bound is 2; upper coverage comes from the bundle. Supported
domain exclusions return minus infinity without clipping into the domain.

The immutable snapshot copies power at 4096 log-k nodes, interpolating log power
in log k. Requested redshifts are selected by value. There is no k extrapolation
or interpolation to unrequested redshifts. Background/growth arrays are copied
too. Only the BaseCosmology subset needed here is implemented; distances and
arbitrary total-matter products are not provided.

Star defaults are z=3 and k=0.009 s/km, distinct from the emulator kp=0.7 /Mpc.
LaCE's 100 logarithmic fitting nodes **include endpoints** of k/kpivot=0.5–2.
For `log P=a0+a1*x+a2*x²`, amplitude is `kp³*exp(a0)/(2*pi²)`, slope a1, and
curvature **2*a2**. This is not primordial running. The compressed reference's
open-endpoint fit is a different convention; it is not silently substituted.

Determinism uses a checksum-validated bundle, 1000 realizations, seed 0, and
latent block zero for every input (common random numbers). Assignment is stable
under separate/batched evaluation, reordering, chunking and repeated redshifts;
nuisance changes cannot resample it. This differs from the native wrapper's
incidental per-row blocks; scientific parity tests explicitly align the latent
convention and projection settings. Preserve the upstream estimator/transforms:
**project mean Arinyo coefficients**, not mean projected latent powers. Latent
scatter is not a calibrated emulator-error covariance. Production realization
counts require an explicitly assessed convergence budget.

The shared scalar observation order is:

```text
Ptotal = (CHCD * Cmul_metals * IC_corr * Plya + Cadd_metals) * Cres
```

Apply this on fine grids, then native measurement rebinning. Ragged grids and
original ordering are retained; padded cells never enter the external API.
Rebin factor one is identity. Full covariance, including cross-redshift elements,
is retained; per-z diagnostic chi² need not sum to the joint chi².

Covariance is frozen and data-scaled at initialization, including optional native
emulator error and **explicit frozen fiducial M(z)**. No unused emulator or CAMB
run is needed. Validate symmetry and positive definiteness; never add arbitrary
jitter or repair determinants with absolute values. The result separates raw
chi2_data, included logdet_cov, ndata, validity and
`loglike=-0.5*(chi2_data+logdet_cov)`. The constant `ndata*log(2*pi)` is omitted.
Fixed logdet can be omitted for inference; future parameter-dependent covariance
must include it and declare dependencies. The full selected data vector is used,
not diagnostic redshift masks.

The MLE example uses derivative-free Powell: default tiny finite-difference steps
can be unreliable with float32 neural predictions. It maximizes only likelihood,
subject to Cobaya hard support bounds, unlike the default posterior maximizer.
