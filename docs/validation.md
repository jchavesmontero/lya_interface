# Validation record and limitations

## Vectorized CM2026/DR1 execution

The interface now performs one native P1D array projection for all unique
redshift/grid requests, with endpoint padding for ragged grids, and selects
cup1d's columnar contaminant/resolution/rebinning kernels for scalar Cobaya
points. Parameter ordering, latent indices, projection quadrature, covariance,
blinding and CM2026 data selection are unchanged. The temporarily disabled
training hull remains a separate explicitly requested configuration change.

The 24 fast checks include scalar/array projection comparisons for unequal
axis lengths, both quadratures, reordered redshifts, duplicate requests,
immutable output and coverage rejection. All 12 real-asset scientific
regressions and both covariance-enabled CM2026 checks passed after these
execution-path changes. The CM2026 test also compares columnar observation
and rebinning directly against the native scalar backend.

No end-to-end performance factor is claimed without a timing measurement.
Cobaya/Nelder–Mead evaluates one parameter point per call; batch kernels do
not make the optimizer evaluate its simplex points concurrently. Upstream
linear interpolation and small ragged-grid packing loops remain.

Validated locally on 2026-10-02 with Python 3.12, Cobaya 3.6.2, CAMB 2.0.0,
NumPy 2.5.1, SciPy 1.18.0, Torch 2.13.0 and pytest 9.1.1. Siblings were installed
editably from the current checkouts without downloading/upgrading dependencies.

| Dependency | HEAD during validation |
|---|---|
| cup1d | a7ad7c536146864f7aeae083502e62f57b0ede91 + external API changes in this work |
| LaCE | 1f0cc5e91d839088ba801e258129277bd006cda5 |
| ForestFlow | 2f604bade597c38e7eaece7012f0bfca459f0ae8 |

A pre-existing ForestFlow training shell-script edit was preserved. No weights,
archives, covariance assets or chains belonging to the user were overwritten.
Validation chains/minimizer/provenance outputs were written under `/tmp`.

## Commands actually exercised

- Final combined `pytest -q` with `LYA_BAO_PACKAGES_PATH` set: 36 passed, with 14 upstream Torch deprecation
  warnings and no skips. This includes verification of the missing CM2026
  covariance error, not an end-to-end covariance-enabled CM2026 fit.

- `pytest -q -m 'not scientific'`: 20 lightweight tests passed, including actual
  Cobaya dependency graph, nuisance/IGM/cosmology invalidation, cached restoration,
  derived-only calculation without weights, and numerical-error propagation.
  Two of these validate tracked Jupytext Python sources, notebook structure,
  code syntax, clean outputs and cell round-trip fidelity without local .ipynb files.
  Recreating both notebooks with `jupytext --sync` from a temporary copy containing
  only their .py sources and the pairing configuration also passed.
- The original small-demo tutorial notebooks were schema-validated with nbformat and their code
  cells executed in order through IPython against the real demo assets. Plot
  code ran with the headless Agg backend; interactive Jupyter rendering was
  not tested. Neither notebook writes analysis outputs or scientific assets.
- Scientific checks: 16 real-asset tests passed: CAMB/LaCE matching,
  real-weight determinism, native/external scalar parity, quadrature boundaries,
  realization sensitivity, snapshot interpolation resolution, injected noiseless nuisance recovery, and the
  cosmology-only bridge at neutrino mass 0 and 0.06 eV.
  The two CM2026-specific native export/blinding checks are described below.
  Two additional real BAO checks are explicitly enabled by `LYA_BAO_PACKAGES_PATH`.
- cup1d `pytest -q -m 'not external_model'`: 38 passed, 1 external regression
  deselected. The full native DESI/GP tutorial regression was not run here.
- Single-point evaluation, 30 accepted-step MCMC after 5 burn-in accepts, and
  likelihood-only Powell minimization all completed with the final 96-node
  configuration. Chain and minimum outputs contain Delta2star, nstar and alphastar.
- Compilation and Git whitespace checks passed. The new GitHub workflows were
  written but not executed remotely in this task.
- cup1d `make docs` passed with warnings treated as errors. A stale ignored
  generated page for the removed emulator interface was preserved under `/tmp`
  before rebuilding; no tracked legacy documentation was removed.

Fast fixtures are deliberately synthetic, not scientific references. Real tests
require existing ForestFlow weights/metadata/transforms, MPG `mpg_emu_cosmo.npy`
and `IGM_histories.npy`, the configured Nyx IGM history (a native MPG prerequisite),
and the two public Karacayli2022 data/covariance files. Missing assets skip with
their paths, or fail when `LYA_REQUIRE_SCIENTIFIC_ASSETS=1`. No asset is generated.

## Measured numerical results

### CM2026 example update

The new examples inherit native CM2026 defaults with `forest_mpg` instead of
`lace_mpg`. Two additional real-asset checks pass: all 53 native fiducial values
and bounds, fixed Planck18 background settings, and the frozen M(z) conversion
table agree with native cup1d; the actual DESI metadata produces exactly the
native blinding offsets. The covariance-missing path fails explicitly, as tested.
The 12 existing real-asset regressions still pass on the preserved small fixture.

The updated walkthrough contains all-redshift native-renderer plots, an optional
likelihood-only Powell minimization, and a final plot at its actual fitted point.
Fast tests exercise both spectra/residual renderers, cache reuse, minimizer
success/budget failure, and diagnostic-only blinding. The full CM2026 notebook
and fit **cannot be run end to end locally** until the corrected covariance file
is supplied. This is not replaced by a covariance-disabled fit. Jupytext source
structure/round trips and local synchronization are checked; only .py is tracked.
The new helpers were also exercised with real weights on the explicitly separate
small regression fixture: 11 initial/redshift panels, a converged 162-evaluation
minimization (chi2≈946.6293), and a replot at the fitted point. This is not a
CM2026 fit or a claim of a global minimum.

### Optional DESI DR2 Lyα BAO check

Cobaya 3.6.2 includes `bao.desi_dr2.desi_bao_lya`. Its official `bao_data` v2.6
release was installed into `/tmp/lya_interface_bao.DE16vi` for validation, without
changing global Cobaya paths or existing scientific assets. The BAO-only runner
at native Planck18 gives loglike=-0.08147437084086256 (chi2=0.16294874168172513).
An independent Gaussian evaluation using DH/rd and DM/rd matches; changing H0
changes the likelihood. A real P1D+BAO check on the explicit small regression
fixture also passes, sharing one CAMB provider and preserving separate terms.

The baseline combined example is optional and not a calibrated CM2026 joint-fit
validation: its P1D covariance is still missing, and P1D/BAO are multiplied under
an unvalidated independence approximation. The default CM2026 background is
fixed, so its BAO term is constant across the baseline sampled parameters.
BAO tests require `LYA_BAO_PACKAGES_PATH` pointing to the official installation;
without this explicit opt-in they skip as optional, not as successful validation.

### Small-fixture numerical reference

The following recorded numerical results use `examples/validation_demo.yaml`
(the former small Karacayli2022 example), not the new full CM2026 configuration.

At the explicit demonstration reference point: 110 bins, raw chi²=2000.6914968,
loglike=-1000.3457484, omitted fixed logdet. Exported stars:
Delta2star=0.3560049506, nstar=-2.2991516703, alphastar=-0.2158193646.
The point is not a claimed best fit. The final bounded demonstration MLE has
chi²≈946.5471 and reaches the intentionally narrow demonstration support edges.
The short chain is a plumbing test, **not a converged posterior**.

One installed-package timing run measured initialization≈0.59 s, cold evaluation
≈5.05 s and an identical cached repeat≈0.000092 s. Hardware/load dependent;
the dependency tests independently verify scientific call counts.

| Projection point | chi², 48 nodes | chi², 96 nodes | chi², 192 nodes |
|---|---:|---:|---:|
| reference | 2000.6152133 | 2000.6914968 | 2000.6915653 |
| lower demo support corner | 961.4830544 | 961.5031875 | 961.5031382 |
| upper demo support corner | 3633.4805339 | 3633.6363962 | 3633.6365632 |

96→192 changes raw chi² by at most 0.000168 at these tested points, below the
initial 0.01 quadrature target. 48 nodes fail that target at the reference and
upper corner; therefore the interface examples use 96. This does not certify
every possible configuration, transverse cutoff or production parameter domain.

Native/external parity aligns the **same snapshot, latent assignment and
quadrature**, comparing actual emulator inputs and contaminated/rebinned powers
to 2e-6 relative tolerance (float32 emulation), with physical inputs agreeing at
1e-13. Independent direct LaCE calculations agree in linear power to 3e-4,
growth to 2e-5 and H(z) to 1e-10; the direct LaCE dense-z interpolation vs exact
provider-redshift convention shifts joint chi² by about -0.0231 to -0.0236 over
the tested As/ns/IGM/H0 points. This is **not** claimed to meet a 0.01 absolute
chi² agreement target. Matched-path algebraic parity and quadrature convergence
are tighter; a production accuracy budget should additionally align the
independent interpolation/transfer sampling. At the reference point, snapshot
grids of 1024/4096/8192 nodes give chi²=2000.691499782/2000.691496756/2000.691496807;
4096→8192 changes chi² by about 5.1e-8. This validates the configured snapshot
resolution at that point, not across an unrestricted production domain.

Using the same 100-node fit grid but removing endpoints changes reference stars
(open minus closed) by approximately (-3.27e-6, -2.12e-4, +1.91e-4) for massless
neutrinos; at 0.06 eV, (-2.67e-6, -2.07e-4, +1.58e-4). The implementation uses
the documented LaCE closed interval. The compressed package itself is not
installed/tested; this measures its stated endpoint-convention difference,
not a full compressed-package regression.

## Important unresolved production choices

With fixed seed/common latent block, realization counts 1000/2000/4000/8000 give
reference chi² of 2000.6915/1997.0440/1998.4550/2000.6606. Repeats at a fixed
count are deterministic, but **1000 realizations are not demonstrated converged
at a 0.01 chi² budget**. The count is an explicit demo/native-compatible estimator
choice, not a production accuracy guarantee. Do not reinterpret this sensitivity
as calibrated emulator covariance or retrain as part of installing the interface.

- The corrected calibrated ForestFlow covariance asset is absent locally;
  nonzero-error production validation remains blocked on the user-supplied asset.
  The demo disables error explicitly, never fabricates calibrated scatter.
- Initial ForestFlow examples admit the documented MPG/domain configuration,
  not arbitrary curvature/radiation/dark-energy/neutrino extensions. The massive
  neutrino bridge check alone does not admit such a full emulator analysis.
- Blinded data are rejected by default. The CM2026 examples explicitly select
  `blinding_policy: native`, which applies the same dataset-seeded native offsets
  to exported star diagnostics only, leaving physical predictions unchanged.
- Multi-group/ragged/full cross-redshift covariance behavior is tested with
  lightweight fixtures; real-asset parity is the documented single public dataset.
- The injected mock test recovers one nuisance amplitude at fixed cosmology/IGM;
  it is not a general multi-parameter cosmological recovery result.
- Wider production data cuts, covariance, priors, realization convergence and
  native spline configurations must be selected and validated explicitly.
- cupix and Vega remain deferred; no adapters, projections or dependencies were
  added for them. The same-data compressed likelihood must not be combined here.

## CI and upstream prerequisite

`tests.yml` provides a lightweight workflow and a separate asset-provisioned
scientific job on a `self-hosted, lya-assets` runner. LaCE and ForestFlow revisions
are pinned above. Supply the **full SHA of the cup1d external-API prerequisite
PR** via `cup1d_ref`; no nonexistent upstream commit is invented before this work
is committed/merged. Scientific CI verifies installed sibling SHAs and makes
missing assets failures, uses existing caches, and uploads provenance. It neither
clones large archives nor installs deferred likelihoods. Once the prerequisite
SHA is available, automatic PR triggers can be enabled against that pin.
