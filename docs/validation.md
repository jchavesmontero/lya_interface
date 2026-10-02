# Validation record and limitations

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

- `pytest -q -m 'not scientific'`: 10 lightweight tests passed, including actual
  Cobaya dependency graph, nuisance/IGM/cosmology invalidation, cached restoration,
  derived-only calculation without weights, and numerical-error propagation.
- `pytest -q -m scientific -s`: 12 real-asset tests passed: CAMB/LaCE matching,
  real-weight determinism, native/external scalar parity, quadrature boundaries,
  realization sensitivity, snapshot interpolation resolution, injected noiseless nuisance recovery, and the
  cosmology-only bridge at neutrino mass 0 and 0.06 eV.
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
- Blinded data are rejected pending an explicit authorized diagnostics policy.
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
