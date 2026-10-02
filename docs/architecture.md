# Architecture

```text
CAMB (cosmological parameters)
  ↓ linear delta_nonu P(k,z), H(z), fsigma8(z), sigma8(z)
LyaCosmology → immutable snapshot + Delta2star/nstar/alphastar
  ↓ snapshot                  ↑ no IGM or nuisance dependencies
ForestFlowTheory ← igm_* → native cup1d IGM histories
  ↓ deterministic mean Arinyo + uncontaminated ragged P1D
Cup1DLikelihood ← p1d_* → native observation model → rebin → covariance
  ↓ data log likelihood only
Cobaya → statistical priors → posterior, sampler or likelihood-only minimizer
```

`initialize` loads static assets/configuration. The likelihood constructs a
static request from cup1d's fine `Rebinning` grids, passing only serialized data
through `get_requirements`. ForestFlow's `must_provide` requests the union of
redshifts from `LyaCosmology`, which requests explicitly linear baryon+CDM power
and matching background/growth from CAMB. No live backend crosses this boundary,
and no circular dependency is needed.

`calculate` stores all point-dependent products in Cobaya state, including derived
outputs. Provider snapshots copy their arrays and never capture a changing
provider. A → B → A restores correct immutable states. Products have explicit
getters or `get_can_provide`; star outputs use `get_can_provide_params`. Cobaya
registers only requested products: diagnostic clients must request
`forestflow_arinyo` before asking the provider for it.

Nuisance-only changes reuse CAMB and ForestFlow. IGM-only changes rerun
emulation/projection but not CAMB. Cosmology changes invalidate all downstream
products. CAMB itself can reuse transfer functions for primordial-only changes.

Requests retain ragged fine grids, original group/redshift ordering, units,
prediction stage, and a hash of resolved scientific configuration/data/covariance.
Emulation is deduplicated by redshift. Identical `(z, grid)` projections within a
point are reused. No unbounded extra parameter cache is introduced; upstream
projection geometry caching is bounded.

The upstream prerequisite is the cup1d API PR delivered with this package:
`ExternalP1DLikelihood`, shared scalar observation algebra, shared covariance
construction and stable Gaussian residuals. No ForestFlow/LaCE source changes
are needed. Native scalar/batch APIs remain supported. The shared covariance
builder also fixes full emulator-covariance nearest-k indexing to search the
selected redshift block, not all redshifts.

The implementation follows [Cobaya's dependency lifecycle](https://cobaya.readthedocs.io/en/stable/theories_and_dependencies.html)
and [cosmological product contracts](https://cobaya.readthedocs.io/en/stable/cosmo_theories_likes.html).
