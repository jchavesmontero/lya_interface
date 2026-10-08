# Architecture

```text
CAMB (cosmological parameters)
  ↓ linear delta_nonu P(k,z), H(z), fsigma8(z), sigma8(z)
LyaCosmology → immutable snapshot + Delta2star/nstar/alphastar
  ↓ snapshot                  ↑ no IGM or nuisance dependencies
ForestFlowTheory ← igm_* → interface-owned IGM histories
  ↓ deterministic mean Arinyo + uncontaminated ragged P1D
Cup1DLikelihood ← p1d_* → native observation model → rebin → covariance
  ↓ data log likelihood only
VegaLikelihood ← BAO nuisances + ForestFlow coefficient factor → native Vega χ²
  ↓ data log likelihood only (native Gaussian priors are registered once upstream)
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

Vega-only consumers request coefficient redshifts without requesting a P1D
projection. The theory evaluates the union of static P1D and Vega redshifts
once per point. Vega receives predicted `bias` and `bias_eta` plus a bias-free
ForestFlow nonlinear factor; its own tracer, Kaiser, metal, peak and smooth
handling is retained. Its template unit conversion requires documented fixed
template `h`, never sampled H0.
When Vega's native `skip-nl-model-in-peak` option is enabled, the ForestFlow
factor is likewise omitted from the peak-only call and retained for the smooth
component.

The upstream prerequisite is the cup1d API PR delivered with this package:
`ExternalP1DLikelihood`, shared scalar observation algebra, shared covariance
construction and stable Gaussian residuals. No ForestFlow/LaCE source changes
are needed. Native scalar/batch APIs remain supported. The shared covariance
builder also fixes full emulator-covariance nearest-k indexing to search the
selected redshift block, not all redshifts.

The implementation follows [Cobaya's dependency lifecycle](https://cobaya.readthedocs.io/en/stable/theories_and_dependencies.html)
and [cosmological product contracts](https://cobaya.readthedocs.io/en/stable/cosmo_theories_likes.html).

## Shared IGM and derived-output contracts

`lya_interface.igm` owns the IGM implementation used by the interface and is
importable with only NumPy, SciPy, and LaCE. cup1d retains its own standalone
IGM implementation by design; a scientific parity test compares the selected
CM2026 histories at representative redshifts.

Set `derived_redshifts` on `ForestFlowTheory` to save scalar shared diagnostics
at those exact redshifts. Names use the exact floating-point hexadecimal
redshift encoding, not rounded decimals, and `derived_output_mapping()` records
each column's quantity, redshift, and units. The outputs include bias,
bias_eta, beta, mean flux, tau_eff, T0, gamma, thermal and pressure scales,
and the ForestFlow pivot inputs. Invalid zero bias produces `NaN` beta rather
than a spurious finite value.
