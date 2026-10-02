# lya_interface

This package implements the full cup1d P1D likelihood in a Cobaya dependency
graph. cupix, Vega, and the compressed likelihood are out of scope.

- Keep cosmology, IGM/projection, and observation/likelihood dependencies separate.
- Consult maintained sibling AGENTS.md and conventions before upstream edits.
- Never create a second CAMB calculation during interface evaluation.
- Keep provider snapshots immutable, point-specific and independent of provider state.
- Reuse upstream kernels and observation/covariance helpers; do not copy them here.
- Do not retrain or regenerate archives, covariance, or bundled weights.
- Native scientific assets are external and trusted; report missing validation.
- Use apply_patch for edits. Preserve native scalar and batch return contracts.
- Run `pytest -q -m 'not scientific'` for fast checks, then configured scientific
  tests, including actual Cobaya routing/restoration. Record exact revisions.
- Examples use demonstration priors and unblinded public data, not production defaults.
