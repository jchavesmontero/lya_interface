# lya_interface

This package implements the full cup1d P1D likelihood in a Cobaya dependency
graph. cupix, Vega, and the compressed likelihood are out of scope.

- Write new or updated docstrings in NumPy/numpydoc style: use a concise summary and appropriate sections such as `Parameters`, `Returns`, `Raises`, `Notes`, and `Examples`, with dashed section underlines. Document parameter types, defaults, array shapes, and scientific units where relevant. Do not add empty or irrelevant sections or perform unrelated bulk docstring rewrites.

- Function signatures should not use a bare `*` to separate positional and keyword-only arguments. Prefer ordinary positional-or-keyword parameters, including optional parameters with defaults. Preserve existing public API compatibility unless a signature change is explicitly requested; do not perform unrelated bulk rewrites. This does not prohibit `*args` or `**kwargs` when needed.

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
- User-facing examples inherit CM2026 with forest_mpg and preserve native blinding.
  Keep the small unblinded validation_demo separate; never substitute it for a
  baseline analysis or silently disable missing calibrated covariance.
