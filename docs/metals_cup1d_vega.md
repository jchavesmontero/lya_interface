# Silicon metals in cup1d (P1D) and Vega (3D correlation function)

The Kaiser silicon model of cup1d (`metal_model_type: Kaiser`,
`cup1d.models.contaminants.metals.si_kaiser.SiKaiser`) and the metal model of the
Vega fork (`vega_fork`, `small scale nl = forestflow`) describe the same 3D
power spectrum. `VegaLikelihood` passes the cup1d histories to Vega, so a joint
P1D + ξ3D fit shares one set of silicon parameters. This page lists what is the
same, what is not, and the approximations each code makes, so that the two can
be kept consistent.

## Shared 3D model

For components `a, b` (Lyα or a silicon line `m`):

```
P_ab(k, mu) = b_a b_b (1 + beta_a mu^2)(1 + beta_b mu^2) P_L(k) D_a D_b W_ab
```

| Term | cup1d (P1D) | Vega fork (ξ) |
|---|---|---|
| Lyα leg | `sqrt(D_NL)`, ForestFlow Arinyo | `forestflow_dnl` (supplied by `lya_interface`) |
| Metal leg | pressure smoothing `exp(-(k/k_p)^2 / 2)`, ForestFlow `k_p` | `forestflow_metal_pressure` |
| Lyα × metal damping `W_am` | Gaussian `exp(-(k_par sigma_m)^2 / 2)` (default) | `forestflow_metal_decorrelation`, same Gaussian |
| Metal autos and metal × metal crosses | not damped, amplitude `b_m b_n` | same |
| SiII × SiII (SiII autos, crosses of SiII 1190/1193/1260) | optional `f exp(-(k_par sigma)^2 / 2)` (`f_SiIIxSiII`, `sig_SiIIxSiII`) | `forestflow_metal_sii_factor`, `forestflow_metal_sii_decorrelation` |
| Lines | SiIII(1207), SiII(1190), SiII(1193), SiII(1260) | same lines, plus CIV(eff) |
| Metal betas | `beta_SiIII`, `beta_SiII` (fiducial fixed at 0.5) | `beta_<line>` from cup1d |
| HCD | not applied to the metal terms (added outside `C_HCD`) | `model-hcd` only in `[model]`, so not applied to metal pairs |

In P1D the transverse integral turns this into moments (`A_j`, `I_j` of
`cup1d.likelihood.external.project_metal_moments`) and the line offsets into
`cos(k dv)`; in Vega the same power is Fourier transformed and the line offsets
enter through the metal distortion matrices.

## Settings that make them consistent

- cup1d: `metal_model_type: Kaiser` with the default Gaussian damping
  (`SiKaiser(damping="gaussian")`). `damping="sigmoid"` (the CM2026 form) has no
  Vega counterpart and breaks the correspondence.
- Vega: `small scale nl = forestflow` in both `[model]` and `[metals]`,
  `use_metal_autos = True` (default; otherwise Vega drops every pair whose names
  both contain "SiII", which includes SiIII(1207)), `fast_metals = False`, and the
  four silicon lines in `in tracer1/2` of `[metals]`.
- `VegaLikelihood`: `metal_model` set to the cup1d configuration of the P1D
  likelihood, so both read the same `p1d_` parameters.

`examples/vega_dr2_lyaxlya_forestflow_kaiser.ini` and
`examples/vega_dr2_lya_forestflow_kaiser_main.ini` follow these settings.

## Parameter routing and conversions

`VegaLikelihood.metal_values` evaluates the cup1d histories at Vega's `z_eff`
for every point:

| cup1d (P1D) | Vega | Conversion |
|---|---|---|
| `b_<line>` (`ln abs(b)`, biases negative) | `bias_<line>` | value at `z_eff`, sign applied |
| `beta_<species>` | `beta_<line>` | value at `z_eff` |
| `sig_<species>` (km/s) | `forestflow_metal_decorrelation[<line>]` (Mpc/h) | `x kms_to_Mpch` |
| `f_SiIIxSiII` | `forestflow_metal_sii_factor` | value at `z_eff`; passed only if free in P1D |
| `sig_SiIIxSiII` (km/s) | `forestflow_metal_sii_decorrelation` (Mpc/h) | `x kms_to_Mpch`; passed only if free in P1D |
| slope of `ln abs(b_<line>)` | `alpha_<line>` | set once from `metal_alpha_reference`, see below |

`kms_to_Mpch = (1 + z_fid) D_H(z_fid) / c`, with `z_fid` and `D_H` the reference
redshift (`ZREF`) and Hubble distance of the data file, i.e. the fixed fiducial
coordinates in which Vega measures separations.

## Redshift evolution of the metal amplitudes

cup1d gives each line its own bias history, `ln abs(b)` interpolated in `z`
between nodes. Vega weights each pixel pair of a metal matrix by the amplitude
evolution of its two legs, `((1 + z) / (1 + z0))^alpha D(z) / D(z0)` with
`z0 = z_eff` (estimator convention, `new_metals = True`). The matrices, and so
the exponents `alpha_<line>`, are fixed when Vega initializes; Vega refuses to
vary them.

The values at `z_eff` agree at every point. The evolution away from `z_eff` is
set by `metal_alpha_reference` of `VegaLikelihood`:

- unset: the `alpha_<line>` of the Vega INI (1 in the examples) are used,
  independently of the cup1d histories;
- a P1D fit saved with `lya_interface.fit.Fit.save` (or a mapping of public
  `p1d_` values): before Vega is built, each `alpha_<line>` is set to the slope
  `d ln abs(b) / d ln(1 + z)` of the cup1d history at that point and `z_eff`
  (`metal_amplitude_exponents`; growth is applied separately in both codes, so
  the exponents describe the same quantity). The derived values are logged and
  kept in `VegaLikelihood.metal_alpha`.

Recommended workflow: fit P1D first, save the fit, then run the joint P1D + ξ3D
fit with

```yaml
likelihood:
  vega:
    metal_model: <cup1d configuration of the P1D likelihood>
    metal_alpha_reference: <P1D fit result written by Fit.save>
```

Remaining differences: the exponents do not follow the sampled point of the joint
fit, and a power law matches a node-interpolated history only to first order
around `z_eff`.

## Other differences

| Item | cup1d | Vega fork | Impact |
|---|---|---|---|
| CIV(eff) | not modelled | auto-correlation included | Vega-only term |
| Redshift of the model | each P1D redshift bin with its own histories and ForestFlow coefficients | one `z_eff` for `forestflow_dnl`, `k_p` and the metal parameters | evolution within the 3D sample only through `alpha` |
| Lyα × metal damping options | Gaussian (default) or sigmoid | Gaussian only | keep Gaussian for joint fits |
| SiII × SiII selection | by species (`SiII`) | exact SiII line names (`SiIII` excluded) | same pairs |
| `fast_metals = True` | — | metal × metal correlations frozen at the first evaluation (only the SiII × SiII factor and decorrelation are recomputed); the metal pressure would not follow `k_p` | keep `fast_metals = False` |
| Units of damping scales | km/s | Mpc/h | converted at the fiducial `z_fid`, not at `z_eff` |

## Approximations of each code

cup1d (P1D):

- The moments are projected on the ForestFlow quadrature; with `n_k_emu` set
  (64 in `cm2026_base_forestmpggp`) they are evaluated on a log-spaced grid and
  interpolated with a cubic spline in `(ln k, ln abs(Q))`, as the Lyα P1D.
- The metal leg carries only pressure smoothing: no `D_NL` (no nonlinear growth
  or line-of-sight suppression of the metals), and no thermal broadening (Si
  lines are `sqrt(28)` times narrower than H).
- Pairs with offsets above `dv_max_kms = 26000` km/s are dropped (no silicon
  pair is).
- The metal terms are additive: rebinned and multiplied by the resolution
  kernel with the rest of the P1D, but not by the HCD contamination.

Vega fork (ξ3D):

- The linear template is the fixed native one; ForestFlow supplies only the
  bias-free nonlinear factor and `k_p` at `z_eff` (see
  [Architecture](architecture.md)). The template `h` converts units; it is never
  the sampled H0.
- The undistorted metal correlation is evaluated at `z0 = z_eff`; the metal
  matrices (estimator convention) carry the redshift evolution with the frozen
  `alpha_<line>` and the effective `r_par` of each true bin.
- The line offsets enter through the metal distortion matrices built from the
  stacked deltas at initialization, not as an exact `cos(k dv)`.
- With `fast_metals = True` (not used here): frozen metal × metal shapes, shared
  Lyα × metal correlations for equal betas, fiducial growth rate in the metal
  terms (see the `Metals` docstring of the fork).
