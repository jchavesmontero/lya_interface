"""Posterior figures of a Cobaya run: best fit, star parameters, nuisances, IGM, contaminants.

Reads the chains of one run (they may still be growing), translates them to
cup1d's native names and calls the figures of
:mod:`cup1d.postprocessing.posterior`. Written to
``<run_dir>/plots/<chain folder>/`` (``--out`` to change it):

* ``best_fit_residuals`` / ``best_fit_p1d``: best-fitting model against every
  data set, one row each (DESI DR1 above, high resolution below). The best
  of the saved minimization (``<run_dir>/best_fit.npy``) and the best chain
  sample; needs the emulator, so it takes a few minutes (``--no-best-fit``
  to skip it).
* ``star_posterior``: Delta2_star--n_star, titles with the 16/50/84 percentiles.
* ``star_literature``: against SDSS/BOSS/eBOSS and DESI DR1 (2D Gaussian);
  only with ``--unblind`` when the data are blinded.
* ``params_vs_star_<group>``: every parameter group against Delta2_star and n_star.
* ``igm_histories``: mean flux, T0, gamma and k_F against z, with DESI DR1
  (ForestFlow's IGM priors) and literature measurements.
* ``metals_z3``, ``hcd_z3``: metal and HCD contamination at z=3.
* ``systematics_vs_z``: resolution coefficients against z.
* ``correlation_matrix``: all parameters.
* ``summary.txt``: 16/50/84 percentiles of every parameter.

Usage (login node; ``$RUN_DIR`` the run folder or a ``chain_<n>`` folder)::

    python scripts/chain_plots.py $RUN_DIR [--prefix chain_<n>/mcmc] [--unblind] [--no-best-fit]
"""
import argparse
import os
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="run folder (with chain_<n> folders) or a chain_<n> folder")
    parser.add_argument("--prefix", default=None,
                        help="chain prefix relative to run_dir (default: the latest chain_<n>/mcmc)")
    parser.add_argument("--out", default=None, help="output folder (default run_dir/plots/<chain folder>)")
    parser.add_argument("--burn", type=float, default=0.3, help="burn-in fraction of every chain")
    parser.add_argument("--max-rows", type=int, default=1_000_000,
                        help="rows kept after thinning (memory: ~0.5 kB per row)")
    parser.add_argument("--model-samples", type=int, default=4000,
                        help="posterior draws for IGM histories and contamination")
    parser.add_argument("--unblind", action="store_true",
                        help="remove the DESI blinding offsets of Delta2_star and n_star "
                             "(only for unblinded data, e.g. DESI DR1 P1D)")
    parser.add_argument("--no-best-fit", action="store_true", help="skip the best-fit P1D (needs the emulator)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="figures to make, among: best_fit " + " ".join(
                            ("star_posterior", "star_literature", "parameters_vs_star", "correlation",
                             "systematics", "igm_histories", "metals", "hcd")))
    parser.add_argument("--label", default=None, help="legend label of this posterior (default: data sets)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import lya_interface.plotting  # noqa: F401  (house style)
    from cup1d.postprocessing.posterior import PosteriorPlotter
    from lya_interface.posterior import (chain_prefix, data_k_range, data_redshift_range, load_posterior,
                                         native_models)

    run_dir, prefix = chain_prefix(args.run_dir, args.prefix)
    out = Path(args.out) if args.out else run_dir / "plots" / prefix.parent.name
    print(f"chains: {prefix}\noutput: {out}", flush=True)

    samples, context = load_posterior(prefix, burn=args.burn, max_rows=args.max_rows, unblind=args.unblind)
    igm, contaminants, syst_znodes, native = native_models(context.native_config)
    samples.label = args.label or " + ".join(native.data_label)
    if samples.blinded:
        print("Delta2_star and n_star are blinded (pass --unblind for unblinded data)", flush=True)
    try:
        k_range = data_k_range(native, 3.0)
    except ValueError:
        k_range = (1e-3, float(getattr(native, "kmax_ikms", None) or 0.04))
    z_range = data_redshift_range(native)

    plotter = PosteriorPlotter(samples, out, igm=igm, contaminants=contaminants, syst_znodes=syst_znodes,
                               z_range=z_range, k_range=k_range, M_of_z=context.M_of_z,
                               n_model_samples=args.model_samples)
    wanted = set(args.only) if args.only else None
    if not args.no_best_fit and (wanted is None or "best_fit" in wanted):
        try:  # the other figures do not need it
            plotter.best_fit = best_fit_figure(plotter, run_dir, context, prefix, out)
        except Exception:
            print(f"best fit failed:\n{traceback.format_exc()}", flush=True)
    skip = () if wanted is None else tuple(n for n in plotter.FIGURES if n not in wanted)
    plotter.plot_all(skip=skip)
    write_summary(samples, context, out / "summary.txt")


def best_fit_figure(plotter, run_dir, context, prefix, out):
    """Plot the best-fitting P1D and return the best point in native names.

    The best of the saved minimization (``run_dir/best_fit.npy``, if any) and
    of the chain sample with the lowest minus log posterior. The model uses
    the run's native cup1d configuration (frozen with the chain).
    """
    from lya_interface.diagnostics import format_goodness_of_fit
    from lya_interface.fit import Fit
    from lya_interface.posterior import best_fit_entries, best_point, native_parameter_name

    saved = run_dir / "best_fit.npy"
    config = context.physical_config(out)
    print(f"building the model for the best fit (OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})",
          flush=True)
    if saved.is_file():
        fit = Fit.from_results(saved, config_path=config)
        candidates = {"minimizer": fit.best_fit.point}
    else:
        fit = Fit(config)
        candidates = {}
    candidates["chain"] = context.best_chain_point
    try:
        label, point = best_point(fit.model, candidates)
        entries, statistics = best_fit_entries(fit.model, point)
    finally:
        fit.close()
    print(f"best fit from the {label}:\n{format_goodness_of_fit(statistics, multiline=True)}", flush=True)
    p1d = statistics.p1d
    plotter.best_fit_p1d(entries, p1d.chi2, p1d.dof, ndata=p1d.ndata,
                         title=f"best fit ({label}), {prefix.parent.name}")
    return {native_parameter_name(name): float(value) for name, value in point.items()}


def write_summary(samples, context, path):
    """Write the 16/50/84 percentiles and prior-edge mass of every parameter."""
    from cup1d.postprocessing.posterior.stats import format_quantiles

    lines = [f"{samples.label}: {context.n_rows} rows after burn-in, {len(samples)} kept, "
             f"effective size {samples.effective_size:.0f} (Kish, ignores autocorrelation)",
             "blinded star parameters" if samples.blinded else "", ""]
    for name in samples.names:
        lo, mid, hi = samples.quantiles(name)
        lines.append(f"{name:>20s}  {mid: .5g}  -{mid - lo:.3g} +{hi - mid:.3g}   "
                     f"${format_quantiles(samples.column(name), samples.weights)}$")
    edges = samples.near_bounds()
    if edges:
        lines += ["", "posterior mass within 2% of a prior edge (> 5%):"]
        lines += [f"{name:>20s}  lower {lo:.0%}  upper {hi:.0%}" for name, lo, hi in edges]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
