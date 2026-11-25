"""Run, test or check a Cobaya MCMC from a production YAML.

The production YAML (e.g. ``<cup1d output root>/DESIY1_QMLE3__Karacayli2022/global_opt/forest_mpg_gp/cobaya_mcmc/cobaya_mcmc.yaml``,
written by ``scripts/prepare_mcmc.py`` or ``notebooks/wip/hires_forestflow_cobaya_mcmc``) overlays an interface
configuration with the starting point, the proposal covariance, the sampler
and the output prefix. Relative paths in it are relative to its folder, so the
script runs from there.

Usage
-----
Test: a short run written to ``test/``, overwriting the previous test, with
the time per step and a forecast. One chain on a laptop or NERSC login node;
under ``srun`` one chain per rank, with frequent convergence checks
(``--learn-every``) so that the checkpoint and the learned covariance are
written too (``MODE=test`` in ``scripts/nersc/mcmc_job.sh``)::

    python scripts/run_mcmc.py <run_dir>/cobaya_mcmc.yaml --test

Production (one chain per MPI rank; resumes from the checkpoint)::

    srun -n 8 python scripts/run_mcmc.py <run_dir>/cobaya_mcmc.yaml

Convergence of the production run (exit code 0 if converged, 1 if not, 2 if
it has not started), used by the Slurm job to resubmit itself; it also reloads
the chains to check them. With ``--test``, of the test run::

    python scripts/run_mcmc.py <run_dir>/cobaya_mcmc.yaml --status
"""
import argparse
import os
import sys
from pathlib import Path
from time import perf_counter


def mpi_rank():
    """MPI rank of this process (0 without MPI)."""
    from cup1d.utils.warn_once import mpi_rank as rank
    return rank()


def read_checkpoint(prefix):
    """Cobaya checkpoint of an output prefix, or None if absent."""
    import yaml
    path = Path(f"{prefix}.checkpoint")
    if not path.is_file():
        return None
    return yaml.safe_load(path.read_text())["sampler"]["mcmc"]


def count_samples(prefix):
    """Accepted points written per chain file ``<prefix>.<i>.txt``."""
    counts = {}
    for chain in sorted(Path(prefix).parent.glob(f"{Path(prefix).name}.*.txt")):
        with open(chain) as handle:
            counts[chain.name] = sum(1 for line in handle if not line.startswith("#"))
    return counts


def acceptance(prefix):
    """Accepted points and total steps (sum of weights, first column) over all chain files."""
    import numpy as np
    accepted, steps = 0, 0.0
    for chain in sorted(Path(prefix).parent.glob(f"{Path(prefix).name}.*.txt")):
        weights = np.loadtxt(chain, usecols=0, ndmin=1)
        accepted += len(weights)
        steps += float(weights.sum())
    return accepted, steps


def status(prefix):
    """Print the convergence status of a run and reload it; return the exit code."""
    checkpoint = read_checkpoint(prefix)
    if checkpoint is None:
        print(f"{prefix}: no checkpoint (not started)")
        return 2
    print(f"{prefix}: converged = {checkpoint.get('converged')}, "
          f"R-1 (last) = {checkpoint.get('Rminus1_last')}, chains = {checkpoint.get('mpi_size')}")
    counts = count_samples(prefix)
    for name, n in counts.items():
        print(f"  {name}: {n} accepted points")
    accepted, steps = acceptance(prefix)
    if steps:
        print(f"  all chains: {accepted} accepted out of {steps:.0f} steps (acceptance {accepted / steps:.2f})")
    for suffix in ("covmat", "progress", "updated.yaml"):
        path = Path(f"{prefix}.{suffix}")
        print(f"  {path.name}: {'written' if path.is_file() else 'MISSING'}")
    try:  # the chains must load as the analysis will load them
        from cobaya import load_samples
        samples = load_samples(str(prefix), skip=0, combined=True)
        physical = [name for name in ("As", "ns", "Delta2star") if name in samples.data.columns]
        print(f"  reloaded: {len(samples)} points; physical columns present: {physical}")
    except Exception as error:  # report, do not fail the status check
        print(f"  reloading failed: {error!r}")
    return 0 if checkpoint.get("converged") else 1


def correlated_start(info, rank, scale):
    """Set this rank's starting point to a draw from the proposal covariance.

    The production YAML gives every parameter an independent Gaussian ``ref``
    around the best fit. With a strongly correlated posterior, independent
    offsets of a fraction of each marginal sigma are many sigma away along the
    narrow directions (Cobaya warns that the initial points are widely
    dispersed, and the chains spend long in burn-in). Here each MPI rank draws
    its start from ``N(best fit, scale**2 * C)``, with ``C`` the proposal
    covariance (unit cube) and the best fit the centre of the YAML ``ref``;
    coordinates beyond a prior edge are reflected back into [0, 1].

    Returns the Mahalanobis distance of the start (in units of C), or None if
    there is no proposal covariance or a parameter has no ``ref`` centre (the
    YAML references are then used as they are).
    """
    import numpy as np
    from lya_interface.mcmc_setup import read_covmat

    mcmc = info["sampler"]["mcmc"]
    covmat = mcmc.get("covmat")
    if not covmat or not Path(covmat).is_file():
        return None
    covariance, names = read_covmat(covmat)
    params = info["params"]
    centre = []
    for name in names:
        ref = params.get(name, {}).get("ref") if isinstance(params.get(name), dict) else None
        if isinstance(ref, dict) and "loc" in ref:
            centre.append(float(ref["loc"]))
        elif isinstance(ref, (int, float)):
            centre.append(float(ref))
        else:
            return None
    centre = np.array(centre)
    seed = mcmc.get("seed")
    rng = np.random.default_rng(None if seed is None else [int(seed), rank])
    chol = np.linalg.cholesky(0.5 * (covariance + covariance.T))
    offset = scale * chol @ rng.standard_normal(len(names))
    start = np.abs(centre + offset)  # reflect at 0
    start = np.where(start > 1, 2 - start, start)  # and at 1
    for name, value in zip(names, start):
        params[name]["ref"] = float(value)
    step = np.linalg.solve(chol, start - centre)
    return float(np.sqrt(step @ step))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("production_yaml", help="production Cobaya YAML (relative paths from its folder)")
    parser.add_argument("--test", action="store_true",
                        help="short single test chain in test/, overwriting the previous one")
    parser.add_argument("--test-samples", type=int, default=200, help="accepted points of the test chain")
    parser.add_argument("--target-steps", type=float, default=50_000,
                        help="steps per chain assumed by the test forecast")
    parser.add_argument("--learn-every", default="5d",
                        help="test: accepted points per chain between convergence checks "
                             "(Cobaya syntax, 'd' = number of parameters; production default 40d)")
    parser.add_argument("--status", action="store_true", help="print convergence of the production run")
    parser.add_argument("--start-scale", type=float, default=0.1,
                        help="each chain starts at a draw from N(best fit, scale^2 * proposal "
                             "covariance); 0 keeps the independent YAML references")
    args = parser.parse_args()

    production_yaml = Path(args.production_yaml).resolve()
    os.chdir(production_yaml.parent)  # covmat, output and interface_base are relative to it

    from lya_interface.configuration import load_configuration
    from lya_interface.mcmc_setup import MCMC_SETTINGS
    info = load_configuration(production_yaml)
    # sampler settings added to the defaults after the YAML was written (e.g.
    # learn_proposal_Rminus1_max); values in the YAML win. Cobaya takes the new
    # value of these on resume.
    mcmc = info["sampler"]["mcmc"]
    added = {key: value for key, value in MCMC_SETTINGS.items() if key not in mcmc}
    mcmc.update(added)
    if args.test:
        info["output"] = "test/mcmc"
    if args.status:
        sys.exit(status(info["output"]))

    if args.test:
        info["resume"] = False
        info["force"] = True
        # seed: Cobaya derives a different seed for every MPI rank from it
        info["sampler"]["mcmc"].update(max_samples=args.test_samples, seed=1,
                                       learn_every=args.learn_every)

    from cobaya.run import run
    rank = mpi_rank()
    if args.start_scale > 0:
        distance = correlated_start(info, rank, args.start_scale)
        if distance is not None:
            print(f"[rank {rank}] start drawn from {args.start_scale} x proposal covariance: "
                  f"distance {distance:.2f} sigma from the best fit", flush=True)
    if rank == 0 and added:
        print(f"sampler settings not in the YAML, from lya_interface.mcmc_setup: {added}", flush=True)
    if rank == 0:
        print(f"{'test' if args.test else 'production'} run of {production_yaml}, output {info['output']}, "
              f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}", flush=True)
    t0 = perf_counter()
    _, sampler = run(info)
    seconds = perf_counter() - t0

    # products() broadcasts over MPI (share_mpi): every rank must call it, or
    # rank 0 waits forever for the others after sampling has finished
    products = sampler.products() if args.test else None
    if args.test and rank == 0:
        chain = products["sample"]
        steps = float(chain["weight"].sum())
        per_step = seconds / steps
        print(f"\ntest run: {seconds:.0f} s, {len(chain)} accepted points out of {steps:.0f} steps "
              f"(acceptance {len(chain) / steps:.2f}), {per_step:.3f} s per step")
        print(f"forecast for {args.target_steps:.0f} steps per chain (an assumption): "
              f"{args.target_steps * per_step / 3600:.1f} h per chain with these threads")
    if not args.test and rank == 0:
        status(info["output"])


if __name__ == "__main__":
    main()
