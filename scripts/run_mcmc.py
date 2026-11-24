"""Run, test or check a Cobaya MCMC from a production YAML.

The production YAML (e.g. ``output/hires_forestflow/cobaya_mcmc/cobaya_mcmc.yaml``,
written by ``notebooks/wip/hires_forestflow_cobaya_mcmc``) overlays an interface
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


def status(prefix):
    """Print the convergence status of a run and reload it; return the exit code."""
    checkpoint = read_checkpoint(prefix)
    if checkpoint is None:
        print(f"{prefix}: no checkpoint (not started)")
        return 2
    print(f"{prefix}: converged = {checkpoint.get('converged')}, "
          f"R-1 (last) = {checkpoint.get('Rminus1_last')}, chains = {checkpoint.get('mpi_size')}")
    for name, n in count_samples(prefix).items():
        print(f"  {name}: {n} accepted points")
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
    args = parser.parse_args()

    production_yaml = Path(args.production_yaml).resolve()
    os.chdir(production_yaml.parent)  # covmat, output and interface_base are relative to it

    from lya_interface.configuration import load_configuration
    info = load_configuration(production_yaml)
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
    if rank == 0:
        print(f"{'test' if args.test else 'production'} run of {production_yaml}, output {info['output']}, "
              f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}", flush=True)
    t0 = perf_counter()
    _, sampler = run(info)
    seconds = perf_counter() - t0

    if args.test and rank == 0:
        chain = sampler.products()["sample"]
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
