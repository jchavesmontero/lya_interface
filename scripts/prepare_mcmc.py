"""Prepare a Cobaya MCMC run folder: best fit, proposal covariance, production YAML.

Batch counterpart of sections 1-2 of ``notebooks/wip/hires_forestflow_cobaya_mcmc``
(both call :func:`lya_interface.mcmc_setup.prepare_run`). It writes, in the run
folder (by default ``<data>/<out_folder of the native cup1d YAML>/cobaya_mcmc``,
with ``<data>`` = ``$CFS/data`` on NERSC):

* ``best_fit.npy``: best fit and Gauss-Newton errors. Minimized only if
  missing (from the cup1d global IC, or the YAML references when the IC file is
  absent), or always with ``--force`` (restarting from the saved best fit).
* ``gauss_newton_covmat.txt`` and ``proposal_covmat.txt``.
* ``cobaya_mcmc.yaml``: production YAML for ``scripts/run_mcmc.py``.
* ``best_fit.png``: P1D and residuals at the best fit.

Usage (NERSC: ``sbatch scripts/nersc/prepare_job.sh``)::

    python scripts/prepare_mcmc.py [examples/hires_forestflow_evaluate.yaml] [--force]
"""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", nargs="?", default=str(ROOT / "examples/hires_forestflow_evaluate.yaml"),
                        help="interface YAML (default: examples/hires_forestflow_evaluate.yaml)")
    parser.add_argument("--run-dir", default=None,
                        help="run folder (default: <data>/<native out_folder>/cobaya_mcmc)")
    parser.add_argument("--force", action="store_true",
                        help="minimize again (from the saved best fit) and recompute everything; "
                             "a covariance learned by a previous chain is not reused")
    parser.add_argument("--covmat-from", default=None,
                        help="proposal covariance of another run (Cobaya .covmat), preferred")
    parser.add_argument("--title", default=None, help="title of the best-fit figure")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    from lya_interface.mcmc_setup import prepare_run

    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    paths = prepare_run(args.config, folder=args.run_dir, force=args.force,
                        covmat_from=args.covmat_from, title=args.title,
                        log=lambda message: print(message, flush=True))
    print(f"\nnext: test the sampler with {paths.production_yaml}", flush=True)


if __name__ == "__main__":
    main()
