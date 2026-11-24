"""Time one likelihood call against the number of OpenMP threads.

Chooses how to split a node between MCMC chains (MPI ranks) and threads per
chain. For each thread count, a fresh process builds the Cobaya model and times
log-posterior calls at distinct points with every parameter moved (no cache
hits; an upper bound on the cost of one MCMC step). It reports, per split of a
node, the chains, the node throughput (likelihood calls per second) and the
memory of all ranks.

Run it on a compute node, not a login node, e.g. on Perlmutter::

    salloc -A <account> -C cpu -q interactive -N 1 -t 00:30:00
    python scripts/thread_scaling.py examples/hires_forestflow_evaluate.yaml \\
        --threads 1 2 4 8 16 32 --cores-per-node 128 --node-memory-gb 512

On a laptop, use the default thread counts and ``--cores-per-node`` equal to
its cores minus two.
"""
import argparse
import json
import os
import subprocess
import sys

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def worker(config, n_calls):
    """Time ``n_calls`` log-posteriors in this process; print a JSON line."""
    import resource
    from time import perf_counter

    import numpy as np
    from cobaya.model import get_model
    from lya_interface.configuration import load_configuration

    info = load_configuration(config)
    t0 = perf_counter()
    model = get_model(info)
    setup = perf_counter() - t0
    names = list(model.parameterization.sampled_params())
    ref = {n: info["params"][n]["ref"] for n in names}
    ref = {n: v["loc"] if isinstance(v, dict) else v for n, v in ref.items()}
    width = {n: info["params"][n]["prior"]["max"] - info["params"][n]["prior"]["min"] for n in names}
    model.logposterior(ref)  # first call: emulator and CAMB set-up
    times = []
    for i in range(n_calls):
        step = 1e-4 * (i + 1) * (-1) ** i
        point = {n: ref[n] + step * width[n] for n in names}
        t0 = perf_counter()
        model.logposterior(point)
        times.append(perf_counter() - t0)
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2  # kB on Linux
    print(json.dumps(dict(seconds=float(np.median(times)), setup=setup, rss_gb=rss_gb)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="interface YAML (e.g. examples/hires_forestflow_evaluate.yaml)")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--calls", type=int, default=10, help="timed calls per thread count")
    parser.add_argument("--cores-per-node", type=int, default=max(1, (os.cpu_count() or 3) - 2))
    parser.add_argument("--node-memory-gb", type=float, default=None,
                        help="node memory; splits that do not fit are flagged")
    parser.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.config, args.calls)
        return

    rows = []
    for n in args.threads:
        env = dict(os.environ, **{name: str(n) for name in THREAD_VARIABLES})
        out = subprocess.run([sys.executable, __file__, args.config, "--worker", str(n), "--calls", str(args.calls)],
                             env=env, capture_output=True, text=True)
        lines = [line for line in out.stdout.splitlines() if line.startswith("{")]
        if out.returncode or not lines:
            print(f"{n} threads failed:\n{out.stderr[-2000:]}")
            continue
        result = json.loads(lines[-1])
        rows.append((n, result))
        print(f"{n:3d} threads: {result['seconds']:.3f} s per call, {result['rss_gb']:.1f} GB, "
              f"setup {result['setup']:.0f} s", flush=True)

    print(f"\nnode of {args.cores_per_node} cores"
          + (f" and {args.node_memory_gb:.0f} GB" if args.node_memory_gb else "") + ":")
    print(f"{'threads':>8s} {'chains':>7s} {'s/call':>8s} {'calls/s per node':>17s} {'memory GB':>10s}")
    best = None
    for n, result in rows:
        chains = args.cores_per_node // n
        if chains == 0:
            continue
        throughput = chains / result["seconds"]
        memory = chains * result["rss_gb"]
        fits = args.node_memory_gb is None or memory < 0.9 * args.node_memory_gb
        print(f"{n:8d} {chains:7d} {result['seconds']:8.3f} {throughput:17.1f} {memory:10.1f}"
              + ("" if fits else "  (does not fit)"))
        if fits and chains >= 4 and (best is None or throughput > best[1]):
            best = (n, throughput, chains)
    if best:
        print(f"\nhighest throughput with >= 4 chains: {best[2]} chains x {best[0]} threads. "
              "Fewer, faster chains reach a given length sooner; more chains give more samples "
              "per hour once they have converged.")


if __name__ == "__main__":
    main()
