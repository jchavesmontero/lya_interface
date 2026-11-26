"""Diagnostics of a (possibly unconverged) Cobaya MCMC from its files on disk.

Light enough for a NERSC login node: it only reads the chain text files (no
model, emulator or MPI). It writes figures and a text summary to a small
folder that is cheap to copy to a laptop:

``summary.txt``
    Sampler settings, convergence history, chains ranked by their mean
    ``-log P`` (stuck chains), parameters ranked by their own R-1 (which mix
    worst), posterior mass at the prior edges, and the learned proposal widths
    relative to the initial ones.
``progress.png``
    Acceptance rate and R-1 at every convergence check.
``logpost_traces.png``
    ``-log P`` of every chain against accepted step; stuck chains in colour.
``worst_traces.png``
    Traces of the parameters with the largest R-1.
``rminus1_per_param.png``
    R-1 of every parameter.

Statistics use the second half of each chain (``--burn``) and the unit-cube
coordinates Cobaya samples (``<name>_cube``, prior range [0, 1]).

Usage (NERSC login node, about a minute for 64 long chains)::

    python scripts/chain_diagnostics.py $RUN_DIR               # latest chain_<n>
    python scripts/chain_diagnostics.py $RUN_DIR --prefix test/mcmc

Output: ``$RUN_DIR/diagnostics/<chain folder>/`` (e.g. ``chain_1``) unless ``--out`` is given.
"""
import argparse
from pathlib import Path

import numpy as np


def read_chains(prefix, columns_wanted=None):
    """Read ``<prefix>.<i>.txt`` into a list of (name, DataFrame), in chain order."""
    import pandas as pd

    files = sorted(Path(prefix).parent.glob(f"{Path(prefix).name}.*.txt"),
                   key=lambda f: int(f.name.split(".")[-2]))
    chains = []
    for file in files:
        with open(file) as handle:
            names = handle.readline().lstrip("#").split()
        usecols = names if columns_wanted is None else [n for n in names if columns_wanted(n)]
        # the last line may be incomplete while a running job writes it
        frame = pd.read_csv(file, sep=r"\s+", skiprows=1, names=names, usecols=usecols,
                            on_bad_lines="skip").apply(pd.to_numeric, errors="coerce").dropna()
        frame = frame.astype({n: np.float32 for n in usecols if n != "weight"})
        chains.append((file.name, frame))
    return chains


def weighted_mean_var(values, weights):
    mean = np.average(values, weights=weights, axis=0)
    var = np.average((values - mean) ** 2, weights=weights, axis=0)
    return mean, var


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="run folder (holds cobaya_mcmc.yaml, chains/, test/)")
    parser.add_argument("--prefix", default=None,
                        help="chain prefix relative to run_dir (default: the latest chain_<n>/mcmc; "
                             "test/mcmc for the test chain)")
    parser.add_argument("--out", default=None, help="output folder (default run_dir/diagnostics/<prefix name>)")
    parser.add_argument("--burn", type=float, default=0.5, help="fraction of each chain skipped in statistics")
    parser.add_argument("--n-worst", type=int, default=8, help="parameters in worst_traces.png")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml

    run_dir = Path(args.run_dir).expanduser().resolve()
    if args.prefix is None and run_dir.name[6:].isdigit() and run_dir.name.startswith("chain_"):
        # given a chain folder itself: use it, with the run folder above it
        args.prefix = f"{run_dir.name}/mcmc"
        run_dir = run_dir.parent
    if args.prefix is None:
        chains = sorted((f for f in run_dir.glob("chain_*") if f.name[6:].isdigit()),
                        key=lambda f: int(f.name[6:]))
        if not chains:
            raise SystemExit(f"no chain_<n> folder in {run_dir}; pass --prefix")
        args.prefix = f"{chains[-1].name}/mcmc"
    prefix = run_dir / args.prefix
    out = Path(args.out) if args.out else run_dir / "diagnostics" / Path(args.prefix).parent.name
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def report(text=""):
        print(text, flush=True)
        lines.append(text)

    report(f"chains: {prefix}")

    # settings and convergence history
    # settings Cobaya used; if that file cannot be read (e.g. being rewritten by
    # a resuming job), those of the production YAML
    mcmc = {}
    for source in (Path(f"{prefix}.updated.yaml"), run_dir / "cobaya_mcmc.yaml"):
        if not source.is_file():
            continue
        try:
            mcmc = yaml.safe_load(source.read_text())["sampler"]["mcmc"]
        except Exception as error:
            report(f"  could not read {source.name}: {type(error).__name__}; trying the next source")
            continue
        report(f"  settings from {source.name}")
        break
    for key in ("covmat", "learn_proposal", "learn_proposal_Rminus1_max", "proposal_scale",
                "Rminus1_stop", "Rminus1_cl_stop", "max_samples"):
        report(f"  {key:>28s}: {mcmc.get(key)}")
    progress_file = Path(f"{prefix}.progress")
    if progress_file.is_file():
        progress = np.atleast_2d(np.loadtxt(progress_file, usecols=(0, 2, 3, 4)))
        report(f"\n{len(progress)} convergence checks; last: N={progress[-1, 0]:.0f}, "
               f"acceptance={progress[-1, 1]:.3f}, R-1={progress[-1, 2]:.3f}, R-1 (95%)={progress[-1, 3]:.3f}")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
        axes[0].plot(progress[:, 0], progress[:, 1], "o-")
        axes[0].axhspan(0.15, 0.35, color="0.92")
        axes[0].set_ylabel("acceptance rate")
        axes[1].semilogy(progress[:, 0], progress[:, 2], "o-", label="means")
        if np.any(np.isfinite(progress[:, 3])):
            axes[1].semilogy(progress[:, 0], progress[:, 3], "s-", label="95% limits")
        axes[1].axhline(mcmc.get("Rminus1_stop") or 0.02, color="k", ls="--", lw=0.8, label="stop")
        axes[1].set_ylabel(r"$R-1$")
        axes[1].legend(frameon=False)
        for ax in axes:
            ax.set_xlabel("accepted points, all chains")
        fig.tight_layout()
        fig.savefig(out / "progress.png", dpi=110)
        plt.close(fig)

    # chains: weights, -log P and the sampled unit-cube coordinates
    chains = read_chains(prefix, lambda n: n in ("weight", "minuslogpost") or n.endswith("_cube"))
    if not chains:
        raise SystemExit(f"no chain files {prefix}.<i>.txt")
    cube = [n for n in chains[0][1].columns if n.endswith("_cube")]
    report(f"\n{len(chains)} chains, {len(cube)} sampled parameters, "
           f"accepted points per chain: {min(len(c) for _, c in chains)}-{max(len(c) for _, c in chains)}")

    # 1. chains ranked by their mean -log P (second half): stuck chains
    rows = []
    for name, frame in chains:
        tail = frame.iloc[int(args.burn * len(frame)):]
        mean = np.average(tail["minuslogpost"], weights=tail["weight"])
        rows.append((name, len(frame), mean, float(tail["minuslogpost"].min())))
    means = np.array([r[2] for r in rows])
    median = np.median(means)
    spread = max(1.4826 * np.median(np.abs(means - median)), 1.0)
    stuck = {r[0] for r in rows if r[2] - median > 5 * spread}
    report(f"\nchains by mean -log P of the second half (median {median:.1f}; "
           f"'stuck' = more than {5 * spread:.1f} above it): {len(stuck)} stuck")
    report(f"  {'chain':>14s} {'points':>8s} {'<-logP>':>9s} {'min':>9s}")
    for name, n, mean, low in sorted(rows, key=lambda r: -r[2]):
        report(f"  {name:>14s} {n:8d} {mean:9.2f} {low:9.2f}{'  STUCK' if name in stuck else ''}")

    fig, ax = plt.subplots(figsize=(10, 4))
    for name, frame in chains:
        step = max(1, len(frame) // 2000)
        ax.plot(np.arange(len(frame))[::step], frame["minuslogpost"].to_numpy()[::step],
                lw=0.6, alpha=0.9 if name in stuck else 0.25,
                color=None if name in stuck else "0.4", label=name if name in stuck else None)
    ax.set_ylim(np.percentile(means, 1) - 20, np.percentile(means, 99) + 40)
    ax.set_xlabel("accepted step")
    ax.set_ylabel(r"$-\log P$")
    if stuck:
        ax.legend(fontsize=7, ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "logpost_traces.png", dpi=110)
    plt.close(fig)

    # 2. R-1 of every parameter: between-chain variance of the means over the
    # mean within-chain variance (second halves)
    chain_means, chain_vars = [], []
    for _, frame in chains:
        tail = frame.iloc[int(args.burn * len(frame)):]
        m, v = weighted_mean_var(tail[cube].to_numpy(np.float64), tail["weight"].to_numpy())
        chain_means.append(m)
        chain_vars.append(v)
    chain_means, chain_vars = np.array(chain_means), np.array(chain_vars)
    rminus1 = chain_means.var(axis=0, ddof=1) / chain_vars.mean(axis=0)
    order = np.argsort(rminus1)[::-1]
    physical = [n[:-len("_cube")] for n in cube]
    report("\nparameters by their own R-1 (worst first):")
    for i in order:
        report(f"  {physical[i]:>24s} {rminus1[i]:8.3f}")

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(range(len(cube)), rminus1[order])
    ax.set_xticks(range(len(cube)))
    ax.set_xticklabels([physical[i] for i in order], rotation=90, fontsize=6)
    ax.set_yscale("log")
    ax.axhline(mcmc.get("Rminus1_stop") or 0.02, color="k", ls="--", lw=0.8)
    ax.set_ylabel(r"$R-1$ per parameter")
    fig.tight_layout()
    fig.savefig(out / "rminus1_per_param.png", dpi=110)
    plt.close(fig)

    worst = order[:args.n_worst]
    fig, axes = plt.subplots(len(worst), 1, figsize=(10, 1.4 * len(worst)), sharex=True)
    for ax, i in zip(np.atleast_1d(axes), worst):
        for name, frame in chains:
            step = max(1, len(frame) // 2000)
            ax.plot(np.arange(len(frame))[::step], frame[cube[i]].to_numpy()[::step],
                    lw=0.5, alpha=0.9 if name in stuck else 0.25,
                    color=None if name in stuck else "0.4")
        ax.set_ylim(0, 1)
        ax.set_ylabel(physical[i].replace("_", r"\_"), fontsize=7)
    np.atleast_1d(axes)[-1].set_xlabel("accepted step (unit-cube value; stuck chains in colour)")
    fig.tight_layout()
    fig.savefig(out / "worst_traces.png", dpi=110)
    plt.close(fig)

    # 3. posterior mass within 2% of a prior edge (all chains, second halves)
    values = np.concatenate([f.iloc[int(args.burn * len(f)):][cube].to_numpy() for _, f in chains])
    weights = np.concatenate([f.iloc[int(args.burn * len(f)):]["weight"].to_numpy() for _, f in chains])
    low = weights @ (values < 0.02) / weights.sum()
    high = weights @ (values > 0.98) / weights.sum()
    report("\nposterior mass within 2% of a prior edge (> 5%):")
    edges = [(physical[i], low[i], high[i]) for i in range(len(cube)) if max(low[i], high[i]) > 0.05]
    for name, lo, hi in sorted(edges, key=lambda e: -max(e[1], e[2])):
        report(f"  {name:>24s}  lower {lo:.0%}  upper {hi:.0%}")
    if not edges:
        report("  none")

    # 4. learned proposal widths relative to the initial ones
    learned_file = Path(f"{prefix}.covmat")
    initial_file = run_dir / (mcmc.get("covmat") or "proposal_covmat.txt")
    if learned_file.is_file() and initial_file.is_file():
        def sigmas(path):
            with open(path) as handle:
                names = handle.readline().lstrip("#").split()
            return dict(zip(names, np.sqrt(np.diag(np.atleast_2d(np.loadtxt(path))))))
        learned, initial = sigmas(learned_file), sigmas(initial_file)
        ratio = {n[:-len("_cube")]: learned[n] / initial[n] for n in cube if n in learned and n in initial}
        same = all(abs(r - 1) < 1e-6 for r in ratio.values())
        if same:
            report("\nlearned proposal identical to the initial one: not learned yet "
                   "(R-1 above learn_proposal_Rminus1_max)")
        else:
            report("\nlearned / initial proposal sigma (largest changes first):")
            for name, r in sorted(ratio.items(), key=lambda item: -abs(np.log(item[1]))):
                report(f"  {name:>24s} {r:7.2f}")

    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}/: summary.txt, progress.png, logpost_traces.png, "
          "rminus1_per_param.png, worst_traces.png", flush=True)


if __name__ == "__main__":
    main()
