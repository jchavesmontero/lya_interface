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


# Cobaya YAML files rewritten in place at every (re)start or convergence check.
# On CFS a rewrite shorter than the old file has left the old tail behind
# (e.g. a last line ".6.2" after "version: 3.6.2"), which breaks the next resume.
COBAYA_YAML_FILES = ("input.yaml", "updated.yaml", "checkpoint")


def load_yaml_tolerant(path):
    """Load a YAML file, dropping trailing lines left over from an older, longer version.

    Returns
    -------
    content, dropped
        The parsed content of the longest valid beginning of the file, and the
        dropped trailing lines (empty if the file is valid).
    """
    import yaml
    lines = Path(path).read_text().splitlines(keepends=True)
    for end in range(len(lines), 0, -1):
        try:
            return yaml.safe_load("".join(lines[:end])), lines[end:]
        except yaml.YAMLError:
            continue
    raise ValueError(f"{path}: no valid YAML at its beginning")


def repair_cobaya_files(prefix):
    """Remove trailing leftovers from Cobaya's YAML files of a prefix (keeps a backup)."""
    from datetime import datetime
    for suffix in COBAYA_YAML_FILES:
        path = Path(f"{prefix}.{suffix}")
        if not path.is_file():
            continue
        _, dropped = load_yaml_tolerant(path)
        if dropped:
            backup = path.with_name(f"{path.name}.corrupt-{datetime.now():%Y%m%dT%H%M%S}")
            text = path.read_text()
            backup.write_text(text)
            path.write_text(text[:len(text) - len("".join(dropped))])
            print(f"repaired {path.name}: removed trailing {''.join(dropped)!r} (backup {backup.name})",
                  flush=True)


def clear_stale_locks(prefix):
    """Remove Cobaya lock files left by a killed job (``<prefix>*.locked``, ``*.lock_err``).

    With ``COBAYA_USE_FILE_LOCKING=False`` (set by the Slurm jobs, as locks are
    unreliable on CFS) Cobaya still refuses to start if ``<prefix>.input.yaml.locked``
    exists, and a cancelled or killed job leaves it behind. The Slurm jobs never
    overlap (each successor waits for the previous one to end), so a lock found
    at start is stale.
    """
    folder, name = Path(prefix).parent, Path(prefix).name
    for pattern in (f"{name}*.locked", f"{name}*.lock_err"):
        for path in folder.glob(pattern):
            path.unlink(missing_ok=True)
            print(f"removed stale lock {path.name}", flush=True)


def read_checkpoint(prefix):
    """Cobaya checkpoint of an output prefix, or None if absent."""
    path = Path(f"{prefix}.checkpoint")
    if not path.is_file():
        return None
    return load_yaml_tolerant(path)[0]["sampler"]["mcmc"]


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


def status(prefix, detailed=False):
    """Print the convergence status of a run; return the exit code (0 converged, 1 not, 2 not started).

    Quick by default: the checkpoint and the last line of the progress file,
    both tiny, so the Slurm job can call it before and after every run. With
    ``detailed`` it also reads every chain (accepted points per chain, overall
    acceptance) and reloads them with Cobaya; for long production chains on
    CFS that takes minutes.
    """
    checkpoint = read_checkpoint(prefix)
    if checkpoint is None:
        print(f"{prefix}: no checkpoint (not started)", flush=True)
        return 2
    print(f"{prefix}: converged = {checkpoint.get('converged')}, "
          f"R-1 (last) = {checkpoint.get('Rminus1_last')}, chains = {checkpoint.get('mpi_size')}", flush=True)
    progress = Path(f"{prefix}.progress")
    if progress.is_file():
        rows = [line.split() for line in progress.read_text().splitlines()
                if line.strip() and not line.startswith("#")]
        if rows and len(rows[-1]) >= 5:
            n, timestamp, rate, rminus1, rminus1_cl = rows[-1][:5]
            print(f"  last check {timestamp}: {float(n):.0f} accepted points (all chains), "
                  f"acceptance {float(rate):.3f}, R-1 {float(rminus1):.3f}, R-1 (95%) {rminus1_cl}",
                  flush=True)
    for suffix in ("covmat", "progress", "updated.yaml"):
        path = Path(f"{prefix}.{suffix}")
        print(f"  {path.name}: {'written' if path.is_file() else 'MISSING'}", flush=True)
    if detailed:
        for name, n in count_samples(prefix).items():
            print(f"  {name}: {n} accepted points", flush=True)
        accepted, steps = acceptance(prefix)
        if steps:
            print(f"  all chains: {accepted} accepted out of {steps:.0f} steps "
                  f"(acceptance {accepted / steps:.2f})", flush=True)
        try:  # the chains must load as the analysis will load them
            from cobaya import load_samples
            samples = load_samples(str(prefix), skip=0, combined=True)
            physical = [name for name in ("As", "ns", "Delta2star") if name in samples.data.columns]
            print(f"  reloaded: {len(samples)} points; physical columns present: {physical}", flush=True)
        except Exception as error:  # report, do not fail the status check
            print(f"  reloading failed: {error!r}", flush=True)
    return 0 if checkpoint.get("converged") else 1


def sync_refs_from_chain(info, prefix):
    """On resume, take every parameter's ``ref`` and ``proposal`` from the existing chains.

    Cobaya's resume check (``is_equal_info``, 3.6.2) is meant to ignore
    ``ref`` and ``proposal`` of parameters but does not, so any change of the
    starting points blocks the resume ("different content of [params:As_cube]").
    The chains' own ``updated.yaml`` holds the values used when the chain
    started (e.g. rank 0's correlated start). Neither matters when resuming.

    Returns the number of parameters updated.
    """
    path = Path(f"{prefix}.updated.yaml")
    if not path.is_file():
        return 0
    old_params = load_yaml_tolerant(path)[0].get("params", {})
    changed = 0
    for name, definition in info["params"].items():
        old = old_params.get(name)
        if not (isinstance(definition, dict) and isinstance(old, dict)):
            continue
        for key in ("ref", "proposal"):
            if key in old and definition.get(key) != old[key]:
                definition[key] = old[key]
                changed += 1
            elif key not in old and key in definition:
                definition.pop(key)
                changed += 1
    return changed


def check_resume(info, prefix):
    """Print how the new run information differs from that of the existing chains.

    Cobaya refuses to resume when they differ ("Old and new run information not
    compatible") but names only the first difference. This lists every sampled
    or derived parameter whose definition differs (after taking ref and
    proposal from the chains, as a resume does), and the sampler options that
    differ. Returns the number of differing parameters.
    """
    path = Path(f"{prefix}.updated.yaml")
    if not path.is_file():
        print(f"{path}: not found (nothing to resume)")
        return 0
    old = load_yaml_tolerant(path)[0]
    sync_refs_from_chain(info, prefix)  # as a resume does
    # Cobaya 3.6.2 compares every key of a parameter, ref and proposal included
    ignored = {"latex", "renames"}

    def clean(definition):
        if not isinstance(definition, dict):
            return {"value": definition}
        return {k: v for k, v in definition.items() if k not in ignored}

    old_params, new_params = old.get("params", {}), info.get("params", {})
    differing = 0
    for name in sorted(set(old_params) | set(new_params)):
        if name not in old_params or name not in new_params:
            print(f"  {name}: only in the {'new' if name in new_params else 'old'} run")
            differing += 1
            continue
        a, b = clean(old_params[name]), clean(new_params[name])
        if a != b:
            differing += 1
            print(f"  {name}:")
            for key in sorted(set(a) | set(b)):
                if a.get(key) != b.get(key):
                    print(f"      {key}: old {a.get(key)!r}")
                    print(f"      {key}: new {b.get(key)!r}")
    old_mcmc, new_mcmc = old.get("sampler", {}).get("mcmc", {}), info["sampler"]["mcmc"]
    for key in sorted(set(new_mcmc)):
        if key in old_mcmc and old_mcmc[key] != new_mcmc[key]:
            print(f"  sampler.mcmc.{key}: old {old_mcmc[key]!r}, new {new_mcmc[key]!r}")
    print(f"{differing} parameters differ")
    return differing


def correlated_start(info, rank, scale):
    """Set this rank's starting point to a draw from the proposal covariance.

    The production YAML gives every parameter an independent Gaussian ``ref``
    around the best fit. With a strongly correlated posterior, independent
    offsets of a fraction of each marginal sigma are many sigma away along the
    narrow directions (Cobaya warns that the initial points are widely
    dispersed, and the chains spend long in burn-in). Here each MPI rank draws
    its start from ``N(best fit, scale**2 * C)``, with ``C`` the proposal
    covariance (unit cube) and the best fit the centre of the YAML ``ref``;
    draws with a coordinate outside [0, 1] are redrawn (reflected after 10000 tries).

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
    # redraw until every coordinate is inside [0, 1]: reflecting at the edges
    # (parameters at a bound, e.g. HCD amplitudes at 0) breaks the correlations
    # and can leave the start many sigma away; reflection only as a fallback
    for _ in range(10000):
        start = centre + scale * chol @ rng.standard_normal(len(names))
        if np.all((start >= 0) & (start <= 1)):
            break
    else:
        start = np.abs(start)
        start = np.where(start > 1, 2 - start, start)
    for name, value in zip(names, start):
        params[name]["ref"] = float(value)
    step = np.linalg.solve(chol, start - centre)
    return float(np.sqrt(step @ step))


CHAIN_PREFIX = "chain_"


def chain_folders(run_dir):
    """Existing ``chain_<n>`` folders of a run folder, in increasing ``n``."""
    folders = [f for f in Path(run_dir).glob(f"{CHAIN_PREFIX}*")
               if f.is_dir() and f.name[len(CHAIN_PREFIX):].isdigit()]
    return sorted(folders, key=lambda f: int(f.name[len(CHAIN_PREFIX):]))


INTERFACE_FILE = "interface.yaml"


def freeze_interface_base(base_path, folder, number):
    """Copy an interface configuration into a chain folder.

    Asset paths relative to the original file become ``<package>:<path>`` (or
    absolute outside the checkouts), so the copy works from the chain folder
    and on other machines. A base that overlays yet another configuration is
    not frozen (the chain keeps pointing at it).

    Returns
    -------
    str
        The ``interface_base`` entry of the chain YAML.
    """
    import yaml
    from lya_interface.configuration import portable_path, resolve_package_path

    base_path = Path(base_path).resolve()
    base = yaml.safe_load(base_path.read_text())
    if "interface_base" in base:
        print(f"{base_path} overlays another configuration: not frozen in chain {number}")
        return str(base_path)
    for section in ("theory", "likelihood"):
        for name, options in base.get(section, {}).items():
            if not (name.startswith("lya_interface.") and isinstance(options, dict)):
                continue
            for key in ("native_config", "covariance_asset", "model_path", "transformation_path",
                        "metal_model", "metal_alpha_reference"):
                value = options.get(key)
                if not isinstance(value, str) or not value or resolve_package_path(value) is not None:
                    continue
                path = Path(value)
                options[key] = portable_path(path if path.is_absolute() else base_path.parent / path, folder)
    (Path(folder) / INTERFACE_FILE).write_text(
        f"# Chain {number}: frozen copy of {base_path}\n" + yaml.safe_dump(base, sort_keys=False))
    return INTERFACE_FILE


def new_chain(production_yaml):
    """Create the next ``chain_<n>`` next to a production YAML and freeze its inputs there.

    The folder gets a copy of the production YAML, with its output set to
    ``mcmc`` inside the folder, of the proposal covariance it names,
    ``prior_bounds.yaml`` with the prior bounds of the chain's unit cube, and
    ``native_config.yaml`` with the complete resolved native cup1d settings
    (which the chain's YAML then uses instead of the checkout's). Runs
    and resumes use that copy, so rewriting the run folder's YAML or covariance
    (e.g. a new prepare job) never changes an existing chain, and a new chain
    never writes into an old one.

    The proposal covariance is in the unit cube of the run folder's
    ``prior_bounds.yaml``; if the priors changed since, it is converted to the
    chain's unit cube.

    Returns
    -------
    pathlib.Path
        The new chain's production YAML.

    Raises
    ------
    FileNotFoundError
        If the proposal covariance is in unit-cube units but the run folder
        does not record its prior bounds (prepare the run again, e.g.
        ``REUSE=1`` in the Slurm job).
    """
    import yaml
    from lya_interface.configuration import (NATIVE_CONFIG_FILE, freeze_native_config, load_configuration,
                                             resolve_interface_base)
    from lya_interface.mcmc_setup import read_covmat, write_covmat
    from lya_interface.unit_cube import (CUBE_SUFFIX, bounds_from_unit_cube_params, covariance_from_unit_cube,
                                         covariance_to_unit_cube, read_prior_bounds, write_prior_bounds)

    production_yaml = Path(production_yaml).resolve()
    run_dir = production_yaml.parent
    existing = chain_folders(run_dir)
    number = int(existing[-1].name[len(CHAIN_PREFIX):]) + 1 if existing else 1
    folder = run_dir / f"{CHAIN_PREFIX}{number}"
    info = yaml.safe_load(production_yaml.read_text())
    info["output"] = "mcmc"
    # the bounds the chain samples with: those of the resolved configuration, now
    resolved = load_configuration(production_yaml)
    bounds = bounds_from_unit_cube_params(resolved["params"])
    covmat = info.get("sampler", {}).get("mcmc", {}).get("covmat")
    proposal = None
    if covmat:
        covariance, names = read_covmat(run_dir / covmat)
        if any(name.endswith(CUBE_SUFFIX) for name in names):
            source = read_prior_bounds(run_dir)  # raises if the run folder does not record them
            changed = sorted(n for n in names if n.endswith(CUBE_SUFFIX)
                             and source.get(n[:-len(CUBE_SUFFIX)]) != bounds.get(n[:-len(CUBE_SUFFIX)]))
            if changed:
                print(f"priors changed since {covmat} was written, converted to the new unit cube: {changed}")
                physical, physical_names = covariance_from_unit_cube(covariance, names, source)
                covariance, names = covariance_to_unit_cube(physical, physical_names, bounds)
        proposal = (covariance, names)
    folder.mkdir()  # fails if it exists: never reuse a chain folder
    if proposal is not None:
        write_covmat(folder / Path(covmat).name, *proposal)
        info["sampler"]["mcmc"]["covmat"] = Path(covmat).name
    if bounds:
        write_prior_bounds(folder, bounds, note=f"of chain {number} (mcmc.*.txt, mcmc.covmat, {covmat})")
    # complete native cup1d settings (and so the fiducial cosmology), frozen
    native_frozen = freeze_native_config(resolved, folder, note=f"for chain {number}") is not None
    if native_frozen:
        info["native_config"] = NATIVE_CONFIG_FILE
    # and the interface configuration it overlays (CAMB settings, fixed
    # cosmology, fiducial M(z), covariance asset): regenerating the example
    # later does not change this chain
    if "interface_base" in info:
        info["interface_base"] = freeze_interface_base(
            resolve_interface_base(info["interface_base"], run_dir), folder, number)
    (folder / production_yaml.name).write_text(
        f"# Chain {number}: frozen copy of {production_yaml.name} (and its proposal covariance)\n"
        "# made when the chain was created; the run and its resumes read this file.\n"
        + yaml.safe_dump(info, sort_keys=False))
    return folder / production_yaml.name


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
    parser.add_argument("--check-resume", action="store_true",
                        help="list how the run information differs from that of the existing chains")
    parser.add_argument("--detailed", action="store_true",
                        help="with --status: also read every chain and reload them (slow for long chains)")
    parser.add_argument("--new-chain", action="store_true",
                        help="create the next chain_<n> folder next to the production YAML, "
                             "print the path of its YAML and exit")
    parser.add_argument("--start-scale", type=float, default=0.1,
                        help="each chain starts at a draw from N(best fit, scale^2 * proposal "
                             "covariance); 0 keeps the independent YAML references")
    args = parser.parse_args()

    production_yaml = Path(args.production_yaml).resolve()
    if args.new_chain:
        print(new_chain(production_yaml))
        return
    # production chains only inside their own chain_<n> folder (see --new-chain),
    # so that an old run is never resumed or overwritten by mistake
    if not (args.test or args.status or args.check_resume) and \
            not production_yaml.parent.name.startswith(CHAIN_PREFIX):
        sys.exit(f"{production_yaml} is not in a {CHAIN_PREFIX}<n> folder: create one with "
                 "--new-chain (or NEW_CHAIN=1 in the Slurm job)")
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
        sys.exit(status(info["output"], detailed=args.detailed))
    if args.check_resume:
        sys.exit(1 if check_resume(info, info["output"]) else 0)

    if args.test:
        info["resume"] = False
        info["force"] = True
        # seed: Cobaya derives a different seed for every MPI rank from it
        info["sampler"]["mcmc"].update(max_samples=args.test_samples, seed=1,
                                       learn_every=args.learn_every)

    from cobaya.run import run
    rank = mpi_rank()
    if rank == 0:
        repair_cobaya_files(info["output"])
        clear_stale_locks(info["output"])
    try:  # the other ranks wait for the repair
        from mpi4py import MPI
        MPI.COMM_WORLD.Barrier()
    except ImportError:
        pass
    # on resume Cobaya continues the chains and ignores the starting points
    resuming = bool(info.get("resume")) and read_checkpoint(info["output"]) is not None
    if resuming:
        synced = sync_refs_from_chain(info, info["output"])
        if rank == 0:
            print(f"resuming {info['output']} from its checkpoint "
                  f"(ref/proposal of the chains kept for {synced} entries)", flush=True)
    if args.start_scale > 0 and not resuming:
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
