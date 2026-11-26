"""Prepare a Cobaya MCMC run: best fit, proposal covariance and production YAML.

Shared by ``notebooks/wip/hires_forestflow_cobaya_mcmc`` (interactive, laptop)
and ``scripts/prepare_mcmc.py`` (batch, e.g. a NERSC job), so both write the
same run folder:

``best_fit.npy``
    :meth:`Fit.save` result: best fit and its Gauss-Newton local errors.
``gauss_newton_covmat.txt``
    Regularized covariance from those errors, physical units.
``proposal_covmat.txt``
    Covariance given to the sampler, unit-cube units (``<name>_cube``).
``prior_bounds.yaml``
    Prior bounds of that unit cube (:func:`lya_interface.unit_cube.read_prior_bounds`).
``cobaya_mcmc.yaml``
    Production YAML run by ``scripts/run_mcmc.py`` (test and production).
``best_fit.png``
    P1D and residuals at the best fit (written by :func:`prepare_run`).
"""
from dataclasses import dataclass
import os
from pathlib import Path
from time import perf_counter

import numpy as np

CHAIN_PREFIX = "chains/mcmc"  # output in the run folder YAML; production runs use
# chain_<n>/ folders (scripts/run_mcmc.py --new-chain) with output "mcmc"
TEST_PREFIX = "test/mcmc"  # test chains, relative to the run folder
MAX_FRACTION = 0.25  # largest proposal sigma, as a fraction of the prior range

# Metropolis settings of the production run (see the notebook for the reasons).
MCMC_SETTINGS = {
    "learn_proposal": True,
    # Cobaya learns the proposal only once R-1 is below this; its default (2)
    # left the hires run on the starting Gauss-Newton covariance for hours at
    # R-1 ~ 3-4. 30 is Cobaya's own value when the starting covariance is
    # incomplete.
    "learn_proposal_Rminus1_max": 30,
    "drag": False,  # speed ratios between blocks < 2: nothing to gain
    "oversample_power": 0,
    "proposal_scale": 1.9,
    "Rminus1_stop": 0.02,  # tighten to 0.01 by resuming
    "Rminus1_cl_stop": 0.2,
    "burn_in": 0,
    "max_tries": 10000,  # rejections in a row before a chain is declared stuck
    "seed": None,
}


@dataclass(frozen=True)
class RunPaths:
    """Files of one run folder."""

    folder: Path

    @property
    def best_fit(self):
        return self.folder / "best_fit.npy"

    @property
    def gn_covmat(self):
        return self.folder / "gauss_newton_covmat.txt"

    @property
    def proposal_covmat(self):
        return self.folder / "proposal_covmat.txt"

    @property
    def production_yaml(self):
        return self.folder / "cobaya_mcmc.yaml"

    @property
    def learned_covmat(self):
        """Covariance learned by the latest production chain (``chain_<n>/mcmc.covmat``), if any."""
        chains = sorted((f for f in self.folder.glob("chain_*") if f.name[6:].isdigit()),
                        key=lambda f: int(f.name[6:]))
        return chains[-1] / "mcmc.covmat" if chains else self.folder / "chain_none/mcmc.covmat"

    @property
    def test_prefix(self):
        return self.folder / TEST_PREFIX

    @property
    def best_fit_figure(self):
        return self.folder / "best_fit.png"


def output_root():
    """Return the data folder holding run folders.

    ``$CFS/data`` when ``CFS`` is set (NERSC), else ``<lya>/data`` next to the
    igmhub checkouts (laptop).
    """
    if "CFS" in os.environ:
        return Path(os.environ["CFS"]) / "data"
    from cup1d.utils.utils import get_path_repo
    return Path(get_path_repo("cup1d")).parents[1] / "data"


def default_run_folder(interface_config):
    """Return ``<output_root>/<out_folder of the native cup1d YAML>/cobaya_mcmc``.

    The native ``out_folder`` (e.g. ``./DESIY1_QMLE3__Karacayli2022/global_opt/forest_mpg_gp``)
    is the folder of the native cup1d results of the same analysis.
    """
    from lya_interface.configuration import load_configuration
    from lya_interface.initial_conditions import _native_args

    args = _native_args(load_configuration(interface_config))
    return output_root() / Path(args.out_folder) / "cobaya_mcmc"


def write_covmat(filename, covariance, cov_names):
    """Write a Cobaya covariance file: ``# name ...`` header and the matrix."""
    np.savetxt(filename, covariance, header=" ".join(cov_names))


def read_covmat(filename):
    """Read a Cobaya covariance file into (matrix, names)."""
    with open(filename) as handle:
        cov_names = handle.readline().lstrip("#").split()
    return np.loadtxt(filename), cov_names


def _prior_width(params, name):
    return params[name]["prior"]["max"] - params[name]["prior"]["min"]


def regularized_covariance(covariance, cov_names, params, max_fraction=MAX_FRACTION, floor=1e-12):
    """Symmetric positive-definite covariance with capped standard deviations.

    Each standard deviation is limited to ``max_fraction`` of its prior range
    (unconstrained or bound-limited directions otherwise get huge proposals);
    correlations are kept. The eigenvalue floor (``floor`` times the largest)
    is applied in units of each prior range: parameters span many orders of
    magnitude (As ~ 1e-9), so a floor in physical units would inflate the
    small ones (it gave sigma(As) ~ 3e-6 > As, so every As/ns step was rejected).
    """
    covariance = 0.5 * (covariance + covariance.T)
    sigma = np.sqrt(np.clip(np.diag(covariance), 0, None))
    width = np.array([_prior_width(params, n) for n in cov_names])
    good = sigma > 0
    correlation = np.eye(len(sigma))
    correlation[np.ix_(good, good)] = covariance[np.ix_(good, good)] / np.outer(sigma[good], sigma[good])
    sigma = np.where(good, np.minimum(sigma, max_fraction * width), 0.01 * width)
    scaled = correlation * np.outer(sigma / width, sigma / width)  # prior-range units
    values, vectors = np.linalg.eigh(scaled)
    values = np.clip(values, floor * values.max(), None)
    return ((vectors * values) @ vectors.T) * np.outer(width, width)


def covmat_problems(covariance, cov_names, params, max_fraction=MAX_FRACTION):
    """Parameters whose proposal sigma exceeds ``max_fraction`` of their prior range."""
    problems = {}
    for name, variance in zip(cov_names, np.diag(covariance)):
        prior_range = _prior_width(params, name)
        if not np.sqrt(variance) <= max_fraction * prior_range * (1 + 1e-6):
            problems[name] = float(np.sqrt(variance) / prior_range)
    return problems


def load_fit(interface_config, paths, log=print):
    """Return a :class:`Fit`, restored from the run folder's best fit if saved."""
    from lya_interface.fit import Fit

    if paths.best_fit.is_file():
        fit = Fit.from_results(paths.best_fit, config_path=interface_config)
        log(f"restored best fit from {paths.best_fit}")
    else:
        fit = Fit(interface_config)
    return fit


def initial_point(fit, log=print):
    """Return the global IC point, or the YAML references if there is no IC file.

    The references of a generated interface YAML are the IC values at the time
    it was generated, so they are a good start when the (git-ignored) IC file
    is not available, e.g. on NERSC.
    """
    from lya_interface.initial_conditions import clip_to_prior, global_ic_point

    try:
        point = global_ic_point(fit.info, fit.reference)
        log("starting from the global IC")
    except FileNotFoundError as error:
        log(f"{error}; starting from the YAML references")
        point = dict(fit.reference)
    point, clipped = clip_to_prior(fit.info, point)
    if clipped:
        log(f"moved to the prior bounds: {clipped}")
    return point


def ensure_best_fit(fit, paths, force=False, run_minimizer=True, log=print):
    """Minimize if needed, estimate the local errors if missing, and save.

    Parameters
    ----------
    fit : Fit
        Possibly restored fit.
    paths : RunPaths
        Run folder.
    force : bool, default=False
        Minimize again even if a best fit is saved, starting from it.
    run_minimizer : bool, default=True
        Minimize when no best fit is saved; if false the chain starts from the
        initial point.

    Returns
    -------
    dict
        Starting point of the chains (best fit, or the initial point).
    """
    if force or fit.best_fit is None:
        if fit.best_fit is not None:
            start = fit.best_fit.point
            log(f"forced minimization: restarting from the saved best fit {paths.best_fit}")
        else:
            start = initial_point(fit, log=log)
        if run_minimizer or force:
            t0 = perf_counter()
            fit.minimize(start)  # also discards the previous local errors
            log(f"minimization: {perf_counter() - t0:.0f} s, success={fit.best_fit.success}")
            fit.save(paths.best_fit)
        else:
            log("no best fit: the chain starts from the initial point")
            return start
    fit.near_bounds()
    if fit.local_errors is None:
        t0 = perf_counter()
        fit.estimate_errors()
        log(f"Gauss-Newton errors: {perf_counter() - t0:.0f} s")
        fit.save(paths.best_fit)
    return fit.best_fit.point


def write_proposal_covmat(fit, paths, covmat_from=None, force=False,
                          max_fraction=MAX_FRACTION, log=print):
    """Write the Gauss-Newton covariance and select and write the proposal one.

    Sources, in order: ``covmat_from``; the covariance learned by a previous
    production chain (skipped when ``force``); the Gauss-Newton one.

    Returns
    -------
    covariance, names
        Proposal covariance in physical units and its parameter names, or
        ``(None, [])`` when the sampler will use the YAML proposal widths.
    """
    from lya_interface.unit_cube import (CUBE_SUFFIX, covariance_from_unit_cube, covariance_to_unit_cube,
                                         flat_prior_bounds, read_prior_bounds, write_prior_bounds)

    params = fit.info["params"]
    if fit.local_errors is not None:
        # cheap: always rebuilt from the saved errors, so a regularization change applies
        errors = fit.local_errors
        write_covmat(paths.gn_covmat,
                     regularized_covariance(errors.covariance, list(errors.names), params,
                                            max_fraction=max_fraction),
                     list(errors.names))
        log(f"wrote {paths.gn_covmat}")

    sources = (covmat_from, paths.gn_covmat) if force else \
        (covmat_from, paths.learned_covmat, paths.gn_covmat)
    for source in sources:
        if source is None or not Path(source).is_file():
            continue
        # physical units, whether the file was learned in the unit cube or not; a
        # unit-cube covariance is converted with the priors of its own run
        covariance, cov_names = read_covmat(source)
        if any(name.endswith(CUBE_SUFFIX) for name in cov_names):
            source_path = Path(source)
            try:
                source_bounds = read_prior_bounds(source_path.parent, prefix=source_path.stem)
            except FileNotFoundError as error:
                log(f"skipping {source}: {error}")
                continue
            covariance, cov_names = covariance_from_unit_cube(covariance, cov_names, source_bounds)
        problems = covmat_problems(covariance, cov_names, params, max_fraction)
        if problems:
            log(f"skipping {source}: sigma / prior range too large for {problems}")
            continue
        write_covmat(paths.proposal_covmat, *covariance_to_unit_cube(covariance, cov_names, params))
        write_prior_bounds(paths.folder, flat_prior_bounds(params),
                           note=f"of {paths.proposal_covmat.name}, written by lya_interface.mcmc_setup")
        log(f"proposal covariance from {source}")
        missing = sorted(set(fit.reference) - set(cov_names))
        if missing:
            log(f"not in the proposal covariance (YAML proposal used): {missing}")
        return covariance, cov_names
    paths.proposal_covmat.unlink(missing_ok=True)
    log("no proposal covariance: Cobaya will use the YAML proposal widths")
    return None, []


def write_production_yaml(fit, paths, interface_config, start_point, proposal_cov,
                          proposal_names, log=print):
    """Write the production YAML: interface base, starting ball, sampler, output.

    The ``ref`` of every parameter is a Gaussian of 0.1 proposal sigma around
    ``start_point`` (physical units). ``scripts/run_mcmc.py`` replaces it, per
    MPI rank, with a correlated draw from 0.1 x the proposal covariance around
    the same centre (``--start-scale``), so the chains are independent and
    start close to the posterior also along its narrow directions.
    """
    import yaml
    from lya_interface.configuration import portable_interface_base

    params = fit.info["params"]
    sigma = {}
    if proposal_cov is not None:
        sigma = dict(zip(proposal_names, np.sqrt(np.diag(proposal_cov))))
    reference_point = {
        name: {"dist": "norm", "loc": float(start_point[name]),
               "scale": float(0.1 * sigma.get(name, params[name]["proposal"]))}
        for name in fit.reference
    }
    mcmc = {"covmat": paths.proposal_covmat.name if paths.proposal_covmat.is_file() else None,
            **MCMC_SETTINGS}
    production = {
        # lya_interface:<path> resolves against the checkout of each machine
        "interface_base": portable_interface_base(interface_config, paths.folder),
        "sample_in_unit_cube": True,
        "reference_point": reference_point,
        "sampler": {"mcmc": mcmc},
        "output": CHAIN_PREFIX,
        "resume": True,
    }
    paths.production_yaml.write_text(
        "# Production Cobaya MCMC written by lya_interface.mcmc_setup.\n"
        "# Test: python <lya_interface>/scripts/run_mcmc.py cobaya_mcmc.yaml --test\n"
        "# Production: sbatch <lya_interface>/scripts/nersc/mcmc_job.sh\n"
        + yaml.safe_dump(production, sort_keys=False)
    )
    log(f"wrote {paths.production_yaml}")


def plot_best_fit(fit, paths, title=None, unblind=False, log=print):
    """Print the goodness of fit and star values; save the best-fit figure.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    from lya_interface.diagnostics import (
        format_goodness_of_fit, goodness_of_fit, plot_point_with_residuals,
    )

    if fit.best_fit is None:
        log("no best fit to check")
        return None
    statistics = fit.best_fit.statistics or goodness_of_fit(fit.model, fit.best_fit.point)
    log(format_goodness_of_fit(statistics, multiline=True))
    log(str(dict(zip(("Delta2_star", "n_star"), fit.star_values(unblind=unblind)))))
    fig, _ = plot_point_with_residuals(fit.model, fit.best_fit.point, ncols=5, title=title)
    fig.savefig(paths.best_fit_figure, dpi=120)
    log(f"wrote {paths.best_fit_figure}")
    return fig


def prepare_run(interface_config, folder=None, force=False, covmat_from=None,
                title=None, log=print):
    """Best fit, local errors, covariances, production YAML and best-fit figure.

    Parameters
    ----------
    interface_config : path-like
        Interface YAML (e.g. ``examples/hires_forestflow_evaluate.yaml``).
    folder : path-like, optional
        Run folder; by default :func:`default_run_folder`.
    force : bool, default=False
        Redo everything: minimize again (from the saved best fit if any),
        re-estimate the errors and ignore a previously learned covariance.
    covmat_from : path-like, optional
        Proposal covariance of another run, preferred over the others.

    Returns
    -------
    RunPaths
    """
    interface_config = Path(interface_config).expanduser().resolve()
    folder = default_run_folder(interface_config) if folder is None else Path(folder).expanduser()
    paths = RunPaths(folder.resolve())
    paths.folder.mkdir(parents=True, exist_ok=True)
    log(f"run folder: {paths.folder}")
    with load_fit(interface_config, paths, log=log) as fit:
        start_point = ensure_best_fit(fit, paths, force=force, log=log)
        covariance, names = write_proposal_covmat(fit, paths, covmat_from=covmat_from,
                                                  force=force, log=log)
        write_production_yaml(fit, paths, interface_config, start_point, covariance, names, log=log)
        plot_best_fit(fit, paths, title=title, log=log)
    return paths
