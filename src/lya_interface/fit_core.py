"""Point evaluation and minimization of a Cobaya graph with the native cup1d likelihood.

Local errors (curvature, emcee) are in :mod:`lya_interface.errors`; the figures
in :mod:`lya_interface.plots`.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
import time
import numpy as np
from lya_interface.parameters import route


@dataclass(frozen=True)
class ComponentEvaluation:
    """Data-only likelihood terms for an arbitrary configured graph.

    ``minus2_loglike`` is the exact convention returned by each component;
    callers must not relabel it as raw data chi-square when a normalization is
    included.  P1D's raw chi-square remains available through
    :func:`evaluate_point`.
    """
    loglikes: dict
    minus2_loglike: dict
    total_minus2_loglike: float


def evaluate_components(model, point):
    """Evaluate every configured data likelihood without assuming P1D.

    Parameters
    ----------
    model : cobaya.model.Model
        Initialized P1D-only, Vega-only, or joint dependency graph.
    point : mapping
        Physical sampled parameter values.

    Returns
    -------
    ComponentEvaluation
        Likelihood-name keyed log likelihoods and their total data objective.

    Raises
    ------
    ValueError
        If the point is outside support or the model domain.
    """
    posterior = model.logposterior(point)
    if not np.isfinite(posterior.logpost):
        raise ValueError("point is outside the prior or admitted model domain")
    loglikes = {name: float(value)
                for name, value in zip(model.likelihood, posterior.loglikes)}
    return ComponentEvaluation(loglikes, {name: -2 * value for name, value in loglikes.items()},
                               -2 * float(sum(loglikes.values())))


def cup1d_component(model):
    """Select the unique cup1d P1D likelihood without relying on ordering.

    Parameters
    ----------
    model : cobaya.model.Model
        Initialized model containing exactly one Cup1DLikelihood.

    Returns
    -------
    Cup1DLikelihood
        The selected native P1D component.

    Raises
    ------
    ValueError
        If the model has zero or multiple cup1d P1D likelihoods.
    """
    from lya_interface.likelihoods.cup1d import Cup1DLikelihood

    matches = [
        component
        for component in model.likelihood.values()
        if isinstance(component, Cup1DLikelihood)
    ]
    if len(matches) != 1:
        raise ValueError("P1D diagnostics require exactly one cup1d likelihood")
    return matches[0]


def evaluate_point(model, point):
    """
    Return the native data result and rebinned model at a sampled point.

    Parameters
    ----------
    model : cobaya.model.Model
        Initialized dependency graph with one cup1d likelihood.
    point : dict
        Physical values of every sampled parameter, keyed by public name.

    Returns
    -------
    result : cup1d.likelihood.external.LikelihoodResult
        Correlated P1D data likelihood, chi-squared and covariance diagnostics.
    powers : dict
        Dataset-indexed lists of rebinned P1D arrays in km/s.

    Raises
    ------
    ValueError
        If the point violates prior support or the admitted model domain.

    Notes
    -----
    Cobaya's priors and model-domain checks are respected. No second cosmology,
    emulator, or native Analysis is constructed.
    """
    posterior = model.logposterior(point)
    if not np.isfinite(posterior.logpost):
        raise ValueError("point is outside the prior or admitted model domain")
    prediction = model.provider.get_result("forestflow_p1d")
    if not prediction["valid"]:
        raise ValueError(prediction["reason"])
    likelihood = cup1d_component(model)
    values = model.parameterization.input_params()
    nuisance = route({k: values[k] for k in likelihood.mapping}, likelihood.mapping)
    moments = prediction.get("metal_moments")
    result = likelihood.backend.evaluate_from_p1d(
        prediction["powers"], prediction["context"], nuisance, moments
    )
    powers = likelihood.backend.apply_observation_model(
        prediction["powers"], prediction["context"], nuisance, moments
    )
    return result, powers


@dataclass(frozen=True)
class PointFit:
    """Store the result of a bounded diagnostic minimization.

    Attributes
    ----------
    point : dict
        Best sampled physical parameter values.
    chi2_data : float
        P1D data chi-squared at ``point``, or ``NaN`` when the configured
        graph has no P1D component (for example Vega-only mode).
    success : bool
        Optimizer convergence flag.
    message : str
        Optimizer termination message.
    evaluations : int
        Number of objective evaluations.
    minus2_loglike_total : float
        Combined likelihood statistic, including every configured likelihood.
    """
    point: dict
    chi2_data: float
    success: bool
    message: str
    evaluations: int
    minus2_loglike_total: float
    statistics: object | None = None


@dataclass(frozen=True)
class GoodnessOfFit:
    """Raw data chi-square, degrees of freedom, and upper-tail probability."""
    chi2: float
    ndata: int
    nfit: int
    probability: float

    @property
    def dof(self):
        return self.ndata - self.nfit


@dataclass(frozen=True)
class JointGoodnessOfFit:
    """Separate P1D, BAO, and combined goodness-of-fit summaries.

    ``datasets`` holds the chi-square and number of points of each P1D data set
    (e.g. ``DESIY1_QMLE3`` and ``Karacayli2022``), which add up to ``p1d``.
    """
    p1d: GoodnessOfFit | None
    bao: GoodnessOfFit | None
    joint: GoodnessOfFit
    datasets: dict = field(default_factory=dict)


def vega_component(model):
    """Return the optional coupled Vega likelihood, if configured."""
    from lya_interface.likelihoods.vega import VegaLikelihood
    return next((x for x in model.likelihood.values() if isinstance(x, VegaLikelihood)), None)


def vega_inputs(model, vega, point):
    """Return Vega's public input values at ``point``, with fixed ones from the model.

    Parameters
    ----------
    model : cobaya.model.Model
        Model holding the fixed parameter values (e.g. ``ap``, ``at`` at fixed cosmology).
    vega : VegaLikelihood
        Vega component of ``model``.
    point : mapping
        Sampled parameter values.

    Returns
    -------
    dict
        Every name of ``vega.public_parameters`` to its value.
    """
    values = {**model.parameterization.constant_params(), **point}
    return {name: values[name] for name in vega.public_parameters}


def _gof(chi2, ndata, nfit):
    from scipy.stats import chi2 as chi2_distribution
    dof = int(ndata) - int(nfit)
    return GoodnessOfFit(float(chi2), int(ndata), int(nfit),
                         float(chi2_distribution.sf(chi2, dof)) if dof > 0 else np.nan)


def goodness_of_fit(model, point):
    """Return raw P1D, Vega/BAO, and joint fit probabilities.

    All reported probabilities use ``N_data - N_sampled`` degrees of freedom.
    This makes the three numbers directly comparable and avoids assigning a
    shared cosmology parameter to just one data set.
    """
    components = evaluate_components(model, point)
    nfit = len(model.parameterization.sampled_params())
    p1d = None
    datasets = {}
    try:
        result, _ = evaluate_point(model, point)
        p1d = _gof(result.chi2_data, result.ndata, nfit)
        datasets = {key: _gof(result.chi2_by_dataset[key], result.ndata_by_dataset[key], 0)
                    for key in result.chi2_by_dataset}
    except (ValueError, KeyError):
        pass
    vega = vega_component(model)
    bao = None
    if vega is not None:
        name = next(name for name, item in model.likelihood.items() if item is vega)
        ndata = sum(data.data_size for data in vega.backend.data.values())
        bao = _gof(-2 * components.loglikes[name], ndata, nfit)
    joint_chi2 = sum(x.chi2 for x in (p1d, bao) if x is not None)
    joint_ndata = sum(x.ndata for x in (p1d, bao) if x is not None)
    return JointGoodnessOfFit(p1d, bao, _gof(joint_chi2, joint_ndata, nfit), datasets)


def _extra_minus2loglike(statistics, minus2loglike, tolerance=1e-6):
    """``; -2logL_total=...`` when it differs from the joint data chi-square, else ``""``.

    They differ when the likelihood includes more than the data chi-square
    (e.g. a covariance log-determinant or a likelihood without a chi-square split).
    """
    if abs(minus2loglike - statistics.joint.chi2) <= tolerance * max(1.0, abs(minus2loglike)):
        return ""
    return f"; -2logL_total={minus2loglike:.6g}"


def format_goodness_of_fit(statistics, *, multiline=False):
    """Format the chi-square of each data set and the joint chi-square and PTE.

    ``chi2_<data set>=chi2/N_data`` for every P1D data set (``chi2_P1D`` when
    the per-data-set split is unavailable) and ``chi2_BAO`` if present, then
    ``chi2_joint=chi2/dof, PTE_joint=p`` with ``dof = N_data - N_sampled``.
    """
    terms = [f"chi2_{label}={value.chi2:.6g}/{value.ndata}"
             for label, value in getattr(statistics, "datasets", {}).items()]
    if not terms and statistics.p1d is not None:
        terms.append(f"chi2_P1D={statistics.p1d.chi2:.6g}/{statistics.p1d.ndata}")
    if statistics.bao is not None:
        terms.append(f"chi2_BAO={statistics.bao.chi2:.6g}/{statistics.bao.ndata}")
    joint = statistics.joint
    joint_term = f"chi2_joint={joint.chi2:.6g}/{joint.dof}, PTE_joint={joint.probability:.4g}"
    return ", ".join(terms) + ("\n" if multiline and terms else "; " if terms else "") + joint_term


def near_prior_bounds(model, point, fraction=0.02):
    """Return the sampled parameters close to a hard bound of their prior.

    Parameters
    ----------
    model : cobaya.model.Model
        Model whose priors define the bounds: flat priors, and the truncation
        edges of truncated Gaussian priors.
    point : mapping
        Physical value of every sampled parameter.
    fraction : float, default=0.02
        Maximum distance to a bound, as a fraction of the prior range.

    Returns
    -------
    list of dict
        As :func:`cup1d.likelihood.parameter.near_prior_bounds` (the shared
        implementation), with ``prior`` the Cobaya distribution name.
    """
    from cup1d.likelihood.parameter import near_prior_bounds as native
    names = list(model.parameterization.sampled_params())
    bounds = dict(zip(names, model.prior.bounds()))
    priors = {name: pdf.dist.name for name, pdf in zip(names, model.prior.pdf)}
    return native({name: point[name] for name in names}, bounds, fraction=fraction, priors=priors)


def format_near_prior_bounds(rows, fraction=0.02):
    """Format :func:`near_prior_bounds` output, one parameter per line."""
    from cup1d.likelihood.parameter import format_near_prior_bounds as native
    return native(rows, fraction)


def minimize_point(
    model,
    initial_point,
    max_evals=10000,
    neval=1000,
    chi2_tol=0.1,
    verbose=True,
    report_every=100,
    optimization_bounds=None,
):
    """
    Run a bounded Nelder-Mead fit of the posterior in the unit cube.

    Parameters
    ----------
    model : cobaya.model.Model
        Initialized graph containing one cup1d likelihood and optional BAO.
    initial_point : dict
        Physical initial values of all sampled parameters.
    max_evals : int, default=10000
        Maximum number of objective evaluations over all restarts; reaching it
        is not a converged fit.
    neval : int, default=1000
        Maximum evaluations of each Nelder--Mead run, as native cup1d.
    chi2_tol : float, default=0.1
        Tolerance on ``-2 log posterior``, as native cup1d.
    verbose : bool, default=True
        Print initial, periodic and final diagnostics without fitted parameters.
    report_every : int, default=100
        Number of optimizer evaluations between progress reports.


    Returns
    -------
    PointFit
        Best valid evaluated point, P1D chi-squared (or ``NaN`` for Vega-only), combined objective,
        evaluation count and optimizer convergence status.

    Raises
    ------
    ValueError
        If the initial point, parameter bounds or reporting interval is invalid.

    Notes
    -----
    The unit cube balances tiny As against order-one coefficients. Bounds are
    the configured Cobaya priors unless finite ``optimization_bounds`` are
    explicitly supplied for otherwise unbounded parameters. Such bounds limit
    only the numerical search box: they never replace or truncate the Cobaya
    posterior priors. Scientific-domain exclusions remain infinite objective
    values. A failed/budget-limited run is not called a best fit.
    The objective is minus the log posterior (every configured likelihood,
    including optional BAO, plus the Cobaya priors), as cup1d's minimizer:
    with Gaussian priors the best fit is the posterior mode, the point a
    joint minimization of several likelihoods must also find. Flat priors
    only add a constant. ``chi2_data`` remains the P1D diagnostic when present and is ``NaN``
    in Vega-only mode, while ``minus2_loglike_total`` reports the combined
    objective. Nothing is written to disk. Multiple starts/convergence checks remain the
    responsibility of the analysis, as for native minimization.
    With ``verbose=True``, print initial/final diagnostics and progress every
    ``report_every`` optimizer evaluations, without displaying fitted parameters.
    """
    if (
        not isinstance(report_every, int)
        or isinstance(report_every, bool)
        or report_every < 1
    ):
        raise ValueError("report_every must be a positive integer")
    names = list(model.parameterization.sampled_params())
    if set(initial_point) != set(names):
        raise ValueError(
            "initial_point must define every sampled parameter exactly once"
        )
    bounds = np.asarray(model.prior.bounds(), dtype=float)
    if optimization_bounds is not None:
        unknown = set(optimization_bounds) - set(names)
        if unknown:
            raise ValueError(f"optimization bounds name unknown sampled parameters: {sorted(unknown)}")
        for index, name in enumerate(names):
            if name not in optimization_bounds:
                continue
            replacement = np.asarray(optimization_bounds[name], dtype=float)
            if replacement.shape != (2,) or not np.all(np.isfinite(replacement)) or replacement[0] >= replacement[1]:
                raise ValueError(f"invalid optimization bounds for {name}")
            bounds[index] = replacement
    if bounds.shape != (len(names), 2) or not np.all(np.isfinite(bounds)):
        raise ValueError(
            "minimization requires finite bounds for every sampled parameter"
        )
    lower, width = bounds[:, 0], np.diff(bounds, axis=1)[:, 0]
    if np.any(width <= 0) or not isinstance(max_evals, int) or max_evals < 1:
        raise ValueError("invalid bounds or evaluation budget")
    start = (np.array([initial_point[n] for n in names]) - lower) / width
    if np.any(start < 0) or np.any(start > 1):
        raise ValueError("initial point lies outside parameter bounds")
    try:
        cup1d_component(model)
    except (ValueError, KeyError):
        initial_result = None
    else:
        initial_result, _ = evaluate_point(model, initial_point)

    def physical(x):
        return dict(zip(names, lower + np.asarray(x) * width))

    best_x = start.copy()
    initial_posterior = model.logposterior(initial_point, return_derived=False)
    best_value = -float(initial_posterior.logpost)  # objective: -log posterior
    best_m2loglike = -2 * float(np.sum(initial_posterior.loglikes))  # reported
    evaluations = 0
    if verbose:
        initial_statistics = goodness_of_fit(model, initial_point)
        print(
            f"Nelder-Mead: {len(names)} parameters, budget={max_evals}; "
            f"initial {format_goodness_of_fit(initial_statistics)}"
            f"{_extra_minus2loglike(initial_statistics, best_m2loglike)}",
            flush=True,
        )

    last_report = time.perf_counter()
    best_report = [None, ""]  # best value and its formatted statistics

    class _Budget(Exception):
        pass

    def objective(x, count=True):
        nonlocal best_x, best_value, best_m2loglike, evaluations, last_report
        if count:
            evaluations += 1
        value, m2loglike = np.inf, np.inf
        if np.all(np.isfinite(x)) and np.all(x >= 0) and np.all(x <= 1):
            posterior = model.logposterior(physical(x), return_derived=False)
            if np.isfinite(posterior.logpost):
                value = -float(posterior.logpost)
                m2loglike = -2 * float(np.sum(posterior.loglikes))
        if np.isfinite(value) and value < best_value:
            best_x, best_value, best_m2loglike = np.array(x, copy=True), value, m2loglike
        if count and evaluations > max_evals:
            raise _Budget
        if verbose and count and evaluations % report_every == 0:
            elapsed = time.perf_counter() - last_report
            # Report the best point so far, not the current trial; recompute
            # its statistics only when it has changed since the last report.
            if best_report[0] != best_value:
                statistics = goodness_of_fit(model, physical(best_x))
                best_report[:] = [best_value, format_goodness_of_fit(statistics)
                                  + _extra_minus2loglike(statistics, best_m2loglike)]
            print(f"Evaluation {evaluations} ({elapsed:.1f} s): best {best_report[1]}", flush=True)
            last_report = time.perf_counter()
        return value

    from cup1d.inference.minimize import nelder_mead_restarts

    try:
        result = nelder_mead_restarts(objective, start, neval=neval, chi2_tol=chi2_tol,
                                      log=print if verbose else None)
        fit = SimpleNamespace(x=result.x, success=result.success, nfev=evaluations,
                              message=result.message)
    except _Budget:
        fit = SimpleNamespace(x=best_x, success=False, nfev=evaluations,
                              message=f"Stopped: evaluation budget of {max_evals} reached.")
    final_valid = np.isfinite(objective(fit.x, count=False))
    point = physical(best_x)
    result = evaluate_point(model, point)[0] if initial_result is not None else None
    statistics = goodness_of_fit(model, point)
    message = str(fit.message)
    if not final_valid:
        message += " Returned point invalid; retained best valid evaluation."
    if verbose:
        print(
            f"Finished: success={bool(fit.success) and final_valid}, evaluations={fit.nfev}; "
            f"{format_goodness_of_fit(statistics)}"
            f"{_extra_minus2loglike(statistics, best_m2loglike)}. {message}",
            flush=True,
        )
    return PointFit(
        point,
        np.nan if result is None else result.chi2_data,
        bool(fit.success) and final_valid,
        message,
        int(fit.nfev),
        -2 * float(np.sum(model.logposterior(point, return_derived=False).loglikes)),
        statistics,
    )


def _lya_cosmology(model):
    """Return the configured LyaCosmology theory component."""
    from lya_interface.theory.lya_cosmology import LyaCosmology
    return next(c for c in model.theory.values() if isinstance(c, LyaCosmology))


def star_values(model, point, unblind=False):
    """Return ``(Delta2star, nstar)`` at a sampled point, blinded by default.

    Parameters
    ----------
    model : cobaya.model.Model
        Graph containing a ``LyaCosmology`` component.
    point : mapping
        Physical sampled values.
    unblind : bool, default=False
        Omit the native DESI blinding offsets. Use only for data sets that
        have been unblinded (e.g. DESI DR1 P1D).

    Returns
    -------
    numpy.ndarray
        ``[Delta2star, nstar]`` at the component's ``(z_star, k_star_ikms)``.
        Unless ``unblind``, they include the native DESI blinding offsets,
        exactly like the exported derived parameters.
    """
    from lya_interface.cup1d_bridge import blind_star_parameters, star_parameters

    model.logposterior(point, return_derived=False)
    lya = _lya_cosmology(model)
    stars = star_parameters(lya.get_lya_cosmology(), lya.z_star, lya.k_star_ikms)
    if not unblind:
        stars = blind_star_parameters(lya._blind, stars)
    return np.array([stars["Delta2star"], stars["nstar"]])


def is_blinded(model):
    """Return whether native blinding offsets apply to the star parameters."""
    blind = _lya_cosmology(model)._blind
    return bool(blind.get("Delta2_star", 0.) or blind.get("n_star", 0.))  # offsets of the plotted pair
