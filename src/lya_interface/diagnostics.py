"""Point diagnostics using the native cup1d likelihood and P1D renderers."""

from dataclasses import dataclass
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
    result = likelihood.backend.evaluate_from_p1d(
        prediction["powers"], prediction["context"], nuisance
    )
    powers = likelihood.backend.apply_observation_model(
        prediction["powers"], prediction["context"], nuisance
    )
    return result, powers


def plot_point(model, point, *, residuals=False, title=None):
    """Plot every selected redshift with cup1d's maintained panel renderers.

    Parameters
    ----------
    model : cobaya.model.Model
        Initialized dependency graph with one cup1d likelihood.
    point : mapping
        Physical sampled values.
    residuals : bool, default=False
        Plot residual panels rather than power spectra.
    title : str, optional
        Prefix for the correlated joint chi-squared title.

    Returns
    -------
    tuple
        Matplotlib figure and axes.

    Error bars and per-bin chi2 are marginal block diagnostics; their sum need
    not equal the correlated joint chi2 displayed in the figure title.
    Neither fitted cosmology nor unblinded derived parameters are displayed.
    """
    from cup1d.likelihood.external import gaussian_residual
    from cup1d.postprocessing.p1d import P1DBin, plot_p1d_spectra, plot_p1d_residuals

    result, powers = evaluate_point(model, point)
    backend = cup1d_component(model).backend
    bins = []
    for key, data in backend.data.items():
        for i, z in enumerate(data.z):
            chi2, _ = gaussian_residual(
                np.asarray(data.Pk_kms[i]) - powers[key][i], backend.chol_Pk_kms[key][i]
            )
            bins.append(
                P1DBin(
                    key,
                    i,
                    float(z),
                    np.asarray(data.k_kms[i]),
                    np.asarray(data.Pk_kms[i]),
                    np.asarray(powers[key][i]),
                    np.sqrt(np.diag(backend.cov_Pk_kms[key][i])),
                    chi2,
                    len(data.k_kms[i]),
                    "C0",
                    0.0,
                )
            )
    renderer = plot_p1d_residuals if residuals else plot_p1d_spectra
    fig, axes = renderer(bins, panels=True, fontsize=14, print_chi2=False)
    prefix = "" if title is None else title + " — "
    fig.suptitle(
        prefix + rf"P1D joint $\chi^2={result.chi2_data:.2f}$, $N={result.ndata}$"
    )
    fig.supxlabel(r"$k_\parallel$ [s/km]")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    return fig, axes


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


def minimize_point(
    model,
    initial_point,
    max_evals=10000,
    xtol=1e-4,
    ftol=1e-6,
    verbose=True,
    report_every=100,
    optimization_bounds=None,
):
    """
    Run a bounded, total-likelihood Nelder-Mead fit in the unit cube.

    Parameters
    ----------
    model : cobaya.model.Model
        Initialized graph containing one cup1d likelihood and optional BAO.
    initial_point : dict
        Physical initial values of all sampled parameters.
    max_evals : int, default=10000
        Maximum number of optimizer objective evaluations.
    xtol : float, default=1e-4
        Simplex convergence tolerance in unit-cube coordinates.
    ftol : float, default=1e-6
        Absolute convergence tolerance on minus the total data log-likelihood.
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
    All configured likelihood terms enter the objective (including optional
    BAO); ``chi2_data`` remains the P1D diagnostic when present and is ``NaN``
    in Vega-only mode, while ``minus2_loglike_total`` reports the combined
    objective. Nothing is written to disk. Multiple starts/convergence checks remain the
    responsibility of the analysis, as for native minimization.
    With ``verbose=True``, print initial/final diagnostics and progress every
    ``report_every`` optimizer evaluations, without displaying fitted parameters.
    """
    from scipy.optimize import minimize

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
    except ValueError:
        initial_result = None
    else:
        initial_result, _ = evaluate_point(model, initial_point)

    def physical(x):
        return dict(zip(names, lower + np.asarray(x) * width))

    best_x = start.copy()
    best_value = -float(np.sum(model.logposterior(initial_point, return_derived=False).loglikes))
    evaluations = 0
    if verbose:
        initial_label = (f"{initial_result.chi2_data:.6g}"
                         if initial_result is not None else "n/a")
        print(
            f"Nelder-Mead: {len(names)} parameters, budget={max_evals}; "
            f"initial chi2_P1D={initial_label}, -2logL_total={2 * best_value:.6g}",
            flush=True,
        )

    def objective(x, count=True):
        nonlocal best_x, best_value, evaluations
        if count:
            evaluations += 1
        value = np.inf
        if np.all(np.isfinite(x)) and np.all(x >= 0) and np.all(x <= 1):
            posterior = model.logposterior(physical(x), return_derived=False)
            if np.isfinite(posterior.logpost):
                value = -float(np.sum(posterior.loglikes))
        if np.isfinite(value) and value < best_value:
            best_x, best_value = np.array(x, copy=True), value
        if verbose and count and evaluations % report_every == 0:
            if np.isfinite(value):
                if initial_result is None:
                    detail = f"chi2_P1D=n/a, -2logL_total={2 * value:.6g}"
                else:
                    current, _ = evaluate_point(model, physical(x))
                    detail = f"chi2_P1D={current.chi2_data:.6g}, -2logL_total={2 * value:.6g}"
            else:
                detail = "invalid trial (-2logL_total=inf)"
            print(
                f"Evaluation {evaluations}: {detail}; best -2logL_total={2 * best_value:.6g}",
                flush=True,
            )
        return value

    simplex = np.tile(start, (len(names) + 1, 1))
    for i in range(len(names)):
        simplex[i + 1, i] += 0.02 if start[i] <= 0.98 else -0.02
    fit = minimize(
        objective,
        start,
        method="Nelder-Mead",
        bounds=[(0.0, 1.0)] * len(names),
        options=dict(
            maxfev=max_evals,
            xatol=xtol,
            fatol=ftol,
            adaptive=True,
            initial_simplex=simplex,
        ),
    )
    final_valid = np.isfinite(objective(fit.x, count=False))
    point = physical(best_x)
    result = evaluate_point(model, point)[0] if initial_result is not None else None
    message = str(fit.message)
    if not final_valid:
        message += " Returned point invalid; retained best valid evaluation."
    if verbose:
        final_label = f"{result.chi2_data:.6g}" if result is not None else "n/a"
        print(
            f"Finished: success={bool(fit.success) and final_valid}, evaluations={fit.nfev}; "
            f"chi2_P1D={final_label}, -2logL_total={2 * best_value:.6g}. {message}",
            flush=True,
        )
    return PointFit(
        point,
        np.nan if result is None else result.chi2_data,
        bool(fit.success) and final_valid,
        message,
        int(fit.nfev),
        -2 * float(np.sum(model.logposterior(point, return_derived=False).loglikes)),
    )
