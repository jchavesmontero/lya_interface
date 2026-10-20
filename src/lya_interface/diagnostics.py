"""Point diagnostics using the native cup1d likelihood and P1D renderers."""

from dataclasses import dataclass
from types import SimpleNamespace
import time
import numpy as np
from lya_interface.parameters import route
import lya_interface.plotting  # noqa: F401  (house matplotlib style)


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
    statistics = goodness_of_fit(model, point)
    prefix = "" if title is None else title + "\n"
    fig.suptitle(
        prefix + format_goodness_of_fit(statistics, multiline=True)
    )
    fig.supxlabel(r"$k_\parallel$ [s/km]")
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
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
    """Separate P1D, BAO, and combined goodness-of-fit summaries."""
    p1d: GoodnessOfFit | None
    bao: GoodnessOfFit | None
    joint: GoodnessOfFit


def vega_component(model):
    """Return the optional coupled Vega likelihood, if configured."""
    from lya_interface.likelihoods.vega import VegaLikelihood
    return next((x for x in model.likelihood.values() if isinstance(x, VegaLikelihood)), None)


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
    try:
        result, _ = evaluate_point(model, point)
        p1d = _gof(result.chi2_data, result.ndata, nfit)
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
    return JointGoodnessOfFit(p1d, bao, _gof(joint_chi2, joint_ndata, nfit))


def format_goodness_of_fit(statistics, *, multiline=False):
    """Format separate data-set and combined chi-square/PTE diagnostics."""
    def one(label, value):
        return "" if value is None else (
            f"chi2_{label}={value.chi2:.6g}/{value.dof}, PTE_{label}={value.probability:.4g}"
        )
    data_terms = "; ".join(filter(None, (one("P1D", statistics.p1d), one("BAO", statistics.bao))))
    joint = one("joint", statistics.joint)
    return data_terms + ("\n" if multiline and data_terms else "; ") + joint


def plot_point_with_residuals(model, point, ncols=4, title=None):
    """Plot ``k P1D / pi`` and the relative residual for every redshift.

    Parameters
    ----------
    model : cobaya.model.Model
        Initialized dependency graph with one cup1d likelihood.
    point : mapping
        Physical sampled values.
    ncols : int, default=4
        Number of redshift columns. Each redshift has a spectrum panel above a
        residual panel sharing the ``k`` axis.
    title : str, optional
        Prefix for the correlated joint chi-squared title.

    Returns
    -------
    tuple
        Matplotlib figure and the array of spectrum axes, one per redshift.

    Notes
    -----
    The model includes contaminants, resolution and rebinning onto the data
    bins. Residuals are ``data / model - 1`` with marginal (diagonal) errors
    ``sigma / model``; the per-redshift chi2 uses the same marginal blocks, so
    their sum need not equal the correlated joint chi2 in the title.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from cup1d.likelihood.external import gaussian_residual

    result, powers = evaluate_point(model, point)
    backend = cup1d_component(model).backend
    entries = []
    for key, data in backend.data.items():
        for i, z in enumerate(data.z):
            k = np.asarray(data.k_kms[i])
            pk = np.asarray(data.Pk_kms[i])
            mod = np.asarray(powers[key][i])
            err = np.sqrt(np.diag(backend.cov_Pk_kms[key][i]))
            chi2, _ = gaussian_residual(pk - mod, backend.chol_Pk_kms[key][i])
            entries.append((float(z), k, pk, mod, err, chi2))
    entries.sort(key=lambda e: e[0])
    nrows = -(-len(entries) // ncols)
    fig = plt.figure(figsize=(4.2 * ncols, 4.2 * nrows))
    outer = fig.add_gridspec(nrows, ncols, left=0.06, right=0.99, bottom=0.05,
                             top=0.88, wspace=0.28, hspace=0.3)
    spectra = []
    for n, (z, k, pk, mod, err, chi2) in enumerate(entries):
        inner = outer[n // ncols, n % ncols].subgridspec(2, 1, height_ratios=(3, 1), hspace=0)
        top = fig.add_subplot(inner[0])
        bottom = fig.add_subplot(inner[1], sharex=top)
        top.errorbar(k, k * pk / np.pi, k * err / np.pi, fmt="o", ms=3, capsize=2, color="C0", label="data")
        top.plot(k, k * mod / np.pi, color="C1", label="model")
        top.set_title(f"z={z:.1f}  $\\chi^2$={chi2:.1f}/{len(k)}", fontsize=10)
        top.tick_params(labelbottom=False)
        bottom.errorbar(k, pk / mod - 1, err / mod, fmt="o", ms=3, capsize=2, color="C0")
        bottom.axhline(0, color="C1", lw=1)
        bottom.axhspan(-0.01, 0.01, color="gray", alpha=0.2, lw=0)
        bottom.set_ylim(-0.1, 0.1)
        bottom.set_yticks([-0.05, 0, 0.05])
        top.yaxis.set_major_locator(MaxNLocator(prune="lower"))
        if n % ncols == 0:
            top.set_ylabel(r"$k P_{1D}/\pi$")
            bottom.set_ylabel("data/model - 1")
        if n >= len(entries) - ncols:
            bottom.set_xlabel(r"$k_\parallel$ [s/km]")
        spectra.append(top)
    spectra[0].legend(fontsize=8)
    prefix = "" if title is None else title + "\n"
    fig.suptitle(prefix + format_goodness_of_fit(goodness_of_fit(model, point), multiline=True))
    return fig, np.array(spectra)


def minimize_point(
    model,
    initial_point,
    max_evals=10000,
    xtol=1e-4,
    ftol=1e-6,
    verbose=True,
    report_every=100,
    optimization_bounds=None,
    stall_evals=200,
    stall_chi2=0.1,
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

    stall_evals : int or None, default=200
        Stop when the best ``-2 log L_total`` has not improved by more than
        ``stall_chi2`` during this many consecutive evaluations. ``None``
        disables the criterion. A stall stop counts as success, like cup1d's
        no-further-improvement stop.
    stall_chi2 : float, default=0.1
        Improvement in ``-2 log L_total`` that resets the stall counter.

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
    except (ValueError, KeyError):
        initial_result = None
    else:
        initial_result, _ = evaluate_point(model, initial_point)

    def physical(x):
        return dict(zip(names, lower + np.asarray(x) * width))

    best_x = start.copy()
    best_value = -float(np.sum(model.logposterior(initial_point, return_derived=False).loglikes))
    evaluations = 0
    if verbose:
        initial_statistics = goodness_of_fit(model, initial_point)
        print(
            f"Nelder-Mead: {len(names)} parameters, budget={max_evals}; "
            f"initial {format_goodness_of_fit(initial_statistics)}; "
            f"-2logL_total={2 * best_value:.6g}",
            flush=True,
        )

    last_report = time.perf_counter()
    stall_reference = [2 * best_value, 0]  # best -2logL at last reset, evaluation count

    class _Stalled(Exception):
        pass

    def objective(x, count=True):
        nonlocal best_x, best_value, evaluations, last_report
        if count:
            evaluations += 1
        value = np.inf
        if np.all(np.isfinite(x)) and np.all(x >= 0) and np.all(x <= 1):
            posterior = model.logposterior(physical(x), return_derived=False)
            if np.isfinite(posterior.logpost):
                value = -float(np.sum(posterior.loglikes))
        if np.isfinite(value) and value < best_value:
            best_x, best_value = np.array(x, copy=True), value
        if count and stall_evals is not None:
            if stall_reference[0] - 2 * best_value > stall_chi2:
                stall_reference[:] = [2 * best_value, evaluations]
            elif evaluations - stall_reference[1] >= stall_evals:
                raise _Stalled
        if verbose and count and evaluations % report_every == 0:
            elapsed = time.perf_counter() - last_report
            if np.isfinite(value):
                detail = format_goodness_of_fit(goodness_of_fit(model, physical(x)))
            else:
                detail = "invalid trial"
            print(
                f"Evaluation {evaluations} ({elapsed:.1f} s): {detail}; "
                f"best -2logL_total={2 * best_value:.6g}",
                flush=True,
            )
            last_report = time.perf_counter()
        return value

    simplex = np.tile(start, (len(names) + 1, 1))
    for i in range(len(names)):
        simplex[i + 1, i] += 0.02 if start[i] <= 0.98 else -0.02
    try:
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
    except _Stalled:
        fit = SimpleNamespace(
            x=best_x, success=True, nfev=evaluations,
            message=f"Stopped: -2logL_total improved by less than {stall_chi2} "
                    f"in {stall_evals} evaluations.",
        )
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
            f"{format_goodness_of_fit(statistics)}; -2logL_total={2 * best_value:.6g}. {message}",
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


@dataclass(frozen=True)
class LocalErrors:
    """Finite-difference Newton covariance around a fitted physical point."""
    errors: dict
    covariance: np.ndarray
    hessian: np.ndarray
    names: tuple


@dataclass(frozen=True)
class PropagatedErrors:
    """Observable covariance obtained by linear propagation of LocalErrors."""
    values: np.ndarray
    covariance: np.ndarray
    errors: np.ndarray


def propagate_local_errors(point, local_errors, evaluate, step_fraction=.25):
    """Propagate a full local parameter covariance through an observable.

    ``evaluate`` receives a physical parameter dictionary and must return a
    one-dimensional numeric observable vector in a stable order. Central
    finite differences use one quarter of each marginalized 1-sigma error;
    the full covariance, including correlations, is then propagated as
    ``J C J^T``.
    """
    if not 0 < step_fraction <= 1:
        raise ValueError("step_fraction must lie in (0, 1]")
    names = local_errors.names
    values = np.asarray(evaluate(point), dtype=float)
    jacobian = np.empty((values.size, len(names)))
    for index, name in enumerate(names):
        sigma = local_errors.errors[name]
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"cannot propagate unconstrained parameter {name}")
        displacement = step_fraction * sigma
        plus, minus = dict(point), dict(point)
        plus[name] += displacement
        minus[name] -= displacement
        jacobian[:, index] = (np.asarray(evaluate(plus)) - np.asarray(evaluate(minus))) / (2 * displacement)
    covariance = jacobian @ local_errors.covariance @ jacobian.T
    return PropagatedErrors(values, covariance, np.sqrt(np.maximum(np.diag(covariance), 0)))


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
    from lya_interface.derived import star_parameters

    model.logposterior(point, return_derived=False)
    lya = _lya_cosmology(model)
    stars = star_parameters(lya.get_lya_cosmology(), lya.z_star, lya.k_star_ikms)
    if unblind:
        return np.array([stars["Delta2star"], stars["nstar"]])
    return np.array([stars["Delta2star"] + lya._blind.get("Delta2_star", 0.),
                     stars["nstar"] + lya._blind.get("n_star", 0.)])


def is_blinded(model):
    """Return whether native blinding offsets apply to the star parameters."""
    blind = _lya_cosmology(model)._blind
    return bool(blind.get("Delta2_star", 0.) or blind.get("n_star", 0.))


def plot_star_ellipse(model, point, local_errors, ax=None, label=None, color=None,
                      levels=(0.6827, 0.9545), unblind=False):
    """Plot the local-error ellipses of ``(n_star, Delta2_star)`` and the minimum.

    Parameters
    ----------
    model : cobaya.model.Model
        Graph containing a ``LyaCosmology`` component.
    point : mapping
        Physical sampled values at the minimum.
    local_errors : LocalErrors
        Local parameter covariance at ``point``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on, e.g. to overlay several fits. A new figure otherwise.
    label : str, optional
        Prefix of the legend entries, useful when overlaying fits.
    color : str, optional
        Colour of this fit; levels are distinguished by line style.
    levels : sequence of float, default=(0.6827, 0.9545)
        Joint two-parameter probability contents of the ellipses.
    unblind : bool, default=False
        Plot values without the native DESI blinding offsets, for unblinded
        data sets such as DESI DR1 P1D. The "(blinded)" labels are then omitted.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    summary : dict
        Blinded values, marginal errors, correlation and the propagated
        :class:`PropagatedErrors`.

    Notes
    -----
    The full parameter covariance is propagated through finite-difference
    derivatives of :func:`star_values`, keeping all correlations. Unless
    ``unblind``, values carry the native DESI blinding offsets, and every legend
    entry and axis label says "(blinded)" whenever an offset applies.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import chi2 as chi2_distribution

    stars = propagate_local_errors(point, local_errors, lambda p: star_values(model, p, unblind))
    delta2, nstar = stars.values
    eigenvalues, eigenvectors = np.linalg.eigh(stars.covariance[::-1, ::-1])  # (n_star, Delta2_star)
    angle = np.linspace(0, 2 * np.pi, 200)
    unit_circle = np.vstack([np.cos(angle), np.sin(angle)])
    suffix = " (blinded)" if is_blinded(model) and not unblind else ""
    prefix = "" if label is None else label + ", "
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    else:
        fig = ax.figure
    if color is None:
        color = ax._get_lines.get_next_color()
    for level, style in zip(levels, ("-", "--", ":", "-.")):
        delta_chi2 = chi2_distribution.ppf(level, 2)
        ellipse = eigenvectors @ (np.sqrt(delta_chi2 * np.maximum(eigenvalues, 0))[:, None] * unit_circle)
        ax.plot(nstar + ellipse[0], delta2 + ellipse[1], style, color=color,
                label=f"{prefix}{100 * level:.0f}%{suffix}")
    ax.plot(nstar, delta2, "x", color=color, ms=10, mew=2, label=f"{prefix}MLE{suffix}")
    ax.set_xlabel(r"$n_\star$" + suffix)
    ax.set_ylabel(r"$\Delta^2_\star$" + suffix)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    summary = {
        "Delta2_star": float(delta2), "n_star": float(nstar),
        "sigma_Delta2_star": float(stars.errors[0]), "sigma_n_star": float(stars.errors[1]),
        "correlation": float(stars.covariance[0, 1] / np.prod(stars.errors)),
        "blinded": bool(suffix), "propagated": stars,
    }
    return fig, ax, summary


def _gauss_newton_hessian(model, names, lower, width, center, step):
    """Unit-cube posterior curvature ``sum J^T C^-1 J`` plus the prior curvature.

    Every Gaussian data term contributes its own ``J^T C^-1 J``: the cup1d P1D
    (per-redshift covariance blocks) and, if present, Vega (global or
    per-correlation inverse covariance). The terms are independent, so their
    curvatures add.

    Parameters
    ----------
    model : cobaya.model.Model
        Graph with a cup1d P1D and/or a Vega likelihood.
    names, lower, width, center
        Sampled names, unit-cube origin and widths, and the expansion point.
    step : float
        Unit-cube derivative step; central unless a bound is within ``step``.

    Returns
    -------
    numpy.ndarray
        Symmetric ``(N, N)`` curvature of ``-log posterior``.

    Raises
    ------
    ValueError
        If the graph has no supported data likelihood, or Vega marginalizes
        template coefficients in the fit (its model is then not a fixed vector).
    """
    from scipy.linalg import cho_solve

    try:
        backend = cup1d_component(model).backend
    except ValueError:
        backend = None
    vega = vega_component(model)
    if backend is None and vega is None:
        raise ValueError("gauss_newton needs a cup1d P1D and/or Vega likelihood")
    if len(model.likelihood) != (backend is not None) + (vega is not None):
        raise ValueError("gauss_newton supports only cup1d P1D and Vega data likelihoods")
    if vega is not None and vega.backend.marginalize_in_fit:
        raise ValueError("gauss_newton does not support Vega marginalize_in_fit")

    def vectors(x):
        point = dict(zip(names, lower + width * x))
        pieces = []
        if backend is not None:
            _, powers = evaluate_point(model, point)
            pieces += [np.asarray(powers[key][i]) for key, data in backend.data.items()
                       for i in range(len(data.z))]
        else:
            model.logposterior(point, return_derived=False)
        if vega is not None:
            values = vega.parameters_for_evaluation({n: point[n] for n in vega.mapping})
            cf = vega.backend.compute_model(values, run_init=False)
            pieces.append(np.concatenate([np.asarray(cf[n])[vega.backend.data[n].model_mask]
                                          for n in vega.backend.corr_items]))
        return pieces

    solvers = []
    if backend is not None:
        solvers += [(lambda j, f=backend.chol_Pk_kms[key][i]: cho_solve(f, j))
                    for key, data in backend.data.items() for i in range(len(data.z))]
    if vega is not None:
        vb = vega.backend
        if vb._use_global_cov:
            invcov = vb.masked_global_invcov
        else:
            from scipy.linalg import block_diag
            invcov = block_diag(*[vb.data[n].inv_masked_cov for n in vb.corr_items])
        solvers.append(lambda j: invcov @ j)

    base = vectors(center)
    jacobian = [np.empty((len(b), len(names))) for b in base]
    for index in range(len(names)):
        h = min(step, center[index], 1. - center[index])
        direction = np.zeros(len(names))
        if h > 0:
            direction[index] = h
            plus, minus = vectors(center + direction), vectors(center - direction)
            derivative = [(p - m) / (2 * h) for p, m in zip(plus, minus)]
        else:
            direction[index] = step if center[index] < .5 else -step
            other = vectors(center + direction)
            derivative = [(o - b) / direction[index] for o, b in zip(other, base)]
        for block, column in zip(jacobian, derivative):
            block[:, index] = column
    hessian = sum(j.T @ solve(j) for j, solve in zip(jacobian, solvers))
    return hessian + _prior_curvature(model, names, lower, width, center, step)


def _prior_curvature(model, names, lower, width, center, step):
    """Diagonal curvature of ``-log prior`` in unit-cube coordinates.

    Cobaya priors are separable, so second differences of ``logprior`` along
    each axis are cheap (no likelihood evaluation). Flat priors give zero.
    """
    def logprior(x):
        return float(model.logprior(dict(zip(names, lower + width * x))))

    curvature = np.zeros(len(names))
    f0 = logprior(center)
    for index in range(len(names)):
        h = min(step, center[index], 1. - center[index])
        if h <= 0:
            continue
        direction = np.zeros(len(names)); direction[index] = h
        curvature[index] = -(logprior(center + direction) - 2 * f0 + logprior(center - direction)) / h**2
    return np.diag(curvature)


def estimate_local_errors(model, point, optimization_bounds=None, step=1.e-3, method="gauss_newton"):
    """Estimate marginalized errors from the local posterior Hessian.

    This is the joint analogue of cup1d's Newton curvature estimate. It uses
    central differences where possible and inward one-sided differences at a
    parameter bound, all in the unit-cube coordinates of
    :func:`minimize_point`. It includes both data terms and Cobaya priors; it
    is a local Gaussian approximation, not a chain.

    Parameters
    ----------
    model : cobaya.model.Model
        Initialized dependency graph.
    point : mapping
        Physical sampled values at the minimum.
    optimization_bounds : mapping, optional
        Replacement ``(min, max)`` bounds defining the unit cube.
    step : float, default=1e-3
        Finite-difference displacement in unit-cube coordinates.
    method : {'gauss_newton', 'finite_difference'}, default='gauss_newton'
        ``'gauss_newton'`` builds the curvature as ``J^T C^-1 J`` from the
        derivatives of the rebinned P1D model (about ``2 N`` evaluations for
        ``N`` parameters, as in cup1d's ``gauss_newton`` method) and adds the
        curvature of the Cobaya priors. It needs a graph whose only data
        likelihoods are cup1d P1D and/or Vega (BAO); each Gaussian term adds its
        own ``J^T C^-1 J``. ``'finite_difference'`` differentiates the full
        posterior (order ``N**2`` evaluations) and works for any graph.

    Raises
    ------
    ValueError
        If the method is unknown, the point is invalid, the curvature is not
        positive semidefinite, or ``'gauss_newton'`` is requested for a graph
        with an unsupported data likelihood.
    """
    if method not in ("gauss_newton", "finite_difference"):
        raise ValueError("method must be 'gauss_newton' or 'finite_difference'")
    names = tuple(model.parameterization.sampled_params())
    bounds = np.asarray(model.prior.bounds(), dtype=float)
    for index, name in enumerate(names):
        if optimization_bounds and name in optimization_bounds:
            bounds[index] = optimization_bounds[name]
    if not np.all(np.isfinite(bounds)) or step <= 0:
        raise ValueError("finite optimization bounds and positive step are required")
    lower, width = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    center = (np.array([point[name] for name in names]) - lower) / width
    def objective(x):
        post = model.logposterior(dict(zip(names, lower + width * x)), return_derived=False)
        return -float(post.logpost) if np.isfinite(post.logpost) else np.inf
    f0 = objective(center)
    if not np.isfinite(f0):
        raise ValueError("cannot estimate local errors at an invalid fitted point")
    size = len(names)
    if method == "gauss_newton":
        hessian = _gauss_newton_hessian(model, names, lower, width, center, step)
    else:
        hessian = np.empty((size, size))
        directions = np.where(center < step, 1., np.where(center > 1 - step, -1., 0.))

        def checked(x):
            value = objective(x)
            if not np.isfinite(value):
                raise ValueError("local curvature stepped outside valid posterior support")
            return value

        for i in range(size):
            ei = np.zeros(size); ei[i] = step
            if directions[i] == 0:
                hessian[i, i] = (checked(center + ei) - 2*f0 + checked(center - ei)) / step**2
            else:
                direction = directions[i] * ei
                hessian[i, i] = (f0 - 2*checked(center + direction) + checked(center + 2*direction)) / step**2
            for j in range(i):
                ej = np.zeros(size); ej[j] = step
                if directions[i] == 0 and directions[j] == 0:
                    value = (checked(center + ei + ej) - checked(center + ei - ej)
                             - checked(center - ei + ej) + checked(center - ei - ej)) / (4*step**2)
                else:
                    di = ei if directions[i] == 0 else directions[i] * ei
                    dj = ej if directions[j] == 0 else directions[j] * ej
                    # Forward/inward mixed derivative. For an interior direction
                    # this remains a first-order local derivative, avoiding an
                    # invalid point across a hard posterior boundary.
                    value = (checked(center + di + dj) - checked(center + di)
                             - checked(center + dj) + f0) / step**2
                hessian[i, j] = hessian[j, i] = value
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues = np.linalg.eigvalsh(hessian)
    tolerance = np.finfo(float).eps * size * max(1., np.max(np.abs(eigenvalues)))
    if np.min(eigenvalues) < -tolerance:
        raise ValueError("local curvature is not positive semidefinite; increase step or inspect the fit")
    covariance_unit = np.linalg.pinv(hessian, hermitian=True, rtol=1.e-10)
    covariance = covariance_unit * np.outer(width, width)
    diagonal = np.diag(covariance)
    if np.any(diagonal < -tolerance):
        raise ValueError("local covariance has negative variances")
    errors = dict(zip(names, np.sqrt(np.maximum(diagonal, 0))))
    return LocalErrors(errors, covariance, hessian, names)


@dataclass(frozen=True)
class EmceeRun:
    """In-memory joint emcee chain; no files are written by this helper."""
    names: tuple
    chain: np.ndarray
    log_probability: np.ndarray
    acceptance_fraction: np.ndarray


def sample_emcee(model, initial_point, *, nwalkers=None, burnin=100, steps=500,
                 optimization_bounds=None, seed=0):
    """Sample the full Cobaya posterior with a serial emcee ensemble.

    ``optimization_bounds`` seed walkers safely; they do not clip the sampled
    posterior. Use an MPI-aware production workflow for long scientific runs.
    """
    import emcee
    names = tuple(model.parameterization.sampled_params())
    bounds = np.asarray(model.prior.bounds(), dtype=float)
    for i, name in enumerate(names):
        if optimization_bounds and name in optimization_bounds:
            bounds[i] = optimization_bounds[name]
    if not np.all(np.isfinite(bounds)):
        raise ValueError("emcee initialization requires finite optimization bounds")
    ndim = len(names); nwalkers = max(2 * ndim + 2, nwalkers or 0)
    lower, width = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    centre = (np.array([initial_point[name] for name in names]) - lower) / width
    rng = np.random.default_rng(seed)
    walkers = np.clip(centre + rng.normal(scale=1.e-3, size=(nwalkers, ndim)), 1.e-8, 1-1.e-8)
    def log_probability(unit):
        point = dict(zip(names, lower + width * unit))
        post = model.logposterior(point, return_derived=False)
        return float(post.logpost) if np.isfinite(post.logpost) else -np.inf
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    state = sampler.run_mcmc(walkers, burnin, progress=True)
    sampler.reset(); sampler.run_mcmc(state, steps, progress=True)
    return EmceeRun(names, sampler.get_chain(), sampler.get_log_prob(), sampler.acceptance_fraction)
