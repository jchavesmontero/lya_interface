"""Local errors of a Cobaya graph: Gauss-Newton curvature, linear propagation, emcee."""

from dataclasses import dataclass
import numpy as np
from lya_interface.fit_core import cup1d_component, evaluate_point, vega_component, vega_inputs


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


def _gauss_newton_hessian(model, names, lower, width, center, step):
    """Unit-cube posterior curvature ``sum J^T C^-1 J`` plus the prior curvature.

    Every Gaussian data term contributes its own ``J^T C^-1 J``: the cup1d P1D,
    with the covariance blocks its likelihood uses (built by
    :mod:`cup1d.inference.curvature`, the same code as cup1d's own error
    estimate: the full covariance of a data set when it has one) and, if
    present, Vega (global or per-correlation inverse covariance). The terms
    are independent, so their curvatures add.

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
        Symmetric ``(N, N)`` curvature of ``-log posterior`` (without the
        flat-prior term, added by :func:`estimate_local_errors`).

    Raises
    ------
    ValueError
        If the graph has no supported data likelihood, or Vega marginalizes
        template coefficients in the fit (its model is then not a fixed vector).
    """
    from cup1d.inference.curvature import (finite_difference_jacobian, flatten_p1d,
                                           gauss_newton_curvature, p1d_blocks)
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

    blocks = p1d_blocks(backend) if backend is not None else []
    solvers = [block.solve for block in blocks]
    if vega is not None:
        vb = vega.backend
        if vb._use_global_cov:
            invcov = vb.masked_global_invcov
        else:
            from scipy.linalg import block_diag
            invcov = block_diag(*[vb.data[n].inv_masked_cov for n in vb.corr_items])
        solvers.append(lambda j: invcov @ j)

    def vectors(x):
        point = dict(zip(names, lower + width * x))
        pieces = []
        if backend is not None:
            _, powers = evaluate_point(model, point)
            pieces += flatten_p1d(powers, blocks)
        else:
            model.logposterior(point, return_derived=False)
        if vega is not None:
            values = vega.parameters_for_evaluation(vega_inputs(model, vega, point))
            cf = vega.backend.compute_model(values, run_init=False)
            pieces.append(np.concatenate([np.asarray(cf[n])[vega.backend.data[n].model_mask]
                                          for n in vega.backend.corr_items]))
        return pieces

    jacobians = finite_difference_jacobian(vectors, center, step)
    hessian = gauss_newton_curvature(jacobians, solvers)
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


def estimate_local_errors(model, point, optimization_bounds=None, step=1.e-4, method="gauss_newton",
                          flat_prior_curvature=True):
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
    step : float, default=1e-4
        Finite-difference displacement in unit-cube coordinates (as native
        cup1d's ``hessian_step``).
    method : {'gauss_newton', 'finite_difference'}, default='gauss_newton'
        ``'gauss_newton'`` builds the curvature as ``J^T C^-1 J`` from the
        derivatives of the rebinned P1D model (about ``2 N`` evaluations for
        ``N`` parameters, as in cup1d's ``gauss_newton`` method) and adds the
        curvature of the Cobaya priors. It needs a graph whose only data
        likelihoods are cup1d P1D and/or Vega (BAO); each Gaussian term adds its
        own ``J^T C^-1 J``. ``'finite_difference'`` differentiates the full
        posterior (order ``N**2`` evaluations) and works for any graph.
    flat_prior_curvature : bool, default=True
        Add every parameter's flat prior as a Gaussian of the same variance
        (:data:`cup1d.inference.curvature.FLAT_PRIOR_CURVATURE` in the unit
        cube), as cup1d's ``estimate_mle_errors`` does. A Gaussian cannot
        truncate at prior edges; without this, directions the data do not
        constrain extend beyond their priors and inflate the errors.

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
    if flat_prior_curvature:
        from cup1d.inference.curvature import FLAT_PRIOR_CURVATURE
        hessian = hessian + FLAT_PRIOR_CURVATURE * np.eye(size)
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
