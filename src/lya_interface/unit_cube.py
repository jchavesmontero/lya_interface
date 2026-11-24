"""Sample flat-prior parameters in the unit cube, as native cup1d does.

Cobaya's MCMC proposes in the sampled parameters' own units, and checks the
proposal covariance (``np.linalg.eigvals``) in those units. With ``As ~ 1e-9``
next to order-one coefficients, the covariance of the physical parameters is
numerically singular even when it is positive definite. Native cup1d (and
:func:`lya_interface.diagnostics.minimize_point`) map every parameter to
``[0, 1]`` instead; this module does the same for Cobaya.

Every sampled parameter ``x`` with a flat prior ``[min, max]`` is replaced by
``x_cube`` with a flat prior ``[0, 1]``, and ``x = min + (max - min) * x_cube``
becomes a Cobaya input parameter computed from it and saved in the chain. The
map is linear, so the prior stays flat in every physical parameter and the
posterior of the physical parameters is unchanged.
"""
import numpy as np

CUBE_SUFFIX = "_cube"


def _flat_bounds(name, definition):
    """Return ``(min, max)`` of a sampled parameter's flat prior."""
    prior = definition["prior"]
    if not isinstance(prior, dict) or set(prior) - {"min", "max", "dist"} or prior.get("dist", "uniform") != "uniform":
        raise ValueError(f"unit-cube sampling requires a flat prior with min/max for {name!r}")
    low, high = float(prior["min"]), float(prior["max"])
    if not np.isfinite(low) or not np.isfinite(high) or not high > low:
        raise ValueError(f"unit-cube sampling requires finite bounds for {name!r}")
    return low, high


def _cube_ref(name, ref, low, width):
    """Translate a Cobaya ``ref`` (number, normal or uniform) to cube units."""
    if ref is None:
        return None
    if not isinstance(ref, dict):
        return (float(ref) - low) / width
    dist = ref.get("dist", "uniform" if "min" in ref else "norm")
    if dist == "norm":
        return {"dist": "norm", "loc": (float(ref["loc"]) - low) / width, "scale": float(ref["scale"]) / width}
    if dist == "uniform":
        return {"min": (float(ref["min"]) - low) / width, "max": (float(ref["max"]) - low) / width}
    raise ValueError(f"cannot translate ref of {name!r} to the unit cube")


def unit_cube_params(params):
    """Return Cobaya ``params`` with every flat-prior parameter sampled in [0, 1].

    Parameters
    ----------
    params : mapping
        Flat Cobaya parameter definitions in physical units.

    Returns
    -------
    dict
        New definitions: ``<name>_cube`` sampled in ``[0, 1]`` (``ref`` and
        ``proposal`` scaled by the prior range) and ``<name>`` a saved input
        parameter ``lambda <name>_cube: min + (max - min) * <name>_cube``.
        Fixed and derived parameters are unchanged.

    Raises
    ------
    ValueError
        If a sampled prior is not flat with finite bounds, a ``ref`` cannot be
        translated, or a ``<name>_cube`` parameter already exists.
    """
    out = {}
    for name, definition in params.items():
        if not (isinstance(definition, dict) and "prior" in definition):
            out[name] = definition
            continue
        cube = name + CUBE_SUFFIX
        if cube in params:
            raise ValueError(f"parameter {cube!r} already exists")
        low, high = _flat_bounds(name, definition)
        width = high - low
        sampled = {"prior": {"min": 0.0, "max": 1.0}}
        ref = _cube_ref(name, definition.get("ref"), low, width)
        if ref is not None:
            sampled["ref"] = ref
        if definition.get("proposal") is not None:
            sampled["proposal"] = float(definition["proposal"]) / width
        out[cube] = sampled
        physical = {"value": f"lambda {cube}: {low!r} + {width!r} * {cube}", "derived": True}
        if "latex" in definition:
            physical["latex"] = definition["latex"]
        out[name] = physical
    return out


def _widths(names, params):
    """Prior ranges of the physical parameters behind ``names`` (physical or cube)."""
    widths = []
    for name in names:
        physical = name[:-len(CUBE_SUFFIX)] if name.endswith(CUBE_SUFFIX) else name
        low, high = _flat_bounds(physical, params[physical])
        widths.append(high - low)
    return np.array(widths)


def covariance_to_unit_cube(covariance, names, params):
    """Convert a physical-parameter covariance to unit-cube names and units.

    Parameters
    ----------
    covariance : array_like, shape (n, n)
        Covariance of the physical parameters ``names``.
    names : sequence of str
        Physical parameter names.
    params : mapping
        Physical Cobaya definitions with the flat priors (before
        :func:`unit_cube_params`).

    Returns
    -------
    covariance : ndarray, shape (n, n)
        Covariance divided by the outer product of the prior ranges.
    names : list of str
        ``<name>_cube`` names.
    """
    widths = _widths(names, params)
    return np.asarray(covariance) / np.outer(widths, widths), [n + CUBE_SUFFIX for n in names]


def covariance_from_unit_cube(covariance, names, params):
    """Convert a covariance with ``<name>_cube`` names back to physical units.

    Names without the suffix are taken as physical already, so this also
    accepts covariances from runs that did not sample in the unit cube.

    Parameters
    ----------
    covariance : array_like, shape (n, n)
        Covariance of unit-cube (or physical) parameters.
    names : sequence of str
        Their names.
    params : mapping
        Physical Cobaya definitions with the flat priors of the current
        configuration (so a covariance learned with other priors is rescaled
        to physical units with that run's priors only if ``params`` holds them).

    Returns
    -------
    covariance : ndarray, shape (n, n)
        Physical covariance.
    names : list of str
        Physical names.
    """
    in_cube = np.array([n.endswith(CUBE_SUFFIX) for n in names])
    widths = np.where(in_cube, _widths(names, params), 1.0)
    physical = [n[:-len(CUBE_SUFFIX)] if cube else n for n, cube in zip(names, in_cube)]
    return np.asarray(covariance) * np.outer(widths, widths), physical
