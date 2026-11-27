"""Sample flat-prior parameters in the unit cube, as native cup1d does.

Cobaya's MCMC proposes in the sampled parameters' own units, and checks the
proposal covariance (``np.linalg.eigvals``) in those units. With ``As ~ 1e-9``
next to order-one coefficients, the covariance of the physical parameters is
numerically singular even when it is positive definite. Native cup1d (and
:func:`lya_interface.fit_core.minimize_point`) map every parameter to
``[0, 1]`` instead; this module does the same for Cobaya.

Every sampled parameter ``x`` with a flat prior ``[min, max]`` is replaced by
``x_cube`` with a flat prior ``[0, 1]``, and ``x = min + (max - min) * x_cube``
becomes a Cobaya input parameter computed from it and saved in the chain. The
map is linear, so the prior stays flat in every physical parameter and the
posterior of the physical parameters is unchanged.
"""
from pathlib import Path
import re

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


PRIOR_BOUNDS_FILE = "prior_bounds.yaml"
_CUBE_LAMBDA = re.compile(
    r"^lambda\s+(?P<cube>\w+)\s*:\s*(?P<low>\S+)\s*\+\s*(?P<width>\S+)\s*\*\s*(?P=cube)\s*$")


def flat_prior_bounds(params):
    """Return the flat-prior bounds of every sampled physical parameter.

    Parameters
    ----------
    params : mapping
        Physical Cobaya definitions (before :func:`unit_cube_params`).

    Returns
    -------
    dict
        ``{name: [min, max]}`` of every parameter with a flat prior.
    """
    return {name: list(_flat_bounds(name, d)) for name, d in params.items()
            if isinstance(d, dict) and isinstance(d.get("prior"), dict) and "min" in d["prior"]}


def bounds_from_unit_cube_params(params):
    """Recover the physical bounds from unit-cube definitions.

    Parameters
    ----------
    params : mapping
        Definitions after :func:`unit_cube_params`, e.g. the ``params`` of a
        Cobaya ``<prefix>.updated.yaml``: ``<name>`` has the value
        ``lambda <name>_cube: min + width * <name>_cube``.

    Returns
    -------
    dict
        ``{name: [min, max]}`` of every parameter sampled in the unit cube.
    """
    bounds = {}
    for name, definition in params.items():
        value = definition.get("value") if isinstance(definition, dict) else None
        match = _CUBE_LAMBDA.match(" ".join(value.split())) if isinstance(value, str) else None
        if match and match.group("cube") == name + CUBE_SUFFIX:
            low, width = float(match.group("low")), float(match.group("width"))
            bounds[name] = [low, low + width]
    return bounds


def write_prior_bounds(folder, bounds, note=""):
    """Write ``prior_bounds.yaml``: the bounds that map a folder's unit-cube products.

    Parameters
    ----------
    folder : path-like
        Folder of the unit-cube products (run or chain folder).
    bounds : mapping
        ``{name: [min, max]}`` physical flat-prior bounds.
    note : str, default=""
        Comment written at the top of the file.

    Returns
    -------
    pathlib.Path
    """
    import yaml

    path = Path(folder) / PRIOR_BOUNDS_FILE
    header = "# Flat-prior bounds: <name> = min + (max - min) * <name>_cube for the\n" \
             "# unit-cube products (chains, covariances) of this folder.\n"
    if note:
        header += f"# {note}\n"
    path.write_text(header + yaml.safe_dump({k: [float(v[0]), float(v[1])] for k, v in bounds.items()},
                                            sort_keys=False))
    return path


def read_prior_bounds(folder, prefix="mcmc"):
    """Read the bounds of a folder's unit-cube products from the folder itself.

    Parameters
    ----------
    folder : path-like
        Run or chain folder.
    prefix : str, default="mcmc"
        Cobaya output prefix in that folder.

    Returns
    -------
    dict
        ``{name: [min, max]}``, from ``prior_bounds.yaml`` or, for runs
        started before it was written, from the unit-cube lambdas that Cobaya
        recorded in ``<prefix>.updated.yaml``.

    Raises
    ------
    FileNotFoundError
        If the folder records neither: the current configuration is never
        used instead, since its priors may differ from those of the run.
    """
    import yaml

    folder = Path(folder)
    path = folder / PRIOR_BOUNDS_FILE
    if path.is_file():
        return {k: [float(v[0]), float(v[1])] for k, v in yaml.safe_load(path.read_text()).items()}
    updated = folder / f"{prefix}.updated.yaml"
    if updated.is_file():
        bounds = bounds_from_unit_cube_params(yaml.safe_load(updated.read_text()).get("params", {}))
        if bounds:
            return bounds
    raise FileNotFoundError(f"no {PRIOR_BOUNDS_FILE} nor unit-cube {updated.name} in {folder}: "
                            "the prior bounds of its unit-cube products are unknown")


def _widths(names, params):
    """Prior ranges of the physical parameters behind ``names`` (physical or cube).

    ``params`` holds Cobaya definitions with flat priors or ``[min, max]`` pairs
    (as returned by :func:`read_prior_bounds`).
    """
    widths = []
    for name in names:
        physical = name[:-len(CUBE_SUFFIX)] if name.endswith(CUBE_SUFFIX) else name
        entry = params[physical]
        low, high = _flat_bounds(physical, entry) if isinstance(entry, dict) else map(float, entry)
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
        :func:`unit_cube_params`), or ``{name: [min, max]}`` bounds.

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
        Bounds of the run the covariance comes from: ``{name: [min, max]}``
        from :func:`read_prior_bounds` of its folder, or physical Cobaya
        definitions with those flat priors. Do not pass the current
        configuration for a covariance of another run: its priors may differ.

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
