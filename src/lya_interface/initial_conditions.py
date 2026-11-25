"""Starting points from cup1d global initial-condition (IC) files."""
from pathlib import Path
import warnings
import numpy as np
from cup1d.utils.ic_files import get_ic_path
from cup1d.utils.otypes import convert_otype, history_otype, interp_ic_history, legacy_otype
from cup1d.utils.utils import split_string
from lya_interface.adapters.cup1d_backend import native_configuration
from lya_interface.names import native_name


def _native_args(info):
    """Return the cup1d arguments of the configuration's ForestFlow theory."""
    forest = next(v for k, v in info["theory"].items() if k.endswith("ForestFlowTheory"))
    return native_configuration(forest["native_config"])[0]


def _any_native_args(info):
    """Return the cup1d arguments of the first component with a ``native_config``, else None."""
    for section in ("theory", "likelihood"):
        for options in info.get(section, {}).values():
            if isinstance(options, dict) and options.get("native_config"):
                return native_configuration(options["native_config"])[0]
    return None


def public_otypes(info, names):
    """Return the otype (log ``exp`` or linear ``const``) of cup1d-owned parameters.

    Parameters
    ----------
    info : dict
        Resolved interface configuration.
    names : iterable of str
        Public parameter names.

    Returns
    -------
    dict
        Public name to the otype of its native history; parameters not owned by
        cup1d (cosmology, Vega nuisances) are omitted, as are all parameters
        when no component has a ``native_config``.
    """
    args = _any_native_args(info)
    if args is None:
        return {}
    out = {}
    for public in names:
        native = native_name(public)
        if native is None:
            continue
        otype = history_otype(args, split_string(native)[0])
        if otype is not None:
            out[public] = otype
    return out


def _prior_min(info, public):
    """Return the lower prior bound of a sampled parameter, or None."""
    definition = info.get("params", {}).get(public)
    if isinstance(definition, dict) and isinstance(definition.get("prior"), dict):
        return definition["prior"].get("min")
    return None


def convert_point_otypes(info, point, saved_otypes):
    """Convert a saved point to the otypes of the configuration.

    Parameters
    ----------
    info : dict
        Resolved interface configuration.
    point : mapping
        Public name to value, in the otypes the point was saved with.
    saved_otypes : mapping or None
        Public name to saved otype. ``None`` (files written before otypes were
        stored) means the CM2026 baseline otypes.

    Returns
    -------
    point : dict
        Copy of ``point`` with converted values. Non-positive linear values
        converted to log take the lower prior bound.
    converted : dict
        Public name to ``(saved_otype, current_otype)`` of converted entries.
    """
    point = dict(point)
    converted = {}
    for public, otype in public_otypes(info, point).items():
        if saved_otypes is not None:
            old = saved_otypes.get(public)
        else:
            old = legacy_otype(split_string(native_name(public))[0])
        if old is None or old == otype:
            continue
        point[public] = convert_otype(point[public], old, otype, floor=_prior_min(info, public))
        converted[public] = (old, otype)
    return point, converted


def _node(args, family, index, public):
    """Return the redshift node of coefficient ``index`` of ``family``."""
    for section in (args.fid_igm, args.fid_cont, args.fid_syst):
        if family + "_znodes" in section:
            return section[family + "_znodes"][int(index)]
    raise ValueError(f"{family}_znodes not found for {public}")


def default_ic_path(info, variation=None):
    """Return the standard cup1d global IC path for a configuration.

    Parameters
    ----------
    info : dict
        Resolved interface configuration.
    variation : str, optional
        Variation name, e.g. ``"kaiser"`` for the Kaiser metal model.

    Returns
    -------
    pathlib.Path
        ``cup1d/data/ics/{data}_{emulator}[_{variation}]_ic_global.npy``.
    """
    args = _native_args(info)
    return Path(get_ic_path(args.data_label, args.emulator_label, "global", variation))


def save_global_ic(info, point, filename=None, variation=None, overwrite=False):
    """Write a best-fit point as a cup1d global initial-condition file.

    Parameters
    ----------
    info : dict
        Resolved interface configuration (its native config gives the nodes).
    point : mapping
        Physical sampled values, e.g. ``best_fit.point``.
    filename : path-like, optional
        Output file; by default :func:`default_ic_path` with ``variation``.
    variation : str, optional
        Variation name used for the default file name.
    overwrite : bool, default=False
        Replace an existing file.

    Returns
    -------
    pathlib.Path
        The written file.

    Raises
    ------
    FileExistsError
        If the file exists and ``overwrite`` is false.

    Notes
    -----
    The format is that of ``Analysis.save_global_ic``/``Likelihood.set_ic_global``:
    one ``{"z": nodes, "val": coefficients, "otype": otype}`` entry per IGM and
    contaminant family, plus ``{"val": value}`` for ``As`` and ``ns``. It can be read back
    with :func:`global_ic_point` and by native cup1d.
    """
    args = _native_args(info)
    path = Path(filename).expanduser() if filename is not None else default_ic_path(info, variation)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    families = {}
    out = {}
    for public, value in point.items():
        if public in ("As", "ns"):
            out[public] = {"val": float(value)}
            continue
        native = native_name(public)
        if native is None:
            continue
        family, index = split_string(native)
        families.setdefault(family, []).append((_node(args, family, index, public), float(value)))
    for family, rows in families.items():
        rows.sort()
        out[family] = {"z": np.array([z for z, _ in rows]), "val": np.array([v for _, v in rows]),
                       "otype": history_otype(args, family)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out, allow_pickle=True)
    return path


def is_global_ic(content):
    """Return whether a loaded ``.npy`` dictionary is a cup1d global IC file."""
    return (isinstance(content, dict) and bool(content)
            and all(isinstance(entry, dict) and "val" in entry for entry in content.values()))


def ic_native_values(args, names, ic):
    """Interpolate a cup1d global IC onto native coefficients.

    Parameters
    ----------
    args : cup1d.configuration.args.Args
        Native arguments that fix the redshift node of every coefficient.
    names : iterable of str
        Native coefficient names (``b_SiIII_0``, ...).
    ic : dict
        Loaded global IC (``{family: {"z": nodes, "val": values}}``).

    Returns
    -------
    values : dict
        Native name to the IC history at its node, for the families in ``ic``,
        converted to the otype configured in ``args``.
    missing : list of str
        Names whose family the IC does not store.
    """
    values, missing = {}, []
    for native in names:
        family, index = split_string(native)
        if family not in ic:
            missing.append(native)
            continue
        znode = _node(args, family, index, native)
        values[native] = interp_ic_history(ic[family], family, znode, history_otype(args, family))
    return values, missing


def global_ic_point(info, reference, ic_file=None):
    """Interpolate a cup1d global IC file onto the sampled parameters.

    Parameters
    ----------
    info : dict
        Resolved interface configuration (``load_configuration``). Its
        ``ForestFlowTheory`` ``native_config`` fixes the data labels, emulator
        and redshift nodes of every coefficient.
    reference : dict
        Sampled parameter values used for entries the IC file does not store
        (``As``, ``ns``, and anything not fitted). Not modified.
    ic_file : path-like, optional
        IC file. By default ``cup1d/data/ics/{data}_{emulator}_ic_global.npy``.

    Returns
    -------
    dict
        Copy of ``reference`` with the IGM and nuisance coefficients replaced by
        the IC histories interpolated at their redshift nodes. ``As``/``ns`` are
        taken from the file only when it stores them.

    Raises
    ------
    FileNotFoundError
        If the IC file does not exist.
    ValueError
        If a coefficient has no node configuration.

    Warns
    -----
    UserWarning
        For coefficients absent from the IC file (e.g. a metal model newer
        than the file); they keep their reference values.

    Notes
    -----
    Mirrors ``Likelihood.set_ic_global``. ``HCD_const`` defaults to zero when
    absent. Histories are converted to the configured otype (log or linear)
    from the one the file records (files without it used the baseline otypes). Values outside the YAML prior bounds are reported by the caller.
    """
    args = _native_args(info)
    if ic_file is None:
        ic_file = get_ic_path(args.data_label, args.emulator_label, "global")
    ic_file = Path(ic_file).expanduser()
    if not ic_file.is_file():
        raise FileNotFoundError(f"Global IC file not found: {ic_file}")
    ic = np.load(ic_file, allow_pickle=True).item()
    point = dict(reference)
    missing = []
    for public in reference:
        if public in ("As", "ns"):
            if public in ic:
                point[public] = float(ic[public]["val"])
            continue
        native = native_name(public)
        if native is None:
            continue
        family, index = split_string(native)
        znode = _node(args, family, index, public)
        if family not in ic:
            if family == "HCD_const":
                point[public] = 0.0
            else:
                missing.append(public)
            continue
        point[public] = interp_ic_history(ic[family], family, znode, history_otype(args, family),
                                          floor=_prior_min(info, public))
    if missing:
        warnings.warn(f"IC file {ic_file.name} has no entry for {missing}; "
                      "keeping their reference values.", stacklevel=2)
    return point
