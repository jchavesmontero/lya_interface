"""Starting points from cup1d global initial-condition (IC) files."""
from pathlib import Path
import warnings
import numpy as np
from cup1d.utils.ic_files import get_ic_path
from cup1d.utils.utils import split_string
from lya_interface.adapters.cup1d_backend import native_configuration


def _native_args(info):
    """Return the cup1d arguments of the configuration's ForestFlow theory."""
    forest = next(v for k, v in info["theory"].items() if k.endswith("ForestFlowTheory"))
    return native_configuration(forest["native_config"])[0]


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
    one ``{"z": nodes, "val": coefficients}`` entry per IGM and contaminant
    family, plus ``{"val": value}`` for ``As`` and ``ns``. It can be read back
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
        if not public.startswith(("igm_", "p1d_")):
            continue
        family, index = split_string(public.split("_", 1)[1])
        families.setdefault(family, []).append((_node(args, family, index, public), float(value)))
    for family, rows in families.items():
        rows.sort()
        out[family] = {"z": np.array([z for z, _ in rows]), "val": np.array([v for _, v in rows])}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out, allow_pickle=True)
    return path


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
    absent. Values outside the YAML prior bounds are reported by the caller.
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
        native = public.split("_", 1)[1] if public.startswith(("igm_", "p1d_")) else None
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
        point[public] = float(np.interp(znode, ic[family]["z"], ic[family]["val"]))
    if missing:
        warnings.warn(f"IC file {ic_file.name} has no entry for {missing}; "
                      "keeping their reference values.", stacklevel=2)
    return point
