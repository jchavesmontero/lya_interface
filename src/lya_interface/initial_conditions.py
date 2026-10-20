"""Starting points from cup1d global initial-condition (IC) files."""
from pathlib import Path
import numpy as np
from cup1d.utils.ic_files import get_ic_path
from cup1d.utils.utils import split_string
from lya_interface.adapters.cup1d_backend import native_configuration


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
        If a coefficient has no node configuration or no IC entry.

    Notes
    -----
    Mirrors ``Likelihood.set_ic_global``. ``HCD_const`` defaults to zero when
    absent. Values outside the YAML prior bounds are reported by the caller.
    """
    forest = next(v for k, v in info["theory"].items() if k.endswith("ForestFlowTheory"))
    args, _, _ = native_configuration(forest["native_config"])
    if ic_file is None:
        ic_file = get_ic_path(args.data_label, args.emulator_label, "global")
    ic_file = Path(ic_file).expanduser()
    if not ic_file.is_file():
        raise FileNotFoundError(f"Global IC file not found: {ic_file}")
    ic = np.load(ic_file, allow_pickle=True).item()
    point = dict(reference)
    for public in reference:
        if public in ("As", "ns"):
            if public in ic:
                point[public] = float(ic[public]["val"])
            continue
        native = public.split("_", 1)[1] if public.startswith(("igm_", "p1d_")) else None
        if native is None:
            continue
        family, index = split_string(native)
        for section in (args.fid_igm, args.fid_cont, args.fid_syst):
            if family + "_znodes" in section:
                znode = section[family + "_znodes"][int(index)]
                break
        else:
            raise ValueError(f"{family}_znodes not found for {public}")
        if family not in ic:
            if family == "HCD_const":
                point[public] = 0.0
                continue
            raise ValueError(f"IC file {ic_file} has no entry for {family}")
        point[public] = float(np.interp(znode, ic[family]["z"], ic[family]["val"]))
    return point
