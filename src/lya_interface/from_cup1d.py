"""Build an interface configuration from a native cup1d YAML.

The native cup1d configuration stays the single source of truth: the generated
Cobaya configuration points ``native_config`` at it (data, emulator, rebinning,
``n_k_emu``, contaminants, covariance factors), and every number the interface
needs separately is read from a native cup1d ``Analysis`` built from it:

* sampled parameters, their prior bounds and fiducial references
  (``Likelihood.free_params``, including the emulator-dependent ``As``/``ns``
  bounds);
* the fixed background cosmology given to CAMB (``fid_cosmo_label``);
* the frozen ``M(z) = H(z)/(1+z)`` used by the emulator covariance.

The configuration has no ``projection`` block, so the P3D-to-P1D quadrature
is ForestFlow's default ``P1DIntegratorKMu()``, the one native cup1d uses.
"""
import os
from pathlib import Path

import numpy as np
from cobaya.yaml import yaml_dump

from lya_interface.adapters.cup1d_backend import native_configuration
from lya_interface.configuration import load_configuration
from lya_interface.names import IGM_PREFIX, nuisance_public_name

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
TEMPLATE = EXAMPLES / "cup1d_forestmpggp_evaluate.yaml"
FORESTFLOW_THEORY = "lya_interface.theory.forestflow.ForestFlowTheory"
CUP1D_LIKELIHOOD = "lya_interface.likelihoods.cup1d.Cup1DLikelihood"
CAMB_FIXED = ("H0", "ombh2", "omch2", "mnu", "nnu", "omk", "tau")
DERIVED = ("Delta2star", "nstar", "alphastar")


def public_names(cup1d_config):
    """Map native cup1d free-parameter names to interface public names.

    Parameters
    ----------
    cup1d_config : str or pathlib.Path
        Native cup1d YAML.

    Returns
    -------
    dict
        Native name to public name, in native order.
    """
    _, igm, nuisance = native_configuration(cup1d_config)
    mapping = {name: name for name in ("As", "ns", "nrun")}
    mapping.update({name: IGM_PREFIX + name for name in igm})
    mapping.update({name: nuisance_public_name(name) for name in nuisance})
    return mapping


def interface_info_from_cup1d(cup1d_config, output_yaml, analysis=None,
                              template=TEMPLATE, proposal_fraction=0.01):
    """Return a Cobaya interface configuration equivalent to a cup1d YAML.

    Parameters
    ----------
    cup1d_config : str or pathlib.Path
        Native cup1d YAML (e.g. ``cup1d/configs/hires/hires_forestflow.yaml``).
    output_yaml : str or pathlib.Path
        Where the configuration will be written; relative paths to the native
        YAML and the covariance asset are expressed relative to its folder.
    analysis : cup1d.Analysis, optional
        Native analysis built from ``cup1d_config``; built here if omitted.
    template : str or pathlib.Path, default: ``examples/cup1d_forestmpggp_evaluate.yaml``
        Interface example supplying the CAMB, cosmology-snapshot and
        ForestFlow component settings that have no native counterpart.
    proposal_fraction : float, default: 0.01
        Cobaya fallback proposal width as a fraction of each prior range; used
        only for parameters missing from a proposal covariance matrix.

    Returns
    -------
    dict
        Configuration with sampled parameters, fixed cosmology, native config,
        fiducial ``M(z)`` and covariance asset taken from cup1d. It has no
        sampler block.

    Raises
    ------
    ValueError
        If a native free parameter has no public name.
    """
    cup1d_config = Path(cup1d_config).resolve()
    out_dir = Path(output_yaml).resolve().parent
    if analysis is None:
        from cup1d import Analysis, Args
        analysis = Analysis(Args.from_yaml(str(cup1d_config), verbose=False))
    like = analysis.fitter.like
    cosmo = like.theory.fid_cosmo["cosmo"]
    background = cosmo.get_background_params()
    primordial = cosmo.get_primordial_params()

    # fixed cosmology given to CAMB: the native fiducial cosmology
    params = {name: float(background[name]) for name in CAMB_FIXED}
    params["nrun"] = float(primordial["nrun"])
    names = public_names(cup1d_config)
    for native, definition in like.free_params.items():
        if native not in names:
            raise ValueError(f"no public name for native parameter {native}")
        low, high = float(definition["min_value"]), float(definition["max_value"])
        params[names[native]] = {
            "prior": {"min": low, "max": high},
            "ref": float(definition["value"]),
            "proposal": proposal_fraction * (high - low),
        }
    for name in DERIVED:
        params[name] = {"derived": True}

    info = load_configuration(template)
    info.pop("output", None)
    info.pop("sampler", None)
    info["params"] = params
    emulator_label = analysis.args.emulator_label

    forest = info["theory"][FORESTFLOW_THEORY]
    forest["native_config"] = os.path.relpath(cup1d_config, out_dir)
    forest["parameter_definitions"] = params  # same object: YAML anchor
    forest["emulator_label"] = emulator_label
    # no projection block: ForestFlow's default k_mu quadrature, as native cup1d
    forest.pop("projection", None)
    domain = forest["cosmology_domain"]
    for name in ("ombh2", "omch2", "mnu", "nnu", "omk", "nrun"):
        domain[name] = [params[name], params[name]]
    low, high = domain.get("H0", (params["H0"], params["H0"]))
    domain["H0"] = [min(low, params["H0"]), max(high, params["H0"])]

    likelihood = info["likelihood"][CUP1D_LIKELIHOOD]
    likelihood["native_config"] = forest["native_config"]
    likelihood["parameter_definitions"] = params
    import forestflow
    covariance = (Path(forestflow.__file__).resolve().parents[1]
                  / "data" / "covariance" / f"l1O_cov_{emulator_label}.npy")
    likelihood["covariance_asset"] = os.path.relpath(covariance, out_dir)
    redshifts = sorted({round(float(z), 6) for data in like.data.values() for z in data.z})
    likelihood["fiducial_M"] = {z: float(cosmo.get_dkms_dMpc(z)) for z in redshifts}
    return info


def write_interface_config(cup1d_config, output_yaml, analysis=None, **options):
    """Write :func:`interface_info_from_cup1d` to ``output_yaml`` with a header.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    output_yaml = Path(output_yaml).resolve()
    info = interface_info_from_cup1d(cup1d_config, output_yaml, analysis=analysis, **options)
    header = (
        "# Generated by lya_interface.from_cup1d from the native cup1d configuration\n"
        f"# {os.path.relpath(Path(cup1d_config).resolve(), output_yaml.parent)}\n"
        "# Do not edit by hand: change the cup1d YAML and regenerate with\n"
        "#   python scripts/config_from_cup1d.py <cup1d_yaml> <this_file>\n"
        "# Parameters, bounds and references come from cup1d's Likelihood.free_params;\n"
        "# the fixed cosmology and fiducial_M from its fid_cosmo_label. No sampler block:\n"
        "# samplers are added by the notebooks/scripts that run them.\n"
    )
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(header + yaml_dump(info))
    return output_yaml


def native_cosmo_priors(cup1d_config):
    """Return native cup1d's flat ``As``/``ns`` prior bounds for a configuration.

    Builds only the native emulator and theory (``set_emulator``/``set_theory``,
    as ``Analysis`` does), so neither the likelihood nor the emulator
    covariance is needed. The bounds depend on the emulator training set, the
    fiducial cosmology and ``cosmo_prior_nsigma``.

    Parameters
    ----------
    cup1d_config : str or pathlib.Path
        Native cup1d YAML.

    Returns
    -------
    dict
        ``{"As": (min, max), "ns": (min, max)}``.
    """
    from cup1d import Args
    from cup1d.emulator.factory import set_emulator
    from cup1d.likelihood.parameters import set_free_likelihood_parameters
    from cup1d.p1ds.factory import set_p1d
    from cup1d.theory.factory import set_theory

    args = Args.from_yaml(str(Path(cup1d_config).resolve()), verbose=False)
    zs = np.unique(np.concatenate([set_p1d(args, label).z for label in args.data_label]))
    free = set_free_likelihood_parameters(args, emulator_label=args.emulator_label)
    theory = set_theory(args, set_emulator(emulator_label=args.emulator_label), free,
                        fid_or_true="fid", use_hull=False, zs=zs)
    return {name: tuple(float(v) for v in theory.cosmo_priors[name]) for name in ("As", "ns")}


def sync_cosmo_priors(interface_yaml, cup1d_config=None):
    """Rewrite the ``As``/``ns`` prior lines of an interface YAML from native cup1d.

    Only the one-line ``prior: {min: ..., max: ...}`` entries of sampled
    ``As``/``ns`` are replaced, so comments and anchors are kept.

    Parameters
    ----------
    interface_yaml : str or pathlib.Path
        Interface YAML with flat ``params``.
    cup1d_config : str or pathlib.Path, optional
        Native cup1d YAML; by default the ForestFlow theory's ``native_config``.

    Returns
    -------
    dict
        Old and new bounds of every replaced parameter.

    Raises
    ------
    ValueError
        If a sampled ``As``/``ns`` prior is not a one-line flow mapping.
    """
    import re

    interface_yaml = Path(interface_yaml)
    if cup1d_config is None:
        info = load_configuration(interface_yaml)
        cup1d_config = info["theory"][FORESTFLOW_THEORY]["native_config"]
    bounds = native_cosmo_priors(cup1d_config)
    text = interface_yaml.read_text()
    changes = {}
    for name, (low, high) in bounds.items():
        pattern = re.compile(rf"(\n  {name}:\n    prior: )\{{min: ([^,]+), max: ([^}}]+)\}}")
        match = pattern.search(text)
        if match is None:
            if re.search(rf"\n  {name}:\n    prior:", text):
                raise ValueError(f"{interface_yaml}: {name} prior is not a one-line mapping")
            continue  # fixed or absent
        changes[name] = ((float(match.group(2)), float(match.group(3))), (low, high))
        text = text[:match.start()] + f"{match.group(1)}{{min: {low!r}, max: {high!r}}}" + text[match.end():]
    interface_yaml.write_text(text)
    return changes
