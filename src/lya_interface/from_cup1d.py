"""Build an interface configuration from a native cup1d YAML.

The native cup1d configuration stays the single source of truth: the generated
Cobaya configuration points ``native_config`` at it (data, emulator, rebinning,
``n_k_emu``, contaminants, covariance factors), and every number the interface
needs separately is read from a native cup1d ``Analysis`` built from it:

* sampled parameters, their prior bounds and fiducial references
  (``Likelihood.free_params``, including the emulator-dependent ``As``/``ns``
  bounds);
* the fixed background cosmology given to CAMB (``fid_cosmo_label``): every
  parameter, also those away from CAMB's defaults (with a warning);
* the star-parameter pivot (``z_star``, ``kp_kms``);
* the frozen ``M(z) = H(z)/(1+z)`` used by the emulator covariance.

The remaining component settings (CAMB accuracy flags, snapshot sampling,
blinding policy) are the defaults of :func:`component_defaults`, not a
hand-written example.

The configuration has no ``projection`` block, so the P3D-to-P1D quadrature
is ForestFlow's default ``P1DIntegratorKMu()``, the one native cup1d uses.
"""
import os
import warnings
from pathlib import Path

import numpy as np
from cobaya.yaml import yaml_dump

from lya_interface.cup1d_bridge import native_configuration
from lya_interface.configuration import load_configuration, portable_path
from lya_interface.names import IGM_PREFIX, nuisance_public_name

P1D_EMULATOR_THEORY = "lya_interface.theory.p1d_emulator.P1DEmulatorTheory"
CUP1D_LIKELIHOOD = "lya_interface.likelihoods.cup1d.Cup1DLikelihood"
LYA_COSMOLOGY = "lya_interface.theory.lya_cosmology.LyaCosmology"
CAMB_FIXED = ("H0", "ombh2", "omch2", "mnu", "nnu", "omk", "tau")
DERIVED = ("Delta2star", "nstar", "alphastar")
# fixed in the domain of the theory (H0 gets a range around its value)
DOMAIN_FIXED = ("ombh2", "omch2", "mnu", "nnu", "omk", "nrun")
H0_DOMAIN = (65.0, 70.0)


def component_defaults():
    """Settings of the Cobaya components that have no native counterpart.

    Returns
    -------
    dict
        ``theory`` and ``likelihood`` blocks; the per-analysis values
        (native config, emulator, pivot, parameters, covariance, ``M(z)``)
        are filled by :func:`interface_info_from_cup1d`.
    """
    return {
        "theory": {
            "camb": {"extra_args": {
                # matter power only, as LaCE (same k sampling)
                "WantCls": False,
                "Want_CMB": False,
                "nonlinear": False,
                "neutrino_hierarchy": "degenerate",
                "num_massive_neutrinos": 1,
            }},
            LYA_COSMOLOGY: {
                "stop_at_error": True,
                "kmax_iMpc": 200.0,
                "snapshot_nodes": 4096,
            },
            P1D_EMULATOR_THEORY: {
                "stop_at_error": True,
                # cINN latent sampling: the bundle default number of
                # realizations (realizations: null) and seed, as native cup1d
                "seed": 0,
            },
        },
        "likelihood": {
            CUP1D_LIKELIHOOD: {
                "stop_at_error": True,
                "blinding_policy": "native",
                "include_logdet": False,
            },
        },
    }


def _camb_defaults(background):
    """CAMB's own value of every optional background/primordial input."""
    import camb

    yhe = camb.bbn.get_predictor().Y_He(
        background["ombh2"] * (camb.constants.COBE_CMBTemp / background["TCMB"]) ** 3,
        background["nnu"] - background["standard_neutrino_neff"],
    ).item()
    return {
        "TCMB": camb.constants.COBE_CMBTemp,
        "standard_neutrino_neff": camb.constants.default_nnu,
        "YHe": yhe,
        "w": -1.0,
        "wa": 0.0,
        "nrunrun": 0.0,
    }


def fixed_cosmology(cosmo):
    """Fixed CAMB parameters reproducing a native LaCE cosmology in Cobaya.

    ``CAMB_FIXED`` and ``nrun`` always; any other input that differs from
    CAMB's default (``TCMB``, ``YHe``, dark energy, ``nrunrun``...) is added
    too, with a warning, so the Cobaya cosmology is never silently different.

    Returns
    -------
    params : dict
        Fixed Cobaya parameters.
    pivot_scalar : float
        Primordial pivot (a CAMB ``extra_args`` setting in Cobaya).
    """
    background = cosmo.get_background_params()
    primordial = cosmo.get_primordial_params()
    params = {name: float(background[name]) for name in CAMB_FIXED}
    params["nrun"] = float(primordial["nrun"])
    values = {**background, **primordial}
    for name, default in _camb_defaults(background).items():
        value = float(values[name])
        if not np.isclose(value, default, rtol=1e-6, atol=1e-12):
            warnings.warn(
                f"the native cosmology sets {name} = {value} (CAMB default {default}): "
                "passed to Cobaya's CAMB as a fixed parameter",
                stacklevel=2,
            )
            params[name] = value
    return params, float(primordial["pivot_scalar"])


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
                              proposal_fraction=0.01):
    """Return a Cobaya interface configuration equivalent to a cup1d YAML.

    Parameters
    ----------
    cup1d_config : str or pathlib.Path
        Native cup1d YAML (e.g. ``cup1d/configs/hires/hires_forestflow.yaml``).
    output_yaml : str or pathlib.Path
        Where the configuration will be written. The native YAML and the
        covariance asset are written as ``<package>:<path>`` when inside a
        package checkout, else relative to its folder.
    analysis : cup1d.Analysis, optional
        Native analysis built from ``cup1d_config``; built here if omitted.
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
        analysis = Analysis(Args.from_yaml(str(cup1d_config), verbose=False), create_output=False)
    like = analysis.fitter.like
    if like.Gauss_priors is not None:
        raise NotImplementedError(
            "the native configuration has Gaussian priors, not translated to Cobaya yet"
        )
    theory = like.theory
    cosmo = theory.fid_cosmo["cosmo"]

    # fixed cosmology given to CAMB: the native fiducial cosmology
    params, pivot_scalar = fixed_cosmology(cosmo)
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

    info = component_defaults()
    info["params"] = params
    camb_args = info["theory"]["camb"]["extra_args"]
    camb_args["pivot_scalar"] = camb_args["pivot_tensor"] = pivot_scalar

    lya_cosmology = info["theory"][LYA_COSMOLOGY]
    lya_cosmology["kmax_iMpc"] = float(cosmo.get_kmax_linP_Mpc())
    lya_cosmology["z_star"] = float(theory.z_star)
    lya_cosmology["k_star_ikms"] = float(theory.kp_kms)

    emulator = info["theory"][P1D_EMULATOR_THEORY]
    # <package>:<path> inside a checkout: the YAML works on every machine
    emulator["native_config"] = portable_path(cup1d_config, out_dir)
    emulator["parameter_definitions"] = params  # same object: YAML anchor
    emulator["emulator_label"] = analysis.args.emulator_label
    # the native training-hull setting (no projection block: ForestFlow's
    # default k_mu quadrature, as native cup1d)
    emulator["enforce_training_hull"] = bool(theory.use_hull and getattr(theory, "hull_reject", True))
    emulator["hull_margin"] = float(theory.hull_margin)
    domain = {name: [params[name], params[name]] for name in DOMAIN_FIXED}
    domain["H0"] = [min(H0_DOMAIN[0], params["H0"]), max(H0_DOMAIN[1], params["H0"])]
    emulator["cosmology_domain"] = domain

    likelihood = info["likelihood"][CUP1D_LIKELIHOOD]
    likelihood["native_config"] = emulator["native_config"]
    likelihood["parameter_definitions"] = params
    from cup1d.likelihood.likelihood import emulator_covariance_path
    covariance = emulator_covariance_path(theory.emulator)  # the file native cup1d reads
    likelihood["covariance_asset"] = portable_path(covariance, out_dir)
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
        "# the fixed cosmology and fiducial_M from its fid_cosmo_label; the other component\n"
        "# settings from lya_interface.from_cup1d.component_defaults. No sampler block:\n"
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
        Native cup1d YAML; by default the P1D emulator theory's ``native_config``.

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
        cup1d_config = info["theory"][P1D_EMULATOR_THEORY]["native_config"]
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
