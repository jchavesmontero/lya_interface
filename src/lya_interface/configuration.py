"""Load complete example configurations and check duplicated registry definitions."""
import os
from copy import deepcopy
from pathlib import Path
from cobaya.yaml import yaml_load_file


def _flatten_parameter_sections(sections):
    """Translate structured interface parameters into Cobaya's flat namespace.

    ``general`` owns cosmology and shared IGM parameters, ``P1D`` owns only
    P1D nuisances, and ``BAO`` owns only Vega nuisances. A record can provide
    ``native`` to select the downstream name; the remaining keys are retained
    as the Cobaya definition. This intentionally rejects implicit aliases.
    """
    if not isinstance(sections, dict) or set(sections) - {"general", "P1D", "BAO"}:
        raise ValueError("parameters must contain only general, P1D, and BAO sections")
    flat, mapping = {}, {}
    for section, values in sections.items():
        if not isinstance(values, dict):
            raise ValueError(f"parameters.{section} must be a mapping")
        for public, record in values.items():
            if not isinstance(record, dict):
                record = {"value": record}
            record = dict(record)
            native = record.pop("native", public)
            transform = record.pop("transform", "identity")
            units = record.pop("units", None)
            status = record.pop("status", None)
            if status == "derived" and "prior" in record:
                raise ValueError(f"derived parameter {public!r} cannot have a prior")
            if public in flat or native in mapping:
                raise ValueError(f"ambiguous parameter mapping for {public!r}/{native!r}")
            flat[public] = record
            mapping[native] = dict(public=public, section=section,
                                   transform=transform, units=units, status=status)
    return flat, mapping


def _fix_parameters(info, fixed):
    """Fix sampled parameters at given values, in ``params`` and every component registry.

    Parameters
    ----------
    info : dict
        Configuration with inherited flat ``params``; modified in place.
    fixed : mapping
        Public parameter name to its fixed value.

    Raises
    ------
    ValueError
        If ``params`` is not flat or a name is not a sampled parameter.
    """
    if not isinstance(fixed, dict) or not isinstance(info.get("params"), dict):
        raise ValueError("fixed_parameters requires inherited flat params")
    not_sampled = sorted(name for name in fixed if not (
        isinstance(info["params"].get(name), dict) and "prior" in info["params"][name]))
    if not_sampled:
        raise ValueError(f"fixed_parameters must name sampled parameters: {not_sampled}")
    # Components may hold their own copy of the definitions (YAML anchors):
    # every copy must agree with params.
    registries = [info["params"]] + [
        options["parameter_definitions"]
        for section in ("theory", "likelihood") for options in info.get(section, {}).values()
        if isinstance(options, dict) and isinstance(options.get("parameter_definitions"), dict)]
    for registry in {id(r): r for r in registries}.values():
        for name, value in fixed.items():
            if name in registry:
                registry[name] = float(value)


def _sample_in_unit_cube(info):
    """Sample every flat-prior parameter in [0, 1] (:mod:`lya_interface.unit_cube`).

    Only Cobaya's ``params`` change; component ``parameter_definitions`` keep
    the physical definitions, which they use for routing.

    Parameters
    ----------
    info : dict
        Configuration with flat ``params``; modified in place.
    """
    from lya_interface.unit_cube import unit_cube_params

    if not isinstance(info.get("params"), dict):
        raise ValueError("sample_in_unit_cube requires flat params")
    for section in ("theory", "likelihood"):
        for options in info.get(section, {}).values():
            if isinstance(options, dict) and isinstance(options.get("parameter_definitions"), dict):
                options["parameter_definitions"] = deepcopy(options["parameter_definitions"])
    info["params"] = unit_cube_params(info["params"])


def _set_references(info, references):
    """Replace the ``ref`` of sampled parameters, in ``params`` and every component registry.

    Parameters
    ----------
    info : dict
        Configuration with inherited flat ``params``; modified in place.
    references : mapping
        Public parameter name to its new Cobaya ``ref``: a number, or a
        mapping such as ``{"dist": "norm", "loc": value, "scale": width}``.

    Raises
    ------
    ValueError
        If ``params`` is not flat or a name is not a sampled parameter.
    """
    if not isinstance(references, dict) or not isinstance(info.get("params"), dict):
        raise ValueError("reference_point requires inherited flat params")
    not_sampled = sorted(name for name in references if not (
        isinstance(info["params"].get(name), dict) and "prior" in info["params"][name]))
    if not_sampled:
        raise ValueError(f"reference_point must name sampled parameters: {not_sampled}")
    registries = [info["params"]] + [
        options["parameter_definitions"]
        for section in ("theory", "likelihood") for options in info.get(section, {}).values()
        if isinstance(options, dict) and isinstance(options.get("parameter_definitions"), dict)]
    for registry in {id(r): r for r in registries}.values():
        for name, ref in references.items():
            if isinstance(registry.get(name), dict):
                registry[name] = {**registry[name], "ref": deepcopy(ref)}


CHECKOUT_PREFIX = "lya_interface:"


def checkout_root():
    """Return the root of the installed lya_interface checkout."""
    import lya_interface
    return Path(lya_interface.__file__).resolve().parents[2]


def resolve_interface_base(value, folder):
    """Resolve an ``interface_base`` entry.

    Parameters
    ----------
    value : str
        ``lya_interface:<path>`` (relative to :func:`checkout_root`), an
        absolute path, or a path relative to ``folder``.
    folder : pathlib.Path
        Folder of the configuration that holds the entry.

    Returns
    -------
    pathlib.Path
    """
    value = str(value)
    if value.startswith(CHECKOUT_PREFIX):
        return checkout_root() / value[len(CHECKOUT_PREFIX):]
    return Path(folder) / value


def portable_interface_base(config_path, folder):
    """Return the ``interface_base`` entry that best survives moving between machines.

    ``lya_interface:<path>`` for a configuration inside the checkout, else the
    path relative to ``folder``.
    """
    config_path = Path(config_path).resolve()
    try:
        return CHECKOUT_PREFIX + config_path.relative_to(checkout_root()).as_posix()
    except ValueError:
        return os.path.relpath(config_path, folder)


def load_configuration(path):
    """Load an interface YAML file and resolve inherited configuration.

    Parameters
    ----------
    path : str or pathlib.Path
        YAML configuration file. Relative science assets are resolved relative
        to this file before the configuration is returned.

    Returns
    -------
    dict
        Deep copy of the resolved Cobaya configuration.

    Raises
    ------
    ValueError
        If an additional likelihood replaces an existing component, a
        component duplicates the public parameter registry, or
        ``fixed_parameters`` names a parameter that is not sampled.

    Notes
    -----
    ``interface_base`` names the configuration this one overlays: a path
    relative to this file, an absolute path, or ``lya_interface:<path>``
    relative to the installed lya_interface checkout (portable between
    machines, e.g. for run folders outside the checkout).

    ``fixed_parameters`` (public name to value) turns sampled parameters of the
    inherited configuration into fixed ones, in ``params`` and in every
    component's ``parameter_definitions``. ``reference_point`` (public name to
    a Cobaya ``ref``) replaces starting references the same way, e.g. to start
    a chain from a best fit. ``sample_in_unit_cube: true`` makes Cobaya sample
    every flat-prior parameter in [0, 1] (``<name>_cube``) and compute the
    physical one from it, as native cup1d; given references stay physical.
    """
    path = Path(path).resolve()
    info = yaml_load_file(str(path))
    if "interface_base" in info:
        base = load_configuration(resolve_interface_base(info.pop("interface_base"), path.parent))
        base.update(info)
        info = base
    additions = info.pop("additional_likelihoods", {})
    if additions:
        likelihoods = info.setdefault("likelihood", {})
        overlap = set(additions) & set(likelihoods)
        if overlap:
            raise ValueError(f"additional likelihoods would replace existing components: {sorted(overlap)}")
        likelihoods.update(additions)
    overrides = info.pop("parameter_overrides", {})
    if overrides:
        if not isinstance(overrides, dict) or not isinstance(info.get("params"), dict):
            raise ValueError("parameter_overrides requires inherited flat params")
        duplicated = set(overrides) & set(info["params"])
        if duplicated:
            raise ValueError(f"parameter_overrides cannot replace existing parameters: {sorted(duplicated)}")
        info["params"] = {**info["params"], **overrides}
    fixed = info.pop("fixed_parameters", {})
    if fixed:
        _fix_parameters(info, fixed)
    references = info.pop("reference_point", {})
    if references:
        _set_references(info, references)
    unit_cube = info.pop("sample_in_unit_cube", False)
    structured = info.pop("parameters", None)
    interface_mapping = None
    p1d_definitions = None
    if structured is not None:
        flat, interface_mapping = _flatten_parameter_sections(structured)
        if "params" in info and info["params"] != flat:
            raise ValueError("use either flat params or structured parameters, not both")
        info["params"] = flat
        info.setdefault("interface_metadata", {})["parameter_mapping"] = interface_mapping
        sections_by_public = {entry["public"]: entry["section"]
                              for entry in interface_mapping.values()}
        p1d_definitions = {name: definition for name, definition in flat.items()
                           if sections_by_public[name] in {"general", "P1D"}}
    for section in ("theory", "likelihood"):
        for name, options in info.get(section, {}).items():
            if name.startswith("lya_interface."):
                if interface_mapping is not None and name.endswith(("ForestFlowTheory", "Cup1DLikelihood")):
                    options["parameter_definitions"] = p1d_definitions
                if interface_mapping is not None and name.endswith("VegaLikelihood"):
                    options.setdefault("parameter_mapping", {
                        entry["public"]: native for native, entry in interface_mapping.items()
                        if entry["section"] == "BAO"})
                if name.endswith("VegaLikelihood"):
                    options.setdefault("parameter_definitions", info["params"])
                if "parameter_definitions" in options and options["parameter_definitions"] != info["params"]:
                    definitions = options["parameter_definitions"]
                    subset_matches = (isinstance(definitions, dict)
                                      and all(info["params"].get(key) == value
                                              for key, value in definitions.items()))
                    if interface_mapping is None and not subset_matches:
                        raise ValueError("component registry and public params definitions differ")
                for key in ("native_config", "covariance_asset", "model_path", "transformation_path", "metal_model",
                            "metal_alpha_reference"):
                    if isinstance(options.get(key), str) and options[key] and not Path(options[key]).is_absolute():
                        candidates = [path.parent / options[key], path.parent.parent / options[key]]
                        # Config-relative assets first; repository-relative
                        # paths are retained for directly runnable examples.
                        suffix = ".pt" if key == "model_path" else ""
                        # A missing calibrated product should still resolve to
                        # its existing asset directory, not a fictitious nested
                        # checkout. Keep the eventual FileNotFoundError exact.
                        fallback = min(candidates, key=lambda p: next(
                            (i for i, parent in enumerate(p.parents) if parent.is_dir()), len(p.parents)))
                        selected = next((p for p in candidates if Path(str(p)+suffix).exists()), fallback)
                        options[key] = str(selected.resolve())
    if unit_cube:
        _sample_in_unit_cube(info)
    return deepcopy(info)
