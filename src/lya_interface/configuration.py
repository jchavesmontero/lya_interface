"""Load complete example configurations and check duplicated registry definitions."""
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
        If an additional likelihood replaces an existing component or a
        component duplicates the public parameter registry.
    """
    path = Path(path).resolve()
    info = yaml_load_file(str(path))
    if "interface_base" in info:
        base = load_configuration(path.parent / info.pop("interface_base"))
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
                for key in ("native_config", "covariance_asset", "model_path", "transformation_path"):
                    if options.get(key) and not Path(options[key]).is_absolute():
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
    return deepcopy(info)
