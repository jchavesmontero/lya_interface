"""Load complete example configurations and check duplicated registry definitions."""
from copy import deepcopy
from pathlib import Path
from cobaya.yaml import yaml_load_file


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
    for section in ("theory", "likelihood"):
        for name, options in info.get(section, {}).items():
            if name.startswith("lya_interface."):
                if "parameter_definitions" in options and options["parameter_definitions"] != info["params"]:
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
