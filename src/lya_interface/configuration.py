"""Load complete example configurations and check duplicated registry definitions."""
from copy import deepcopy
from pathlib import Path
from cobaya.yaml import yaml_load_file


def load_configuration(path):
    path = Path(path).resolve()
    info = yaml_load_file(str(path))
    if "interface_base" in info:
        base = load_configuration(path.parent / info.pop("interface_base"))
        base.update(info)
        info = base
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
                        selected = next((p for p in candidates if Path(str(p)+suffix).exists()), candidates[0])
                        options[key] = str(selected.resolve())
    return deepcopy(info)
