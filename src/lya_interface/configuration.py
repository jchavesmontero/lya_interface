"""Load complete example configurations and check duplicated registry definitions."""
import os
from copy import deepcopy
from pathlib import Path
from cobaya.yaml import yaml_load_file


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


NATIVE_CONFIG_FILE = "native_config.yaml"
# asset paths ``<package>:<path>`` are relative to that package's checkout
PACKAGE_ROOTS = ("lace", "forestflow", "cup1d")


def package_root(name):
    """Root of the installed checkout of a package (``lace``, ``forestflow``, ``cup1d``)."""
    import importlib.util

    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(name)
    return Path(spec.origin).resolve().parents[1]


def resolve_package_path(value):
    """``<package>:<path>`` to an absolute path, or ``None`` for any other value."""
    value = str(value)
    if value.startswith(CHECKOUT_PREFIX):
        return checkout_root() / value[len(CHECKOUT_PREFIX):]
    for name in PACKAGE_ROOTS:
        if value.startswith(name + ":"):
            return package_root(name) / value[len(name) + 1:]
    return None


def portable_path(path, folder):
    """``<package>:<path>`` for a file inside a package checkout (portable
    between machines), else the path relative to ``folder``."""
    path = Path(path).resolve()
    for name in PACKAGE_ROOTS:
        try:
            return f"{name}:" + path.relative_to(package_root(name)).as_posix()
        except (ValueError, ModuleNotFoundError):
            continue
    try:
        return CHECKOUT_PREFIX + path.relative_to(checkout_root()).as_posix()
    except ValueError:
        return os.path.relpath(path, folder)


def native_config_path(info):
    """Return the native cup1d YAML shared by the components of a configuration.

    Parameters
    ----------
    info : mapping
        Resolved configuration (:func:`load_configuration`).

    Returns
    -------
    str or None
        The path, or None if no component has one.

    Raises
    ------
    ValueError
        If components use different native configurations.
    """
    paths = {options["native_config"] for section in ("theory", "likelihood")
             for name, options in info.get(section, {}).items()
             if name.startswith("lya_interface.") and isinstance(options, dict) and options.get("native_config")}
    if len(paths) > 1:
        raise ValueError(f"components use different native configurations: {sorted(paths)}")
    return next(iter(paths), None)


def freeze_native_config(info, folder, note=""):
    """Write the complete resolved native cup1d settings of a configuration to a folder.

    Parameters
    ----------
    info : mapping
        Resolved configuration (:func:`load_configuration`).
    folder : path-like
        Destination, e.g. a chain folder.
    note : str, default=""
        Comment written at the top of the file.

    Returns
    -------
    pathlib.Path or None
        ``<folder>/native_config.yaml``: every cup1d setting (CM2026 defaults
        merged with the native YAML), so the run no longer depends on the
        native YAML or the defaults of the cup1d checkout. It sets the
        fiducial cosmology, hence ``H(z)/(1+z)``. None without a native
        configuration.
    """
    from cup1d import Args

    source = native_config_path(info)
    if source is None:
        return None
    args = Args.from_yaml(source, verbose=False)
    return args.write_yaml(Path(folder) / NATIVE_CONFIG_FILE,
                           header=f"Resolved cup1d settings frozen from {source}\n" + note)


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
    ``native_config`` (a path relative to this file or absolute) replaces the
    native cup1d YAML of every component, e.g. the copy frozen with a chain
    (:func:`freeze_native_config`).
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
    native = info.pop("native_config", None)
    if native is not None:
        native = Path(native)
        native = str((native if native.is_absolute() else path.parent / native).resolve())
        for section in ("theory", "likelihood"):
            for name, options in info.get(section, {}).items():
                if name.startswith("lya_interface.") and isinstance(options, dict) and "native_config" in options:
                    options["native_config"] = native
    unit_cube = info.pop("sample_in_unit_cube", False)
    for section in ("theory", "likelihood"):
        for name, options in info.get(section, {}).items():
            if name.startswith("lya_interface."):
                if name.endswith("VegaLikelihood"):
                    options.setdefault("parameter_definitions", info["params"])
                if "parameter_definitions" in options and options["parameter_definitions"] != info["params"]:
                    definitions = options["parameter_definitions"]
                    subset_matches = (isinstance(definitions, dict)
                                      and all(info["params"].get(key) == value
                                              for key, value in definitions.items()))
                    if not subset_matches:
                        raise ValueError("component registry and public params definitions differ")
                for key in ("native_config", "covariance_asset", "metal_model", "metal_alpha_reference"):
                    packaged = resolve_package_path(options[key]) if isinstance(options.get(key), str) else None
                    if packaged is not None:
                        options[key] = str(packaged.resolve())
                    elif isinstance(options.get(key), str) and options[key] and not Path(options[key]).is_absolute():
                        candidates = [path.parent / options[key], path.parent.parent / options[key]]
                        # Config-relative assets first; repository-relative
                        # paths are retained for directly runnable examples.
                        # A missing calibrated product should still resolve to
                        # its existing asset directory, not a fictitious nested
                        # checkout. Keep the eventual FileNotFoundError exact.
                        fallback = min(candidates, key=lambda p: next(
                            (i for i, parent in enumerate(p.parents) if parent.is_dir()), len(p.parents)))
                        selected = next((p for p in candidates if p.exists()), fallback)
                        options[key] = str(selected.resolve())
    if unit_cube:
        _sample_in_unit_cube(info)
    return deepcopy(info)
