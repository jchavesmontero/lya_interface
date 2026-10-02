"""Record numerical configuration and installed checkout/asset identities."""
import hashlib
import importlib
import importlib.metadata
from pathlib import Path
import subprocess
from dataclasses import asdict


def file_identity(path):
    """Return the resolved path and SHA-256 digest of a file.

    Parameters
    ----------
    path : str or pathlib.Path
        Existing asset whose exact bytes identify a scientific input.

    Returns
    -------
    dict
        Resolved ``path`` and hexadecimal ``sha256`` digest.
    """
    p = Path(path).resolve()
    digest = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(p), "sha256": digest.hexdigest()}


def collect(info, model=None):
    """Collect reproducibility metadata for a resolved interface run.

    Parameters
    ----------
    info : dict
        Resolved Cobaya configuration.
    model : cobaya.model.Model, optional
        Initialized model from which provider requests and BAO assets are
        collected.

    Returns
    -------
    dict
        Configuration, dependency revisions, data assets and, when available,
        resolved model contracts.
    """
    result = {"configuration": info, "dependencies": {}, "assets": []}
    for name in ("cobaya", "camb", "numpy", "scipy", "torch", "cup1d", "forestflow", "lace"):
        module = importlib.import_module(name)
        root = Path(module.__file__).resolve().parent
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True)
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = getattr(module, "__version__", "uninstalled checkout")
        result["dependencies"][name] = {"version": version, "revision": revision.stdout.strip() or None,
                                        "dirty": bool(dirty.stdout.strip()) if revision.returncode == 0 else None}
    for category in ("theory", "likelihood"):
        for name, options in info.get(category, {}).items():
            if not options:
                continue
            for key in ("native_config", "covariance_asset", "transformation_path"):
                if options.get(key):
                    result["assets"].append(file_identity(options[key]))
            if name.endswith("ForestFlowTheory"):
                import forestflow
                key = options.get("model_key", "forest_mpg_fix")
                if options.get("emulator_label"):
                    from cup1d.emulator.factory import _EMULATOR_ALIASES
                    key = _EMULATOR_ALIASES[options["emulator_label"]]
                prefix = options.get("model_path") or str(Path(forestflow.__file__).resolve().parents[1] / "data" / "emulator_models" / key)
                for suffix in (".pt", "_metadata.npy", "_transf.npy", "_manifest.json"):
                    if Path(prefix + suffix).exists():
                        result["assets"].append(file_identity(prefix + suffix))
    history_paths = []
    if any(isinstance(options, dict) and options.get("native_config")
           for options in info.get("likelihood", {}).values()):
        from cup1d.utils.utils import get_path_repo
        from lace.configuration import get_nyx_path
        history_paths = [Path(get_path_repo("lace"))/"data/sim_suites/Australia20"/name
                         for name in ("IGM_histories.npy", "mpg_emu_cosmo.npy")]
        history_paths.append(Path(get_nyx_path())/"IGM_histories.npy")
    for path in history_paths:
        if path.is_file():
            result["assets"].append(file_identity(path))
    for options in info.get("likelihood", {}).values():
        if isinstance(options, dict) and options.get("native_config"):
            from cup1d import Args
            args = Args.from_yaml(options["native_config"], verbose=False)
            for label in args.data_label:
                if label.startswith("DESIY1"):
                    from cup1d.p1ds.observations.data_DESIY1 import set_p1d_filename
                    result["assets"].append(file_identity(args.p1d_fname or set_p1d_filename(label)))
                    continue
                directory = Path(get_path_repo("cup1d"))/"data/p1d_measurements"/label
                if directory.is_dir():
                    result["assets"] += [file_identity(p) for p in sorted(directory.iterdir()) if p.is_file()]
    if model is not None:
        from cobaya.likelihoods.base_classes import BAO
        for component in model.likelihood.values():
            if isinstance(component, BAO):
                directory = Path(component.path) if component.path else Path(component.packages_path) / "data"
                for field in ("measurements_file", "cov_file", "invcov_file", "grid_file"):
                    if getattr(component, field, None):
                        result["assets"].append(file_identity(directory / getattr(component, field)))
        result["registries"] = {name: [asdict(p) for p in component.registry.entries.values()]
            for name, component in model.likelihood.items() if hasattr(component,"registry")}
        result["requests"] = {name: component.backend.get_prediction_request().to_dict()
            for name, component in model.likelihood.items() if hasattr(component,"backend")}
        result["resolved_bundles"] = {name: dict(input_labels=component.emulator.input_labels,
            output_labels=component.emulator.output_labels, domain=component.emulator.model_domain,
            estimator="transformed-space mean Arinyo parameters, then project",
            latent_assignment="common random block 0 for every redshift/input")
            for name, component in model.theory.items() if hasattr(component,"emulator")}
    return result
