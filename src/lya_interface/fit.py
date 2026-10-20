"""Minimization runs on the Cobaya graph that can be saved and restored.

This is the interface counterpart of cup1d's ``Analysis`` / ``Analysis.from_results``:
a :class:`Fit` owns one configuration and its Cobaya model, runs the bounded
minimizer and the local-error estimate, and writes a small result file from which
:meth:`Fit.from_results` rebuilds the same model with the fit restored.
"""
from pathlib import Path
import warnings

import numpy as np
from cobaya.model import get_model

from lya_interface.configuration import load_configuration
from lya_interface.diagnostics import (
    LocalErrors, PointFit, estimate_local_errors, goodness_of_fit, minimize_point,
    plot_star_ellipse, star_values,
)
from lya_interface.provenance import file_identity

FORMAT_VERSION = 1
_ASSET_KEYS = ("native_config", "covariance_asset")


def _examples_directory():
    """Return the ``examples`` directory of this checkout."""
    import lya_interface
    return Path(lya_interface.__file__).resolve().parents[2] / "examples"


def _asset_digests(info):
    """Return SHA-256 digests of the configuration-owned scientific assets."""
    digests = {}
    for section in ("theory", "likelihood"):
        for name, options in info.get(section, {}).items():
            if not isinstance(options, dict):
                continue
            for key in _ASSET_KEYS:
                path = options.get(key)
                if path and Path(path).is_file():
                    digests[f"{name}.{key}"] = file_identity(path)["sha256"]
    return digests


class Fit:
    """Configuration, Cobaya model and minimization state of one fit.

    Parameters
    ----------
    config_file : str or pathlib.Path
        Interface YAML resolved with :func:`load_configuration`.
    drop_derived : bool, default=True
        Remove ``derived: true`` parameter columns before building the model.
        The minimizer evaluates without derived values, which Cobaya cannot
        collect afterwards; star parameters are computed explicitly instead.

    Attributes
    ----------
    config_path : pathlib.Path
        Resolved configuration file.
    info : dict
        Configuration given to Cobaya.
    model : cobaya.model.Model
        The dependency graph. Close it with :meth:`close`.
    reference : dict
        Reference value of every sampled parameter.
    best_fit : PointFit or None
        Result of :meth:`minimize`, or the restored fit.
    local_errors : LocalErrors or None
        Result of :meth:`estimate_errors`, or the restored errors.

    Raises
    ------
    FileNotFoundError
        If a configured calibrated covariance asset is missing; it is never
        silently disabled.
    """

    def __init__(self, config_file, drop_derived=True):
        self.config_path = Path(config_file).expanduser().resolve()
        self.drop_derived = bool(drop_derived)
        self.info = load_configuration(self.config_path)
        if self.drop_derived:
            for name, definition in tuple(self.info["params"].items()):
                if isinstance(definition, dict) and definition.get("derived") is True:
                    self.info["params"].pop(name)
        for options in self.info.get("likelihood", {}).values():
            asset = options.get("covariance_asset") if isinstance(options, dict) else None
            if asset and not Path(asset).is_file():
                raise FileNotFoundError(f"Calibrated covariance required: {asset}")
        self.reference = {
            name: definition["ref"]
            for name, definition in self.info["params"].items()
            if isinstance(definition, dict) and "prior" in definition
        }
        self.model = get_model(self.info)
        self.best_fit = None
        self.local_errors = None
        self.initial_point = None
        self.error_method = None

    def minimize(self, initial_point=None, **options):
        """Run :func:`minimize_point` and store the result.

        Parameters
        ----------
        initial_point : dict, optional
            Physical start of every sampled parameter. Defaults to the restored
            best fit if any, else the references.
        **options
            Forwarded to :func:`minimize_point` (``max_evals``, ``stall_evals``...).

        Returns
        -------
        PointFit
            The new fit. Previously estimated local errors are discarded.
        """
        if initial_point is None:
            initial_point = self.best_fit.point if self.best_fit is not None else self.reference
        self.initial_point = dict(initial_point)
        self.best_fit = minimize_point(self.model, self.initial_point, **options)
        self.local_errors = None
        return self.best_fit

    def estimate_errors(self, method="gauss_newton", **options):
        """Run :func:`estimate_local_errors` at the best fit and store them.

        Parameters
        ----------
        method : {'gauss_newton', 'finite_difference'}, default='gauss_newton'
            Curvature estimator.
        **options
            Forwarded to :func:`estimate_local_errors`.

        Returns
        -------
        LocalErrors

        Raises
        ------
        ValueError
            If no best fit is available.
        """
        if self.best_fit is None:
            raise ValueError("run minimize() or restore a fit before estimating errors")
        self.local_errors = estimate_local_errors(
            self.model, self.best_fit.point, method=method, **options
        )
        self.error_method = method
        return self.local_errors

    def star_values(self, unblind=False):
        """Return ``[Delta2star, nstar]`` at the best fit, blinded unless ``unblind``."""
        if self.best_fit is None:
            raise ValueError("run minimize() or restore a fit first")
        return star_values(self.model, self.best_fit.point, unblind)

    def plot_star_ellipse(self, ax=None, label=None, color=None, levels=(0.6827, 0.9545),
                          unblind=False):
        """Plot the ``(n_star, Delta2_star)`` local-error ellipses at the best fit.

        Local errors are estimated with Gauss-Newton first if not available.
        Arguments and returns as :func:`plot_star_ellipse`; pass ``ax`` to
        overlay several fits.
        """
        if self.best_fit is None:
            raise ValueError("run minimize() or restore a fit first")
        if self.local_errors is None:
            self.estimate_errors()
        return plot_star_ellipse(self.model, self.best_fit.point, self.local_errors,
                                 ax=ax, label=label, color=color, levels=levels,
                                 unblind=unblind)

    def save(self, filename):
        """Write the fit to a NumPy result file.

        The file stores the configuration path and digests of the configuration
        and its scientific assets, the best-fit point and summary, the initial
        point and, when available, the local errors. Chains, models and
        emulators are never written.

        Parameters
        ----------
        filename : str or pathlib.Path
            Output ``.npy`` file.

        Raises
        ------
        ValueError
            If there is no fit to save.
        """
        if self.best_fit is None:
            raise ValueError("nothing to save: run minimize() first")
        fit = self.best_fit
        payload = {
            "format_version": FORMAT_VERSION,
            "result_type": "minimizer",
            "config_path": str(self.config_path),
            "config_sha256": file_identity(self.config_path)["sha256"],
            "asset_sha256": _asset_digests(self.info),
            "drop_derived": self.drop_derived,
            "initial_point": self.initial_point,
            "best_fit": {
                "point": dict(fit.point),
                "chi2_data": float(fit.chi2_data),
                "success": bool(fit.success),
                "message": str(fit.message),
                "evaluations": int(fit.evaluations),
                "minus2_loglike_total": float(fit.minus2_loglike_total),
            },
            "local_errors": None if self.local_errors is None else {
                "method": self.error_method,
                "errors": dict(self.local_errors.errors),
                "covariance": np.asarray(self.local_errors.covariance),
                "hessian": np.asarray(self.local_errors.hessian),
                "names": tuple(self.local_errors.names),
            },
        }
        np.save(Path(filename).expanduser(), payload, allow_pickle=True)

    @classmethod
    def from_results(cls, filename, config_path=None):
        """Rebuild a fit from a result file written by :meth:`save`.

        Parameters
        ----------
        filename : str or pathlib.Path
            Result file. A plain point dictionary saved with ``np.save`` (older
            notebooks) is also accepted, in which case ``config_path`` is required.
        config_path : str or pathlib.Path, optional
            Configuration to use instead of the recorded one, e.g. for a result
            produced on another machine. If omitted and the recorded file is
            missing, a file with the same name under this checkout's
            ``examples`` directory is used.

        Returns
        -------
        Fit
            New fit with its model built, ``best_fit`` restored (goodness of fit
            recomputed) and ``local_errors`` restored when saved.

        Warns
        -----
        UserWarning
            When another configuration than the recorded one is used, when the
            configuration or its assets changed since the fit, or when the
            recomputed ``-2 log L`` differs from the saved one.

        Raises
        ------
        FileNotFoundError
            If no configuration can be found.
        ValueError
            If the file is not a supported result or its point does not match
            the sampled parameters.
        """
        path = Path(filename).expanduser().resolve()
        payload = np.load(path, allow_pickle=True).item()
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported result file: {path}")
        if "format_version" not in payload:
            if config_path is None:
                raise ValueError(f"{path} holds only a point; pass config_path=")
            payload = {"format_version": FORMAT_VERSION, "config_path": str(config_path),
                       "best_fit": {"point": payload}}
        if payload["format_version"] != FORMAT_VERSION:
            raise ValueError(f"Unsupported result format {payload['format_version']}: {path}")

        recorded = Path(payload["config_path"]).expanduser()
        if config_path is not None:
            selected = Path(config_path).expanduser()
            if selected.resolve() != recorded.resolve():
                warnings.warn(f"Using config_path={selected} instead of the recorded "
                              f"configuration ({recorded}).", stacklevel=2)
        elif recorded.is_file():
            selected = recorded
        else:
            matches = sorted(_examples_directory().rglob(recorded.name))
            if len(matches) != 1:
                raise FileNotFoundError(f"Recorded configuration does not exist: {recorded}; "
                                        "pass config_path= to override")
            selected = matches[0]
            warnings.warn(f"Recorded configuration ({recorded}) does not exist; falling back "
                          f"to local file {selected}. It may differ from the one used for the fit.",
                          stacklevel=2)
        if not selected.is_file():
            raise FileNotFoundError(f"Configuration not found: {selected}")

        fit = cls(selected, drop_derived=payload.get("drop_derived", True))
        if "config_sha256" in payload and \
                file_identity(selected)["sha256"] != payload["config_sha256"]:
            warnings.warn(f"Configuration {selected} changed since the fit was saved.",
                          stacklevel=2)
        current = _asset_digests(fit.info)
        changed = sorted(key for key, digest in payload.get("asset_sha256", {}).items()
                         if current.get(key) != digest)
        if changed:
            warnings.warn(f"Assets changed since the fit was saved: {changed}", stacklevel=2)

        saved = payload["best_fit"]
        point = {name: float(value) for name, value in saved["point"].items()}
        if set(point) != set(fit.reference):
            fit.close()
            raise ValueError(
                "Saved point does not match the sampled parameters: "
                f"missing {sorted(set(fit.reference) - set(point))}, "
                f"extra {sorted(set(point) - set(fit.reference))}"
            )
        statistics = goodness_of_fit(fit.model, point)
        minus2 = -2 * float(np.sum(fit.model.logposterior(point, return_derived=False).loglikes))
        stored = saved.get("minus2_loglike_total")
        if stored is not None and not np.isclose(minus2, stored, rtol=1e-8, atol=1e-6):
            warnings.warn(f"Recomputed -2logL_total={minus2:.6g} differs from the saved "
                          f"{stored:.6g}.", stacklevel=2)
        fit.best_fit = PointFit(
            point,
            statistics.p1d.chi2 if statistics.p1d is not None else np.nan,
            bool(saved.get("success", True)),
            str(saved.get("message", f"Restored from {path.name}")),
            int(saved.get("evaluations", 0)),
            minus2,
            statistics,
        )
        fit.initial_point = payload.get("initial_point")
        errors = payload.get("local_errors")
        if errors is not None:
            fit.local_errors = LocalErrors(dict(errors["errors"]), np.asarray(errors["covariance"]),
                                           np.asarray(errors["hessian"]), tuple(errors["names"]))
            fit.error_method = errors.get("method")
        fit.results_path = str(path)
        return fit

    def close(self):
        """Close the Cobaya model."""
        self.model.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
