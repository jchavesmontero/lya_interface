"""Cobaya chains of a run as cup1d posterior samples, for cup1d's posterior figures.

The figures live in :mod:`cup1d.postprocessing.posterior`, since the
analysis is a 1D one; this module only translates a Cobaya run (weighted
chains with public names) into what they need: :class:`PosteriorSamples` in
native cup1d names, the static native IGM and contaminant models, and the
best-fit P1D bins. Joint analyses with other data would add their own figures
here.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from lya_interface.cup1d_bridge import DERIVED_ALIASES
from lya_interface.names import native_name

CHAIN_FOLDER_PREFIX = "chain_"


def chain_prefix(run_dir, prefix=None):
    """Return the run folder and the chain prefix to read.

    Parameters
    ----------
    run_dir : path-like
        Run folder (with ``chain_<n>`` folders), or a ``chain_<n>`` folder.
    prefix : str, optional
        Chain prefix relative to ``run_dir``, e.g. ``chain_2/mcmc``; by
        default ``mcmc`` in the given chain folder or in the latest
        ``chain_<n>``.

    Returns
    -------
    run_dir : pathlib.Path
        The run folder.
    prefix : pathlib.Path
        Absolute chain prefix (``<prefix>.<i>.txt`` are the chains).

    Raises
    ------
    FileNotFoundError
        If no chain folder is found.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    name = run_dir.name
    if prefix is None and name.startswith(CHAIN_FOLDER_PREFIX) and name[len(CHAIN_FOLDER_PREFIX):].isdigit():
        return run_dir.parent, run_dir / "mcmc"
    if prefix is None:
        chains = sorted((f for f in run_dir.glob(CHAIN_FOLDER_PREFIX + "*")
                         if f.name[len(CHAIN_FOLDER_PREFIX):].isdigit()),
                        key=lambda f: int(f.name[len(CHAIN_FOLDER_PREFIX):]))
        if not chains:
            raise FileNotFoundError(f"no {CHAIN_FOLDER_PREFIX}<n> folder in {run_dir}; pass a prefix")
        return run_dir, chains[-1] / "mcmc"
    return run_dir, run_dir / prefix


def run_configuration(prefix):
    """Return the physical-parameter configuration of a chain.

    The chain YAML (``cobaya_mcmc.yaml`` next to the chains, else in the run
    folder) samples the unit cube; its ``interface_base`` holds the physical
    flat priors, the native cup1d configuration and the fiducial conversions.

    Parameters
    ----------
    prefix : path-like
        Chain prefix.

    Returns
    -------
    info : dict
        Resolved interface configuration (physical parameters).
    path : pathlib.Path
        The interface YAML it was read from.

    Raises
    ------
    FileNotFoundError
        If no chain YAML is found.
    """
    from lya_interface.configuration import load_configuration, resolve_interface_base

    prefix = Path(prefix)
    candidates = [prefix.parent / "cobaya_mcmc.yaml", prefix.parent.parent / "cobaya_mcmc.yaml"]
    chain_yaml = next((path for path in candidates if path.is_file()), None)
    if chain_yaml is None:
        raise FileNotFoundError(f"no cobaya_mcmc.yaml in {[str(p.parent) for p in candidates]}")
    production = yaml.safe_load(chain_yaml.read_text())
    if "interface_base" in production:
        path = resolve_interface_base(production["interface_base"], chain_yaml.parent).resolve()
    else:
        path = chain_yaml
    return load_configuration(path), path


def run_native_config(prefix, log=print):
    """Return the native cup1d YAML a chain was run with.

    Parameters
    ----------
    prefix : path-like
        Chain prefix.
    log : callable, default=print

    Returns
    -------
    path : str
        The chain's frozen ``native_config.yaml``, or, for chains started
        before it was written, the native YAML recorded by Cobaya in
        ``<prefix>.updated.yaml`` (a file of the checkout, which may have
        changed since: a warning is logged).

    Raises
    ------
    FileNotFoundError
        If the chain records neither.
    """
    from lya_interface.configuration import NATIVE_CONFIG_FILE, native_config_path

    prefix = Path(prefix)
    frozen = prefix.parent / NATIVE_CONFIG_FILE
    if frozen.is_file():
        return str(frozen)
    updated = prefix.parent / f"{prefix.name}.updated.yaml"
    path = native_config_path(yaml.safe_load(updated.read_text())) if updated.is_file() else None
    if path is None:
        raise FileNotFoundError(f"no {NATIVE_CONFIG_FILE} nor {updated.name} with a native_config "
                                f"in {prefix.parent}")
    if not Path(path).is_file() and "/cup1d/" in path:
        # recorded on another machine (e.g. NERSC): same file of this cup1d checkout
        from cup1d.utils.utils import get_path_repo
        path = str(Path(get_path_repo("cup1d")) / path.rsplit("/cup1d/", 1)[1])
    log(f"warning: {prefix.parent.name} has no frozen {NATIVE_CONFIG_FILE}; using {path} from the "
        "checkout, as recorded by Cobaya (it may have changed since the chain started)")
    return path


@dataclass
class RunContext:
    """What the figures need besides the samples.

    Attributes
    ----------
    info : dict
        Physical interface configuration.
    config_path : pathlib.Path
        Interface YAML of ``info``.
    native_config : str
        Native cup1d YAML of the run (:func:`run_native_config`).
    fid_cosmo_label : str
        Fiducial cosmology of that configuration.
    blinding : dict
        Native star-parameter offsets included in the chain values.
    best_chain_point : dict
        Public physical values of the sample with the lowest minus log
        posterior among all rows read (before thinning).
    best_chain_minuslogpost : float
    n_rows : int
        Rows read after burn-in (before thinning).
    """
    info: dict
    config_path: Path
    native_config: str
    fid_cosmo_label: str
    blinding: dict = field(default_factory=dict)
    best_chain_point: dict = field(default_factory=dict)
    best_chain_minuslogpost: float = np.inf
    n_rows: int = 0
    _cosmology: object = field(default=None, repr=False)

    def M_of_z(self, z):
        """``H(z)/(1+z)`` [km/s/Mpc] of the run's fiducial cosmology."""
        if self._cosmology is None:
            from lace.cosmo.cosmology import Cosmology
            self._cosmology = Cosmology(cosmo_label=self.fid_cosmo_label)
        return float(np.atleast_1d(self._cosmology.get_dkms_dMpc(np.atleast_1d(float(z))))[0])

    def physical_config(self, folder):
        """Write an interface YAML with the run's native config; return its path.

        Parameters
        ----------
        folder : path-like
            Where to write ``run_config.yaml`` (an overlay of
            :attr:`config_path`), e.g. to build the model with :class:`Fit`.

        Returns
        -------
        pathlib.Path
        """
        path = Path(folder) / "run_config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"interface_base": str(Path(self.config_path).resolve()),
                                        "native_config": str(Path(self.native_config).resolve())}))
        return path


def _sampled_bounds(info):
    """Public name to (min, max) of every flat-prior sampled parameter."""
    return {name: (float(d["prior"]["min"]), float(d["prior"]["max"]))
            for name, d in info["params"].items()
            if isinstance(d, dict) and isinstance(d.get("prior"), dict) and "min" in d["prior"]}


def native_parameter_name(public):
    """Native cup1d name of a public chain column (``igm_tau_eff_0`` -> ``tau_eff_0``)."""
    if public in DERIVED_ALIASES:
        return DERIVED_ALIASES[public]
    return native_name(public) or public


def read_chains(prefix, columns, burn=0.3, max_rows=1_000_000, log=print):
    """Read Cobaya chains ``<prefix>.<i>.txt``: burn-in removed, thinned.

    Parameters
    ----------
    prefix : path-like
        Chain prefix.
    columns : sequence of str
        Columns to keep besides ``weight`` and ``minuslogpost``.
    burn : float, default=0.3
        Fraction of the rows of every chain discarded as burn-in.
    max_rows : int, default=1_000_000
        Approximate maximum number of rows kept in total: every chain is
        thinned by an integer step (rows keep their weights), which keeps
        memory bounded on login nodes.
    log : callable, default=print
        Progress messages.

    Returns
    -------
    values : numpy.ndarray, shape (n, len(columns))
        float64 values of ``columns``.
    weights, minuslogpost : numpy.ndarray, shape (n,)
    best : dict
        ``columns`` at the row with the lowest ``minuslogpost`` of all rows
        after burn-in, before thinning, plus its ``minuslogpost``.
    n_rows : int
        Rows after burn-in, before thinning.

    Raises
    ------
    FileNotFoundError
        If there are no chain files.
    KeyError
        If a requested column is missing from a chain.

    Notes
    -----
    The last line of a chain that a running job is writing may be incomplete;
    it is skipped, so this can be run while the chains grow.
    """
    import pandas as pd

    prefix = Path(prefix)
    files = sorted(prefix.parent.glob(f"{prefix.name}.*.txt"), key=lambda f: int(f.name.split(".")[-2]))
    files = [f for f in files if f.name.split(".")[-2].isdigit()]
    if not files:
        raise FileNotFoundError(f"no chains {prefix}.<i>.txt")
    wanted = ["weight", "minuslogpost", *columns]
    per_chain = max(max_rows // len(files), 1)
    blocks, best, n_rows = [], {"minuslogpost": np.inf}, 0
    for file in files:
        with open(file) as handle:
            names = handle.readline().lstrip("#").split()
        missing = [c for c in wanted if c not in names]
        if missing:
            raise KeyError(f"{file.name} has no columns {missing}")
        frame = pd.read_csv(file, sep=r"\s+", skiprows=1, names=names, usecols=wanted,
                            on_bad_lines="skip").apply(pd.to_numeric, errors="coerce").dropna()
        frame = frame.iloc[int(burn * len(frame)):]
        n_rows += len(frame)
        if len(frame) == 0:
            continue
        row = frame["minuslogpost"].idxmin()
        if frame.at[row, "minuslogpost"] < best["minuslogpost"]:
            best = {c: float(frame.at[row, c]) for c in ["minuslogpost", *columns]}
        step = max(-(-len(frame) // per_chain), 1)
        blocks.append(frame.iloc[::step][wanted].to_numpy(dtype=np.float64))
    data = np.concatenate(blocks)
    log(f"read {len(files)} chains: {n_rows} rows after {burn:.0%} burn-in, kept {len(data)}")
    return data[:, 2:], data[:, 0], data[:, 1], best, n_rows


def load_posterior(prefix, burn=0.3, max_rows=1_000_000, unblind=False, label="", log=print):
    """Load a Cobaya run as cup1d :class:`PosteriorSamples`.

    Parameters
    ----------
    prefix : path-like
        Chain prefix (see :func:`chain_prefix`).
    burn : float, default=0.3
        Burn-in fraction of every chain.
    max_rows : int, default=1_000_000
        Approximate number of rows kept (see :func:`read_chains`).
    unblind : bool, default=False
        Remove the native DESI blinding offsets from ``Delta2_star`` and
        ``n_star``. Only for data sets that have been unblinded (e.g. DESI DR1
        P1D); otherwise they stay blinded and are marked as such.
    label : str, default=""
        Legend label of the posterior.
    log : callable, default=print
        Progress messages.

    Returns
    -------
    samples : cup1d.postprocessing.posterior.PosteriorSamples
        Physical sampled parameters and ``Delta2_star``, ``n_star`` (if
        derived in the chains), in native cup1d names, with the prior bounds
        recorded by the run (:func:`lya_interface.unit_cube.read_prior_bounds`).
    context : RunContext
        Configuration, the run's native config and fiducial cosmology,
        blinding and the best chain sample.

    Raises
    ------
    FileNotFoundError
        If the chain folder records no prior bounds.
    """
    from cup1d.postprocessing.posterior import PosteriorSamples
    from lya_interface.cup1d_bridge import native_blinding

    from lya_interface.unit_cube import read_prior_bounds

    prefix = Path(prefix)
    info, config_path = run_configuration(prefix)
    # the run's own priors (prior_bounds.yaml or Cobaya's record), never the
    # current configuration, whose priors may have changed since
    bounds = read_prior_bounds(prefix.parent, prefix.name)
    current = _sampled_bounds(info)
    changed = sorted(n for n in bounds if n in current and not np.allclose(bounds[n], current[n]))
    if changed:
        log(f"priors changed since this run (the run's are used): {changed}")
    sampled = list(bounds)
    derived = [name for name in ("Delta2star", "nstar") if name in info["params"]]
    values, weights, minuslogpost, best, n_rows = read_chains(prefix, sampled + derived, burn, max_rows, log)
    from cup1d import Args

    native_config = run_native_config(prefix, log)
    fid_cosmo_label = Args.from_yaml(native_config, verbose=False).fid_cosmo_label
    blinding = native_blinding(native_config)
    blinded = any(blinding.get(key, 0.0) for key in ("Delta2_star", "n_star"))
    names = [native_parameter_name(name) for name in sampled + derived]
    if unblind and blinded:
        stars = [name for name in names if name in ("Delta2_star", "n_star")]
        columns = blinding.unblind({name: values[:, names.index(name)] for name in stars})
        for name in stars:
            values[:, names.index(name)] = columns[name]
        log("star parameters unblinded")
    samples = PosteriorSamples(
        values, names, weights,
        bounds={native_parameter_name(name): bound for name, bound in bounds.items()},
        minuslogpost=minuslogpost, sampled=[native_parameter_name(n) for n in sampled],
        label=label, blinded=blinded and not unblind,
    )
    context = RunContext(info, config_path, native_config, fid_cosmo_label, blinding,
                         {name: best[name] for name in sampled}, best["minuslogpost"], n_rows)
    return samples, context


def native_models(native_config):
    """Static native cup1d IGM and contaminant models of a configuration.

    No emulator, cosmology or likelihood is built.

    Parameters
    ----------
    native_config : path-like
        Native cup1d YAML.

    Returns
    -------
    igm : lya_interface.igm.IGM
        IGM histories (same model the likelihood uses).
    contaminants : cup1d.models.contaminants.model_contaminants.Contaminants
    syst_znodes : dict
        Systematics family (e.g. ``R_coeff``) to its node redshifts.
    args : cup1d.Args
        Resolved native settings.
    """
    from cup1d.models.contaminants.model_contaminants import Contaminants
    from lya_interface.cup1d_bridge import make_igm, native_configuration

    args, igm_names, nuisance = native_configuration(native_config)
    igm = make_igm(args, igm_names)
    contaminants = Contaminants(free_param_names=nuisance, pars_cont=args.fid_cont,
                                ic_correction=args.ic_correction)
    syst_znodes = {family: np.asarray(args.fid_syst[family + "_znodes"], dtype=float)
                   for family in args.syst_params if family + "_znodes" in args.fid_syst}
    return igm, contaminants, syst_znodes, args


def data_k_range(args, z=3.0):
    """Wavenumber range (s/km) of the data at a redshift, over every data set.

    Parameters
    ----------
    args : cup1d.Args
        Resolved native settings.
    z : float, default=3.0

    Returns
    -------
    tuple of float
        ``(min, max)`` of the data wavenumbers at ``z`` (tolerance 1e-3).

    Raises
    ------
    ValueError
        If no data set has that redshift.
    """
    from cup1d.p1ds.factory import set_p1d

    ks = []
    for label in args.data_label:
        data = set_p1d(args, label)
        for zz, k in zip(data.z, data.k_kms):
            if abs(float(zz) - z) < 1e-3:
                ks.append(np.asarray(k))
    if not ks:
        raise ValueError(f"no data at z={z}")
    return float(min(k.min() for k in ks)), float(max(k.max() for k in ks))


def data_redshift_range(args):
    """Smallest and largest data redshift over every data set.

    Parameters
    ----------
    args : cup1d.Args
        Resolved native settings.

    Returns
    -------
    tuple of float
    """
    from cup1d.p1ds.factory import set_p1d

    zs = np.concatenate([np.asarray(set_p1d(args, label).z, dtype=float) for label in args.data_label])
    return float(zs.min()), float(zs.max())


def best_fit_entries(model, point):
    """P1D bins and goodness of fit of the model at one point.

    Parameters
    ----------
    model : cobaya.model.Model
        Initialized graph with one cup1d likelihood.
    point : mapping
        Public physical sampled values.

    Returns
    -------
    entries : list of cup1d.postprocessing.p1d.P1DBin
        One bin per data set and redshift; ``chi2`` is the marginal block
        chi-squared of that bin and ``dof`` its number of data points.
    statistics : lya_interface.fit_core.JointGoodnessOfFit
        Correlated chi-squared of the fit.
    """
    from cup1d.likelihood.external import gaussian_residual
    from cup1d.postprocessing.p1d import P1DBin
    from lya_interface.fit_core import cup1d_component, evaluate_point, goodness_of_fit

    _, powers = evaluate_point(model, point)
    backend = cup1d_component(model).backend
    entries = []
    for key, data in backend.data.items():
        for i, z in enumerate(data.z):
            pk = np.asarray(data.Pk_kms[i])
            mod = np.asarray(powers[key][i])
            chi2, _ = gaussian_residual(pk - mod, backend.chol_Pk_kms[key][i])
            entries.append(P1DBin(key, i, float(z), np.asarray(data.k_kms[i]), pk, mod,
                                  np.sqrt(np.diag(backend.cov_Pk_kms[key][i])), float(chi2),
                                  len(pk), "C0", 0.0))
    return entries, goodness_of_fit(model, point)


def best_point(model, candidates, log=print):
    """Return the candidate point with the highest posterior.

    Parameters
    ----------
    model : cobaya.model.Model
    candidates : mapping
        Label (e.g. ``"minimizer"``, ``"chain"``) to public physical point.
    log : callable, default=print

    Returns
    -------
    label : str
    point : dict
    """
    scored = {}
    for label, point in candidates.items():
        logpost = model.logposterior(point, return_derived=False).logpost
        log(f"best-fit candidate {label}: -log posterior = {-logpost:.3f}")
        if np.isfinite(logpost):
            scored[label] = logpost
    if not scored:
        raise ValueError("no candidate best-fit point inside the prior")
    label = max(scored, key=scored.get)
    return label, dict(candidates[label])
