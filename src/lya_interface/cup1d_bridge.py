"""lya_interface's bridge to cup1d: native configuration, observation backend,
P1D projection, star parameters and the failure classes.

Everything the Cobaya components need from a native cup1d YAML is built here
(no ``Analysis``, no emulator): the resolved settings and coefficient ownership
(:func:`native_configuration`, :func:`registry_for`), the IGM histories
(:func:`make_igm`), the static likelihood backend (:func:`make_backend`), the
blinding offsets (:func:`native_blinding`), the projection of Arinyo
coefficients to P1D (:func:`project_request`) and the star parameters
(:func:`star_parameters`). The prediction contracts are cup1d's, re-exported.
"""
from pathlib import Path

import numpy as np
from cup1d import Args
from cup1d.likelihood.external import ExternalP1DLikelihood
from cup1d.likelihood.parameters import set_free_likelihood_parameters
from cup1d.models.contaminants.model_contaminants import Contaminants
from cup1d.models.contaminants.model_systematics import Systematics
from cup1d.p1ds.factory import is_synthetic_data_label, set_p1d
from cup1d.theory.projection import (PredictionContext, PredictionGroup,  # noqa: F401  (re-exported)
                                     PredictionRequest, fingerprint)
from lya_interface.igm import IGM
from lya_interface.parameters import ParameterRegistry
from lya_interface.provenance import file_identity

# public star-parameter names to native (LaCE/cup1d) names; LaCE's 100 log
# nodes include the endpoints of the fit window
DERIVED_ALIASES = {"Delta2star": "Delta2_star", "nstar": "n_star", "alphastar": "alpha_star"}


class ModelDomainError(ValueError):
    """Signal that a sampled point lies outside the admitted physical model domain.

    Likelihood code may convert this exception into a rejected point because
    the requested prediction is scientifically unsupported rather than a
    numerical-coverage failure.
    """


class NumericalCoverageError(ValueError):
    """Signal insufficient configured numerical coverage for a valid point.

    Unlike :class:`ModelDomainError`, this exception identifies a configuration
    error (for example an inadequate linear-power range) and must not be
    silently converted into a likelihood rejection.
    """


# ------------------------------------------------------- native configuration

def _serializable(value):
    """Convert NumPy containers recursively to fingerprintable Python values.

    Parameters
    ----------
    value : object
        Mapping, NumPy scalar/array, or already JSON-compatible value.

    Returns
    -------
    object
        Recursively serializable equivalent.
    """
    if isinstance(value, dict):
        return {k: _serializable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def native_configuration(path):
    """Load native cup1d settings and partition public coefficient ownership.

    Parameters
    ----------
    path : str or pathlib.Path
        Native cup1d YAML configuration.

    Returns
    -------
    args, igm, nuisance : tuple
        Resolved Args plus ordered native IGM and nuisance coefficient names.

    Raises
    ------
    ValueError
        If native statistical priors would be silently duplicated outside
        Cobaya.
    """
    args = Args.from_yaml(str(Path(path).resolve()), verbose=False)
    if args.use_star_priors or getattr(args, "prior_Gauss_rms", None) is not None or getattr(args, "Gauss_priors", None) or args.fid_igm.get("Gauss_priors") or args.fid_cont.get("Gauss_priors") or args.fid_syst.get("Gauss_priors"):
        raise ValueError("export native statistical priors explicitly to Cobaya; implicit native priors are unsupported")
    names = set_free_likelihood_parameters(args, emulator_label=args.emulator_label)
    igm = [n for n in names if n.rsplit("_", 1)[0] in args.igm_params]
    nuisance = [n for n in names if n not in igm and n not in {"As", "ns", "nrun"}]
    return args, igm, nuisance


def registry_for(args, definitions, igm, nuisance):
    """Build an explicit public-to-native ownership registry.

    Returns
    -------
    ParameterRegistry
        Registry with coefficient history metadata for IGM and nuisances.
    """
    metadata = {}
    for name in igm + nuisance:
        family = name.rsplit("_", 1)[0]
        section = args.fid_igm if name in igm else args.fid_syst if family in args.syst_params else args.fid_cont
        metadata[name] = dict(units="dimensionless coefficient",
            transformation=f"native {section[family+'_ztype']} / {section[family+'_otype']} history" +
            ("; multiplies fiducial physical history" if name in igm else "; enters native contaminant model"))
    return ParameterRegistry(definitions, igm, nuisance, backend_metadata=metadata)


def make_igm(args, igm_names):
    """Native cup1d IGM histories of a resolved configuration.

    The one construction used by the Cobaya theory
    (:class:`lya_interface.theory.p1d_emulator.P1DEmulatorTheory`) and the
    chain figures (:func:`lya_interface.posterior.native_models`).

    Parameters
    ----------
    args : cup1d.Args
        Resolved native settings (:func:`native_configuration`).
    igm_names : list of str
        Native free IGM coefficients (:func:`native_configuration`).

    Returns
    -------
    lya_interface.igm.IGM
        Static IGM model.
    """
    return IGM(free_param_names=igm_names, pars_igm=args.fid_igm)


def native_blinding(path):
    """
    Resolve the same dataset-dependent diagnostic offsets as native cup1d.

    This reads static data metadata only, never initializes CAMB or Analysis.
    Offsets are applied to exported star diagnostics, not physical predictions.

    Returns
    -------
    cup1d.utils.blinding.Blinding
        Offsets decided by cup1d's own rule (``Blinding.from_data``), a
        read-only mapping with ``blind``/``unblind`` methods.
    """
    from cup1d.utils.blinding import Blinding
    args, _, _ = native_configuration(path)
    # generator: data sets are read until the first one requesting blinding
    return Blinding.from_data(set_p1d(args, name) for name in args.data_label)


def make_backend(path, definitions, covariance_asset=None, fiducial_M=None,
                 include_logdet=False, blinding_policy="reject"):
    """Construct the static cup1d observation/covariance backend.

    Parameters
    ----------
    path : str or pathlib.Path
        Native cup1d YAML configuration.
    definitions : mapping
        Explicit Cobaya public parameter definitions.
    covariance_asset : str or pathlib.Path, optional
        ForestFlow relative emulator-covariance product.
    fiducial_M : mapping
        Frozen ``H(z)/(1+z)`` values in km/s/Mpc for every data redshift.
    include_logdet : bool, default=False
        Include fixed covariance normalization in likelihood values.
    blinding_policy : {'reject', 'native'}, default='reject'
        Whether static native diagnostic blinding is allowed.

    Returns
    -------
    ExternalP1DLikelihood, ParameterRegistry
        Static observation backend and public ownership registry.

    Raises
    ------
    ValueError
        If static data, blinding, conversions, or covariance policy is invalid.
    FileNotFoundError
        If calibrated emulator error is required but no asset is supplied.
    """
    args, igm, nuisance = native_configuration(path)
    registry = registry_for(args, definitions, igm, nuisance)
    if any(is_synthetic_data_label(n) for n in args.data_label):
        raise ValueError("synthetic data require an explicit prepared static dataset, not a hidden native theory")
    data = {n: set_p1d(args, n) for n in args.data_label}
    if blinding_policy not in {"reject", "native"}:
        raise ValueError("blinding_policy must be 'reject' or 'native'")
    if blinding_policy != "native" and any(d.apply_blinding for d in data.values()):
        raise ValueError("blinded data require blinding_policy='native' to preserve native diagnostic blinding")
    zs = sorted({float(z) for d in data.values() for z in d.z})
    if fiducial_M is None or set(map(float, fiducial_M)) != set(zs):
        raise ValueError("supply frozen fiducial_M keyed by every selected data redshift; no implicit CAMB initialization")
    conversions = {float(z): float(m) for z, m in fiducial_M.items()}
    if any(not np.isfinite(m) or m <= 0 for m in conversions.values()):
        raise ValueError("fiducial_M must be finite and positive")
    if covariance_asset is None:
        if np.any(args.cov_factor["val_emu"] != 0):
            raise FileNotFoundError("nonzero emulator error requires an explicit ForestFlow/data/covariance asset")
        # Explicitly disabled calibrated error: zero relative covariance.
        error = dict(zz_zk=np.array(zs), k_Mpc_zk=np.ones(len(zs)), cov_zk=np.zeros((len(zs), len(zs))))
    else:
        error = np.load(Path(covariance_asset).resolve(), allow_pickle=True).item()
    # resolution correction only for these data sets, as native cup1d
    syst_data_labels = getattr(args, "syst_data_labels", None)
    # one fine grid per redshift for data sets with the same redshifts, as native cup1d
    shared_k_grid = bool(getattr(args, "shared_k_grid", True))
    identity = fingerprint(_serializable(dict(native=file_identity(path),
        fiducial_M=conversions, covariance_asset=file_identity(covariance_asset) if covariance_asset else None,
        cov_factor=args.cov_factor, emu_cov_type=args.emu_cov_type, rebin=args.k_rebin_factor,
        fid_cont=args.fid_cont, fid_syst=args.fid_syst, ic_correction=args.ic_correction,
        syst_data_labels=syst_data_labels,
        shared_k_grid=shared_k_grid,
        data={key: dict(z=d.z, k=[k.tolist() for k in d.k_kms], power=[p.tolist() for p in d.Pk_kms],
                       covariance=[c.tolist() for c in d.cov_Pk_kms],
                       full_covariance=None if d.full_cov_Pk_kms is None else d.full_cov_Pk_kms.tolist()) for key,d in data.items()})))
    backend = ExternalP1DLikelihood(data,
        Contaminants(free_param_names=nuisance, pars_cont=args.fid_cont, ic_correction=args.ic_correction),
        Systematics(free_param_names=nuisance, pars_syst=args.fid_syst),
        cov_factor=args.cov_factor, emulator_covariance=error,
        fiducial_conversion=lambda z: conversions[float(z)], emu_cov_type=args.emu_cov_type,
        k_rebin_factor=args.k_rebin_factor, configuration_id=identity,
        include_logdet=include_logdet, syst_data_labels=syst_data_labels,
        shared_k_grid=shared_k_grid)
    return backend, registry


# ----------------------------------------------------------------- projection
def project_request(request, snapshot, model, linear, arinyo, integrator, kmax_iMpc,
                    moments=False, n_k=None):
    """
    Project unique (z, k) requests together, preserving ragged output grids.

    Parameters
    ----------
    request : PredictionRequest
        Dataset groups with redshifts and ragged wavenumber grids in s/km.
    snapshot : CobayaCosmologySnapshot
        Immutable linear power and velocity conversions for the current point.
    model : forestflow.model.arinyo.ArinyoModel
        Native analytic flux-power model.
    linear : forestflow.model.linear.LinearTheoryGrid
        Linear-theory grid associated with the same snapshot.
    arinyo : mapping
        Redshift-indexed dictionaries of physical Arinyo coefficients.
    integrator : forestflow.statistics.p1d.P1DIntegrator
        Native transverse quadrature and integration geometry.
    kmax_iMpc : float
        Maximum admitted parallel wavenumber in inverse Mpc.
    moments : bool, default=False
        Also return the Kaiser metal moments, from the same single pass.
    n_k : int, optional
        Project on ``n_k`` log-spaced k per row and interpolate, as cup1d's
        ``n_k_emu`` (``cup1d.theory.projection.project_fused``).

    Returns
    -------
    mapping or tuple
        Immutable dataset-indexed tuples of P1D arrays in km/s, retaining
        the requested redshift ordering and individual grid lengths; with
        ``moments``, the pair ``(p1d, metal_moments)``.

    Raises
    ------
    ModelDomainError
        If parallel wavenumbers exceed the trained P1D coverage.
    FloatingPointError
        If the projected power is non-finite or non-positive.

    Notes
    -----
    Padding repeats each row's last valid k, never zero or an out-of-range
    sentinel. It is removed before returning immutable provider products.
    The single native P1D projection integrates only along the transverse
    axis; parameter arrays have explicit singleton k and transverse axes.
    """
    from cup1d.theory.projection import project_fused
    try:
        powers, metal_moments = project_fused(request, snapshot, arinyo,
                                              kmax_iMpc=kmax_iMpc, integrator=integrator,
                                              model=model, linear=linear,
                                              moments=moments, n_k=n_k)
        return (powers, metal_moments) if moments else powers
    except NumericalCoverageError:
        # A provider-coverage failure is distinct from a rejected emulator
        # point and must remain visible to callers instead of being converted
        # to a likelihood-domain rejection.
        raise
    except ValueError as error:
        raise ModelDomainError(str(error)) from error


# ------------------------------------------------------------ star parameters
def blind_star_parameters(blinding, stars):
    """Star parameters in public names with the native blinding offsets added.

    Parameters
    ----------
    blinding : cup1d.utils.blinding.Blinding
        Native offsets (:func:`native_blinding`).
    stars : mapping
        Values keyed by public names (``Delta2star``, ``nstar``, ``alphastar``).

    Returns
    -------
    dict
        The same keys, blinded through ``blinding.blind`` (cup1d's offsets
        and conventions; no offsets are added here).
    """
    native = blinding.blind({DERIVED_ALIASES[name]: value for name, value in stars.items()})
    return {name: native[DERIVED_ALIASES[name]] for name in stars}


def star_parameters(snapshot, z_star=3., k_star_ikms=.009):
    """Compute public linear-power star parameters from a snapshot.

    Parameters
    ----------
    snapshot : lya_interface.cosmology.CobayaCosmologySnapshot
        Immutable point-specific cosmology.
    z_star : float, default=3.0
        Pivot redshift.
    k_star_ikms : float, default=0.009
        Velocity-space pivot wavenumber in s/km.

    Returns
    -------
    dict
        Public linear ``bc`` power summaries at ``(z_star, k_star_ikms)``:
        ``Delta2star`` is ``k**3 P(k)/(2*pi**2)``, ``nstar`` is the local
        logarithmic slope, and ``alphastar`` is the running returned by
        LaCE's finite-window quadratic log-power fit. These are not literal
        point derivatives.
    """
    native = snapshot.get_linP_kms_params(z_star, k_star_ikms)
    return {public: native[backend] for public, backend in DERIVED_ALIASES.items()}
