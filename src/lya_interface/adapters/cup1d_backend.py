"""Static native configuration loading; deliberately no Analysis or emulator."""
from pathlib import Path
import numpy as np
from cup1d import Args
from cup1d.likelihood.parameters import set_free_likelihood_parameters
from cup1d.models.contaminants.model_contaminants import Contaminants
from cup1d.models.contaminants.model_systematics import Systematics
from cup1d.models.igm.model_igm import IGM
from cup1d.p1ds.factory import set_p1d, is_synthetic_data_label
from cup1d.likelihood.external import ExternalP1DLikelihood
from lya_interface.parameters import ParameterRegistry
from lya_interface.provenance import file_identity
from cup1d.likelihood.external import fingerprint


def _serializable(value):
    if isinstance(value, dict):
        return {k: _serializable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def native_configuration(path):
    args = Args.from_yaml(str(Path(path).resolve()), verbose=False)
    if args.use_star_priors or getattr(args, "prior_Gauss_rms", None) is not None or getattr(args, "Gauss_priors", None) or args.fid_igm.get("Gauss_priors") or args.fid_cont.get("Gauss_priors") or args.fid_syst.get("Gauss_priors"):
        raise ValueError("export native statistical priors explicitly to Cobaya; implicit native priors are unsupported")
    names = set_free_likelihood_parameters(args, emulator_label=args.emulator_label)
    igm = [n for n in names if n.rsplit("_", 1)[0] in args.igm_params]
    nuisance = [n for n in names if n not in igm and n not in {"As", "ns", "nrun"}]
    return args, igm, nuisance


def registry_for(args, definitions, igm, nuisance):
    metadata = {}
    for name in igm + nuisance:
        family = name.rsplit("_", 1)[0]
        section = args.fid_igm if name in igm else args.fid_syst if family in args.syst_params else args.fid_cont
        metadata[name] = dict(units="dimensionless coefficient",
            transformation=f"native {section[family+'_ztype']} / {section[family+'_otype']} history" +
            ("; multiplies fiducial physical history" if name in igm else "; enters native contaminant model"))
    return ParameterRegistry(definitions, igm, nuisance, backend_metadata=metadata)


def make_igm(path, definitions):
    args, igm, nuisance = native_configuration(path)
    registry = registry_for(args, definitions, igm, nuisance)
    return IGM(free_param_names=igm, pars_igm=args.fid_igm), registry


def make_backend(path, definitions, *, covariance_asset=None, fiducial_M=None,
                 include_logdet=False):
    args, igm, nuisance = native_configuration(path)
    registry = registry_for(args, definitions, igm, nuisance)
    if any(is_synthetic_data_label(n) for n in args.data_label):
        raise ValueError("synthetic data require an explicit prepared static dataset, not a hidden native theory")
    data = {n: set_p1d(args, n) for n in args.data_label}
    if any(d.apply_blinding for d in data.values()):
        raise ValueError("provider-derived cosmological diagnostics are unblinded: use an explicitly authorized unblinded analysis, not a blinded native dataset")
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
    identity = fingerprint(_serializable(dict(native=file_identity(path),
        fiducial_M=conversions, covariance_asset=file_identity(covariance_asset) if covariance_asset else None,
        cov_factor=args.cov_factor, emu_cov_type=args.emu_cov_type, rebin=args.k_rebin_factor,
        fid_cont=args.fid_cont, fid_syst=args.fid_syst, ic_correction=args.ic_correction,
        data={key: dict(z=d.z, k=[k.tolist() for k in d.k_kms], power=[p.tolist() for p in d.Pk_kms],
                       covariance=[c.tolist() for c in d.cov_Pk_kms],
                       full_covariance=None if d.full_cov_Pk_kms is None else d.full_cov_Pk_kms.tolist()) for key,d in data.items()})))
    backend = ExternalP1DLikelihood(data,
        Contaminants(free_param_names=nuisance, pars_cont=args.fid_cont, ic_correction=args.ic_correction),
        Systematics(free_param_names=nuisance, pars_syst=args.fid_syst),
        cov_factor=args.cov_factor, emulator_covariance=error,
        fiducial_conversion=lambda z: conversions[float(z)], emu_cov_type=args.emu_cov_type,
        k_rebin_factor=args.k_rebin_factor, configuration_id=identity,
        include_logdet=include_logdet)
    return backend, registry
