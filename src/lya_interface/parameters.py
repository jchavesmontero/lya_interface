"""Explicit public/backend ownership. Statistical priors belong to Cobaya."""
from dataclasses import dataclass
import numpy as np

COSMOLOGY = {"As", "ns", "nrun", "H0", "ombh2", "omch2", "mnu", "tau", "nnu", "omk"}


@dataclass(frozen=True)
class Parameter:
    """Immutable ownership and transformation record for one public parameter.

    Attributes
    ----------
    public, backend : str
        Cobaya-facing name and the corresponding native consumer name.
    owner : str
        Subsystem responsible for the parameter.
    units, transformation, status : str
        Display units, coordinate conversion description, and sampling role.
    definition : object
        Original Cobaya parameter definition.
    consumer : str
        Native component which consumes ``backend``.
    prior, reference, proposal : object, optional
        Sampling metadata copied from the public definition.
    """
    public: str
    backend: str
    owner: str
    units: str
    transformation: str
    status: str
    definition: object
    consumer: str
    prior: object = None
    reference: object = None
    proposal: object = None


class ParameterRegistry:
    """Map public Cobaya names onto cosmology, IGM, and nuisance backends.

    Parameters
    ----------
    definitions : dict
        Public Cobaya parameter definitions.
    igm_names, nuisance_names : sequence of str
        Native cup1d coefficient names owned by each subsystem.
    backend_metadata : mapping, optional
        Units and transformation descriptions keyed by backend name.
    """
    def __init__(self, definitions, igm_names, nuisance_names, backend_metadata=None):
        """Build immutable ownership records for explicit Cobaya parameters.

        Parameters
        ----------
        definitions : mapping
            Public parameter definitions. Every IGM and nuisance coefficient
            must be explicit; a definition with ``prior`` is sampled.
        igm_names, nuisance_names : sequence of str
            Native cup1d names owned by ForestFlow and the P1D likelihood.
        backend_metadata : mapping, optional
            Units and transformations keyed by native parameter name.

        Raises
        ------
        ValueError
            If names conflict, a definition is unused/missing, derived values
            are sampled, or the ``logA``/``As`` transformation is inconsistent.
        """
        self.entries = {}
        if len(set(igm_names + nuisance_names)) != len(igm_names + nuisance_names):
            raise ValueError("duplicate or conflicting backend coefficient names")
        aliases = {"igm_" + n: (n, "IGM", "ForestFlowTheory") for n in igm_names}
        aliases.update({"p1d_" + n: (n, "nuisance", "Cup1DLikelihood") for n in nuisance_names})
        aliases.update({n: (n, "cosmology", "CAMB") for n in COSMOLOGY})
        aliases.update({n: (n, "derived", "LyaCosmology") for n in ("Delta2star", "nstar", "alphastar")})
        aliases["logA"] = ("As", "cosmology", "CAMB")
        if "logA" in definitions and "As" in definitions and isinstance(definitions["As"], dict) and "prior" in definitions["As"]:
            raise ValueError("do not independently sample both logA and As")
        for name, definition in definitions.items():
            if name not in aliases:
                raise ValueError(f"unknown/unused parameter {name!r}")
            native, owner, consumer = aliases[name]
            status = "sampled" if isinstance(definition, dict) and "prior" in definition else "fixed"
            if owner == "derived":
                if status == "sampled":
                    raise ValueError(f"cannot independently sample derived parameter {name}")
                status = "derived"
            units = "dimensionless coefficient" if owner in {"IGM", "nuisance"} else {"H0": "km/s/Mpc", "mnu": "eV"}.get(native, "dimensionless")
            metadata = (backend_metadata or {}).get(native, {})
            settings = definition if isinstance(definition, dict) else {}
            self.entries[name] = Parameter(name, native, owner, metadata.get("units", units),
                "As = 1e-10 * exp(logA)" if name == "logA" else metadata.get("transformation", "identity"),
                status, definition, consumer, settings.get("prior"), settings.get("ref"), settings.get("proposal"))
        for name in aliases:
            if name.startswith(("igm_", "p1d_")) and name not in self.entries:
                raise ValueError(f"missing explicit parameter definition {name}")
        if "logA" in definitions:
            d = definitions.get("As")
            if not isinstance(d, dict) or d.get("value") != "lambda logA: 1e-10 * np.exp(logA)":
                raise ValueError("logA requires explicit exponential As transformation")

    def mapping(self, owner):
        """Return public-to-backend parameter names for one owner.

        Parameters
        ----------
        owner : str
            Ownership category such as ``'IGM'``, ``'nuisance'``, or
            ``'cosmology'``.

        Returns
        -------
        dict
            Public names mapped to native backend names in registry order.
        """
        return {p.public: p.backend for p in self.entries.values() if p.owner == owner}


def route(values, mapping):
    """Route public values into a backend-named parameter dictionary.

    Parameters
    ----------
    values : mapping
        Public parameter values.
    mapping : mapping
        Public-to-backend parameter-name mapping.

    Returns
    -------
    dict
        Backend values for entries present in both mappings.
    """
    if set(values) != set(mapping):
        raise ValueError(f"parameter routing mismatch: expected {sorted(mapping)}, got {sorted(values)}")
    result = {mapping[k]: float(v) for k, v in values.items()}
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("parameters must be finite")
    return result
