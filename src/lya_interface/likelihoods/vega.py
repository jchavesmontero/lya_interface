"""Vega correlation-function adapter using shared ForestFlow coefficients."""
import inspect
from types import MappingProxyType

import numpy as np
from cobaya.likelihood import Likelihood


class VegaLikelihood(Likelihood):
    """Evaluate Vega data chi-square with a shared ForestFlow correction.

    The native Vega template remains fixed.  ForestFlow supplies only the
    bias-free nonlinear factor at Vega's configured effective redshift; this
    is a documented hybrid, not a cosmology-consistent replacement template.
    """
    native_config: str | None = None
    parameter_definitions: dict = {}
    parameter_mapping: dict = {}
    velocity_parameters: list = []
    metal_model: str | None = None
    growth_from_cosmology: bool = True
    fixed_native_parameters: dict = {}
    template_h: float | None = None
    use_forestflow: bool = True
    lya_tracer: str = "LYA"
    prior_policy: str = "reject"

    def initialize(self):
        """Load static Vega data, template and configuration once."""
        if not self.native_config:
            raise ValueError("VegaLikelihood requires native_config")
        if self.use_forestflow and (self.template_h is None or self.template_h <= 0):
            raise ValueError("ForestFlow Vega mode requires positive documented template_h")
        from vega.vega_interface import VegaInterface

        self.backend = VegaInterface(self.native_config)
        self.zeff = float(self.backend.fiducial["z_eff"])
        self.mapping = dict(self.parameter_mapping)
        # A public parameter may feed several native ones (e.g. one SiII
        # decorrelation for the three SiII lines).
        self._targets = {public: (native,) if isinstance(native, str) else tuple(native)
                         for public, native in self.mapping.items()}
        unknown = set(self.velocity_parameters) - set(self.mapping)
        if unknown:
            raise ValueError(f"velocity_parameters not in parameter_mapping: {sorted(unknown)}")
        # Velocity (km/s) to Vega's comoving Mpc/h, with the fixed fiducial
        # cosmology of the data coordinates: (1 + z_fid) D_H(z_fid) / c.
        self.kms_to_Mpch = None
        if self.velocity_parameters:
            fiducial = self.backend.fiducial
            self.kms_to_Mpch = (1 + fiducial["z_fiducial"]) * fiducial["DH"] / 299792.458
        natives = [native for targets in self._targets.values() for native in targets]
        if len(set(natives)) != len(natives):
            raise ValueError("a native Vega parameter is mapped more than once")
        self.fixed_native = dict(self.fixed_native_parameters)
        overlap = set(natives) & set(self.fixed_native)
        if overlap:
            raise ValueError(f"native Vega parameters cannot be both mapped and fixed: {sorted(overlap)}")
        # Kaiser silicon model shared with P1D: Vega's silicon parameters are
        # the cup1d histories (same public parameters) evaluated at z_eff.
        self.metal_parameters = {}
        self._kaiser = None
        if self.metal_model:
            self._setup_metal_model(natives)
        self.params = {name: None for name in self.public_parameters}
        self.native_priors = MappingProxyType(dict(self.backend.priors))
        if self.prior_policy not in {"reject", "external"}:
            raise ValueError("prior_policy must be 'reject' or 'external'")
        if self.native_priors and self.prior_policy == "reject":
            raise ValueError("native Vega priors must be migrated to Cobaya and prior_policy='external'")
        if self.prior_policy == "external":
            missing = (set(self.native_priors) - set(natives)
                       - set(self.fixed_native))
            if missing:
                raise ValueError(f"native Vega priors have no interface mapping: {sorted(missing)}")
            inverse = {native: public for public, targets in self._targets.items()
                       for native in targets}
            absent = []
            for native, prior in self.native_priors.items():
                if native in self.fixed_native:
                    continue
                definition = self.parameter_definitions.get(inverse[native])
                if isinstance(definition, dict) and "prior" in definition:
                    continue
                # A fixed native-prior parameter is permitted only at its
                # Gaussian mean.  Its omitted prior is then a constant and
                # cannot affect a fit; any other fixed value would silently
                # discard a meaningful prior penalty.
                if np.isscalar(definition) and np.isclose(float(definition), float(prior[0])):
                    continue
                absent.append(native)
            for native, value in self.fixed_native.items():
                if native not in self.native_priors:
                    raise ValueError(f"fixed native Vega parameter has no native Gaussian prior: {native}")
                if not np.isclose(float(value), float(self.native_priors[native][0])):
                    raise ValueError(
                        f"fixed native Vega parameter {native} must equal its native Gaussian mean"
                    )
            if absent:
                raise ValueError(
                    "native Vega priors must be sampled in Cobaya or fixed at their native mean: "
                    f"{sorted(absent)}"
                )
        if self.use_forestflow:
            # These Ly-alpha quantities are ForestFlow outputs in coupled
            # mode.  Remove native sampled/default entries rather than
            # relying on later value precedence; ``logp`` supplies bias and
            # bias_eta for this exact point, and Vega derives beta from them.
            owned = (f"bias_{self.lya_tracer}",
                     f"bias_eta_{self.lya_tracer}",
                     f"beta_{self.lya_tracer}")
            for name in owned:
                self.backend.params.pop(name, None)
                self.backend.sample_params["limits"].pop(name, None)

    _SILICON = {"SiIII": "SiIII(1207)", "SiIIa": "SiII(1190)",
                "SiIIb": "SiII(1193)", "SiIIc": "SiII(1260)"}

    def _setup_metal_model(self, natives):
        """Load the cup1d Kaiser silicon model whose histories feed Vega.

        Parameters
        ----------
        natives : list of str
            Native Vega names already set by ``parameter_mapping``.

        Raises
        ------
        ValueError
            If the cup1d configuration does not use the Kaiser metal model, or a
            silicon parameter is also set by ``parameter_mapping``.
        """
        from cup1d.models.contaminants.metals.si_kaiser import FAMILIES
        from cup1d.models.contaminants.model_contaminants import Contaminants
        from lya_interface.adapters.cup1d_backend import native_configuration

        args, _, nuisance = native_configuration(self.metal_model)
        if args.fid_cont.get("metal_model_type") != "Kaiser":
            raise ValueError("metal_model must be a cup1d configuration with metal_model_type: Kaiser")
        self._kaiser = Contaminants(free_param_names=nuisance,
                                    pars_cont=args.fid_cont).metal_models["Si_kaiser"]
        # Same public names as the P1D likelihood (``p1d_<native>``).
        self.metal_parameters = {"p1d_" + name: name for name in nuisance
                                 if name.rsplit("_", 1)[0] in FAMILIES}
        derived = [f"{kind}_{line}" for line in self._SILICON.values()
                   for kind in ("bias", "beta")]
        overlap = set(derived) & set(natives)
        if overlap:
            raise ValueError(f"silicon parameters set twice (metal_model and mapping): {sorted(overlap)}")
        fiducial = self.backend.fiducial
        self.kms_to_Mpch = (1 + fiducial["z_fiducial"]) * fiducial["DH"] / 299792.458

    @property
    def public_parameters(self):
        """Public input parameters: mapped BAO nuisances and shared metal ones."""
        return list(self.mapping) + list(self.metal_parameters)

    def metal_values(self, params_values):
        """Return Vega's silicon parameters from the shared Kaiser histories.

        Parameters
        ----------
        params_values : mapping
            Public values; the shared ``p1d_`` metal parameters are used.

        Returns
        -------
        dict
            ``bias_<line>`` and ``beta_<line>`` at Vega's effective redshift,
            and ``forestflow_metal_decorrelation``: the line-of-sight
            decorrelation of Lya x each line (Mpc/h), which the Vega fork
            applies to Lya x metal correlations only.

        Notes
        -----
        In Vega's estimator convention the metal correlation is evaluated at
        ``z0 = z_eff`` and the amplitude evolution exponents are frozen into the
        metal matrices, so only the values at ``z_eff`` are needed.
        """
        native = {cup1d: float(params_values[public])
                  for public, cup1d in self.metal_parameters.items()}
        values = {}
        decorrelation = {}
        for line, (bias, beta, sigma) in self._kaiser.coefficients(self.zeff, native).items():
            name = self._SILICON[line]
            values[f"bias_{name}"] = bias
            values[f"beta_{name}"] = beta
            decorrelation[name] = sigma * self.kms_to_Mpch
        values["forestflow_metal_decorrelation"] = decorrelation
        return values

    def get_can_support_params(self):
        """Return the interface-owned BAO nuisance and shared metal names."""
        return self.public_parameters

    def get_requirements(self):
        """Request only a coefficient product at Vega's effective redshift."""
        if not self.use_forestflow:
            return {}
        requirements = {"forestflow_coefficients": {"redshifts": [self.zeff]}}
        if self.growth_from_cosmology:
            requirements["lya_cosmology"] = {"redshifts": [self.zeff]}
        return requirements

    def initialize_with_provider(self, provider):
        """Store Cobaya's immutable result provider."""
        self.provider = provider

    def _forestflow_correction(self, coefficients):
        """Evaluate the shared correction on every native Vega grid."""
        from forestflow.model.arinyo import ArinyoModel

        coeff = coefficients[self.zeff]
        corrections = {}
        for name, model in self.backend.models.items():
            core = model.Pk_core
            # Vega templates conventionally use h/Mpc and (Mpc/h)^3.  Convert
            # to the ForestFlow Mpc convention exactly once using provenance,
            # never a sampled H0.
            h = float(self.template_h)
            k_iMpc = h * core.k_grid
            linear_mpc3 = core._pk_fid / h**3
            correction = ArinyoModel.nonlinear_correction(
                linear_mpc3[None, :], k_iMpc[None, :], core.muk_grid, coeff
            )
            # Vega's correlation-function integration grid reaches much
            # higher k than the ForestFlow training use case.  There the
            # physical exponential damping can underflow to an exact zero.
            # Preserve that limiting prediction at machine precision while
            # keeping the strictly-positive factor required by Vega's
            # Ly-alpha square-root handling.
            corrections[name] = np.maximum(correction, np.finfo(float).tiny)
        return corrections

    def _metal_pressure(self, coefficients):
        """Pressure smoothing of one metal leg, ``exp(-(k/k_p)^2 / 2)``.

        Uses ForestFlow's Arinyo ``k_p`` at Vega's effective redshift, as the
        P1D Kaiser metal model, on every Vega (mu, k) grid (k converted from
        h/Mpc with the documented template ``h``).
        """
        k_p = coefficients[self.zeff]["kp"]
        h = float(self.template_h)
        out = {}
        for name, model in self.backend.models.items():
            core = model.Pk_core
            shape = np.broadcast_shapes(core.muk_grid.shape, core.k_grid.shape)
            # Underflow at Vega's highest k is kept at machine precision, as
            # for the ForestFlow nonlinear correction.
            pressure = np.maximum(np.exp(-0.5 * (h * core.k_grid / k_p) ** 2), np.finfo(float).tiny)
            out[name] = np.broadcast_to(pressure, shape)
        return out

    def parameters_for_evaluation(self, params_values):
        """Build native Vega parameters for one interface point.

        This public helper is intended for diagnostics and plotting.  It
        applies the same routing and ForestFlow injection as :meth:`logp`
        without evaluating a likelihood or mutating backend defaults.
        """
        if set(params_values) != set(self.public_parameters):
            raise ValueError("Vega parameter routing mismatch")
        values = dict(self.fixed_native)
        if self._kaiser is not None:
            values.update(self.metal_values(params_values))
        for name, value in params_values.items():
            if name in self.metal_parameters:
                continue
            value = float(value) * (self.kms_to_Mpch if name in self.velocity_parameters else 1.0)
            values.update({native: value for native in self._targets[name]})
        if self.use_forestflow:
            coefficients = self.provider.get_result("forestflow_coefficients")
            if coefficients is None:
                raise ValueError("ForestFlow coefficients are unavailable for Vega evaluation")
            if self.growth_from_cosmology:
                # Growth rate of the sampled cosmology (not the template's), so
                # beta = bias_eta f / bias is the one used by the P1D likelihood.
                snapshot = self.provider.get_result("lya_cosmology")
                if snapshot is not None:  # always present in a real graph
                    values["growth_rate"] = float(
                        np.atleast_1d(snapshot.compute_growth_rate(self.zeff))[0])
            values[f"bias_{self.lya_tracer}"] = coefficients[self.zeff]["bias"]
            values[f"bias_eta_{self.lya_tracer}"] = coefficients[self.zeff]["bias_eta"]
            values["forestflow_dnl"] = self._forestflow_correction(coefficients)
            if self._kaiser is not None:
                values["forestflow_metal_pressure"] = self._metal_pressure(coefficients)
        return values

    def logp(self, _derived=None, **params_values):
        """Return Vega's data-only Gaussian log likelihood.

        Native Gaussian priors are intentionally excluded because their
        migration to Cobaya must be explicit in the interface configuration.
        """
        if set(params_values) != set(self.public_parameters):
            raise ValueError("Vega parameter routing mismatch")
        try:
            values = self.parameters_for_evaluation(params_values)
        except ValueError as error:
            if "coefficients are unavailable" in str(error):
                return -np.inf
            raise
        chi2 = self.data_chi2(values)
        return -0.5 * chi2 if np.isfinite(chi2) else -np.inf

    def data_chi2(self, values):
        """Return Vega's data-only chi-square for native parameter values.

        Vega versions whose ``chi2`` accepts ``include_priors`` are asked to
        omit the Gaussian priors; otherwise (e.g. the vega_fork ``p1d_int``
        branch) the native prior chi-square is subtracted from the total.
        Either way the priors are left to Cobaya.
        """
        if "include_priors" in inspect.signature(self.backend.chi2).parameters:
            return self.backend.chi2(values, include_priors=False)
        return self.backend.chi2(values) - self.backend.compute_prior_chi2(values)
