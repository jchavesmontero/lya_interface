"""Vega correlation-function adapter using shared ForestFlow coefficients."""
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
        self.params = {name: None for name in self.mapping}
        self.native_priors = MappingProxyType(dict(self.backend.priors))
        if self.prior_policy not in {"reject", "external"}:
            raise ValueError("prior_policy must be 'reject' or 'external'")
        if self.native_priors and self.prior_policy == "reject":
            raise ValueError("native Vega priors must be migrated to Cobaya and prior_policy='external'")
        if self.prior_policy == "external":
            missing = set(self.native_priors) - set(self.mapping.values())
            if missing:
                raise ValueError(f"native Vega priors have no interface mapping: {sorted(missing)}")
            inverse = {native: public for public, native in self.mapping.items()}
            absent = [native for native in self.native_priors
                      if not isinstance(self.parameter_definitions.get(inverse[native]), dict)
                      or "prior" not in self.parameter_definitions[inverse[native]]]
            if absent:
                raise ValueError(f"native Vega priors are not explicitly registered in Cobaya: {sorted(absent)}")
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

    def get_can_support_params(self):
        """Return the interface-owned BAO nuisance names."""
        return list(self.mapping)

    def get_requirements(self):
        """Request only a coefficient product at Vega's effective redshift."""
        if not self.use_forestflow:
            return {}
        return {"forestflow_coefficients": {"redshifts": [self.zeff]}}

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
            corrections[name] = ArinyoModel.nonlinear_correction(
                linear_mpc3[None, :], k_iMpc[None, :], core.muk_grid, coeff
            )
        return corrections

    def parameters_for_evaluation(self, params_values):
        """Build native Vega parameters for one interface point.

        This public helper is intended for diagnostics and plotting.  It
        applies the same routing and ForestFlow injection as :meth:`logp`
        without evaluating a likelihood or mutating backend defaults.
        """
        if set(params_values) != set(self.mapping):
            raise ValueError("Vega parameter routing mismatch")
        values = {self.mapping[name]: float(value) for name, value in params_values.items()}
        if self.use_forestflow:
            coefficients = self.provider.get_result("forestflow_coefficients")
            if coefficients is None:
                raise ValueError("ForestFlow coefficients are unavailable for Vega evaluation")
            values[f"bias_{self.lya_tracer}"] = coefficients[self.zeff]["bias"]
            values[f"bias_eta_{self.lya_tracer}"] = coefficients[self.zeff]["bias_eta"]
            values["forestflow_dnl"] = self._forestflow_correction(coefficients)
        return values

    def logp(self, _derived=None, **params_values):
        """Return Vega's data-only Gaussian log likelihood.

        Native Gaussian priors are intentionally excluded because their
        migration to Cobaya must be explicit in the interface configuration.
        """
        if set(params_values) != set(self.mapping):
            raise ValueError("Vega parameter routing mismatch")
        try:
            values = self.parameters_for_evaluation(params_values)
        except ValueError as error:
            if "coefficients are unavailable" in str(error):
                return -np.inf
            raise
        chi2 = self.backend.chi2(values, include_priors=False)
        return -0.5 * chi2 if np.isfinite(chi2) else -np.inf
