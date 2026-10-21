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
    metal_alpha_reference: str | dict | None = None

    def initialize(self):
        """Load static Vega data, template and configuration once.

        With ``metal_model`` and ``metal_alpha_reference`` (a P1D fit saved by
        :meth:`lya_interface.fit.Fit.save`, a mapping of public ``metal_``
        values, or a cup1d global IC file), the amplitude exponent ``alpha_<line>`` of each silicon line is
        set once, before Vega builds its metal matrices, to the logarithmic
        slope ``d ln|b| / d ln(1 + z)`` of the cup1d bias history at that
        reference point and Vega's ``z_eff`` (see
        :func:`metal_amplitude_exponents`). Vega freezes these exponents at
        initialization, so they do not follow the sampled point.
        """
        if not self.native_config:
            raise ValueError("VegaLikelihood requires native_config")
        if self.use_forestflow and (self.template_h is None or self.template_h <= 0):
            raise ValueError("ForestFlow Vega mode requires positive documented template_h")
        from vega.vega_interface import VegaInterface

        self.metal_parameters = {}
        self._kaiser = None
        self._sii_x_sii_free = False
        self.metal_alpha = {}
        if self.metal_model:
            self._load_metal_model()
        overrides = None
        if self.metal_alpha_reference is not None:
            if not self.metal_model:
                raise ValueError("metal_alpha_reference requires metal_model")
            configured = set(self.fixed_native_parameters)
            for native in self.parameter_mapping.values():
                configured |= {native} if isinstance(native, str) else set(native)
            alphas = {f"alpha_{name}" for name in self._SILICON.values()}
            if alphas & configured:
                raise ValueError("alpha_<silicon line> set both by metal_alpha_reference and by "
                                 f"parameter_mapping/fixed_native_parameters: {sorted(alphas & configured)}")
            zeff = main_config_zeff(self.native_config)
            native = self._native_metal_reference(self.metal_alpha_reference)
            exponents = metal_amplitude_exponents(self._kaiser, native, zeff)
            self.metal_alpha = {self._SILICON[line]: alpha for line, alpha in exponents.items()}
            overrides = {f"alpha_{name}": alpha for name, alpha in self.metal_alpha.items()}
            self.log.info("silicon alpha_<line> from the P1D reference at z_eff = %s: %s", zeff,
                          {name: round(alpha, 4) for name, alpha in self.metal_alpha.items()})

        # parameter_overrides is a vega_fork extension: pass it only when needed
        if overrides:
            self.backend = VegaInterface(self.native_config, parameter_overrides=overrides)
        else:
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

    def _load_metal_model(self):
        """Load the cup1d Kaiser silicon model whose histories feed Vega.

        Raises
        ------
        ValueError
            If the cup1d configuration does not use the Kaiser metal model.
        """
        from cup1d.models.contaminants.metals.si_kaiser import FAMILIES
        from cup1d.models.contaminants.model_contaminants import Contaminants
        from lya_interface.adapters.cup1d_backend import native_configuration
        from lya_interface.names import nuisance_public_name

        args, _, nuisance = native_configuration(self.metal_model)
        if args.fid_cont.get("metal_model_type") != "Kaiser":
            raise ValueError("metal_model must be a cup1d configuration with metal_model_type: Kaiser")
        self._metal_args = args
        self._kaiser = Contaminants(free_param_names=nuisance,
                                    pars_cont=args.fid_cont).metal_models["Si_kaiser"]
        # Same public parameters as the P1D likelihood (``metal_<native>``).
        self.metal_parameters = {nuisance_public_name(name): name for name in nuisance
                                 if name.rsplit("_", 1)[0] in FAMILIES}
        # The SiII x SiII factor and decorrelation are passed to Vega only when free in P1D
        self._sii_x_sii_free = any(name.rsplit("_", 1)[0] in ("f_SiIIxSiII", "sig_SiIIxSiII")
                                   for name in self.metal_parameters.values())

    def _native_metal_point(self, point):
        """Return the native cup1d metal values of a public point.

        Parameters
        ----------
        point : mapping
            Public parameter values; it must hold every shared silicon bias
            (``metal_b_<line>_<i>``), the only histories the exponents depend
            on. Other shared metal parameters it lacks (e.g. the SiII x SiII
            factor and decorrelation of a fit saved before they were free) are
            left out, so they keep their fiducial values.

        Returns
        -------
        dict
            Native cup1d name to value.

        Raises
        ------
        ValueError
            If a shared silicon bias is missing.
        """
        missing = sorted(set(self.metal_parameters) - set(point))
        biases = [name for name in missing if self.metal_parameters[name].startswith("b_")]
        if biases:
            raise ValueError(f"metal_alpha_reference lacks the shared silicon biases {biases}")
        if missing:
            self.log.info("metal_alpha_reference: %s not in the reference point; fiducial values",
                          missing)
        return {native: float(point[public]) for public, native in self.metal_parameters.items()
                if public in point}

    def metal_alpha_at(self, point):
        """Return the silicon exponents ``alpha_<line>`` implied by a public point.

        Compare them with :attr:`metal_alpha`, the exponents frozen in Vega's
        metal matrices, to check that a best fit is consistent with them.

        Parameters
        ----------
        point : mapping
            Public parameter values with every shared silicon bias.

        Returns
        -------
        dict
            Vega line name (``SiIII(1207)``, ...) to exponent at Vega's ``z_eff``.
        """
        exponents = metal_amplitude_exponents(self._kaiser, self._native_metal_point(point), self.zeff)
        return {self._SILICON[line]: alpha for line, alpha in exponents.items()}

    def _native_metal_reference(self, reference):
        """Return the native cup1d metal values of ``metal_alpha_reference``.

        Parameters
        ----------
        reference : str, pathlib.Path or mapping
            A :meth:`lya_interface.fit.Fit.save` result or a mapping of public
            values (see :func:`reference_point`), or a cup1d global IC file
            (:func:`lya_interface.initial_conditions.save_global_ic`), which is
            interpolated at the redshift nodes of the metal model. Shared metal
            families absent from the IC file (e.g. silicon betas fixed when it
            was written) keep their fiducial values.

        Returns
        -------
        dict
            Native cup1d name to value.
        """
        if not isinstance(reference, dict):
            from pathlib import Path
            from lya_interface.initial_conditions import ic_native_values, is_global_ic

            content = np.load(Path(reference).expanduser(), allow_pickle=True).item()
            if is_global_ic(content):
                native, missing = ic_native_values(self._metal_args, self.metal_parameters.values(),
                                                   content)
                if missing:
                    self.log.info("metal_alpha_reference: %s not in the IC file; fiducial values",
                                  missing)
                return native
        return self._native_metal_point(reference_point(reference))

    def _setup_metal_model(self, natives):
        """Check the silicon routing to Vega and set the velocity conversion.

        Parameters
        ----------
        natives : list of str
            Native Vega names already set by ``parameter_mapping``.

        Raises
        ------
        ValueError
            If a silicon parameter is also set by ``parameter_mapping``.
        """
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
            Public values; the shared ``metal_`` parameters are used.

        Returns
        -------
        dict
            ``bias_<line>`` and ``beta_<line>`` at Vega's effective redshift,
            and ``forestflow_metal_decorrelation``: the line-of-sight
            decorrelation of Lya x each line (Mpc/h), which the Vega fork
            applies to Lya x metal correlations only. When the P1D model frees
            the SiII x SiII amplitude factor and decorrelation (``f_SiIIxSiII``,
            ``sig_SiIIxSiII``), also ``forestflow_metal_sii_factor`` and
            ``forestflow_metal_sii_decorrelation`` (Mpc/h), which the Vega fork
            applies to the SiII autos and SiII x SiII crosses only; otherwise
            they are left unset, so Vega's SiII x SiII terms follow the biases.

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
        if self._sii_x_sii_free:
            values["forestflow_metal_sii_factor"] = float(
                self._kaiser.get_value("f_SiIIxSiII", self.zeff, native))
            values["forestflow_metal_sii_decorrelation"] = float(
                self._kaiser.get_value("sig_SiIIxSiII", self.zeff, native)) * self.kms_to_Mpch
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


def reference_point(reference):
    """Return the public parameter values of a reference point.

    Parameters
    ----------
    reference : str, pathlib.Path or mapping
        A result file written by :meth:`lya_interface.fit.Fit.save` (its best-fit
        point is used), or a mapping of public parameter values. Names from
        before the ``metal_``/``xi3d_`` prefixes are translated
        (:func:`lya_interface.names.translate_legacy`).

    Returns
    -------
    dict
        Public parameter name to value.

    Raises
    ------
    ValueError
        If the file holds no best-fit point.
    """
    from lya_interface.names import translate_legacy

    if isinstance(reference, dict):
        return translate_legacy(reference)
    from pathlib import Path

    content = np.load(Path(reference).expanduser(), allow_pickle=True).item()
    if "best_fit" in content:
        return translate_legacy(content["best_fit"]["point"])
    raise ValueError(f"{reference} is not a Fit result file (no best_fit point)")


def main_config_zeff(native_config):
    """Return the effective redshift ``zeff`` of a Vega main config, without building Vega.

    Parameters
    ----------
    native_config : str
        Vega main INI, resolved as Vega does.

    Returns
    -------
    float
        ``zeff`` of its ``[data sets]`` section.
    """
    import configparser

    from vega import utils

    config = configparser.ConfigParser()
    config.optionxform = lambda option: option
    config.read(utils.find_file(native_config))
    return config["data sets"].getfloat("zeff")


def metal_amplitude_exponents(kaiser, native, z, dz=1e-3):
    """Return the amplitude exponent of each silicon line, Vega's ``b ~ (1 + z)**alpha``.

    The exponent is the logarithmic slope of the cup1d bias history,
    ``alpha = d ln|b| / d ln(1 + z)``, by a central difference at ``z``. In
    both codes the growth of the matter perturbations is applied separately, so
    the exponents describe the same quantity; they agree with the cup1d history
    to first order around ``z``.

    Parameters
    ----------
    kaiser : cup1d.models.contaminants.metals.si_kaiser.SiKaiser
        Kaiser silicon model with the cup1d histories.
    native : mapping
        Native cup1d coefficient values (``b_<line>_<i>``, ...).
    z : float
        Redshift of the slope (Vega's effective redshift).
    dz : float, default=1e-3
        Half step of the central difference.

    Returns
    -------
    dict
        cup1d line name (``SiIII``, ``SiIIa``, ...) to exponent.
    """
    upper = kaiser.coefficients(z + dz, native)
    lower = kaiser.coefficients(z - dz, native)
    step = np.log((1 + z + dz) / (1 + z - dz))
    return {line: float(np.log(abs(upper[line][0]) / abs(lower[line][0])) / step)
            for line in upper}
