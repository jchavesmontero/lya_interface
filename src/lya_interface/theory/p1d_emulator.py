"""Cobaya theory of the P1D emulators of cup1d: ForestFlow (Arinyo + projection) or LaCE (direct P1D)."""
from types import MappingProxyType
import numpy as np
from cobaya.theory import Theory
from lya_interface.cup1d_bridge import (ModelDomainError, PredictionContext, PredictionRequest,
                                        make_igm, native_configuration, project_request,
                                        registry_for)
from lya_interface.parameters import route


DIAGNOSTIC_QUANTITIES = ("bias", "bias_eta", "beta", "mean_flux", "tau_eff",
                         "T0_K", "gamma", "sigT_kms", "kF_kms", "Delta2_p", "n_p")


def diagnostic_column_name(quantity, redshift):
    """Return a collision-free scalar derived-column name for a redshift.

    The exact IEEE-754 hexadecimal representation, rather than a rounded
    decimal, makes distinct requested redshifts distinct output columns.
    """
    token = float(redshift).hex().replace("-", "m").replace(".", "p")
    return f"{quantity}_z_{token}"


class P1DEmulatorTheory(Theory):
    """Provide deterministic raw-P1D products to Cobaya graphs.

    ``emulator_label`` is any label of :func:`cup1d.emulator.factory.set_emulator`:
    a ForestFlow bundle (``forest_mpg_cinn``, ``forest_mpg_gp``: emulated
    Arinyo coefficients projected to P1D) or a LaCE Gaussian process
    (``lace_mpg``: P1D emulated directly). The emulator, its inputs
    (:func:`cup1d.theory.emulator_inputs.emulator_inputs` with the Cobaya
    cosmology snapshot), the Arinyo coefficients
    (:meth:`cup1d.emulator.factory.Emulator.arinyo`, one latent block for
    every row) and the training hull are those of native cup1d. With LaCE
    there are no Arinyo coefficients: the coefficient and Arinyo products are
    ``None`` and the bias diagnostics NaN.
    """
    stop_at_error: bool = True
    native_config: str | None = None
    parameter_definitions: dict = {}
    emulator_label: str = "forest_mpg_cinn"
    # cINN latent sampling (ForestFlow's P1DEmulator options); None: the
    # bundle default number of realizations, as native cup1d
    realizations: int | None = None
    seed: int = 0
    projection: dict = {}
    # as native cup1d's Theory: hull of the training points expanded by
    # 1 + hull_margin about their mean (cup1d.emulator.hull.Hull)
    hull_margin: float = 0.05
    enforce_training_hull: bool = True
    cosmology_domain: dict = {}
    derived_redshifts: list = []
    n_k_projection: int | None = None

    def initialize(self):
        """Load IGM models, the cup1d emulator and the native P1D integrator.

        Raises
        ------
        ValueError
            If configuration, emulator options, domain definitions, or
            emulator inputs are incompatible.
        """
        from cup1d.emulator.factory import FORESTFLOW_LABELS, set_emulator
        if not self.native_config or not self.parameter_definitions:
            raise ValueError("native_config and explicit parameter_definitions are required")
        if self.realizations is not None and (not isinstance(self.realizations, int) or self.realizations < 1):
            raise ValueError("realizations must be a positive integer (or null: the bundle default)")
        native_args, igm_names, nuisance = native_configuration(self.native_config)
        self.igm = make_igm(native_args, igm_names)
        self.registry = registry_for(native_args, self.parameter_definitions, igm_names, nuisance)
        # The Kaiser metal model of the P1D likelihood needs k_perp moments
        # from the same projection; compute them only when it is selected.
        self.metal_moments = native_args.fid_cont.get("metal_model_type") == "Kaiser"
        # As native cup1d's n_k_emu: project on this many log-spaced k per
        # redshift and interpolate to the rebinning grid (None: every k).
        self.n_k = (self.n_k_projection if self.n_k_projection is not None
                    else getattr(native_args, "n_k_emu", None))
        self.mapping = self.registry.mapping("IGM")
        if not {"H0", "ombh2", "omch2", "mnu", "nnu", "omk", "nrun"} <= set(self.cosmology_domain):
            raise ValueError("declare the admitted background, radiation, neutrino and running domain explicitly")
        for name, bounds in self.cosmology_domain.items():
            if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
                raise ValueError(f"invalid supported domain for {name}")
        self.params = {k: None for k in self.mapping}
        # the emulator of native cup1d; the cINN latent-sampling options
        # reach ForestFlow's P1DEmulator (sampling_options)
        options = {}
        if self.emulator_label in FORESTFLOW_LABELS:
            options = dict(seed=self.seed, Nrealizations=self.realizations)
        self.set_emulator(set_emulator(self.emulator_label, **options))
        if self.metal_moments and not self.emulator.forestflow:
            raise ValueError("the Kaiser metal model needs a ForestFlow emulator")
        from cup1d.theory.projection import make_integrator
        # ``rule: k_mu`` selects the split Gauss-Legendre ln k rule (cup1d's
        # native default); otherwise the ln k_perp rule.
        self.integrator = make_integrator(self.projection)
        self.derived_redshifts = tuple(float(z) for z in self.derived_redshifts)
        if len(set(self.derived_redshifts)) != len(self.derived_redshifts):
            raise ValueError("derived_redshifts must be unique")
        self.calls = 0
        self.request = None

    def set_emulator(self, emulator):
        """Use a cup1d emulator (or a model wrapped by ``as_emulator``).

        Sets the emulator inputs, pivot, maximum ``k_parallel`` and redshift
        coverage.

        Parameters
        ----------
        emulator : cup1d.emulator.factory.Emulator or object
            Emulator; another object is wrapped by
            :func:`cup1d.emulator.factory.as_emulator`.
        """
        from cup1d.emulator.factory import as_emulator
        self.emulator = as_emulator(emulator)
        self.input_labels = list(self.emulator.emu_params)
        required = {"Delta2_p", "n_p", "mF", "gamma", "sigT_Mpc", "kF_Mpc"}
        if not required <= set(self.input_labels) or set(self.input_labels) - required - {"alpha_p"}:
            raise ValueError("unsupported emulator input labels")
        self.kp_iMpc = float(self.emulator.kp_Mpc)
        # ForestFlow: P1D fitting cut of the bundle; LaCE: its k range
        self.kmax_iMpc = float(self.emulator.kmax_1d_iMpc if self.emulator.forestflow
                               else self.emulator.kmax_Mpc)
        self.zmax = float(getattr(self.emulator, "zmax", 4.6))

    def get_can_support_params(self):
        """Declare public IGM coefficient names.

        Returns
        -------
        list of str
            Registry-mapped IGM parameters accepted by this component.
        """
        return list(self.mapping)

    def get_can_provide(self):
        """Declare raw P1D and Arinyo provider products.

        Returns
        -------
        list of str
            ``forestflow_p1d`` and ``forestflow_arinyo``.
        """
        return ["forestflow_p1d", "forestflow_arinyo", "forestflow_coefficients",
                "forestflow_diagnostics"]

    def get_can_provide_params(self):
        """Declare stable scalar diagnostics requested in configuration."""
        return [diagnostic_column_name(quantity, z) for z in self.derived_redshifts
                for quantity in DIAGNOSTIC_QUANTITIES]

    def derived_output_mapping(self):
        """Return column-to-quantity/redshift/unit provenance for saved output."""
        units = {"bias": "dimensionless", "bias_eta": "dimensionless",
                 "beta": "dimensionless", "mean_flux": "dimensionless",
                 "tau_eff": "dimensionless", "T0_K": "K", "gamma": "dimensionless",
                 "sigT_kms": "km/s", "kF_kms": "s/km", "Delta2_p": "dimensionless",
                 "n_p": "dimensionless"}
        return {diagnostic_column_name(quantity, z): {"quantity": quantity,
                "redshift": z, "units": units[quantity]}
                for z in self.derived_redshifts for quantity in DIAGNOSTIC_QUANTITIES}

    def get_requirements(self):
        """Request cosmology values used for explicit domain admission.

        Returns
        -------
        dict
            Required background, radiation, neutrino, and running parameters.
        """
        return {k: None for k in self.cosmology_domain}

    def must_provide(self, **requirements):
        """Install one immutable prediction request and the training hull.

        Parameters
        ----------
        **requirements
            Cobaya requests containing serialized ``forestflow_p1d`` grids.

        Returns
        -------
        dict
            Required immutable Lya cosmology redshifts.

        Raises
        ------
        ValueError
            If requests conflict or request no redshift.
        """
        super().must_provide(**requirements)
        coefficient_request = requirements.get("forestflow_coefficients", {})
        coefficient_redshifts = tuple(float(z) for z in coefficient_request.get("redshifts", ()))
        if "forestflow_p1d" in requirements:
            request = PredictionRequest.from_dict(requirements["forestflow_p1d"])
            if self.request and request.identity != self.request.identity:
                raise ValueError("multiple conflicting prediction requests; combine data groups in one likelihood")
            self.request = request
        p1d_redshifts = () if self.request is None else self.request.redshifts
        self.redshifts = tuple(sorted(set(p1d_redshifts) | set(coefficient_redshifts)
                                     | set(self.derived_redshifts)))
        if not self.redshifts:
            raise ValueError("the P1D emulator theory requires coefficient or P1D redshifts")
        if self.enforce_training_hull:
            from cup1d.emulator.hull import Hull
            from cup1d.emulator.utils_sims import get_training_hc
            # the construction of native cup1d (Theory.__init__, set_fid_cosmo)
            points = get_training_hc(self.emulator.suite)[1]
            self.hull = Hull(zs=np.array(self.redshifts), data_hull=points,
                             suite=self.emulator.suite, extra_factor=1.0 + self.hull_margin)
        return {"lya_cosmology": {"redshifts": list(self.redshifts)}}

    def initialize_with_provider(self, provider):
        """Store the Cobaya provider supplying immutable cosmology snapshots.

        Parameters
        ----------
        provider : cobaya.theory.Provider
            Provider from which ``lya_cosmology`` is retrieved per point.
        """
        self.provider = provider

    def calculate(self, state, want_derived=True, **params_values):
        """Evaluate the emulator inputs, Arinyo coefficients (ForestFlow) and raw P1D.

        Parameters
        ----------
        state : dict
            Cobaya state updated with immutable ``forestflow_p1d`` and
            ``forestflow_arinyo`` products.
        want_derived : bool, default=True
            Accepted Cobaya flag; this component exports no derived values.
        **params_values
            Physical cosmology and public IGM coefficient values.

        Notes
        -----
        Inputs are admitted against explicit cosmology bounds, redshift
        coverage, physical IGM constraints, and optionally the native hull.
        The Arinyo coefficients come from ``emulator.arinyo`` (as native
        cup1d): a cINN draws latent block zero for every redshift, so the
        output is invariant to request ordering and batching. Domain failures
        produce a valid immutable rejection product instead of invoking a
        second CAMB calculation.
        """
        from forestflow.model.arinyo import ArinyoModel
        from forestflow.model.linear import LinearTheoryGrid
        snapshot = self.provider.get_result("lya_cosmology")
        try:
            for name, bounds in self.cosmology_domain.items():
                value = params_values[name]
                if not bounds[0] <= value <= bounds[1]:
                    raise ModelDomainError(f"{name} outside explicitly admitted cosmology domain")
            # union of the P1D, coefficient and diagnostic redshifts
            zs = self.redshifts
            if min(zs) < 2. or max(zs) > self.zmax:
                raise ModelDomainError("redshift outside MPG bundle coverage")
            native = route({k: v for k, v in params_values.items() if k not in self.cosmology_domain}, self.mapping)
            from cup1d.theory.emulator_inputs import emulator_inputs
            z_array = np.array(zs)
            M = snapshot.get_dkms_diMpc(z_array)
            # the same mapping as native cup1d (Theory.get_emulator_calls)
            columns = emulator_inputs(self.input_labels, z_array,
                                      [snapshot.get_linP_Mpc_params(z, self.kp_iMpc) for z in zs],
                                      M, self.igm, like_params=native)
            inputs = []
            for i in range(len(zs)):
                row = {k: float(columns[k][i]) for k in self.input_labels}
                if not all(np.isfinite(v) for v in row.values()) or not 0 < row["mF"] < 1 or any(row[k] <= 0 for k in ("Delta2_p", "gamma", "sigT_Mpc", "kF_Mpc")):
                    raise ModelDomainError("nonphysical IGM/emulator input")
                inputs.append(row)
            flux = columns["mF"]
            gamma = columns["gamma"]
            thermal = columns["sigT_Mpc"] * M     # km/s
            pressure = columns["kF_Mpc"] / M      # s/km
            if self.enforce_training_hull and not self.hull.in_hulls(
                    np.column_stack([columns[k] for k in self.hull.params])):
                raise ModelDomainError("outside native training pairwise hull")
            arinyo = None
            if self.emulator.forestflow:
                ari = self.emulator.arinyo({k: columns[k] for k in self.input_labels})
                arinyo = {z: MappingProxyType({k: float(v[i]) for k, v in ari.items()})
                          for i, z in enumerate(zs)}
            self.calls += 1
            diagnostics = {}
            get_t0 = getattr(self.igm.models["T_model"], "get_T0", None)
            t0 = (get_t0(z_array, like_params=native) if get_t0 else np.full(len(zs), np.nan))
            for index, z in enumerate(zs):
                coeff = arinyo[z] if arinyo is not None else None
                bias = coeff["bias"] if coeff is not None else np.nan
                bias_eta = coeff["bias_eta"] if coeff is not None else np.nan
                beta = np.nan if (coeff is None or bias == 0) else snapshot.get_growth_rate(z) * bias_eta / bias
                diagnostics[z] = MappingProxyType(dict(
                    z=float(z), mean_flux=float(flux[index]),
                    tau_eff=float(-np.log(flux[index])), gamma=float(gamma[index]),
                    sigT_kms=float(thermal[index]), kF_kms=float(pressure[index]),
                    T0_K=float(t0[index]), Delta2_p=float(inputs[index]["Delta2_p"]),
                    n_p=float(inputs[index]["n_p"]), dkms_diMpc=float(M[index]),
                    bias=float(bias), bias_eta=float(bias_eta),
                    beta=float(beta), arinyo=coeff,
                ))
            state["forestflow_coefficients"] = MappingProxyType(arinyo) if arinyo is not None else None
            state["forestflow_diagnostics"] = MappingProxyType(diagnostics)
            moments = None
            if self.request is None:
                powers = None
            elif not self.emulator.forestflow:
                from cup1d.theory.projection import emulate_request
                try:
                    powers = emulate_request(self.request, snapshot, self.emulator,
                                             dict(zip(zs, inputs)), kmax_iMpc=self.kmax_iMpc,
                                             n_k=self.n_k)
                except ValueError as error:
                    raise ModelDomainError(str(error)) from error
            else:
                model = ArinyoModel(fiducial_cosmology=snapshot)
                linear = LinearTheoryGrid(z=z_array, cosmology=snapshot)
                # One pass for the Lya P1D and, with the Kaiser metal model,
                # its metal moments; optionally on a coarse k grid.
                projected = project_request(
                    self.request, snapshot, model, linear, arinyo, self.integrator,
                    self.kmax_iMpc,
                    moments=self.metal_moments, n_k=self.n_k,
                )
                powers, moments = projected if self.metal_moments else (projected, None)
            if powers is not None:
                context = PredictionContext(self.request.identity, zs, tuple(flux), tuple(M))
                state["forestflow_p1d"] = MappingProxyType(dict(valid=True, powers=powers, context=context,
                    inputs=tuple(MappingProxyType(r) for r in inputs), cosmology_id=snapshot.identity,
                    metal_moments=moments))
            else:
                state["forestflow_p1d"] = MappingProxyType(dict(valid=True, powers=None, context=None,
                    inputs=tuple(MappingProxyType(r) for r in inputs), cosmology_id=snapshot.identity))
            state["forestflow_arinyo"] = MappingProxyType(arinyo) if arinyo is not None else None
        except ModelDomainError as error:
            state["forestflow_p1d"] = MappingProxyType(dict(valid=False, reason=str(error)))
            state["forestflow_arinyo"] = None
            state["forestflow_coefficients"] = None
            state["forestflow_diagnostics"] = None
        state["derived"] = ({column: diagnostics[z][quantity]
                             for z in self.derived_redshifts
                             for quantity in DIAGNOSTIC_QUANTITIES
                             for column in [diagnostic_column_name(quantity, z)]}
                            if want_derived and state["forestflow_diagnostics"] is not None else {})
