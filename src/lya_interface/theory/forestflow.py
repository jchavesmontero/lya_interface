"""Native IGM histories and P1D: ForestFlow (Arinyo + projection) or LaCE (direct P1D)."""
from types import MappingProxyType
import numpy as np
from cobaya.theory import Theory
from lya_interface.adapters.cup1d_backend import make_igm
from lya_interface.contracts import PredictionRequest, PredictionContext, ModelDomainError
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


# Emulators predicting P1D directly (cup1d.emulator.factory labels); the others
# are ForestFlow bundles (Arinyo coefficients projected to P1D).
P1D_EMULATORS = ("lace_mpg",)
FORESTFLOW_EMULATORS = ("forest_mpg_cinn", "forest_mpg_gp")


class ForestFlowTheory(Theory):
    """Provide deterministic raw-P1D products to Cobaya graphs.

    ``emulator_label`` selects a ForestFlow bundle (``forest_mpg_cinn``,
    ``forest_mpg_gp``: emulated Arinyo coefficients projected to P1D) or the
    LaCE Gaussian process (``lace_mpg``: P1D emulated directly). Both take
    their inputs from :func:`cup1d.theory.emulator_inputs.emulator_inputs`
    with the Cobaya cosmology snapshot, as native cup1d does with its own
    cosmology. With LaCE there are no Arinyo coefficients: the coefficient and
    Arinyo products are ``None`` and the bias diagnostics NaN.
    """
    stop_at_error: bool = True
    native_config: str | None = None
    parameter_definitions: dict = {}
    emulator_label: str = "forest_mpg_cinn"
    model_path: str | None = None
    transformation_path: str | None = None
    realizations: int = 1000
    seed: int = 0
    projection: dict = {}
    hull_factor: float = 1.
    enforce_training_hull: bool = True
    cosmology_domain: dict = {}
    derived_redshifts: list = []
    n_k_projection: int | None = None

    def initialize(self):
        """Load IGM models, emulator bundle, and native P1D integrator.

        Raises
        ------
        ValueError
            If configuration, emulator aliases/assets, domain definitions, or
            bundle input/output conventions are incompatible.
        """
        from forestflow.emulator import load_emulator
        if not self.native_config or not self.parameter_definitions:
            raise ValueError("native_config and explicit parameter_definitions are required")
        if not isinstance(self.realizations, int) or self.realizations < 1:
            raise ValueError("realizations must be a positive integer")
        if bool(self.model_path) != bool(self.transformation_path):
            raise ValueError("custom model_path and transformation_path must be supplied together; otherwise the default emulator is used")
        self.igm, self.registry = make_igm(self.native_config, self.parameter_definitions)
        from lya_interface.adapters.cup1d_backend import native_configuration
        # The Kaiser metal model of the P1D likelihood needs k_perp moments
        # from the same projection; compute them only when it is selected.
        native_args = native_configuration(self.native_config)[0]
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
        if self.emulator_label not in FORESTFLOW_EMULATORS + P1D_EMULATORS:
            raise ValueError(f"emulator_label must be one of {FORESTFLOW_EMULATORS + P1D_EMULATORS}")
        self.p1d_emulator = self.emulator_label in P1D_EMULATORS
        if self.p1d_emulator:
            from cup1d.emulator.factory import set_emulator
            if self.model_path:
                raise ValueError("custom model_path/transformation_path apply only to the cINN emulator")
            if self.metal_moments:
                raise ValueError("the Kaiser metal model needs a ForestFlow emulator")
            self.emulator = set_emulator(self.emulator_label)
            self.input_labels = list(self.emulator.emu_params)
            self.kp_iMpc = float(self.emulator.kp_Mpc)
            self.kmax_iMpc = float(self.emulator.kmax_Mpc)
            self.zmax = float(getattr(self.emulator, "zmax", 4.6))
        else:
            if self.emulator_label == "forest_mpg_gp":
                if self.model_path:
                    raise ValueError("custom model_path/transformation_path apply only to the cINN emulator")
                self.emulator = load_emulator("forest_mpg_gp")
            elif self.model_path:  # custom cINN assets
                self.emulator = load_emulator(None, model_path=self.model_path, transf_file=self.transformation_path,
                    Nrealizations=self.realizations, compile_model=True)
            else:
                self.emulator = load_emulator(self.emulator_label, Nrealizations=self.realizations, compile_model=True)
            if set(self.emulator.output_labels) != {"bias", "bias_eta", "q1", "q2", "kvav", "av", "bv", "kp"}:
                raise ValueError("bundle does not provide the supported named Arinyo parameters")
            self.input_labels = list(self.emulator.input_labels)
            self.kp_iMpc = float(self.emulator.kp_iMpc)
            self.kmax_iMpc = float(self.emulator.kmax_1d_iMpc)
            self.zmax = float(self.emulator.zmax)
        required = {"Delta2_p", "n_p", "mF", "gamma", "sigT_Mpc", "kF_Mpc"}
        if not required <= set(self.input_labels) or set(self.input_labels) - required - {"alpha_p"}:
            raise ValueError("unsupported emulator input labels")
        if self.kp_iMpc != .7:
            raise ValueError("native training-hull assets use kp=0.7 /Mpc; a different pivot needs matching domain assets")
        if set(self.emulator.list_sim_cube) != {f"mpg_{i}" for i in range(30)}:
            raise ValueError("this domain bridge requires the full MPG cube; held-out/reduced/other suites need matching training-domain assets")
        from cup1d.likelihood.external import make_integrator
        # ``rule: k_mu`` selects the split Gauss-Legendre ln k rule (cup1d's
        # native default); otherwise the ln k_perp rule.
        self.integrator = make_integrator(self.projection)
        self.derived_redshifts = tuple(float(z) for z in self.derived_redshifts)
        if len(set(self.derived_redshifts)) != len(self.derived_redshifts):
            raise ValueError("derived_redshifts must be unique")
        self.calls = 0
        self.request = None

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
        """Install one immutable prediction request and its training hull.

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
            If requests conflict, lack a P1D contract, or training-domain
            labels do not match the bundle.
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
            raise ValueError("ForestFlow requires coefficient or P1D redshifts")
        from cup1d.utils.utils_sims import get_training_hc
        from cup1d.utils.hull import Hull
        labels, points, _, _ = get_training_hc(self.emulator.list_sim_cube[0][:3], emu_params=self.input_labels)
        if set(labels) != set(self.input_labels):
            raise ValueError("training domain labels differ from model inputs")
        self.domain_labels = labels
        self.hull = Hull(zs=np.array(self.redshifts), data_hull=points,
                         suite=self.emulator.list_sim_cube[0][:3], extra_factor=self.hull_factor)
        return {"lya_cosmology": {"redshifts": list(self.redshifts)}}

    def initialize_with_provider(self, provider):
        """Store the Cobaya provider supplying immutable cosmology snapshots.

        Parameters
        ----------
        provider : cobaya.theory.Provider
            Provider from which ``lya_cosmology`` is retrieved per point.
        """
        self.provider = provider

    def _emulator_settings(self):
        """Fill emulator settings missing in subclasses with their own ``initialize``.

        Such subclasses (e.g. test fixtures) set only ``emulator``: a
        ForestFlow-like bundle exposing ``input_labels``, ``kp_iMpc``,
        ``kmax_1d_iMpc`` and ``zmax``.
        """
        if not hasattr(self, "p1d_emulator"):
            self.p1d_emulator = False
        if not hasattr(self, "input_labels"):
            self.input_labels = list(self.emulator.input_labels)
        if not hasattr(self, "kp_iMpc"):
            self.kp_iMpc = float(self.emulator.kp_iMpc)
        if not hasattr(self, "kmax_iMpc"):
            self.kmax_iMpc = float(self.emulator.kmax_1d_iMpc)
        if not hasattr(self, "zmax"):
            self.zmax = float(self.emulator.zmax)

    def calculate(self, state, want_derived=True, **params_values):
        """Evaluate deterministic Arinyo parameters and raw ForestFlow P1D.

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
        A stable latent block zero is used at every redshift, so output is
        invariant to request ordering and batching. Domain failures produce a
        valid immutable rejection product instead of invoking a second CAMB
        calculation.
        """
        from forestflow.model.arinyo import ArinyoModel
        from forestflow.model.linear import LinearTheoryGrid
        from lya_interface.projection import project_request
        self._emulator_settings()
        snapshot = self.provider.get_result("lya_cosmology")
        try:
            for name, bounds in self.cosmology_domain.items():
                value = params_values[name]
                if not bounds[0] <= value <= bounds[1]:
                    raise ModelDomainError(f"{name} outside explicitly admitted cosmology domain")
            # Third-party/legacy subclasses that implement ``must_provide``
            # themselves historically set only ``request``.  Keep that
            # public extension point working while the native implementation
            # uses the full union of P1D, coefficient, and diagnostic grids.
            zs = getattr(self, "redshifts", self.request.redshifts)
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
            if self.enforce_training_hull and not self.hull.in_hulls(np.array([[row[k] for k in self.domain_labels] for row in inputs])):
                raise ModelDomainError("outside native training pairwise hull")
            if self.p1d_emulator:
                arinyo = None
            # Redshift-independent common random numbers: stable latent block
            # zero for each input, invariant under batching/order/chunking.
            elif getattr(self.emulator, "uses_sampling", True):
                ari = self.emulator.evaluate(inputs, Nrealizations=self.realizations,
                    seed=self.seed, latent_indices=[0]*len(zs))
            else:  # deterministic emulator: no latent draws
                ari = self.emulator.evaluate(inputs)
            self.calls += 1
            if not self.p1d_emulator:
                arinyo = {z: MappingProxyType({k: float(np.atleast_1d(ari[k])[i]) for k in self.emulator.output_labels}) for i, z in enumerate(zs)}
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
            elif self.p1d_emulator:
                from cup1d.likelihood.external import emulate_request
                try:
                    powers = emulate_request(self.request, snapshot, self.emulator,
                                             dict(zip(zs, inputs)), kmax_iMpc=self.kmax_iMpc,
                                             n_k=getattr(self, "n_k", None))
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
                    moments=getattr(self, "metal_moments", False),
                    n_k=getattr(self, "n_k", None),
                )
                powers, moments = projected if getattr(self, "metal_moments", False) else (projected, None)
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
