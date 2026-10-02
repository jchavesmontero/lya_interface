"""Native IGM histories, deterministic mean Arinyo parameters and P1D."""
from types import MappingProxyType
import numpy as np
from cobaya.theory import Theory
from lya_interface.adapters.cup1d_backend import make_igm
from lya_interface.contracts import PredictionRequest, PredictionContext, ModelDomainError
from lya_interface.cosmology import readonly
from lya_interface.parameters import route


class ForestFlowTheory(Theory):
    stop_at_error: bool = True
    native_config: str | None = None
    parameter_definitions: dict = {}
    model_key: str = "forest_mpg_fix"
    model_path: str | None = None
    transformation_path: str | None = None
    realizations: int = 1000
    seed: int = 0
    projection: dict = {}
    hull_factor: float = 1.
    cosmology_domain: dict = {}

    def initialize(self):
        from forestflow.emulator.p3d_cinn import P3DEmulator
        from forestflow.statistics.p1d import P1DIntegrator
        if not self.native_config or not self.parameter_definitions:
            raise ValueError("native_config and explicit parameter_definitions are required")
        if not isinstance(self.realizations, int) or self.realizations < 1:
            raise ValueError("realizations must be a positive integer")
        if bool(self.model_path) != bool(self.transformation_path):
            raise ValueError("custom model_path and transformation_path must be supplied together; otherwise use model_key")
        self.igm, self.registry = make_igm(self.native_config, self.parameter_definitions)
        self.mapping = self.registry.mapping("IGM")
        if not {"H0", "ombh2", "omch2", "mnu", "nnu", "omk", "nrun"} <= set(self.cosmology_domain):
            raise ValueError("declare the admitted background, radiation, neutrino and running domain explicitly")
        for name, bounds in self.cosmology_domain.items():
            if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
                raise ValueError(f"invalid supported domain for {name}")
        self.params = {k: None for k in self.mapping}
        self.emulator = P3DEmulator(key=None if self.model_path else self.model_key,
            model_path=self.model_path, transf_file=self.transformation_path, Nrealizations=self.realizations)
        required = {"Delta2_p", "n_p", "mF", "gamma", "sigT_Mpc", "kF_Mpc"}
        if not required <= set(self.emulator.input_labels) or set(self.emulator.input_labels) - required - {"alpha_p"}:
            raise ValueError("unsupported ForestFlow input labels")
        if set(self.emulator.output_labels) != {"bias", "bias_eta", "q1", "q2", "kvav", "av", "bv", "kp"}:
            raise ValueError("bundle does not provide the supported named Arinyo parameters")
        if self.emulator.kp_iMpc != .7:
            raise ValueError("native training-hull assets use kp=0.7 /Mpc; a different pivot needs matching domain assets")
        if set(self.emulator.list_sim_cube) != {f"mpg_{i}" for i in range(30)}:
            raise ValueError("this domain bridge requires the full MPG cube; held-out/reduced/other suites need matching training-domain assets")
        self.integrator = P1DIntegrator(**self.projection)
        self.calls = 0
        self.request = None

    def get_can_support_params(self):
        return list(self.mapping)

    def get_can_provide(self):
        return ["forestflow_p1d", "forestflow_arinyo"]

    def get_requirements(self):
        # Provider values constrain admitted cosmology without creating CAMB.
        return {k: None for k in self.cosmology_domain}

    def must_provide(self, **requirements):
        super().must_provide(**requirements)
        if "forestflow_p1d" in requirements:
            request = PredictionRequest.from_dict(requirements["forestflow_p1d"])
            if self.request and request.identity != self.request.identity:
                raise ValueError("multiple conflicting prediction requests; combine data groups in one likelihood")
            self.request = request
            from cup1d.utils.utils_sims import get_training_hc
            from cup1d.utils.hull import Hull
            labels, points, _, _ = get_training_hc(self.emulator.list_sim_cube[0][:3], emu_params=self.emulator.input_labels)
            if set(labels) != set(self.emulator.input_labels):
                raise ValueError("training domain labels differ from model inputs")
            self.domain_labels = labels
            self.hull = Hull(zs=np.array(request.redshifts), data_hull=points,
                             suite=self.emulator.list_sim_cube[0][:3], extra_factor=self.hull_factor)
        if self.request is None:
            raise ValueError("ForestFlow projection requires a static cup1d request")
        return {"lya_cosmology": {"redshifts": list(self.request.redshifts)}}

    def initialize_with_provider(self, provider):
        self.provider = provider

    def calculate(self, state, want_derived=True, **params_values):
        from forestflow.model.arinyo import ArinyoModel
        from forestflow.model.linear import LinearTheoryGrid
        from forestflow.statistics.p1d import P1D_kms
        snapshot = self.provider.get_result("lya_cosmology")
        try:
            for name, bounds in self.cosmology_domain.items():
                value = params_values[name]
                if not bounds[0] <= value <= bounds[1]:
                    raise ModelDomainError(f"{name} outside explicitly admitted cosmology domain")
            zs = self.request.redshifts
            if min(zs) < 2. or max(zs) > self.emulator.zmax:
                raise ModelDomainError("redshift outside MPG bundle coverage")
            native = route({k: v for k, v in params_values.items() if k not in self.cosmology_domain}, self.mapping)
            models = self.igm.models
            flux = np.exp(-models["F_model"].get_tau_eff(np.array(zs), like_params=native))
            gamma = models["T_model"].get_gamma(np.array(zs), like_params=native)
            thermal = models["T_model"].get_sigT_kms(np.array(zs), like_params=native)
            pressure = models["P_model"].get_kF_kms(np.array(zs), like_params=native)
            M = snapshot.get_dkms_diMpc(np.array(zs))
            inputs = []
            for i, z in enumerate(zs):
                values = snapshot.get_linP_Mpc_params(z, self.emulator.kp_iMpc)
                values.update(mF=flux[i], gamma=gamma[i], sigT_Mpc=thermal[i]/M[i], kF_Mpc=pressure[i]*M[i])
                row = {k: float(values[k]) for k in self.emulator.input_labels}
                if not all(np.isfinite(v) for v in row.values()) or not 0 < row["mF"] < 1 or any(row[k] <= 0 for k in ("Delta2_p", "gamma", "sigT_Mpc", "kF_Mpc")):
                    raise ModelDomainError("nonphysical IGM/emulator input")
                inputs.append(row)
            if not self.hull.in_hulls(np.array([[row[k] for k in self.domain_labels] for row in inputs])):
                raise ModelDomainError("outside native training pairwise hull")
            # Redshift-independent common random numbers: stable latent block
            # zero for each input, invariant under batching/order/chunking.
            ari = self.emulator.evaluate(inputs, Nrealizations=self.realizations,
                seed=self.seed, latent_indices=[0]*len(zs))
            self.calls += 1
            arinyo = {z: MappingProxyType({k: float(np.atleast_1d(ari[k])[i]) for k in self.emulator.output_labels}) for i, z in enumerate(zs)}
            model = ArinyoModel(fiducial_cosmology=snapshot)
            linear = LinearTheoryGrid(z=np.array(zs), cosmology=snapshot)
            powers = {}
            projected = {}
            for group in self.request.groups:
                rows = []
                for z, k in zip(group.redshifts, group.k_ikms):
                    key = (z, k)
                    if key in projected:
                        rows.append(projected[key])
                        continue
                    m = M[zs.index(z)]
                    if max(k)*m > self.emulator.kmax_1d_iMpc:
                        raise ModelDomainError("parallel k outside trained P1D coverage")
                    snapshot.validate_k(np.sqrt((np.asarray(k)*m)**2 + self.integrator.k_perp_iMpc[-1]**2))
                    snapshot.validate_k(np.sqrt((np.asarray(k)*m)**2 + self.integrator.k_perp_iMpc[0]**2))
                    p = P1D_kms(linear, z, np.asarray(k), model.P3D_Mpc_kpar_kperp, m,
                                arinyo[z], integrator=self.integrator)
                    if not np.all(np.isfinite(p)) or np.any(p <= 0):
                        raise FloatingPointError("non-finite/nonpositive Arinyo projection")
                    projected[key] = readonly(p)
                    rows.append(projected[key])
                powers[group.identifier] = tuple(rows)
            context = PredictionContext(self.request.identity, zs, tuple(flux), tuple(M))
            state["forestflow_p1d"] = MappingProxyType(dict(valid=True, powers=MappingProxyType(powers), context=context,
                inputs=tuple(MappingProxyType(r) for r in inputs), cosmology_id=snapshot.identity))
            state["forestflow_arinyo"] = MappingProxyType(arinyo)
        except ModelDomainError as error:
            state["forestflow_p1d"] = MappingProxyType(dict(valid=False, reason=str(error)))
            state["forestflow_arinyo"] = None
        state["derived"] = {}
