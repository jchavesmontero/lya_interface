from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from cup1d.likelihood.external import ExternalP1DLikelihood, PredictionContext, PredictionRequest
from lya_interface.cosmology import CobayaCosmologySnapshot
from lya_interface.contracts import NumericalCoverageError
from lya_interface.derived import star_parameters
from lya_interface.parameters import ParameterRegistry, route
from lya_interface.configuration import _flatten_parameter_sections


class Contaminants:
    def get_contamination(self, zs, ks, mean_flux, M, **kwargs):
        return {name: [np.full(len(k), value) for k in ks] for name, value in
                dict(cont_HCD=2., cont_mul_metals=3., IC_corr=5., cont_add_metals=7.).items()}


class Systematics:
    def get_contamination(self, zs, ks, **kwargs):
        return [np.full(len(k), 11.) for k in ks]


def backend(include_logdet=False, rebin=1, emu_type="block", emu_factor=0., error=None):
    ks = [np.array([.001, .002]), np.array([.001, .002, .003])]
    full_cov = np.eye(5)*2
    full_cov[0, 3] = full_cov[3, 0] = .8
    data = SimpleNamespace(z=np.array([3., 2.]), k_kms=ks,
        k_kms_min=[k*.9 for k in ks], k_kms_max=[k*1.1 for k in ks],
        Pk_kms=[np.array([4., 5.]), np.array([6., 7., 8.])], Pksmooth_kms=None,
        covstat_Pk_kms=[full_cov[:2,:2], full_cov[2:,2:]],
        cov_Pk_kms=[full_cov[:2,:2], full_cov[2:,2:]],
        full_Pk_kms=np.array([4.,5.,6.,7.,8.]), full_cov_stat_Pk_kms=full_cov,
        full_cov_Pk_kms=full_cov, full_zs=np.array([3.,3.,2.,2.,2.]),
        full_k_kms=np.concatenate(ks))
    return ExternalP1DLikelihood({"a": data}, Contaminants(), Systematics(),
        cov_factor=dict(z=[2.,3.], val_stat=[1.,1.], val_syst=[1.,1.], val_emu=[emu_factor,emu_factor], val_full=[1.,1.]),
        emulator_covariance=error if error is not None else dict(zz_zk=np.array([2.,3.]), k_Mpc_zk=np.array([.1,.2]), cov_zk=np.eye(2)),
        fiducial_conversion=lambda z: 70., configuration_id="fixture", include_logdet=include_logdet,
        k_rebin_factor=rebin, emu_cov_type=emu_type)


def test_external_order_ragged_full_covariance_and_determinant():
    b = backend(True)
    r = b.get_prediction_request()
    assert PredictionRequest.from_dict(r.to_dict()) == r
    ctx = PredictionContext(r.identity, (2.,3.), (.8,.7), (65.,70.))
    p = {"a": [np.array([1.,2.]), np.array([3.,4.,5.])]}
    model = b.apply_observation_model(p, ctx, {"R_coeff_0": 0.})
    np.testing.assert_array_equal(np.concatenate(model["a"]), (30*np.arange(1,6)+7)*11)
    result = b.evaluate_from_p1d(p, ctx, {"R_coeff_0":0.})
    residual = b.data["a"].full_Pk_kms - np.concatenate(model["a"])
    expected = residual @ np.linalg.solve(b.data["a"].full_cov_Pk_kms, residual)
    assert result.chi2_data == pytest.approx(expected, rel=1e-14)
    assert result.logdet_cov == pytest.approx(np.linalg.slogdet(b.data["a"].full_cov_Pk_kms)[1])
    assert result.loglike == -.5*(result.chi2_data+result.logdet_cov)
    assert result.ndata == 5 and result.valid
    with pytest.raises(ValueError, match="identity"):
        b.evaluate_from_p1d(p, PredictionContext("wrong", (2.,3.), (.8,.7), (65.,70.)), {})
    with pytest.raises(ValueError, match="malformed"):
        b.evaluate_from_p1d({"a":[np.ones(1), np.ones(3)]}, ctx, {})


def test_rebin_and_shared_native_covariance():
    b = backend(rebin=4, emu_type="full")
    r = b.get_prediction_request()
    p = {"a": [np.ones(len(k)) for k in r.groups[0].k_ikms]}
    ctx = PredictionContext(r.identity,(2.,3.),(.8,.7),(65.,70.))
    for row in b.apply_observation_model(p,ctx,{} )["a"]:
        np.testing.assert_allclose(row, 37.)


def test_nonzero_full_emulator_covariance_uses_selected_z_k_indices():
    relative=np.eye(5)*.001
    relative[2,4]=relative[4,2]=.0002
    error=dict(zz_zk=np.array([2.,2.,2.,3.,3.]),k_Mpc_zk=np.array([.05,.16,.23,.07,.14]),cov_zk=relative)
    b=backend(emu_type="full",emu_factor=.5,error=error)
    indices=[3,4,0,1,2]
    power=b.data["a"].full_Pk_kms
    expected=relative[np.ix_(indices,indices)]*power[:,None]*power[None,:]*.5**2
    np.testing.assert_allclose(b.emu_full_cov_Pk_kms["a"],expected,rtol=1e-14)
    np.testing.assert_allclose(b.full_cov_Pk_kms["a"],b.data["a"].full_cov_Pk_kms+expected,rtol=1e-14)


def test_serialized_grid_fingerprints_detect_tampering():
    request=backend().get_prediction_request().to_dict()
    request["groups"][0]["k_ikms"][0][0]*=.99
    with pytest.raises(ValueError,match="fingerprint"):
        PredictionRequest.from_dict(request)


def test_analytic_star_units_and_immutable_snapshot():
    k = np.geomspace(1e-4,200.,4000)
    zs = (4.,0.,3.,2.)
    pivot = .72
    x = np.log(k/pivot)
    p = np.exp(2.-2.3*x+.1*x*x)
    s = CobayaCosmologySnapshot(zs,k,np.tile(p,(4,1)),[400.,70.,320.,220.],[1.,.5,.9,.95])
    derived = star_parameters(s,3.,.009)
    assert derived["nstar"] == pytest.approx(-2.3,abs=1e-12)
    assert derived["alphastar"] == pytest.approx(.2,abs=1e-12)
    assert derived["Delta2star"] == pytest.approx(pivot**3*np.exp(2)/(2*np.pi**2),rel=1e-12)
    np.testing.assert_allclose(s.get_linP_kms(3.,k/80.),p*80**3,rtol=1e-12)
    with pytest.raises(ValueError):
        s.power_Mpc3.flags.writeable = True
    with pytest.raises(NumericalCoverageError):
        s.get_linP_Mpc(3.,[201.])
    with pytest.raises(NumericalCoverageError):
        s.get_growth_rate(3.1)
    with pytest.raises(ValueError,match="total matter"):
        s.get_linP_Mpc(3.,[1.],species="bcnu")
    covariance_Mpc = np.ones((2,2))
    M = np.array([70.,80.])
    np.testing.assert_array_equal(covariance_Mpc*M[:,None]*M[None,:], [[4900.,5600.],[5600.,6400.]])


def test_registry_errors():
    defs = {"As": {"prior":{"min":1e-9,"max":3e-9}}, "igm_tau_eff_0":0., "p1d_R_coeff_0":0.}
    r = ParameterRegistry(defs,["tau_eff_0"],["R_coeff_0"])
    assert route({"igm_tau_eff_0":.2},r.mapping("IGM")) == {"tau_eff_0":.2}
    for bad in ({**defs,"garbage":1.}, {k:v for k,v in defs.items() if k!="igm_tau_eff_0"}, {**defs,"logA":{"prior":{"min":2.,"max":4.}}}):
        with pytest.raises(ValueError):
            ParameterRegistry(bad,["tau_eff_0"],["R_coeff_0"])


def test_structured_parameter_sections_have_explicit_ownership():
    flat, mapping = _flatten_parameter_sections({
        "general": {"H0": {"value": 67.0, "units": "km/s/Mpc"}},
        "P1D": {"p1d_metal": {"native": "metal", "prior": {"min": -1, "max": 1}}},
        "BAO": {"ap": {"prior": {"min": .8, "max": 1.2}}},
    })
    assert set(flat) == {"H0", "p1d_metal", "ap"}
    assert mapping["metal"]["section"] == "P1D"
    assert mapping["ap"]["section"] == "BAO"


def test_structured_parameter_rejects_native_collision():
    with pytest.raises(ValueError, match="ambiguous"):
        _flatten_parameter_sections({"P1D": {"a": {"native": "same"}}, "BAO": {"b": {"native": "same"}}})


def test_shared_igm_has_no_runtime_cup1d_dependency():
    root = Path(__file__).parents[1] / "src" / "lya_interface" / "igm"
    for source in root.glob("*.py"):
        text = source.read_text()
        assert "from cup1d" not in text
        assert "import cup1d" not in text


def test_shared_igm_history_preserves_scalar_and_batch_contracts():
    from lya_interface.igm.mean_flux_class import MeanFlux
    fid = {"tau_eff_z": np.array([2., 3., 4., 5.]),
           "tau_eff": np.array([.1, .2, .35, .55])}
    options = {"tau_eff_ztype": "interp_lin", "tau_eff_otype": "exp",
               "tau_eff_znodes": [2., 3., 4., 5.]}
    model = MeanFlux(free_param_names=["tau_eff_0", "tau_eff_1", "tau_eff_2", "tau_eff_3"],
                     fid_igm=fid, fid_vals={}, prop_coeffs=options,
                     flat_priors={"tau_eff": [[-1., 1.]]}, Gauss_priors=None)
    z = np.array([2.5, 3.5])
    scalar = model.get_tau_eff(z, {f"tau_eff_{i}": 0. for i in range(4)})
    batch = model.get_value_batch("tau_eff", z, {f"tau_eff_{i}": np.zeros(2) for i in range(4)})
    np.testing.assert_allclose(scalar, model.fid_interp["tau_eff"](z))
    np.testing.assert_allclose(batch, np.ones((2, 2)))


def test_diagnostic_column_names_are_collision_free_for_distinct_floats():
    from lya_interface.theory.forestflow import diagnostic_column_name
    a = diagnostic_column_name("bias", 2.3)
    b = diagnostic_column_name("bias", np.nextafter(2.3, 3.0))
    assert a != b
    assert a.startswith("bias_z_")


def test_derived_output_mapping_has_units_and_exact_redshift_identity():
    from lya_interface.theory.forestflow import ForestFlowTheory, diagnostic_column_name
    theory = ForestFlowTheory.__new__(ForestFlowTheory)
    theory.derived_redshifts = (2.3, np.nextafter(2.3, 3.0))
    mapping = theory.derived_output_mapping()
    first = diagnostic_column_name("T0_K", 2.3)
    second = diagnostic_column_name("T0_K", np.nextafter(2.3, 3.0))
    assert mapping[first] == {"quantity": "T0_K", "redshift": 2.3, "units": "K"}
    assert first != second


def test_structured_configuration_routes_component_definitions(tmp_path):
    from lya_interface.configuration import load_configuration
    config = tmp_path / "joint.yaml"
    config.write_text("""
parameters:
  general:
    H0: 67.0
  P1D:
    p1d_x: {native: x, value: 0.0}
  BAO:
    ap: {native: ap, value: 1.0}
theory:
  lya_interface.theory.forestflow.ForestFlowTheory: {}
likelihood:
  lya_interface.likelihoods.cup1d.Cup1DLikelihood: {}
  lya_interface.likelihoods.vega.VegaLikelihood: {}
""")
    info = load_configuration(config)
    forest = info["theory"]["lya_interface.theory.forestflow.ForestFlowTheory"]
    cup = info["likelihood"]["lya_interface.likelihoods.cup1d.Cup1DLikelihood"]
    vega = info["likelihood"]["lya_interface.likelihoods.vega.VegaLikelihood"]
    assert set(forest["parameter_definitions"]) == {"H0", "p1d_x"}
    assert cup["parameter_definitions"] == forest["parameter_definitions"]
    assert vega["parameter_mapping"] == {"ap": "ap"}
    assert vega["parameter_definitions"] == info["params"]


def test_provenance_accepts_vega_ini_without_cup1d_yaml_loading(tmp_path):
    """Vega native assets are INIs, not inputs for ``cup1d.Args``."""
    from lya_interface.provenance import collect

    ini = tmp_path / "vega.ini"
    ini.write_text("[data sets]\nzeff = 2.3\n")
    result = collect({"likelihood": {
        "lya_interface.likelihoods.vega.VegaLikelihood": {
            "native_config": str(ini),
        }
    }})
    assert result["assets"] == [
        {"path": str(ini.resolve()),
         "sha256": __import__("hashlib").sha256(ini.read_bytes()).hexdigest()}
    ]


def test_forestflow_vega_mode_removes_stale_native_beta(monkeypatch):
    """Shared bias/bias_eta must win over Vega's native beta default."""
    import sys
    from types import ModuleType
    from lya_interface.likelihoods.vega import VegaLikelihood

    class Backend:
        def __init__(self, _path):
            self.fiducial = {"z_eff": 2.3}
            self.priors = {}
            self.params = {"bias_LYA": -.1, "bias_eta_LYA": -.2,
                           "beta_LYA": 1.7, "ap": 1.}
            self.sample_params = {"limits": {"bias_LYA": [-1., 0.],
                                               "bias_eta_LYA": [-1., 0.],
                                               "beta_LYA": [0., 5.]}}
            self.models = {}

    module = ModuleType("vega.vega_interface")
    module.VegaInterface = Backend
    monkeypatch.setitem(sys.modules, "vega.vega_interface", module)
    adapter = object.__new__(VegaLikelihood)
    adapter.native_config = "fixture.ini"
    adapter.parameter_definitions = {"ap": {"value": 1.}}
    adapter.parameter_mapping = {"ap": "ap"}
    adapter.template_h = .67
    adapter.use_forestflow = True
    adapter.lya_tracer = "LYA"
    adapter.prior_policy = "reject"
    adapter.initialize()
    assert adapter.get_requirements() == {"forestflow_coefficients": {"redshifts": [2.3]}}
    for name in ("bias_LYA", "bias_eta_LYA", "beta_LYA"):
        assert name not in adapter.backend.params
        assert name not in adapter.backend.sample_params["limits"]
    adapter.provider = SimpleNamespace(get_result=lambda name: {
        2.3: {"bias": -.14, "bias_eta": -.21}
    } if name == "forestflow_coefficients" else None)
    values = adapter.parameters_for_evaluation({"ap": 1.})
    assert values["ap"] == 1.
    assert values["bias_LYA"] == -.14
    assert values["bias_eta_LYA"] == -.21
    assert values["forestflow_dnl"] == {}


def test_vega_external_prior_allows_only_fixed_native_mean(monkeypatch):
    """An internal fixed nuisance is safe only at its constant-prior mean."""
    import sys
    from types import ModuleType
    from lya_interface.likelihoods.vega import VegaLikelihood

    class Backend:
        def __init__(self, _path):
            self.fiducial = {"z_eff": 2.3}
            self.priors = {"drp_QSO": (0., 1.)}
            self.params = {}
            self.sample_params = {"limits": {}}
            self.models = {}

    module = ModuleType("vega.vega_interface")
    module.VegaInterface = Backend
    monkeypatch.setitem(sys.modules, "vega.vega_interface", module)
    adapter = object.__new__(VegaLikelihood)
    adapter.native_config = "fixture.ini"
    adapter.use_forestflow = False
    adapter.parameter_mapping = {}
    adapter.parameter_definitions = {}
    adapter.fixed_native_parameters = {"drp_QSO": 0.}
    adapter.prior_policy = "external"
    adapter.lya_tracer = "LYA"
    adapter.initialize()

    rejected = object.__new__(VegaLikelihood)
    rejected.native_config = "fixture.ini"
    rejected.use_forestflow = False
    rejected.parameter_mapping = {}
    rejected.parameter_definitions = {}
    rejected.fixed_native_parameters = {"drp_QSO": .1}
    rejected.prior_policy = "external"
    rejected.lya_tracer = "LYA"
    with pytest.raises(ValueError, match="must equal its native Gaussian mean"):
        rejected.initialize()


def test_projection_analytic_normalization_convergence():
    from forestflow.statistics.p1d import P1DIntegrator
    # P3D = exp(-k_perp^2), integral exactly (exp(-a²)-exp(-b²))/(4*pi).
    expected = (np.exp(-.01**2)-np.exp(-4.**2))/(4*np.pi)
    errors=[]
    for n in (24,48,96):
        integrator=P1DIntegrator(.01,4.,n)
        value=integrator(None,3.,np.array([.1,.2]),lambda lin,z,kp,kt,p:np.exp(-kt**2))
        errors.append(abs(value[0,0]-expected))
    assert errors[-1] < errors[0] and errors[-1] < 1e-7
