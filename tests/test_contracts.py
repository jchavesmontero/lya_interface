from types import SimpleNamespace
import numpy as np
import pytest
from cup1d.likelihood.external import ExternalP1DLikelihood, PredictionContext, PredictionRequest
from lya_interface.cosmology import CobayaCosmologySnapshot
from lya_interface.contracts import NumericalCoverageError
from lya_interface.derived import star_parameters
from lya_interface.parameters import ParameterRegistry, route


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
