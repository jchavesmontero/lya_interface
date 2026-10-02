"""Real assets, actual Cobaya graph, and independent LaCE calculations."""
from copy import deepcopy
from pathlib import Path
import os
import numpy as np
import pytest
from cobaya.model import get_model
from lya_interface.configuration import load_configuration
from lya_interface.parameters import route

ROOT = Path(__file__).resolve().parents[1]


def unavailable(message):
    if os.environ.get("LYA_REQUIRE_SCIENTIFIC_ASSETS") == "1":
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture(scope="module")
def model():
    import forestflow
    asset = Path(forestflow.__file__).resolve().parents[1]/"data/emulator_models/forest_mpg_fix.pt"
    if not asset.exists():
        unavailable(f"missing real model asset {asset}")
    from lace.configuration import get_nyx_path
    from cup1d.utils.utils import get_path_repo
    needed = [asset.with_name("forest_mpg_fix_metadata.npy"), asset.with_name("forest_mpg_fix_transf.npy"),
              Path(get_nyx_path())/"IGM_histories.npy"]
    needed += [Path(get_path_repo("lace"))/"data/sim_suites/Australia20"/name for name in ("IGM_histories.npy", "mpg_emu_cosmo.npy")]
    needed += [Path(get_path_repo("cup1d"))/"data/p1d_measurements/Karacayli2022"/name for name in
               ("final-conservative-p1d-karacayli_etal2021.txt", "final-conservative-covariance-karacayli_etal2021.txt")]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        unavailable("missing scientific assets: " + ", ".join(missing))
    info = load_configuration(ROOT/"examples/validation_demo.yaml")
    info["params"]["H0"] = dict(prior=dict(min=66.,max=69.),ref=67.66,proposal=.1)
    for section in ("theory","likelihood"):
        for name, options in info[section].items():
            if "parameter_definitions" in options:
                options["parameter_definitions"] = deepcopy(info["params"])
    with get_model(info) as m:
        yield m


def point(**overrides):
    return dict(As=2.105209331337507e-9, ns=.9665, igm_tau_eff_0=0.,p1d_f_Lya_SiIII_0=-4.,H0=67.66,**overrides)


@pytest.mark.scientific
def test_actual_graph_routing_prior_and_cache_restoration(model):
    p=point()
    post=model.logposterior(p)
    assert np.isfinite(post.logpost)
    cosmology = model.theory["lya_interface.theory.lya_cosmology.LyaCosmology"]
    forest = model.theory["lya_interface.theory.forestflow.ForestFlowTheory"]
    a=model.provider.get_result("lya_cosmology")
    original=model.provider.get_result("forestflow_p1d")
    calls=(cosmology.calls,forest.calls)
    nuisance={**p,"p1d_f_Lya_SiIII_0":-4.01}
    model.logposterior(nuisance)
    assert (cosmology.calls,forest.calls)==calls
    model.logposterior({**p,"igm_tau_eff_0":.001})
    assert cosmology.calls==calls[0] and forest.calls==calls[1]+1
    model.logposterior({**p,"H0":67.7})
    assert cosmology.calls==calls[0]+1 and forest.calls==calls[1]+2
    b=model.provider.get_result("lya_cosmology")
    assert a.identity!=b.identity
    assert np.isfinite(a.get_linP_Mpc(3.,.7))
    # Cobaya holds three cached states; restore exact original snapshot and P1D.
    restored=model.logposterior(p)
    assert (cosmology.calls,forest.calls)==(calls[0]+1,calls[1]+2)
    assert model.provider.get_result("lya_cosmology") is a
    assert model.provider.get_result("forestflow_p1d") is original
    np.testing.assert_array_equal(post.derived,restored.derived)
    assert restored.loglikes == pytest.approx(post.loglikes)
    like=next(iter(model.likelihood.values()))
    fixed=dict(model.parameterization.input_params())
    result=like.backend.evaluate_from_p1d(original["powers"],original["context"],route({k:fixed[k] for k in like.mapping},like.mapping))
    assert result.loglike==pytest.approx(sum(post.loglikes),abs=1e-10)
    assert result.chi2_data==-2*result.loglike
    assert post.logpost != result.loglike  # only Cobaya added the demonstration priors


@pytest.mark.scientific
@pytest.mark.parametrize("change", [{},{"As":2.12e-9,"ns":.967},{"H0":67.8,"igm_tau_eff_0":.002}])
def test_matching_native_inputs_linear_power_growth_and_projection(model,change):
    from lace.cosmo.cosmology import Cosmology
    from forestflow.model.arinyo import ArinyoModel
    from forestflow.model.linear import LinearTheoryGrid
    from forestflow.statistics.p1d import P1D_kms, P1DIntegrator
    p={**point(),**change}
    model.logposterior(p)
    snapshot=model.provider.get_result("lya_cosmology")
    forest=model.theory["lya_interface.theory.forestflow.ForestFlowTheory"]
    pred=model.provider.get_result("forestflow_p1d")
    assert pred["valid"]
    native=Cosmology(cosmo_params_dict=dict(As=p["As"],ns=p["ns"],H0=p["H0"],ombh2=.02242,omch2=.11933,mnu=0.,nnu=3.046,tau=.054,nrun=0.,pivot_scalar=.05), camb_kmax_Mpc=200./1.001)
    native.CAMBparams.WantCls = False
    native.CAMBparams.Want_CMB = False
    ks=np.geomspace(.01,100.,200)
    for z in (2.2,3.,4.2):
        np.testing.assert_allclose(snapshot.get_linP_Mpc(z,ks),native.get_linP_Mpc(z,ks),rtol=3e-4)
        np.testing.assert_allclose(snapshot.get_dkms_diMpc(z),native.get_dkms_diMpc(z),rtol=1e-10)
        np.testing.assert_allclose(snapshot.get_growth_rate(z),native.get_growth_rate(z),rtol=2e-5)
        assert snapshot.get_linP_kms_params(z,.009)["Delta2_star"]==pytest.approx(native.get_linP_kms_params(z,.009)["Delta2_star"],rel=3e-4)
    zs=forest.request.redshifts
    for z,row in zip(zs,pred["inputs"]):
        summaries=native.get_linP_Mpc_params(z,forest.emulator.kp_iMpc)
        for name in ("Delta2_p","n_p"):
            assert row[name]==pytest.approx(summaries[name],rel=3e-4)
    # Same estimator and stable latent identity: separate, batched, reordered.
    inputs=[dict(r) for r in pred["inputs"]]
    ari=forest.emulator.evaluate(inputs,Nrealizations=forest.realizations,seed=forest.seed,latent_indices=[0]*len(zs))
    perm=np.arange(len(zs))[::-1]
    reordered=forest.emulator.evaluate([inputs[i] for i in perm],Nrealizations=forest.realizations,seed=forest.seed,latent_indices=[0]*len(zs))
    single=forest.emulator.evaluate(inputs[3],Nrealizations=forest.realizations,seed=forest.seed,latent_indices=[0])
    for name in ari:
        np.testing.assert_allclose(reordered[name][perm],ari[name],rtol=2e-6,atol=1e-7)
        np.testing.assert_allclose(single[name],ari[name][3],rtol=2e-6,atol=1e-7)
    arinyo=forest.get_result("forestflow_arinyo")
    native_model=ArinyoModel(native)
    linear=LinearTheoryGrid(z=np.array(zs),cosmology=native)
    like=next(iter(model.likelihood.values()))
    fine={}
    for g in forest.request.groups:
        fine[g.identifier]=[P1D_kms(linear,z,np.array(k),native_model.P3D_Mpc_kpar_kperp,
             native.get_dkms_diMpc(z),arinyo[z],integrator=forest.integrator) for z,k in zip(g.redshifts,g.k_ikms)]
        for a,b in zip(pred["powers"][g.identifier],fine[g.identifier]):
            np.testing.assert_allclose(a,b,rtol=5e-4)
    fixed=dict(model.parameterization.input_params())
    nuisance=route({k:fixed[k] for k in like.mapping},like.mapping)
    reference=like.backend.evaluate_from_p1d(fine,pred["context"],nuisance)
    external=like.backend.evaluate_from_p1d(pred["powers"],pred["context"],nuisance)
    print("independent LaCE-minus-provider chi2, point",change,":",reference.chi2_data-external.chi2_data)
    assert abs(reference.chi2_data-external.chi2_data)<1.

    # Exercise maintained native cup1d scalar forward path with explicit test
    # injection of this same snapshot and latent convention. This bypasses
    # only its obsolete primordial-only cosmology setup, not its kernels.
    from cup1d.theory.theory import Theory as NativeTheory
    from forestflow.emulator.p1d import P1DEmulator

    class InjectedArinyo(ArinyoModel):
        def P1D_Mpc(self,linear,z,k,params):
            from forestflow.statistics.p1d import P1D_Mpc
            return P1D_Mpc(linear,z,k,self.P3D_Mpc_kpar_kperp,params,integrator=forest.integrator)

    class InjectedEmulator(P1DEmulator):
        def __init__(self):
            self.emulator=forest.emulator
            self.emu_params=self.emulator.input_labels
            self.emulator_label=forest.model_key
            self.model_Arinyo=InjectedArinyo(snapshot)
            self._prediction_cache=None

        def set_linear_theory(self,z,new_cosmo_params=None):
            self.linear=LinearTheoryGrid(z=np.unique(z),cosmology=snapshot)

        def _evaluate_emulator(self,calls,**kwargs):
            return self.emulator.evaluate(calls,Nrealizations=forest.realizations,seed=forest.seed,latent_indices=[0]*len(calls))

    class InjectedTheory(NativeTheory):
        def get_cosmology(self,like_params=None):
            return snapshot

        def get_blob(self,cosmo=None):
            return (0.,)*6  # native blob is irrelevant to data likelihood parity

    native_theory=InjectedTheory(InjectedEmulator(),model_igm=forest.igm,model_cont=like.backend.model_cont,
                                model_syst=like.backend.model_syst,use_hull=False)
    native_theory.star_priors=None
    igm_native=route({k:fixed[k] for k in forest.mapping},forest.mapping)
    for g in forest.request.groups:
        native_inputs,native_M=native_theory.get_emulator_calls(np.array(g.redshifts),like_params=igm_native)
        for i,z in enumerate(g.redshifts):
            for name in forest.emulator.input_labels:
                assert native_inputs[name][i]==pytest.approx(pred["inputs"][zs.index(z)][name],rel=1e-13)
        native_observed=native_theory.get_p1d_kms(np.array(g.redshifts),[np.array(k) for k in g.k_ikms],
                     like_params={**igm_native,**nuisance},return_blob=False)
        external_observed=like.backend.apply_observation_model(pred["powers"],pred["context"],nuisance)
        for a,b in zip(like.backend.Rebin_data.rebinning(g.identifier,native_observed),external_observed[g.identifier]):
            np.testing.assert_allclose(a,b,rtol=2e-6,atol=1e-6)


@pytest.mark.scientific
@pytest.mark.parametrize("change", [{},{"As":1.9001e-9,"ns":.95001,"igm_tau_eff_0":-.0299},{"As":2.2999e-9,"ns":.97999,"igm_tau_eff_0":.0299}])
def test_projection_convergence(model,change):
    from forestflow.model.arinyo import ArinyoModel
    from forestflow.model.linear import LinearTheoryGrid
    from forestflow.statistics.p1d import P1D_kms,P1DIntegrator
    p={**point(),**change}
    model.logposterior(p)
    snapshot=model.provider.get_result("lya_cosmology")
    forest=model.theory["lya_interface.theory.forestflow.ForestFlowTheory"]
    pred=model.provider.get_result("forestflow_p1d")
    assert pred["valid"]
    ari=forest.get_result("forestflow_arinyo")
    linear=LinearTheoryGrid(z=np.array(forest.request.redshifts),cosmology=snapshot)
    arinyo=ArinyoModel(snapshot)
    like=next(iter(model.likelihood.values()))
    fixed=dict(model.parameterization.input_params())
    nuisance=route({k:fixed[k] for k in like.mapping},like.mapping)
    chi=[]
    for n in (48,96,192):
        integrator=P1DIntegrator(n_k_perp=n)
        powers={g.identifier:[P1D_kms(linear,z,np.array(k),arinyo.P3D_Mpc_kpar_kperp,
                snapshot.get_dkms_diMpc(z),ari[z],integrator=integrator) for z,k in zip(g.redshifts,g.k_ikms)] for g in forest.request.groups}
        chi.append(like.backend.evaluate_from_p1d(powers,pred["context"],nuisance).chi2_data)
    print("projection chi2 48/96/192, point",change,":",chi)
    assert abs(chi[-1]-chi[-2])<.01


@pytest.mark.scientific
def test_realization_count_sensitivity_is_not_error_covariance(model):
    from forestflow.model.arinyo import ArinyoModel
    from forestflow.model.linear import LinearTheoryGrid
    from forestflow.statistics.p1d import P1D_kms
    model.logposterior(point())
    s=model.provider.get_result("lya_cosmology")
    forest=model.theory["lya_interface.theory.forestflow.ForestFlowTheory"]
    pred=model.provider.get_result("forestflow_p1d")
    inputs=[dict(r) for r in pred["inputs"]]
    zs=forest.request.redshifts
    arinyo=ArinyoModel(s)
    linear=LinearTheoryGrid(z=np.array(zs),cosmology=s)
    like=next(iter(model.likelihood.values()))
    fixed=model.parameterization.input_params()
    nuisance=route({k:fixed[k] for k in like.mapping},like.mapping)
    values=[]
    for n in (1000,2000,4000,8000):
        ari=forest.emulator.evaluate(inputs,Nrealizations=n,seed=forest.seed,latent_indices=[0]*len(zs))
        rows={z:{k:np.atleast_1d(ari[k])[i] for k in ari} for i,z in enumerate(zs)}
        powers={g.identifier:[P1D_kms(linear,z,np.array(k),arinyo.P3D_Mpc_kpar_kperp,
            s.get_dkms_diMpc(z),rows[z],integrator=forest.integrator) for z,k in zip(g.redshifts,g.k_ikms)] for g in forest.request.groups}
        values.append(like.backend.evaluate_from_p1d(powers,pred["context"],nuisance).chi2_data)
    print("realization-count chi2 1000/2000/4000/8000:",values)
    assert np.all(np.isfinite(values))
    assert values[0]==pytest.approx(like.backend.evaluate_from_p1d(pred["powers"],pred["context"],nuisance).chi2_data,rel=1e-12)


@pytest.mark.scientific
def test_snapshot_interpolation_resolution(model):
    from lya_interface.cosmology import CobayaCosmologySnapshot
    from forestflow.model.arinyo import ArinyoModel
    from forestflow.model.linear import LinearTheoryGrid
    from forestflow.statistics.p1d import P1D_kms
    model.logposterior(point())
    original=model.provider.get_result("lya_cosmology")
    pk=model.provider.get_Pk_interpolator(nonlinear=False,var_pair=("delta_nonu","delta_nonu"))
    forest=model.theory["lya_interface.theory.forestflow.ForestFlowTheory"]
    pred=model.provider.get_result("forestflow_p1d")
    ari=forest.get_result("forestflow_arinyo")
    like=next(iter(model.likelihood.values()))
    fixed=model.parameterization.input_params()
    nuisance=route({k:fixed[k] for k in like.mapping},like.mapping)
    values=[]
    for n in (1024,4096,8192):
        k=np.geomspace(original.k_iMpc[0],original.k_iMpc[-1],n)
        s=CobayaCosmologySnapshot(original.redshifts,k,[pk.P(z,k) for z in original.redshifts],
            original.hubble,original.growth,primordial=original.primordial,mnu=original.mnu)
        arinyo=ArinyoModel(s)
        linear=LinearTheoryGrid(z=np.array(forest.request.redshifts),cosmology=s)
        powers={g.identifier:[P1D_kms(linear,z,np.array(k),arinyo.P3D_Mpc_kpar_kperp,
            s.get_dkms_diMpc(z),ari[z],integrator=forest.integrator) for z,k in zip(g.redshifts,g.k_ikms)] for g in forest.request.groups}
        values.append(like.backend.evaluate_from_p1d(powers,pred["context"],nuisance).chi2_data)
    print("snapshot-node chi2 1024/4096/8192:",values)
    assert abs(values[-1]-values[-2])<.001


@pytest.mark.scientific
def test_noiseless_mock_recovers_injected_nuisance(model):
    """Small real-weight mock regression; not a cosmological identifiability claim."""
    from scipy.optimize import minimize_scalar
    from cup1d.likelihood.external import gaussian_residual
    p=point()
    model.logposterior(p)
    prediction=model.provider.get_result("forestflow_p1d")
    like=next(iter(model.likelihood.values()))
    fixed=dict(model.parameterization.input_params())
    nuisance=route({k:fixed[k] for k in like.mapping},like.mapping)
    injected=like.backend.apply_observation_model(prediction["powers"],prediction["context"],nuisance)
    def objective(value):
        candidate={**nuisance,"f_Lya_SiIII_0":value}
        observed=like.backend.apply_observation_model(prediction["powers"],prediction["context"],candidate)
        total=0.
        for key in injected:
            residual=np.concatenate(observed[key])-np.concatenate(injected[key])
            total+=gaussian_residual(residual,like.backend.full_chol_Pk_kms[key])[0]
        return total
    result=minimize_scalar(objective,bounds=(-4.2,-3.8),method="bounded",options={"xatol":1e-9})
    assert result.success and abs(result.x+4.)<1e-6 and result.fun<1e-10


@pytest.mark.scientific
@pytest.mark.parametrize("mass", [0., .06])
def test_cosmology_only_neutrino_growth_and_fit_conventions(mass):
    from lace.cosmo.cosmology import Cosmology
    info=load_configuration(ROOT/"examples/validation_demo.yaml")
    del info["theory"]["lya_interface.theory.forestflow.ForestFlowTheory"]
    info["params"]={k:v for k,v in info["params"].items() if not k.startswith(("igm_","p1d_"))}
    info["params"]["mnu"]=mass
    info["theory"]["camb"]["extra_args"]["num_massive_neutrinos"]=int(mass>0)
    info["likelihood"]={"stars":{"external":lambda Delta2star,nstar,alphastar:0.}}
    p=dict(As=2.105209331337507e-9,ns=.9665)
    with get_model(info) as m:
        m.logposterior(p)
        s=m.theory["lya_interface.theory.lya_cosmology.LyaCosmology"].get_result("lya_cosmology")
        assert s.get_mnu()==mass
        native=Cosmology(cosmo_params_dict=dict(**p,H0=67.66,ombh2=.02242,omch2=.11933,mnu=mass,nnu=3.046,tau=.054),camb_kmax_Mpc=200./1.001)
        native.CAMBparams.WantCls=False
        native.CAMBparams.Want_CMB=False
        for z in (2.,3.,4.6):
            np.testing.assert_allclose(s.get_growth_rate(z),native.get_growth_rate(z),rtol=2e-5)
            np.testing.assert_allclose(s.get_linP_Mpc(z,np.geomspace(.01,100.,40)),native.get_linP_Mpc(z,np.geomspace(.01,100.,40)),rtol=3e-4)
        ratios=np.geomspace(.5,2.,100)
        kp=.009*s.get_dkms_diMpc(3.)
        powers=s.get_linP_Mpc(3.,ratios*kp)
        closed=np.polyfit(np.log(ratios),np.log(powers),2)
        opened=np.polyfit(np.log(ratios[1:-1]),np.log(powers[1:-1]),2)
        difference=np.array([kp**3*(np.exp(opened[2])-np.exp(closed[2]))/(2*np.pi**2),opened[1]-closed[1],2*(opened[0]-closed[0])])
        print("open-minus-closed star fit (Delta2,n,alpha), mnu=",mass,":",difference)
        assert max(abs(difference))<.001
