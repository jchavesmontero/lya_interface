"""Fast actual-Cobaya lifecycle tests, explicitly synthetic provider/assets."""
import numpy as np
import pytest
from cobaya.model import get_model
from cobaya.theory import Theory
from lya_interface.theory.lya_cosmology import LyaCosmology
from lya_interface.theory.forestflow import ForestFlowTheory
from lya_interface.likelihoods.cup1d import Cup1DLikelihood
from lya_interface.contracts import PredictionRequest
from test_contracts import backend


class Spectrum:
    kmin=1e-4
    kmax=200.
    def __init__(self,A,ns):
        self.A,self.ns=A,ns
    def P(self,z,k):
        return self.A*(1+z)**-2*np.asarray(k)**self.ns


class LinearFixture(Theory):
    params={"As":None,"ns":None,"H0":None,"mnu":None,"nrun":None}
    def initialize(self):
        self.calls=0
    def must_provide(self,**requirements):
        super().must_provide(**requirements)
    def calculate(self,state,want_derived=True,**values):
        state["pk"]=Spectrum(values["As"]*1e11,values["ns"]-3.)
        state["H"]=values["H0"]
        state["derived"]={}
        self.calls+=1
    def get_Pk_interpolator(self,**kwargs):
        assert kwargs["nonlinear"] is False
        assert kwargs["var_pair"]==("delta_nonu","delta_nonu")
        return self.current_state["pk"]
    def get_Hubble(self,z,units="km/s/Mpc"):
        return self.current_state["H"]*(1+np.asarray(z))**1.5
    def get_fsigma8(self,z):
        return .4/(1+np.asarray(z))
    def get_sigma8_z(self,z):
        return .5/(1+np.asarray(z))


class Histories:
    def get_tau_eff(self,z,like_params=None):
        return np.full(len(z),.4)
    def get_gamma(self,z,like_params=None):
        return np.full(len(z),like_params["gamma_0"])
    def get_sigT_kms(self,z,like_params=None):
        return np.full(len(z),10.)
    def get_kF_kms(self,z,like_params=None):
        return np.full(len(z),.15)


class EmulatorFixture:
    input_labels=["Delta2_p","n_p","mF","gamma","sigT_Mpc","kF_Mpc"]
    output_labels=["bias","bias_eta","q1","q2","kvav","av","bv","kp"]
    kp_iMpc=.7
    zmax=4.6
    kmax_1d_iMpc=6.
    def evaluate(self,rows,**kwargs):
        params=dict(bias=-.15,bias_eta=-.2,q1=.4,q2=0.,kvav=.6,av=.3,bv=1.5,kp=10.)
        return {name:np.array([value*r["gamma"] if name=="bias" else value for r in rows]) for name,value in params.items()}


class AlwaysInside:
    def in_hulls(self,points):
        return True


class ForestFixture(ForestFlowTheory):
    def initialize(self):
        from types import SimpleNamespace
        from forestflow.statistics.p1d import P1DIntegrator
        self.mapping={"igm_gamma_0":"gamma_0"}
        self.params={k:None for k in self.mapping}
        histories=Histories()
        self.igm=SimpleNamespace(models={k:histories for k in ("F_model","T_model","P_model")})
        self.emulator=EmulatorFixture()
        self.integrator=P1DIntegrator(n_k_perp=24)
        self.hull=AlwaysInside()
        self.domain_labels=self.emulator.input_labels
        self.calls=0
        self.request=None
    def must_provide(self,**requirements):
        if "forestflow_p1d" in requirements:
            self.request=PredictionRequest.from_dict(requirements["forestflow_p1d"])
        return {"lya_cosmology":{"redshifts":list(self.request.redshifts)}}


class LikelihoodFixture(Cup1DLikelihood):
    def initialize(self):
        self.backend=backend()
        self.mapping={"p1d_R_coeff_0":"R_coeff_0"}
        self.params={k:None for k in self.mapping}


def test_real_cobaya_dependency_graph_and_exact_restore():
    info=dict(params=dict(As=dict(prior=dict(min=1e-9,max=3e-9)),ns=.96,H0=dict(prior=dict(min=60,max=80)),mnu=0.,nrun=0.,
        igm_gamma_0=dict(prior=dict(min=1.,max=2.)),p1d_R_coeff_0=dict(prior=dict(min=-.1,max=.1)),
        Delta2star=dict(derived=True),nstar=dict(derived=True),alphastar=dict(derived=True)),
        theory={"linear":{"external":LinearFixture},"snapshot":{"external":LyaCosmology},"forest":{"external":ForestFixture}},
        likelihood={"data":{"external":LikelihoodFixture}})
    with get_model(info) as model:
        a=dict(As=2e-9,H0=70.,igm_gamma_0=1.4,p1d_R_coeff_0=0.)
        first=model.logposterior(a)
        snap=model.provider.get_result("lya_cosmology")
        pred=model.provider.get_result("forestflow_p1d")
        linear=model.theory["linear"]
        forest=model.theory["forest"]
        assert (linear.calls,forest.calls)==(1,1)
        model.logposterior({**a,"p1d_R_coeff_0":.01})
        assert (linear.calls,forest.calls)==(1,1)
        model.logposterior({**a,"igm_gamma_0":1.41})
        assert (linear.calls,forest.calls)==(1,2)
        second=model.logposterior({**a,"H0":70.1})
        assert (linear.calls,forest.calls)==(2,3)
        assert second.derived[0]!=first.derived[0]
        restored=model.logposterior(a)
        assert (linear.calls,forest.calls)==(2,3)
        assert model.provider.get_result("lya_cosmology") is snap
        assert model.provider.get_result("forestflow_p1d") is pred
        np.testing.assert_array_equal(first.derived,restored.derived)
        # Prior density changes posterior only, not the data calculation.
        loglike=float(model.loglikes(a,return_derived=False)[0])
        assert first.logpost!=loglike


def test_derived_only_does_not_initialize_forestflow():
    info=dict(params=dict(As=2e-9,ns=.96,H0=70.,mnu=0.,nrun=0.,Delta2star=dict(derived=True),nstar=dict(derived=True),alphastar=dict(derived=True)),
        theory={"linear":{"external":LinearFixture},"snapshot":{"external":LyaCosmology}},
        likelihood={"diagnostic":{"external":lambda Delta2star,nstar,alphastar:0.}})
    with get_model(info) as model:
        result=model.logposterior({})
        assert len(result.derived)==3 and np.all(np.isfinite(result.derived))
        assert len(model.theory)==2


def test_numerical_coverage_is_not_silently_rejected():
    from lya_interface.contracts import NumericalCoverageError
    info=dict(params=dict(As=2e-9,ns=.96,H0=70.,mnu=0.,nrun=0.,igm_gamma_0=1.4,p1d_R_coeff_0=0.),
        theory={"linear":{"external":LinearFixture},"snapshot":{"external":LyaCosmology,"kmax_iMpc":50.},"forest":{"external":ForestFixture}},
        likelihood={"data":{"external":LikelihoodFixture}})
    with get_model(info) as model:
        with pytest.raises(NumericalCoverageError,match="provider range"):
            model.logposterior({})
