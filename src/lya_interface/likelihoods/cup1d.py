"""Cobaya full P1D data likelihood; never adds cup1d statistical priors."""
import numpy as np
from cobaya.likelihood import Likelihood
from lya_interface.adapters.cup1d_backend import make_backend
from lya_interface.parameters import route


class Cup1DLikelihood(Likelihood):
    stop_at_error: bool = True
    native_config: str | None = None
    parameter_definitions: dict = {}
    covariance_asset: str | None = None
    fiducial_M: dict = {}
    include_logdet: bool = False

    def initialize(self):
        if not self.native_config or not self.parameter_definitions:
            raise ValueError("native_config and explicit parameter_definitions are required")
        self.backend, self.registry = make_backend(self.native_config, self.parameter_definitions,
            covariance_asset=self.covariance_asset, fiducial_M=self.fiducial_M, include_logdet=self.include_logdet)
        self.mapping = self.registry.mapping("nuisance")
        self.params = {k: None for k in self.mapping}

    def get_can_support_params(self):
        return list(self.mapping)

    def get_requirements(self):
        return {"forestflow_p1d": self.backend.get_prediction_request().to_dict()}

    def initialize_with_provider(self, provider):
        self.provider = provider

    def logp(self, _derived=None, **params_values):
        prediction = self.provider.get_result("forestflow_p1d")
        if not prediction["valid"]:
            return -np.inf
        result = self.backend.evaluate_from_p1d(prediction["powers"], prediction["context"], route(params_values, self.mapping))
        return result.loglike
