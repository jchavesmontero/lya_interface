"""Cobaya full P1D data likelihood; never adds cup1d statistical priors."""
import numpy as np
from cobaya.likelihood import Likelihood
from lya_interface.adapters.cup1d_backend import make_backend
from lya_interface.parameters import route


class Cup1DLikelihood(Likelihood):
    """Cobaya likelihood evaluating cup1d P1D data without native priors.

    The likelihood consumes ForestFlow raw P1D predictions, applies the
    frozen cup1d observation model and contracts the calibrated covariance.
    """
    stop_at_error: bool = True
    native_config: str | None = None
    parameter_definitions: dict = {}
    covariance_asset: str | None = None
    fiducial_M: dict = {}
    include_logdet: bool = False
    blinding_policy: str = "reject"

    def initialize(self):
        """Build the static cup1d backend and nuisance parameter registry.

        Raises
        ------
        ValueError
            If either the native cup1d YAML path or explicit parameter
            definitions are not configured.
        """
        if not self.native_config or not self.parameter_definitions:
            raise ValueError("native_config and explicit parameter_definitions are required")
        self.backend, self.registry = make_backend(self.native_config, self.parameter_definitions,
            covariance_asset=self.covariance_asset, fiducial_M=self.fiducial_M,
            include_logdet=self.include_logdet, blinding_policy=self.blinding_policy)
        self.mapping = self.registry.mapping("nuisance")
        self.params = {k: None for k in self.mapping}

    def get_can_support_params(self):
        """Declare public nuisance parameters accepted by this likelihood.

        Returns
        -------
        list of str
            Cobaya nuisance names in the registry's stable mapping order.
        """
        return list(self.mapping)

    def get_requirements(self):
        """Request the immutable raw-P1D contract from ForestFlow theory.

        Returns
        -------
        dict
            Cobaya requirements containing serialized ``forestflow_p1d``
            grids and, for native blinding, the native-configuration request.
        """
        requirements = {"forestflow_p1d": self.backend.get_prediction_request().to_dict()}
        if self.blinding_policy == "native":
            requirements["lya_blinding"] = {"native_config": self.native_config}
        return requirements

    def initialize_with_provider(self, provider):
        """Store Cobaya's result provider after graph initialization.

        Parameters
        ----------
        provider : cobaya.theory.Provider
            Provider used to retrieve the immutable ``forestflow_p1d`` result.
        """
        self.provider = provider

    def logp(self, _derived=None, **params_values):
        """Evaluate the P1D data log likelihood for one Cobaya point.

        Parameters
        ----------
        _derived : dict, optional
            Cobaya-derived-parameter sink; unused by this likelihood.
        **params_values
            Physical public nuisance values declared by
            :meth:`get_can_support_params`. They are routed to cup1d's native
            named coefficients, not interpreted as sampler-cube coordinates.

        Returns
        -------
        float
            Correlated Gaussian P1D log likelihood. It contains no cup1d
            statistical prior; Cobaya owns priors and posterior assembly.

        Notes
        -----
        If ForestFlow rejects the point or cannot provide a valid raw-P1D
        prediction, this method returns ``-numpy.inf``. The fixed covariance
        log determinant is included only when configured at initialization.
        """
        prediction = self.provider.get_result("forestflow_p1d")
        if not prediction["valid"]:
            return -np.inf
        result = self.backend.evaluate_from_p1d(prediction["powers"], prediction["context"], route(params_values, self.mapping))
        return result.loglike
