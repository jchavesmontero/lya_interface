"""Convenience API around the actual Cobaya dependency graph."""
import numpy as np
from cobaya.model import get_model
from .configuration import load_configuration


class IGMHubCobaya:
    """Provide the high-level Cobaya interface for the configured P1D analysis."""
    def __init__(self, config_file):
        """Load an interface configuration and construct its Cobaya graph.

        Parameters
        ----------
        config_file : str or pathlib.Path
            Interface YAML file resolved through :func:`load_configuration`.
        """
        self.config = load_configuration(config_file)
        self.model = get_model(self.config)

    def loglike(self, **sampled_parameters):
        """Evaluate the sum of configured likelihood terms.

        Parameters
        ----------
        **sampled_parameters
            Public physical sampled values accepted by the configuration.

        Returns
        -------
        float
            Sum of likelihood log values, excluding Cobaya priors.
        """
        return float(np.sum(self.model.loglikes(sampled_parameters, return_derived=False)))

    def close(self):
        """Close the underlying Cobaya model and release its resources."""
        self.model.close()


Interface = IGMHubCobaya
