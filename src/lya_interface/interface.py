"""Convenience API around the actual Cobaya dependency graph."""
import numpy as np
from cobaya.model import get_model
from .configuration import load_configuration


class IGMHubCobaya:
    def __init__(self, config_file):
        self.config = load_configuration(config_file)
        self.model = get_model(self.config)

    def loglike(self, **sampled_parameters):
        return float(np.sum(self.model.loglikes(sampled_parameters, return_derived=False)))

    def close(self):
        self.model.close()


Interface = IGMHubCobaya
