"""Shared upstream request/context contracts and explicit failure classes."""
from cup1d.likelihood.external import PredictionRequest, PredictionContext, PredictionGroup


class ModelDomainError(ValueError):
    """A scientifically unsupported sampled point, suitable for rejection."""


class NumericalCoverageError(ValueError):
    """Configured numerical range is insufficient; never silently reject."""
