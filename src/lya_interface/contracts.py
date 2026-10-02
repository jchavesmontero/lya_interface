"""Shared upstream request/context contracts and explicit failure classes."""
from cup1d.likelihood.external import PredictionRequest, PredictionContext, PredictionGroup


class ModelDomainError(ValueError):
    """Signal that a sampled point lies outside the admitted physical model domain.

    Likelihood code may convert this exception into a rejected point because
    the requested prediction is scientifically unsupported rather than a
    numerical-coverage failure.
    """


class NumericalCoverageError(ValueError):
    """Signal insufficient configured numerical coverage for a valid point.

    Unlike :class:`ModelDomainError`, this exception identifies a configuration
    error (for example an inadequate linear-power range) and must not be
    silently converted into a likelihood rejection.
    """
