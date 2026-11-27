"""Route scalar Cobaya points through cup1d's columnar observation kernels."""
from cup1d.likelihood.external import ExternalP1DLikelihood


class VectorizedP1DLikelihood(ExternalP1DLikelihood):
    """
    Preserve the scalar likelihood contract using native batch kernels.

    Cobaya supplies one parameter point at a time. A singleton parameter
    axis selects cup1d's columnar contaminant/resolution/rebinning path
    (``ExternalP1DLikelihood.columnar``), without constructing native theory
    or duplicating any physical model. The Kaiser metal model (which needs
    metal moments) has no columnar kernel and uses the scalar path.
    The inherited correlated likelihood contraction is already array based.
    """

    columnar = True
