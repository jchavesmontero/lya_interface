"""Compatibility wrapper for cup1d's public coefficient projection route."""

from lya_interface.contracts import ModelDomainError, NumericalCoverageError


def project_request(request, snapshot, model, linear, arinyo, integrator, kmax_iMpc):
    """
    Project unique (z, k) requests together, preserving ragged output grids.

    Parameters
    ----------
    request : PredictionRequest
        Dataset groups with redshifts and ragged wavenumber grids in s/km.
    snapshot : CobayaCosmologySnapshot
        Immutable linear power and velocity conversions for the current point.
    model : forestflow.model.arinyo.ArinyoModel
        Native analytic flux-power model.
    linear : forestflow.model.linear.LinearTheoryGrid
        Linear-theory grid associated with the same snapshot.
    arinyo : mapping
        Redshift-indexed dictionaries of physical Arinyo coefficients.
    integrator : forestflow.statistics.p1d.P1DIntegrator
        Native transverse quadrature and integration geometry.
    kmax_iMpc : float
        Maximum admitted parallel wavenumber in inverse Mpc.

    Returns
    -------
    mapping
        Immutable dataset-indexed tuples of P1D arrays in km/s, retaining
        the requested redshift ordering and individual grid lengths.

    Raises
    ------
    ModelDomainError
        If parallel wavenumbers exceed the trained P1D coverage.
    FloatingPointError
        If the projected power is non-finite or non-positive.

    Notes
    -----
    Padding repeats each row's last valid k, never zero or an out-of-range
    sentinel. It is removed before returning immutable provider products.
    The single native P1D projection integrates only along the transverse
    axis; parameter arrays have explicit singleton k and transverse axes.
    """
    from cup1d.likelihood.external import project_arinyo_request
    try:
        return project_arinyo_request(request, snapshot, arinyo,
                                      kmax_iMpc=kmax_iMpc, integrator=integrator,
                                      model=model, linear=linear)
    except NumericalCoverageError:
        # A provider-coverage failure is distinct from a rejected emulator
        # point and must remain visible to callers instead of being converted
        # to a likelihood-domain rejection.
        raise
    except ValueError as error:
        raise ModelDomainError(str(error)) from error
