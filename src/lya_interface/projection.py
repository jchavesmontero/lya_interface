"""Coalesced redshift projection using ForestFlow's native array kernel."""
from types import MappingProxyType

import numpy as np

from lya_interface.cosmology import readonly
from lya_interface.contracts import ModelDomainError


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
    from forestflow.statistics.p1d import P1D_Mpc

    keys = list(dict.fromkeys(
        (z, k) for group in request.groups for z, k in zip(group.redshifts, group.k_ikms)
    ))
    zs = np.array([z for z, _ in keys])
    lengths = np.array([len(k) for _, k in keys])
    conversion = np.asarray(snapshot.get_dkms_diMpc(zs))
    k_iMpc = np.empty((len(keys), int(lengths.max())))
    for i, (_, k) in enumerate(keys):
        row = np.asarray(k) * conversion[i]
        if row.max() > kmax_iMpc:
            raise ModelDomainError("parallel k outside trained P1D coverage")
        k_iMpc[i, :len(row)] = row
        k_iMpc[i, len(row):] = row[-1]
    for cutoff in (integrator.k_perp_iMpc[0], integrator.k_perp_iMpc[-1]):
        snapshot.validate_k(np.sqrt(k_iMpc**2 + cutoff**2))
    parameters = {
        name: np.array([arinyo[z][name] for z in zs])[:, None, None]
        for name in next(iter(arinyo.values()))
    }
    projected = P1D_Mpc(
        linear, zs, k_iMpc, model.P3D_Mpc_kpar_kperp, parameters,
        integrator=integrator,
    ) * conversion[:, None]
    if not np.all(np.isfinite(projected)) or np.any(projected <= 0):
        raise FloatingPointError("non-finite/nonpositive Arinyo projection")
    by_key = {key: readonly(projected[i, :lengths[i]]) for i, key in enumerate(keys)}
    return MappingProxyType({
        group.identifier: tuple(by_key[(z, k)] for z, k in zip(group.redshifts, group.k_ikms))
        for group in request.groups
    })
