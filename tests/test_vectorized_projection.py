"""Ragged redshift projection agrees with the native scalar kernel."""
from types import SimpleNamespace

import numpy as np
import pytest

from forestflow.statistics.p1d import P1DIntegrator, P1D_kms
from lya_interface.projection import project_request
from lya_interface.contracts import ModelDomainError


@pytest.mark.parametrize("method", ["simpson", "gauss_legendre"])
def test_ragged_projection_is_one_call_and_restores_duplicates_and_order(method):
    calls = []

    def kernel(linear, z, kpar, kperp, parameters):
        calls.append(kpar.shape)
        return parameters["amplitude"] * np.exp(-kpar**2 - kperp**2)

    snapshot = SimpleNamespace(
        get_dkms_diMpc=lambda z: np.asarray(z) + 10,
        validate_k=lambda k: None,
    )
    groups = [
        SimpleNamespace(identifier="a", redshifts=(3., 2.),
                        k_ikms=((.01, .02, .03), (.01, .04))),
        SimpleNamespace(identifier="b", redshifts=(3., 3.),
                        k_ikms=((.01, .02, .03), (.01, .03, .05, .07))),
    ]
    request = SimpleNamespace(groups=groups)
    arinyo = {2.: {"amplitude": 2.}, 3.: {"amplitude": 3.}}
    integrator = P1DIntegrator(n_k_perp=17, method=method)
    result = project_request(request, snapshot, SimpleNamespace(P3D_Mpc_kpar_kperp=kernel),
                             None, arinyo, integrator, 6.)
    assert len(calls) == 1
    assert calls[0] == (3, 4, 17)
    assert result["a"][0] is result["b"][0]
    for group in groups:
        for z, k, actual in zip(group.redshifts, group.k_ikms, result[group.identifier]):
            expected = P1D_kms(None, z, k, kernel, z + 10, arinyo[z], integrator=integrator)
            np.testing.assert_allclose(actual, expected, rtol=1e-13)
            assert actual.shape == (len(k),)
            with pytest.raises(ValueError):
                actual.setflags(write=True)
    with pytest.raises(ModelDomainError, match="coverage"):
        project_request(request, snapshot, SimpleNamespace(P3D_Mpc_kpar_kperp=kernel),
                        None, arinyo, integrator, .01)
