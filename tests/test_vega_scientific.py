"""Native Vega reference and data-only adapter regression checks."""
from pathlib import Path

import pytest


REFERENCE = Path("/home/jchaves/Proyectos/projects/lya/data/mocks_bao/baseline/fits/main_lya.ini")


@pytest.mark.scientific
def test_vega_adapter_matches_native_data_only_reference():
    """The public adapter removes Vega priors without changing data chi2."""
    if not REFERENCE.is_file():
        pytest.skip(f"missing native Vega reference: {REFERENCE}")
    from lya_interface.likelihoods.vega import VegaLikelihood
    adapter = object.__new__(VegaLikelihood)
    adapter.native_config = str(REFERENCE)
    adapter.use_forestflow = False
    adapter.parameter_mapping = {"ap": "ap", "at": "at", "beta_hcd": "beta_hcd",
                                 "L0_hcd": "L0_hcd", "drp_QSO": "drp_QSO"}
    adapter.parameter_definitions = {
        "ap": {"prior": {"min": .8, "max": 1.2}},
        "at": {"prior": {"min": .8, "max": 1.2}},
        "beta_hcd": {"prior": {"dist": "norm", "loc": .5, "scale": .09}},
        "L0_hcd": {"prior": {"dist": "norm", "loc": 5., "scale": 2.}},
        "drp_QSO": {"prior": {"dist": "norm", "loc": 0., "scale": 1.}},
    }
    adapter.prior_policy = "external"
    adapter.lya_tracer = "LYA"
    adapter.initialize()
    point = dict(ap=1.0, at=1.0, beta_hcd=.5, L0_hcd=5.3, drp_QSO=0.0)
    data_chi2 = adapter.backend.chi2(point, include_priors=False)
    assert -2 * adapter.logp(**point) == pytest.approx(data_chi2)
    assert adapter.backend.chi2(point) - data_chi2 == pytest.approx(.0225)
