"""Optional real DESI DR2 BAO checks with the official external data product."""
from copy import deepcopy
from pathlib import Path
import os
import numpy as np
import pytest
from cobaya.model import get_model
from lya_interface.configuration import load_configuration
from lya_interface.diagnostics import evaluate_point

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bao_packages():
    path = os.environ.get("LYA_BAO_PACKAGES_PATH")
    if not path:
        pytest.skip("optional BAO check: set LYA_BAO_PACKAGES_PATH to installed official BAO data")
    for name in ("mean", "cov"):
        asset = Path(path) / f"data/bao_data/desi_bao_dr2/desi_gaussian_bao_Lya_GCcomb_{name}.txt"
        if not asset.is_file():
            pytest.fail(f"requested BAO validation is missing {asset}")
    return path


@pytest.mark.scientific
def test_real_bao_distances_and_independent_gaussian(bao_packages):
    info = load_configuration(ROOT / "examples/desidr2_lya_bao_only.yaml")
    info["packages_path"] = bao_packages
    cov = np.loadtxt(Path(bao_packages) / "data/bao_data/desi_bao_dr2/desi_gaussian_bao_Lya_GCcomb_cov.txt")
    with get_model(info) as model:
        first = model.logposterior(dict(H0=67.66))
        likelihood = model.likelihood["bao.desi_dr2.desi_bao_lya"]
        rd = model.provider.get_param("rdrag")
        z = likelihood.data["z"].to_numpy()
        distances = {
            "DM_over_rs": (1+z) * model.provider.get_angular_diameter_distance(z) / rd,
            "DH_over_rs": 299792.458 / model.provider.get_Hubble(z) / rd,
        }
        prediction = np.array([distances[name][i] for i, name in enumerate(likelihood.data["observable"])])
        residual = prediction - likelihood.data["value"].to_numpy()
        expected = -.5 * residual @ np.linalg.solve(cov, residual)
        np.testing.assert_allclose(sum(first.loglikes), expected, rtol=1e-10, atol=1e-12)
        changed = model.logposterior(dict(H0=68.))
        assert np.isfinite(changed.logpost) and changed.loglikes != first.loglikes


@pytest.mark.scientific
def test_real_bao_and_p1d_share_camb_and_report_separate_terms(bao_packages):
    # Explicit small P1D regression fixture, NOT a covariance-enabled CM2026 fit.
    info = load_configuration(ROOT / "examples/validation_demo.yaml")
    info["packages_path"] = bao_packages
    info["likelihood"]["bao.desi_dr2.desi_bao_lya"] = None
    info["params"]["H0"] = dict(prior=dict(min=66., max=69.), ref=67.66, proposal=.1)
    for section in ("theory", "likelihood"):
        for options in info[section].values():
            if isinstance(options, dict) and "parameter_definitions" in options:
                options["parameter_definitions"] = deepcopy(info["params"])
    point = {k: d["ref"] for k, d in info["params"].items() if isinstance(d, dict) and "prior" in d}
    with get_model(info) as model:
        assert list(model.theory).count("camb") == 1
        posterior = model.logposterior(point)
        result, _ = evaluate_point(model, point)
        terms = dict(zip(model.likelihood, posterior.loglikes))
        p1d = "lya_interface.likelihoods.cup1d.Cup1DLikelihood"
        np.testing.assert_allclose(terms[p1d], result.loglike, rtol=1e-12)
        assert np.isfinite(terms["bao.desi_dr2.desi_bao_lya"])
        changed = model.logposterior({**point, "H0": 67.8})
        assert np.isfinite(changed.logpost)
        assert changed.loglikes[-1] != posterior.loglikes[-1]
