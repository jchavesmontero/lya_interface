"""Optional external BAO composition; fixtures do not claim DESI validation."""
from pathlib import Path
import numpy as np
import pytest
from cobaya.component import get_component_class
from cobaya.likelihood import Likelihood
from cobaya.model import get_model
from lya_interface.configuration import load_configuration
from lya_interface.diagnostics import evaluate_point, evaluate_components, cup1d_component, minimize_point
from test_diagnostics import configuration

ROOT = Path(__file__).resolve().parents[1]


def test_bao_configuration_retains_p1d_and_shared_camb():
    baseline = load_configuration(ROOT / "examples/cup1d_evaluate.yaml")
    joint = load_configuration(ROOT / "examples/cup1d_desidr2_lya_bao.yaml")
    name = "bao.desi_dr2.desi_bao_lya"
    assert set(joint["likelihood"]) == set(baseline["likelihood"]) | {name}
    for section in ("params", "theory"):
        assert joint[section] == baseline[section]
    cls = get_component_class(name, kind="likelihood")
    assert "desi_bao_dr2" in cls.get_defaults()["measurements_file"]
    assert "additional_likelihoods" not in joint and "interface_base" not in joint


def test_additional_likelihood_cannot_overwrite_existing_component(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("likelihood:\n  test: null\nadditional_likelihoods:\n  test: null\n")
    with pytest.raises(ValueError, match="replace existing"):
        load_configuration(path)


class DistanceFixture(Likelihood):
    """Synthetic distance term; not the real BAO measurements."""
    def get_requirements(self):
        return {"Hubble": {"z": [3.]}}

    def logp(self, **params):
        h = float(np.atleast_1d(self.provider.get_Hubble([3.]))[0])
        return -.5 * ((h-560.)/10.)**2 - 2.


class VegaOnlyFixture(Likelihood):
    """Minimal non-P1D component exercising neutral diagnostics."""
    def get_can_support_params(self):
        return ["ap"]

    def logp(self, ap, **params):
        return -3.5


def test_component_diagnostics_support_vega_only_graphs():
    config = {"params": {"ap": {"prior": {"min": .8, "max": 1.2}, "ref": 1.}},
              "likelihood": {"vega": {"external": VegaOnlyFixture}}}
    with get_model(config) as model:
        result = evaluate_components(model, {"ap": 1.})
        assert result.loglikes == {"vega": -3.5}
        assert result.minus2_loglike == {"vega": 7.0}
        assert result.total_minus2_loglike == 7.0
        fit = minimize_point(model, {"ap": 1.}, max_evals=8, verbose=False)
        assert np.isnan(fit.chi2_data)
        assert fit.minus2_loglike_total == pytest.approx(7.0)


def test_combined_diagnostics_select_p1d_independent_of_order():
    info = configuration()
    info["likelihood"] = {"distance": {"external": DistanceFixture}, **info["likelihood"]}
    with get_model(info) as model:
        point = dict(p1d_R_coeff_0=0.)
        result, _ = evaluate_point(model, point)
        post = model.logposterior(point)
        assert cup1d_component(model) is model.likelihood["data"]
        assert sum(post.loglikes) == pytest.approx(result.loglike - 2.)
        fit = minimize_point(model, point, max_evals=300)
        assert fit.success
        assert fit.minus2_loglike_total == pytest.approx(fit.chi2_data + 4.)
        assert model.theory["linear"].calls == 1
