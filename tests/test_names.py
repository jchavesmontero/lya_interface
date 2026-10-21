"""Public parameter prefixes and the translator for results saved before them."""
import numpy as np
import pytest
from lya_interface.names import current_name, native_name, nuisance_public_name, translate_legacy


def test_kaiser_silicon_coefficients_are_shared_metal_parameters():
    for native in ("b_SiIII_0", "b_SiIIc_1", "beta_SiII_0", "sig_SiIII_1", "f_SiIIxSiII_0", "sig_SiIIxSiII_1"):
        assert nuisance_public_name(native) == "metal_" + native
        assert native_name("metal_" + native) == native
    for native in ("R_coeff_3", "HCD_damp1_0", "f_Lya_SiIII_0", "s_SiIIa_SiIIb_0"):
        assert nuisance_public_name(native) == "p1d_" + native
    assert native_name("igm_tau_eff_2") == "tau_eff_2"
    assert native_name("xi3d_bias_hcd") is None and native_name("ap") is None


def test_legacy_names_are_translated():
    old = {"p1d_b_SiIII_0": -4.2, "p1d_sig_SiIIxSiII_1": 1.0, "p1d_R_coeff_0": 0.01,
           "bias_hcd": -0.03, "bias_CIV_eff": -0.02, "bias_SiII_1190": -0.005,
           "ap": 1.0, "igm_kF_kms_0": 1.0}
    assert translate_legacy(old) == {
        "metal_b_SiIII_0": -4.2, "metal_sig_SiIIxSiII_1": 1.0, "p1d_R_coeff_0": 0.01,
        "xi3d_bias_hcd": -0.03, "xi3d_bias_CIV_eff": -0.02, "xi3d_bias_SiII_1190": -0.005,
        "ap": 1.0, "igm_kF_kms_0": 1.0}
    current = translate_legacy(old)
    assert translate_legacy(current) == current
    assert current_name("xi3d_L0_hcd") == "xi3d_L0_hcd"
    with pytest.raises(ValueError, match="both translate"):
        translate_legacy({"bias_hcd": 1.0, "xi3d_bias_hcd": 1.0})


def test_reference_point_translates_a_saved_fit(tmp_path):
    from lya_interface.likelihoods.vega import reference_point

    np.save(tmp_path / "old.npy", {"format_version": 1, "best_fit": {"point": {"p1d_b_SiIIa_1": -4.0}}},
            allow_pickle=True)
    assert reference_point(str(tmp_path / "old.npy")) == {"metal_b_SiIIa_1": -4.0}


def test_near_prior_bounds_reports_flat_and_truncated_edges():
    from cobaya.model import get_model
    from lya_interface.diagnostics import format_near_prior_bounds, near_prior_bounds

    model = get_model({"likelihood": {"one": lambda a, b, c: 0}, "params": {
        "a": {"prior": {"min": 0, "max": 1}},
        "b": {"prior": {"dist": "truncnorm", "a": -2, "b": 3, "loc": 0, "scale": 1}},
        "c": {"prior": {"dist": "norm", "loc": 0, "scale": 1}}}})
    rows = near_prior_bounds(model, {"a": 0.995, "b": -1.95, "c": 100.0})
    assert [(r["name"], r["side"], r["prior"]) for r in rows] == [("a", "upper", "uniform"),
                                                                    ("b", "lower", "truncnorm")]
    assert "a " in format_near_prior_bounds(rows) and near_prior_bounds(model, {"a": .5, "b": 0, "c": 0}) == []
