"""Silicon parameters passed from the shared P1D Kaiser model to Vega."""
import numpy as np
import pytest


def _adapter(free):
    """VegaLikelihood with only what ``metal_values`` needs; ``free`` are native P1D names."""
    from cup1d.configuration.args import Args
    from cup1d.models.contaminants.metals.si_kaiser import FAMILIES, SiKaiser
    from lya_interface.likelihoods.vega import VegaLikelihood

    prop_coeffs = {}
    for name, otype in Args._KAISER_OTYPES.items():
        prop_coeffs[name + "_ztype"] = "pivot"
        prop_coeffs[name + "_otype"] = otype
    fid_vals = {name: [0.0, Args._NULL_VALUES[name]] for name in FAMILIES}
    adapter = object.__new__(VegaLikelihood)
    adapter._kaiser = SiKaiser(free_param_names=free, fid_vals=fid_vals, prop_coeffs=prop_coeffs,
                               flat_priors=Args._CONT_FLAT_PRIORS)
    adapter.metal_parameters = {"metal_" + name: name for name in free}
    adapter._sii_x_sii_free = any(name.rsplit("_", 1)[0] in ("f_SiIIxSiII", "sig_SiIIxSiII")
                                  for name in free)
    adapter.zeff = 2.3
    adapter.kms_to_Mpch = 0.0125  # Mpc/h per km/s
    return adapter


def test_sii_x_sii_factor_and_decorrelation_are_passed_in_mpch_when_free():
    adapter = _adapter(["b_SiIIa_0", "sig_SiII_0", "f_SiIIxSiII_0", "sig_SiIIxSiII_0"])
    values = adapter.metal_values({"metal_b_SiIIa_0": np.log(0.01), "metal_sig_SiII_0": np.log(200.0),
                                   "metal_f_SiIIxSiII_0": np.log(3.0),
                                   "metal_sig_SiIIxSiII_0": np.log(80.0)})
    np.testing.assert_allclose(values["forestflow_metal_sii_factor"], 3.0, rtol=1e-12)
    np.testing.assert_allclose(values["forestflow_metal_sii_decorrelation"], 80.0 * 0.0125, rtol=1e-12)
    np.testing.assert_allclose(values["forestflow_metal_decorrelation"]["SiII(1190)"], 200.0 * 0.0125,
                               rtol=1e-12)


def test_sii_x_sii_parameters_are_left_unset_when_not_free_in_p1d():
    adapter = _adapter(["b_SiIIa_0", "sig_SiII_0"])
    values = adapter.metal_values({"metal_b_SiIIa_0": np.log(0.01), "metal_sig_SiII_0": np.log(200.0)})
    assert "forestflow_metal_sii_factor" not in values
    assert "forestflow_metal_sii_decorrelation" not in values


def test_metal_amplitude_exponents_are_the_log_slope_of_the_cumetal_history():
    """For ln|b| linear in z the exponent at z is (1 + z) times the slope."""
    from cup1d.configuration.args import Args
    from cup1d.models.contaminants.metals.si_kaiser import FAMILIES, SiKaiser
    from lya_interface.likelihoods.vega import metal_amplitude_exponents

    prop_coeffs = {}
    for name, otype in Args._KAISER_OTYPES.items():
        prop_coeffs[name + "_otype"] = otype
        prop_coeffs[name + "_ztype"] = "interp_lin" if name == "b_SiIII" else "pivot"
    prop_coeffs["b_SiIII_znodes"] = [2.0, 3.0]
    fid_vals = {name: [0.0, Args._NULL_VALUES[name]] for name in FAMILIES}
    fid_vals["b_SiIII"] = [np.log(0.01), np.log(0.02)]
    kaiser = SiKaiser(free_param_names=["b_SiIII_0", "b_SiIII_1"], fid_vals=fid_vals,
                      prop_coeffs=prop_coeffs, flat_priors=Args._CONT_FLAT_PRIORS)
    native = {"b_SiIII_0": np.log(0.01), "b_SiIII_1": np.log(0.03)}
    alpha = metal_amplitude_exponents(kaiser, native, 2.33)
    np.testing.assert_allclose(alpha["SiIII"], (1 + 2.33) * np.log(3.0), rtol=1e-6)
    assert alpha["SiIIa"] == 0.0  # constant (inactive) history


def test_reference_point_from_a_fit_file_or_a_mapping(tmp_path):
    from lya_interface.likelihoods.vega import reference_point

    point = {"metal_b_SiIII_0": -4.2, "H0": 67.0}
    np.save(tmp_path / "fit.npy", {"format_version": 1, "best_fit": {"point": point}}, allow_pickle=True)
    assert reference_point(str(tmp_path / "fit.npy")) == point
    assert reference_point(point) == point
    np.save(tmp_path / "other.npy", {"point": point}, allow_pickle=True)
    with pytest.raises(ValueError, match="best_fit"):
        reference_point(str(tmp_path / "other.npy"))


def test_reference_must_hold_every_shared_silicon_bias():
    """The exponents depend only on the biases; other missing metal parameters stay fiducial."""
    import logging

    adapter = _adapter(["b_SiIIa_0", "sig_SiII_0", "f_SiIIxSiII_0"])
    adapter.log = logging.getLogger(__name__)
    with pytest.raises(ValueError, match="lacks the shared silicon biases"):
        adapter._native_metal_point({"metal_sig_SiII_0": 5.0})
    assert adapter._native_metal_point({"metal_b_SiIIa_0": -4.0}) == {"b_SiIIa_0": -4.0}
    assert adapter._native_metal_point({"metal_b_SiIIa_0": -4.0, "metal_sig_SiII_0": 5.0,
                                        "metal_f_SiIIxSiII_0": 0.1, "H0": 67.0}) == {
        "b_SiIIa_0": -4.0, "sig_SiII_0": 5.0, "f_SiIIxSiII_0": 0.1}


def test_metal_alpha_reference_requires_a_metal_model():
    from lya_interface.likelihoods.vega import VegaLikelihood

    adapter = object.__new__(VegaLikelihood)
    adapter.native_config = "main.ini"
    adapter.use_forestflow = False
    adapter.metal_model = None
    adapter.metal_alpha_reference = {"metal_b_SiIII_0": -4.2}
    with pytest.raises(ValueError, match="requires metal_model"):
        adapter.initialize()


def test_metal_alpha_reference_from_a_global_ic_file_or_a_fit_file(tmp_path):
    """IC histories are interpolated at the metal nodes; families absent from the file are left out."""
    import logging
    from types import SimpleNamespace

    adapter = _adapter(["b_SiIIa_1", "sig_SiII_0"])
    adapter.log = logging.getLogger(__name__)
    adapter._metal_args = SimpleNamespace(fid_igm={}, fid_syst={},
                                          fid_cont={"b_SiIIa_znodes": [2.2, 3.0], "sig_SiII_znodes": [2.2]})
    ic = {"ns": {"val": 0.96}, "b_SiIIa": {"z": np.array([2.0, 4.0]), "val": np.array([-4.0, -3.0])}}
    np.save(tmp_path / "ic.npy", ic, allow_pickle=True)
    assert adapter._native_metal_reference(str(tmp_path / "ic.npy")) == {"b_SiIIa_1": -3.5}
    point = {"metal_b_SiIIa_1": -4.0, "metal_sig_SiII_0": 5.0}
    np.save(tmp_path / "fit.npy", {"format_version": 1, "best_fit": {"point": point}}, allow_pickle=True)
    assert adapter._native_metal_reference(str(tmp_path / "fit.npy")) == {"b_SiIIa_1": -4.0,
                                                                          "sig_SiII_0": 5.0}


def test_fit_stores_a_fit_reference_as_its_metal_values_and_keeps_an_ic_file_as_a_path(tmp_path):
    """A stored point survives its file being overwritten by the fit that started from it."""
    from lya_interface.fit import _metal_alpha_reference

    point = {"p1d_b_SiIIa_1": -4.0, "metal_f_SiIIxSiII_0": 0.1, "H0": 67.0}
    np.save(tmp_path / "fit.npy", {"format_version": 1, "best_fit": {"point": point}}, allow_pickle=True)
    expected = {"metal_b_SiIIa_1": -4.0, "metal_f_SiIIxSiII_0": 0.1}
    assert _metal_alpha_reference(tmp_path / "fit.npy") == expected
    assert _metal_alpha_reference(point) == expected
    np.save(tmp_path / "ic.npy", {"b_SiIIa": {"z": np.array([2.0, 4.0]), "val": np.array([-4.0, -3.0])}},
            allow_pickle=True)
    assert _metal_alpha_reference(str(tmp_path / "ic.npy")) == str((tmp_path / "ic.npy").resolve())
