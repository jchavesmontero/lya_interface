"""CM2026 examples preserve the maintained native configuration and alias."""
from pathlib import Path
import numpy as np
from cup1d import Args
from cup1d.utils.utils import get_path_repo
from lya_interface.configuration import load_configuration
from lya_interface.cup1d_bridge import native_configuration, registry_for, _serializable

ROOT = Path(__file__).resolve().parents[1]


def test_cm2026_settings_and_full_parameter_registry():
    info = load_configuration(ROOT / "examples/cup1d_evaluate.yaml")
    forest = info["theory"]["lya_interface.theory.p1d_emulator.P1DEmulatorTheory"]
    like = info["likelihood"]["lya_interface.likelihoods.cup1d.Cup1DLikelihood"]
    actual, igm, nuisance = native_configuration(forest["native_config"])
    baseline = Args.from_yaml(Path(get_path_repo("cup1d")) / "configs/cm2026/cm2026_base.yaml", verbose=False)
    for name in ("data_label", "z_min", "z_max", "kmax_ikms",
                 "fid_igm", "fid_cont", "fid_syst",
                 "ic_correction", "cov_syst_type", "emu_cov_type"):
        assert _serializable(getattr(actual, name)) == _serializable(getattr(baseline, name))
    # Intentional difference: the fiducial cosmology of the Vega DR2 template.
    # The converged numerical settings are cup1d's defaults, inherited as is.
    assert actual.fid_cosmo_label == "Planck18_noBAO_wnu"
    for name in ("k_rebin_factor", "n_k_emu", "cosmo_prior_nsigma"):
        assert getattr(actual, name) == getattr(baseline, name)
    assert (actual.k_rebin_factor, actual.n_k_emu, actual.cosmo_prior_nsigma) == (32, 64, 2.0)
    for name in actual.cov_factor:
        np.testing.assert_array_equal(actual.cov_factor[name], baseline.cov_factor[name])
    assert actual.emulator_label == forest["emulator_label"] == "forest_mpg_cinn"
    assert like["blinding_policy"] == "native"
    import forestflow
    assert Path(like["covariance_asset"]) == Path(forestflow.__file__).resolve().parents[1] / "data/covariance/l1O_cov_forest_mpg_cinn.npy"
    registry = registry_for(actual, info["params"], igm, nuisance)
    assert len(registry.mapping("IGM")) == 16
    assert len(registry.mapping("nuisance")) == 35
    assert all(p.status == "sampled" for p in registry.entries.values() if p.owner in {"IGM", "nuisance"})


def test_missing_covariance_directory_resolves_to_sibling_checkout(tmp_path):
    examples = tmp_path / "lya_interface" / "examples"
    examples.mkdir(parents=True)
    (tmp_path / "ForestFlow").mkdir()  # No scientific assets or covariance directory.
    configuration = examples / "point.yaml"
    configuration.write_text(
        "likelihood:\n  lya_interface.likelihoods.cup1d.Cup1DLikelihood:\n"
        "    covariance_asset: ../ForestFlow/data/covariance/missing.npy\n"
    )
    info = load_configuration(configuration)
    assert Path(info["likelihood"]["lya_interface.likelihoods.cup1d.Cup1DLikelihood"]["covariance_asset"]) == tmp_path / "ForestFlow/data/covariance/missing.npy"


