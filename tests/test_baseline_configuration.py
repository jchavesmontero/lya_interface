"""CM2026 examples preserve the maintained native configuration and alias."""
from pathlib import Path
import numpy as np
from cup1d import Args
from cup1d.utils.utils import get_path_repo
from lya_interface.configuration import load_configuration
from lya_interface.adapters.cup1d_backend import native_configuration, registry_for, _serializable

ROOT = Path(__file__).resolve().parents[1]


def test_cm2026_settings_and_full_parameter_registry():
    info = load_configuration(ROOT / "examples/cup1d_evaluate.yaml")
    forest = info["theory"]["lya_interface.theory.forestflow.ForestFlowTheory"]
    like = info["likelihood"]["lya_interface.likelihoods.cup1d.Cup1DLikelihood"]
    actual, igm, nuisance = native_configuration(forest["native_config"])
    baseline = Args.from_yaml(Path(get_path_repo("cup1d")) / "configs/cm2026/cm2026_base.yaml", verbose=False)
    for name in ("data_label", "z_min", "z_max", "kmax_ikms", "k_rebin_factor",
                 "fid_cosmo_label", "fid_igm", "fid_cont", "fid_syst",
                 "ic_correction", "cov_syst_type", "emu_cov_type"):
        assert _serializable(getattr(actual, name)) == _serializable(getattr(baseline, name))
    for name in actual.cov_factor:
        np.testing.assert_array_equal(actual.cov_factor[name], baseline.cov_factor[name])
    assert actual.emulator_label == forest["emulator_label"] == "forest_mpg"
    assert like["blinding_policy"] == "native"
    import forestflow
    assert Path(like["covariance_asset"]) == Path(forestflow.__file__).resolve().parents[1] / "data/covariance/l1O_cov_forest_mpg_fix.npy"
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
