"""Real native baseline metadata/export checks; no covariance is fabricated."""
from pathlib import Path
import numpy as np
import pytest
from cobaya.model import get_model
from lya_interface.configuration import load_configuration
from lya_interface.adapters.cup1d_backend import native_configuration, native_blinding
from test_scientific import unavailable

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.scientific
def test_cm2026_export_matches_native_fiducial_parameters_and_bounds():
    from cup1d.emulator.factory import set_emulator
    from cup1d.theory.factory import set_theory
    from cup1d.likelihood.parameters import set_free_likelihood_parameters
    import forestflow
    weights = Path(forestflow.__file__).resolve().parents[1] / "data/emulator_models/forest_mpg_fix.pt"
    if not weights.is_file():
        unavailable(f"missing native baseline weights {weights}")
    info = load_configuration(ROOT / "examples/cup1d_evaluate.yaml")
    path = info["theory"]["lya_interface.theory.forestflow.ForestFlowTheory"]["native_config"]
    args, igm, nuisance = native_configuration(path)
    names = set_free_likelihood_parameters(args, args.emulator_label)
    # Native setup is a reference check only, never invoked inside evaluation.
    theory = set_theory(args, set_emulator(args.emulator_label), names,
                        use_hull=False, zs=np.arange(2.2, 4.21, .2))
    native = theory.get_parameters()
    for name, value in theory.fid_cosmo["cosmo"].get_background_params().items():
        if name in info["params"] and not isinstance(info["params"][name], dict):
            np.testing.assert_allclose(info["params"][name], value, rtol=1e-12)
    for z, conversion in info["likelihood"]["lya_interface.likelihoods.cup1d.Cup1DLikelihood"]["fiducial_M"].items():
        np.testing.assert_allclose(conversion, theory.fid_cosmo["cosmo"].get_dkms_dMpc(float(z)), rtol=1e-12)
    for name in names:
        public = "igm_" + name if name in igm else "p1d_" + name if name in nuisance else name
        definition = info["params"][public]
        np.testing.assert_allclose(definition["ref"], native[name]["value"], rtol=1e-12)
        np.testing.assert_allclose(
            [definition["prior"]["min"], definition["prior"]["max"]],
            [native[name]["min_value"], native[name]["max_value"]], rtol=1e-12
        )


@pytest.mark.scientific
def test_cm2026_native_blinding_and_required_covariance():
    from cup1d.p1ds.factory import set_p1d
    from cup1d.utils.blinding import set_blinding
    from lya_interface.adapters.cup1d_backend import make_backend
    info = load_configuration(ROOT / "examples/cup1d_evaluate.yaml")
    like = info["likelihood"]["lya_interface.likelihoods.cup1d.Cup1DLikelihood"]
    args, _, _ = native_configuration(like["native_config"])
    data = set_p1d(args, args.data_label[0])
    assert data.apply_blinding
    seed = int.from_bytes(data.blinding.encode("utf-8"), byteorder="big")
    assert native_blinding(like["native_config"]) == set_blinding(True, seed)
    with pytest.raises(ValueError, match="blinding_policy"):
        make_backend(like["native_config"], info["params"],
                     covariance_asset=like["covariance_asset"], fiducial_M=like["fiducial_M"])
    asset = Path(like["covariance_asset"])
    if not asset.is_file():
        # Missing calibrated covariance must not fall back to disabled error.
        with pytest.raises(FileNotFoundError, match="l1O_cov_forest_mpg_fix"):
            get_model(info)
    else:
        with get_model(info) as model:
            point = {k: d["ref"] for k, d in info["params"].items()
                     if isinstance(d, dict) and "prior" in d}
            assert np.isfinite(model.logposterior(point).logpost)
            # Compare the columnar path against native scalar observation
            # and rebinning on the actual covariance-enabled DR1 baseline.
            from cup1d.likelihood.external import ExternalP1DLikelihood
            from lya_interface.diagnostics import cup1d_component
            from lya_interface.parameters import route
            component = cup1d_component(model)
            prediction = model.provider.get_result("forestflow_p1d")
            values = model.parameterization.input_params()
            nuisance = route({name: values[name] for name in component.mapping}, component.mapping)
            actual = component.backend.apply_observation_model(
                prediction["powers"], prediction["context"], nuisance,
            )
            reference = ExternalP1DLikelihood.apply_observation_model(
                component.backend, prediction["powers"], prediction["context"], nuisance,
            )
            for group in actual:
                for a, b in zip(actual[group], reference[group]):
                    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
