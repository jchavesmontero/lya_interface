"""Unit-cube sampling keeps flat physical priors and translates covariances."""
import numpy as np
import pytest

from lya_interface.unit_cube import (CUBE_SUFFIX, covariance_from_unit_cube,
                                     covariance_to_unit_cube, unit_cube_params)

PARAMS = {
    "As": {"prior": {"min": 1e-9, "max": 3e-9}, "ref": {"dist": "norm", "loc": 2e-9, "scale": 1e-11},
           "proposal": 2e-11, "latex": "A_s"},
    "igm_tau_eff_0": {"prior": {"min": -0.2, "max": 0.2}, "ref": 0.1, "proposal": 0.004},
    "H0": 67.66,
    "Delta2star": {"derived": True},
}


def test_params_are_sampled_in_unit_cube():
    out = unit_cube_params(PARAMS)
    assert out["H0"] == 67.66 and out["Delta2star"] == {"derived": True}
    cube = out["As" + CUBE_SUFFIX]
    assert cube["prior"] == {"min": 0.0, "max": 1.0}
    assert cube["ref"] == pytest.approx({"dist": "norm", "loc": 0.5, "scale": 0.005})
    assert cube["proposal"] == pytest.approx(0.01)
    assert out["igm_tau_eff_0" + CUBE_SUFFIX]["ref"] == pytest.approx(0.75)
    physical = eval(out["As"]["value"])  # the Cobaya lambda
    assert physical(0.25) == pytest.approx(1.5e-9, rel=1e-14)
    assert out["As"]["derived"] is True and out["As"]["latex"] == "A_s"


def test_covariance_round_trip_and_conditioning():
    names = ["As", "igm_tau_eff_0"]
    covariance = np.array([[1e-21, 1e-12], [1e-12, 4e-3]])
    cube, cube_names = covariance_to_unit_cube(covariance, names, PARAMS)
    assert cube_names == ["As_cube", "igm_tau_eff_0_cube"]
    assert np.all(np.linalg.eigvals(cube) > 0)  # Cobaya's check
    back, back_names = covariance_from_unit_cube(cube, cube_names, PARAMS)
    assert back_names == names
    np.testing.assert_allclose(back, covariance, rtol=1e-12)
    # physical names pass through unchanged
    same, same_names = covariance_from_unit_cube(covariance, names, PARAMS)
    assert same_names == names and np.array_equal(same, covariance)


def test_non_flat_prior_is_rejected():
    with pytest.raises(ValueError, match="flat prior"):
        unit_cube_params({"x": {"prior": {"dist": "norm", "loc": 0, "scale": 1}}})


def test_prior_bounds_are_stored_and_read_from_the_run(tmp_path):
    """A run's unit cube is read from its own folder, never the current configuration."""
    import yaml
    from lya_interface.unit_cube import (bounds_from_unit_cube_params, flat_prior_bounds,
                                         read_prior_bounds, write_prior_bounds)

    bounds = flat_prior_bounds(PARAMS)
    assert set(bounds) == {"As", "igm_tau_eff_0"}
    # the lambdas Cobaya records (e.g. in <prefix>.updated.yaml) give the same bounds
    same = lambda x, y: set(x) == set(y) and all(np.allclose(x[k], y[k], rtol=1e-12) for k in x)
    assert same(bounds_from_unit_cube_params(unit_cube_params(PARAMS)), bounds)
    with pytest.raises(FileNotFoundError):
        read_prior_bounds(tmp_path)
    (tmp_path / "mcmc.updated.yaml").write_text(yaml.safe_dump({"params": unit_cube_params(PARAMS)}))
    assert same(read_prior_bounds(tmp_path), bounds)
    written = {"As": [0.5e-9, 4e-9], "igm_tau_eff_0": [-0.2, 0.2]}
    write_prior_bounds(tmp_path, written)
    assert read_prior_bounds(tmp_path) == written  # prior_bounds.yaml takes precedence


def test_covariance_moves_between_unit_cubes_of_different_priors():
    """A covariance learned with one prior is rescaled, not reused, in another unit cube."""
    old = {"a": [-1.0, 3.0], "b": [0.1, 0.2]}
    new = {"a": [-2.0, 6.0], "b": [0.1, 0.2]}
    physical = np.array([[0.04, 0.001], [0.001, 1e-4]])
    cube, names = covariance_to_unit_cube(physical, ["a", "b"], old)
    back, back_names = covariance_from_unit_cube(cube, names, old)
    moved, _ = covariance_to_unit_cube(back, back_names, new)
    assert np.allclose(back, physical)
    assert moved[0, 0] == pytest.approx(cube[0, 0] / 4)
    assert moved[1, 1] == pytest.approx(cube[1, 1])
