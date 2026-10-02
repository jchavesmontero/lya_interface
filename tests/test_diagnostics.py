"""Actual Cobaya routing with explicit synthetic assets for point diagnostics."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cobaya.model import get_model
from test_cobaya_graph import LinearFixture, ForestFixture, LikelihoodFixture
from lya_interface.theory.lya_cosmology import LyaCosmology
from lya_interface.diagnostics import evaluate_point, plot_point, minimize_point


def configuration(likelihood=LikelihoodFixture):
    return dict(params=dict(As=2e-9, ns=.96, H0=70., mnu=0., nrun=0.,
        igm_gamma_0=1.4, p1d_R_coeff_0=dict(prior=dict(min=-.05, max=.05), ref=0.),
        Delta2star=dict(derived=True), nstar=dict(derived=True), alphastar=dict(derived=True)),
        theory={"linear": {"external": LinearFixture},
                "snapshot": {"external": LyaCosmology},
                "forest": {"external": ForestFixture}},
        likelihood={"data": {"external": likelihood}})


def test_point_plots_all_bins_without_recalculating_theory():
    with get_model(configuration()) as model:
        point = dict(p1d_R_coeff_0=0.)
        result, powers = evaluate_point(model, point)
        calls = model.theory["forest"].calls
        for residuals in (False, True):
            fig, axes = plot_point(model, point, residuals=residuals, title="test")
            data = next(iter(model.likelihood.values())).backend.data
            assert len(axes) == sum(len(d.z) for d in data.values())
            for ax, expected in zip(axes, (p for rows in powers.values() for p in rows)):
                assert len(ax.lines[0].get_xdata()) == len(expected)
            assert "joint" in fig._suptitle.get_text()
            plt.close(fig)
        assert result.valid and model.theory["forest"].calls == calls


def test_minimization_returns_real_fitted_point_and_reports_budget_failure(capsys):
    with get_model(configuration()) as model:
        point = dict(p1d_R_coeff_0=0.)
        initial, _ = evaluate_point(model, point)
        budget_limited = minimize_point(model, point, max_evals=1)
        assert not budget_limited.success
        fit = minimize_point(model, point, max_evals=300, report_every=2)
        output = capsys.readouterr().out
        assert "Evaluation 2:" in output
        assert "chi2_P1D=" in output and "best -2logL_total=" in output
        assert "Finished:" in output
        assert fit.success
        assert fit.chi2_data <= initial.chi2_data + 1e-8
        checked, _ = evaluate_point(model, fit.point)
        np.testing.assert_allclose(checked.chi2_data, fit.chi2_data)


def test_native_blinding_shifts_diagnostics_not_scientific_predictions(monkeypatch):
    from lya_interface.adapters import cup1d_backend
    offsets = dict(Delta2_star=.05, n_star=.01, alpha_star=.005)
    monkeypatch.setattr(cup1d_backend, "native_blinding", lambda path: offsets)

    class Blinded(LikelihoodFixture):
        def get_requirements(self):
            requirements = super().get_requirements()
            requirements["lya_blinding"] = dict(native_config="synthetic metadata")
            return requirements

    point = dict(p1d_R_coeff_0=0.)
    with get_model(configuration()) as plain, get_model(configuration(Blinded)) as blinded:
        a, b = plain.logposterior(point), blinded.logposterior(point)
        np.testing.assert_allclose(b.loglikes, a.loglikes)
        np.testing.assert_allclose(np.asarray(b.derived)-a.derived, list(offsets.values()))
        assert plain.provider.get_result("lya_cosmology").identity == blinded.provider.get_result("lya_cosmology").identity
        # Restoring cached states must not accumulate the offsets.
        blinded.logposterior(dict(p1d_R_coeff_0=.01))
        np.testing.assert_array_equal(blinded.logposterior(point).derived, b.derived)
