"""Evaluate or run P1D, BAO-only, or joint checks; retain native P1D blinding."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from cobaya.model import get_model
from cobaya.run import run
from lya_interface.configuration import load_configuration
from lya_interface.provenance import collect
from lya_interface.diagnostics import evaluate_point


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configuration", nargs="?", default="examples/cup1d_evaluate.yaml")
    parser.add_argument("--provenance", default=None)
    parser.add_argument("--output", default=None, help="override sampler output prefix")
    parser.add_argument("--packages-path", default=None, help="Cobaya external-data installation directory (for BAO)")
    parser.add_argument("--point", default=None, help="JSON mapping overriding sampled reference values")
    args = parser.parse_args()
    info = load_configuration(args.configuration)
    if args.packages_path:
        info["packages_path"] = str(Path(args.packages_path).resolve())
    if args.output:
        info["output"] = args.output
    if args.provenance:
        Path(args.provenance).write_text(json.dumps(collect(info), indent=2))
    if "sampler" in info:
        updated, sampler = run(info)
        if args.provenance:
            Path(args.provenance).write_text(json.dumps(collect(info, sampler.model), indent=2))
        return
    point = {name: definition["ref"] for name, definition in info["params"].items()
             if isinstance(definition, dict) and "prior" in definition}
    if args.point:
        point.update(json.loads(args.point))
    start = perf_counter()
    with get_model(info) as model:
        initialized = perf_counter()
        posterior = model.logposterior(point)
        cold = perf_counter()
        repeat = model.logposterior(point)
        warm = perf_counter()
        if args.provenance:
            Path(args.provenance).write_text(json.dumps(collect(info, model), indent=2))
        diagnostics = dict(loglike_total=float(sum(posterior.loglikes)),
            loglikes=dict(zip(model.likelihood, map(float, posterior.loglikes))),
            minus2_logposterior=-2*posterior.logpost, derived=dict(zip(model.parameterization.derived_params(), posterior.derived)),
            seconds=dict(initialization=initialized-start, cold=cold-initialized, warm=warm-cold))
        if any(hasattr(component, "backend") for component in model.likelihood.values()):
            result, _ = evaluate_point(model, point)
            diagnostics.update(loglike=result.loglike, chi2_data=result.chi2_data,
                               logdet_cov=result.logdet_cov, ndata=result.ndata)
        print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
