"""Evaluate or run the complete demonstration; outputs unblinded diagnostics."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from cobaya.model import get_model
from cobaya.run import run
from lya_interface.configuration import load_configuration
from lya_interface.provenance import collect
from lya_interface.parameters import route


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configuration", nargs="?", default="examples/cup1d_evaluate.yaml")
    parser.add_argument("--provenance", default=None)
    parser.add_argument("--output", default=None, help="override sampler output prefix")
    parser.add_argument("--point", default=None, help="JSON mapping overriding sampled reference values")
    args = parser.parse_args()
    info = load_configuration(args.configuration)
    if args.output:
        info["output"] = args.output
    provenance = collect(info)
    if args.provenance:
        Path(args.provenance).write_text(json.dumps(provenance, indent=2))
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
        prediction = model.provider.get_result("forestflow_p1d")
        if not prediction["valid"]:
            raise ValueError(prediction["reason"])
        likelihood = next(iter(model.likelihood.values()))
        if args.provenance:
            Path(args.provenance).write_text(json.dumps(collect(info, model), indent=2))
        nuisance = {k: point.get(k, info["params"][k]) for k in likelihood.mapping}
        result = likelihood.backend.evaluate_from_p1d(prediction["powers"], prediction["context"], route(nuisance, likelihood.mapping))
        print(json.dumps(dict(loglike=result.loglike, chi2_data=result.chi2_data,
            logdet_cov=result.logdet_cov, ndata=result.ndata,
            minus2_logposterior=-2*posterior.logpost, derived=dict(zip(model.parameterization.derived_params(), posterior.derived)),
            seconds=dict(initialization=initialized-start, cold=cold-initialized, warm=warm-cold)), indent=2))


if __name__ == "__main__":
    main()
