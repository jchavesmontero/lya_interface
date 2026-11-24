"""Write an interface Cobaya configuration equivalent to a native cup1d YAML.

Usage
-----
python scripts/config_from_cup1d.py ../cup1d/configs/hires/hires_forestflow.yaml \
    examples/hires_forestflow_evaluate.yaml

Hand-written examples: update only their As/ns flat priors from native cup1d
(emulator training set and cosmo_prior_nsigma), keeping comments:

python scripts/config_from_cup1d.py --sync-cosmo-priors examples/cup1d_evaluate.yaml ...
"""
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from lya_interface.from_cup1d import sync_cosmo_priors, write_interface_config


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sync-cosmo-priors", action="store_true",
                        help="update As/ns priors of the given interface YAMLs in place")
    parser.add_argument("paths", nargs="+",
                        help="<cup1d_yaml> <output_yaml>, or interface YAMLs with --sync-cosmo-priors")
    args = parser.parse_args()
    if args.sync_cosmo_priors:
        for path in args.paths:
            for name, (old, new) in sync_cosmo_priors(path).items():
                print(f"{path}: {name} {old} -> {new}")
        return
    if len(args.paths) != 2:
        parser.error("expected <cup1d_yaml> <output_yaml>")
    path = write_interface_config(*args.paths)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
