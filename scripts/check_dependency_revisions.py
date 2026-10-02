"""Read-only validation for an asset-provisioned pinned scientific CI runner."""
import argparse
import importlib
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cup1d",required=True)
    parser.add_argument("--lace",default="1f0cc5e91d839088ba801e258129277bd006cda5")
    parser.add_argument("--forestflow",default="2f604bade597c38e7eaece7012f0bfca459f0ae8")
    args=parser.parse_args()
    for name,revision in vars(args).items():
        if len(revision)!=40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError(f"{name}: provide a full pinned commit SHA")
        module=importlib.import_module(name)
        root=Path(module.__file__).resolve().parent
        actual=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
        if actual!=revision:
            raise RuntimeError(f"{name}: installed {actual}, required {revision}")
        print(name,actual)
    from cup1d.likelihood.external import ExternalP1DLikelihood


if __name__=="__main__":
    main()
