"""Extensible derived registry; LaCE's 100 log nodes include endpoints."""
DERIVED_ALIASES = {"Delta2star": "Delta2_star", "nstar": "n_star", "alphastar": "alpha_star"}


def star_parameters(snapshot, z_star=3., k_star_ikms=.009):
    native = snapshot.get_linP_kms_params(z_star, k_star_ikms)
    return {public: native[backend] for public, backend in DERIVED_ALIASES.items()}
