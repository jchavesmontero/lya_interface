"""Extensible derived registry; LaCE's 100 log nodes include endpoints."""
DERIVED_ALIASES = {"Delta2star": "Delta2_star", "nstar": "n_star", "alphastar": "alpha_star"}


def blind_star_parameters(blinding, stars):
    """Star parameters in public names with the native blinding offsets added.

    Parameters
    ----------
    blinding : cup1d.utils.blinding.Blinding
        Native offsets (:func:`lya_interface.adapters.cup1d_backend.native_blinding`).
    stars : mapping
        Values keyed by public names (``Delta2star``, ``nstar``, ``alphastar``).

    Returns
    -------
    dict
        The same keys, blinded through ``blinding.blind`` (cup1d's offsets
        and conventions; no offsets are added here).
    """
    native = blinding.blind({DERIVED_ALIASES[name]: value for name, value in stars.items()})
    return {name: native[DERIVED_ALIASES[name]] for name in stars}


def star_parameters(snapshot, z_star=3., k_star_ikms=.009):
    """Compute public linear-power star parameters from a snapshot.

    Parameters
    ----------
    snapshot : lya_interface.cosmology.CobayaCosmologySnapshot
        Immutable point-specific cosmology.
    z_star : float, default=3.0
        Pivot redshift.
    k_star_ikms : float, default=0.009
        Velocity-space pivot wavenumber in s/km.

    Returns
    -------
    dict
        Public linear ``bc`` power summaries at ``(z_star, k_star_ikms)``:
        ``Delta2star`` is ``k**3 P(k)/(2*pi**2)``, ``nstar`` is the local
        logarithmic slope, and ``alphastar`` is the running returned by
        LaCE's finite-window quadratic log-power fit. These are not literal
        point derivatives.
    """
    native = snapshot.get_linP_kms_params(z_star, k_star_ikms)
    return {public: native[backend] for public, backend in DERIVED_ALIASES.items()}
