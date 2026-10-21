"""Public parameter names: prefixes by role, and a translator for older names.

Prefixes
--------
``igm_``
    IGM history coefficients (cup1d), shared by P1D and xi3D through ForestFlow.
``metal_``
    Kaiser silicon coefficients (cup1d ``Si_kaiser`` families), one parameter
    used by both the P1D likelihood and Vega (at its effective redshift).
``p1d_``
    P1D-only nuisances (CM2026 metal templates, HCD, resolution).
``xi3d_``
    xi3D-only Vega nuisances. The BAO dilations ``ap`` and ``at`` keep their names.

Results written before the ``metal_``/``xi3d_`` prefixes (Kaiser metals were
``p1d_<family>_<i>``, Vega nuisances had their bare names) are read through
:func:`translate_legacy`.
"""
from cup1d.models.contaminants.metals.si_kaiser import FAMILIES as METAL_FAMILIES
from cup1d.utils.utils import split_string

IGM_PREFIX = "igm_"
METAL_PREFIX = "metal_"
P1D_PREFIX = "p1d_"
XI3D_PREFIX = "xi3d_"
# Prefixes of public names owned by cup1d (native name after the prefix).
CUP1D_PREFIXES = (IGM_PREFIX, METAL_PREFIX, P1D_PREFIX)

# Vega public names used before the xi3d_ prefix, old -> new.
LEGACY_XI3D_NAMES = {name: XI3D_PREFIX + name for name in (
    "bias_hcd", "beta_hcd", "L0_hcd", "bias_CIV_eff", "desi_inst_sys_amp",
    "bias_QSO", "drp_QSO", "qso_rad_strength", "sigma_velo_disp_lorentz_QSO",
    "bias_SiIII_1207", "bias_SiII_1190", "bias_SiII_1193", "bias_SiII_1260",
)}


def is_metal(native):
    """Return whether a native cup1d coefficient belongs to the shared Kaiser silicon model."""
    return split_string(native)[0] in METAL_FAMILIES


def nuisance_public_name(native):
    """Return the public name of a native cup1d nuisance coefficient.

    Parameters
    ----------
    native : str
        Native coefficient, e.g. ``b_SiIII_0`` or ``R_coeff_3``.

    Returns
    -------
    str
        ``metal_<native>`` for the shared Kaiser silicon model, else ``p1d_<native>``.
    """
    return (METAL_PREFIX if is_metal(native) else P1D_PREFIX) + native


def native_name(public):
    """Return the native cup1d name of an ``igm_``/``metal_``/``p1d_`` public name, else None."""
    for prefix in CUP1D_PREFIXES:
        if public.startswith(prefix):
            return public[len(prefix):]
    return None


def current_name(name):
    """Translate one public name written before the ``metal_``/``xi3d_`` prefixes.

    Parameters
    ----------
    name : str
        Public name, old or current.

    Returns
    -------
    str
        The current name (unchanged if it is not an old one).
    """
    if name in LEGACY_XI3D_NAMES:
        return LEGACY_XI3D_NAMES[name]
    if name.startswith(P1D_PREFIX) and is_metal(name[len(P1D_PREFIX):]):
        return METAL_PREFIX + name[len(P1D_PREFIX):]
    return name


def translate_legacy(values):
    """Return a copy of a public-name mapping with old names translated.

    Parameters
    ----------
    values : mapping
        Public name to value, e.g. a best-fit point saved before the renaming.

    Returns
    -------
    dict
        Same values keyed by current names.

    Raises
    ------
    ValueError
        If an old and a current name collide.
    """
    out = {}
    for name, value in values.items():
        new = current_name(name)
        if new in out:
            raise ValueError(f"{name!r} and another entry both translate to {new!r}")
        out[new] = value
    return out
