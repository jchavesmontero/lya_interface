"""Immutable, point-specific LaCE-compatible provider snapshot."""
from dataclasses import dataclass, field
import hashlib
import numpy as np
from scipy.interpolate import CubicSpline
from lace.cosmo.base_cosmology import BaseCosmology
from .contracts import NumericalCoverageError


def readonly(array):
    """Return an immutable float64 array detached from the input buffer.

    Parameters
    ----------
    array : array_like
        Numerical values to copy into an immutable contiguous representation.

    Returns
    -------
    numpy.ndarray
        Read-only array with ``float64`` dtype.
    """
    # A bytes-backed array cannot have WRITEABLE re-enabled by consumers.
    a = np.asarray(array, dtype=np.float64)
    return np.frombuffer(a.tobytes(), dtype=np.float64).reshape(a.shape)


@dataclass(frozen=True)
class CobayaCosmologySnapshot(BaseCosmology):
    """Immutable, point-specific linear cosmology supplied by Cobaya.

    Parameters
    ----------
    redshifts : tuple of float
        Provider redshifts at which all background and linear-power products
        are sampled.
    k_iMpc : array_like
        Strictly increasing comoving wavenumber grid in 1/Mpc.
    power_Mpc3 : array_like
        Linear ``delta_nonu`` power with shape ``(nz, nk)`` in Mpc^3.
    hubble, growth : array_like
        Hubble rate in km/s/Mpc and growth rate at ``redshifts``.
    """
    redshifts: tuple
    k_iMpc: object
    power_Mpc3: object
    hubble: object
    growth: object
    primordial: tuple = ()
    mnu: float = 0.
    _splines: tuple = field(init=False, repr=False)
    identity: str = field(init=False)

    def __post_init__(self):
        """Freeze provider arrays, validate coverage, and build splines.

        Raises
        ------
        ValueError
            If redshift, linear-power, background, or growth arrays have
            incompatible shapes or non-finite/non-positive values.
        """
        object.__setattr__(self, "redshifts", tuple(float(z) for z in self.redshifts))
        for name in ("k_iMpc", "power_Mpc3", "hubble", "growth"):
            object.__setattr__(self, name, readonly(getattr(self, name)))
        k, p = self.k_iMpc, self.power_Mpc3
        if len(set(self.redshifts)) != len(self.redshifts) or k.ndim != 1 or len(k) < 4 or np.any(np.diff(k) <= 0) or p.shape != (len(self.redshifts), len(k)):
            raise ValueError("malformed cosmology grid")
        if self.hubble.shape != (len(self.redshifts),) or self.growth.shape != self.hubble.shape:
            raise ValueError("background/growth redshift shape mismatch")
        if any(np.any(~np.isfinite(a)) or np.any(a <= 0) for a in (k, p, self.hubble, self.growth)):
            raise ValueError("cosmology products must be finite and positive")
        splines = tuple(CubicSpline(np.log(k), np.log(row), extrapolate=False) for row in p)
        for spline in splines:
            spline.c = readonly(spline.c)
            spline.x = readonly(spline.x)
        object.__setattr__(self, "_splines", splines)
        digest = hashlib.sha256(repr((self.redshifts, self.primordial, self.mnu)).encode())
        for a in (k, p, self.hubble, self.growth):
            digest.update(a.tobytes())
        object.__setattr__(self, "identity", digest.hexdigest())

    def _index(self, z):
        """Return the exact requested-redshift index.

        Raises
        ------
        NumericalCoverageError
            If Cobaya did not request the supplied redshift.
        """
        try:
            return self.redshifts.index(float(z))
        except ValueError as error:
            raise NumericalCoverageError(f"redshift {z} was not requested from CAMB") from error

    def validate_k(self, k):
        """Validate wavenumbers against stored 1/Mpc linear-power coverage.

        Parameters
        ----------
        k : array-like
            Comoving wavenumbers in 1/Mpc.

        Raises
        ------
        NumericalCoverageError
            If any value is non-finite or outside the stored closed interval.
        """
        k = np.asarray(k)
        if np.any(~np.isfinite(k)) or np.any(k < self.k_iMpc[0]) or np.any(k > self.k_iMpc[-1]):
            raise NumericalCoverageError(f"k lies outside snapshot [{self.k_iMpc[0]}, {self.k_iMpc[-1]}] /Mpc; increase configured provider range")

    def compute_linP_Mpc(self, z, k_Mpc, species="bc"):
        """Interpolate stored baryon-plus-CDM linear power in Mpc cubed.

        Parameters
        ----------
        z : float
            One exact provider redshift.
        k_Mpc : array-like
            In-range wavenumbers in 1/Mpc.
        species : {'bc'}, default='bc'
            Snapshot contains only Cobaya's ``delta_nonu`` (bc) spectrum.

        Returns
        -------
        ndarray
            Interpolated linear power in Mpc cubed.

        Raises
        ------
        ValueError
            If total-matter species is requested.
        NumericalCoverageError
            If redshift or wavenumbers were not supplied by Cobaya.
        """
        if species != "bc":
            raise ValueError("snapshot contains linear delta_nonu (bc), not total matter")
        self.validate_k(k_Mpc)
        return np.exp(self._splines[self._index(z)](np.log(k_Mpc)))

    def _background(self, z, values):
        """Gather exact-redshift background values with input shape preserved.

        Parameters
        ----------
        z : float or array-like
            Requested snapshot redshifts.
        values : ndarray
            One stored value per snapshot redshift.

        Returns
        -------
        ndarray
            Values reshaped like ``z``.
        """
        zs = np.asarray(z)
        return np.asarray([values[self._index(v)] for v in zs.flat]).reshape(zs.shape)

    def compute_hubble_parameter(self, z):
        """Return stored Hubble rate in km/s/Mpc at exact snapshot redshifts."""
        return self._background(z, self.hubble)

    def compute_growth_rate(self, z):
        """Return stored logarithmic linear growth rate at exact redshifts."""
        return self._background(z, self.growth)

    def get_kmax_linP_Mpc(self):
        """Return the largest supported linear wavenumber in 1/Mpc."""
        return float(self.k_iMpc[-1])

    def get_mnu(self):
        """Return the summed neutrino mass in eV."""
        return self.mnu

    def get_primordial_params(self):
        """Return the primordial parameters used to build this snapshot."""
        return dict(self.primordial)
