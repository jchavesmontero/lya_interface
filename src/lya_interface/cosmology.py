"""Immutable, point-specific LaCE-compatible provider snapshot."""
from dataclasses import dataclass, field
import hashlib
import numpy as np
from scipy.interpolate import CubicSpline
from lace.cosmo.base_cosmology import BaseCosmology
from .contracts import NumericalCoverageError


def readonly(array):
    # A bytes-backed array cannot have WRITEABLE re-enabled by consumers.
    a = np.asarray(array, dtype=np.float64)
    return np.frombuffer(a.tobytes(), dtype=np.float64).reshape(a.shape)


@dataclass(frozen=True)
class CobayaCosmologySnapshot(BaseCosmology):
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
        try:
            return self.redshifts.index(float(z))
        except ValueError as error:
            raise NumericalCoverageError(f"redshift {z} was not requested from CAMB") from error

    def validate_k(self, k):
        k = np.asarray(k)
        if np.any(~np.isfinite(k)) or np.any(k < self.k_iMpc[0]) or np.any(k > self.k_iMpc[-1]):
            raise NumericalCoverageError(f"k lies outside snapshot [{self.k_iMpc[0]}, {self.k_iMpc[-1]}] /Mpc; increase configured provider range")

    def compute_linP_Mpc(self, z, k_Mpc, species="bc"):
        if species != "bc":
            raise ValueError("snapshot contains linear delta_nonu (bc), not total matter")
        self.validate_k(k_Mpc)
        return np.exp(self._splines[self._index(z)](np.log(k_Mpc)))

    def _background(self, z, values):
        zs = np.asarray(z)
        return np.asarray([values[self._index(v)] for v in zs.flat]).reshape(zs.shape)

    def compute_hubble_parameter(self, z):
        return self._background(z, self.hubble)

    def compute_growth_rate(self, z):
        return self._background(z, self.growth)

    def get_kmax_linP_Mpc(self):
        return float(self.k_iMpc[-1])

    def get_mnu(self):
        return self.mnu

    def get_primordial_params(self):
        return dict(self.primordial)
