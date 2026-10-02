"""Cosmology-only Cobaya Theory. Never depends on IGM or contaminants."""
import numpy as np
from cobaya.theory import Theory
from lya_interface.cosmology import CobayaCosmologySnapshot
from lya_interface.derived import star_parameters, DERIVED_ALIASES


class LyaCosmology(Theory):
    stop_at_error: bool = True
    z_star: float = 3.
    k_star_ikms: float = .009
    kmax_iMpc: float = 200.
    snapshot_nodes: int = 4096
    redshifts: list = [0., 2., 4.6]

    def initialize(self):
        if self.kmax_iMpc <= 0 or self.snapshot_nodes < 100:
            raise ValueError("invalid snapshot range/resolution")
        self._zs = set(float(z) for z in self.redshifts) | {float(self.z_star)}
        self.calls = 0

    def get_can_provide(self):
        return ["lya_cosmology"]

    def get_can_provide_params(self):
        return list(DERIVED_ALIASES)

    def must_provide(self, **requirements):
        super().must_provide(**requirements)
        if "lya_cosmology" in requirements:
            self._zs.update(requirements["lya_cosmology"].get("redshifts", []))
        zs = sorted(self._zs)
        return {"Pk_interpolator": {"z": zs, "k_max": self.kmax_iMpc,
                "nonlinear": False, "vars_pairs": [["delta_nonu", "delta_nonu"]]},
                "Hubble": {"z": zs}, "fsigma8": {"z": zs}, "sigma8_z": {"z": zs}}

    def get_requirements(self):
        return {name: None for name in ("As", "ns", "nrun", "mnu")}

    def initialize_with_provider(self, provider):
        self.provider = provider

    def calculate(self, state, want_derived=True, **params_values_dict):
        pk = self.provider.get_Pk_interpolator(nonlinear=False, var_pair=("delta_nonu", "delta_nonu"))
        zs = tuple(sorted(self._zs))
        # Copy provider interpolation: no captured provider or live CAMB object.
        k = np.geomspace(pk.kmin, min(pk.kmax, self.kmax_iMpc), self.snapshot_nodes)
        snapshot = CobayaCosmologySnapshot(zs, k, [pk.P(z, k) for z in zs],
            self.provider.get_Hubble(zs, units="km/s/Mpc"),
            self.provider.get_fsigma8(zs) / self.provider.get_sigma8_z(zs),
            primordial=tuple((name, float(params_values_dict[name])) for name in ("As", "ns", "nrun")),
            mnu=float(params_values_dict["mnu"]))
        state["lya_cosmology"] = snapshot
        state["derived"] = star_parameters(snapshot, self.z_star, self.k_star_ikms) if want_derived else {}
        self.calls += 1

    def get_lya_cosmology(self):
        return self.current_state["lya_cosmology"]
