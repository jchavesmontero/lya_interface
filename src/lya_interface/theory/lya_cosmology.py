"""Cosmology-only Cobaya Theory. Never depends on IGM or contaminants."""
import numpy as np
from cobaya.theory import Theory
from lya_interface.cosmology import CobayaCosmologySnapshot
from lya_interface.derived import star_parameters, DERIVED_ALIASES


class LyaCosmology(Theory):
    """Provide immutable linear-cosmology snapshots and derived star parameters."""
    stop_at_error: bool = True
    z_star: float = 3.
    k_star_ikms: float = .009
    kmax_iMpc: float = 200.
    snapshot_nodes: int = 4096
    redshifts: list = [0., 2., 4.6]

    def initialize(self):
        """Validate snapshot settings and initialize requested-redshift state.

        Raises
        ------
        ValueError
            If linear coverage or snapshot resolution is invalid.
        """
        if self.kmax_iMpc <= 0 or self.snapshot_nodes < 100:
            raise ValueError("invalid snapshot range/resolution")
        self._zs = set(float(z) for z in self.redshifts) | {float(self.z_star)}
        self.calls = 0
        self._blinding_config = None
        self._blind = {}

    def get_can_provide(self):
        """Declare immutable cosmology and diagnostic-blinding products.

        Returns
        -------
        list of str
            ``lya_cosmology`` and ``lya_blinding`` provider products.
        """
        return ["lya_cosmology", "lya_blinding"]

    def get_can_provide_params(self):
        """Declare derived public linear-power star parameters.

        Returns
        -------
        list of str
            Public aliases for amplitude, slope, and running.
        """
        return list(DERIVED_ALIASES)

    def must_provide(self, **requirements):
        """Merge cosmology/redshift and optional native-blinding requirements.

        Parameters
        ----------
        **requirements
            Cobaya dependency requests, including optional ``lya_cosmology``
            redshifts and ``lya_blinding`` native configuration.

        Returns
        -------
        dict
            CAMB linear-Pk, Hubble, and growth requirements for every merged
            redshift.

        Raises
        ------
        ValueError
            If components request conflicting native blinding configurations.
        """
        super().must_provide(**requirements)
        if "lya_blinding" in requirements:
            from lya_interface.adapters.cup1d_backend import native_blinding
            path = requirements["lya_blinding"]["native_config"]
            if self._blinding_config is not None and self._blinding_config != path:
                raise ValueError("conflicting native diagnostic blinding configurations")
            if self._blinding_config is None:
                self._blind = native_blinding(path)
                self._blinding_config = path
        if "lya_cosmology" in requirements:
            self._zs.update(requirements["lya_cosmology"].get("redshifts", []))
        zs = sorted(self._zs)
        return {"Pk_interpolator": {"z": zs, "k_max": self.kmax_iMpc,
                "nonlinear": False, "vars_pairs": [["delta_nonu", "delta_nonu"]]},
                "Hubble": {"z": zs}, "fsigma8": {"z": zs}, "sigma8_z": {"z": zs}}

    def get_requirements(self):
        """Request primordial and neutrino input parameters from Cobaya.

        Returns
        -------
        dict
            Required ``As``, ``ns``, ``nrun``, and ``mnu`` inputs.
        """
        return {name: None for name in ("As", "ns", "nrun", "mnu")}

    def initialize_with_provider(self, provider):
        """Store the Cobaya provider used to build frozen snapshots.

        Parameters
        ----------
        provider : cobaya.theory.Provider
            Provider supplying linear Pk, Hubble, and growth products.
        """
        self.provider = provider

    def calculate(self, state, want_derived=True, **params_values_dict):
        """Create one frozen snapshot and optional blinded diagnostic values.

        Parameters
        ----------
        state : dict
            Cobaya state updated with provider products.
        want_derived : bool, default=True
            Calculate exported star parameters when true.
        **params_values_dict
            Physical primordial and neutrino parameters.

        Notes
        -----
        The provider interpolation is copied into arrays; snapshots never
        retain a live provider/CAMB object. Blinding affects derived display
        values only, never predictions.
        """
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
        stars = star_parameters(snapshot, self.z_star, self.k_star_ikms) if want_derived else {}
        # As in native cup1d, blinding affects exported diagnostics only.
        # Keep the snapshot and every scientific prediction unshifted.
        state["derived"] = {name: value + self._blind.get(DERIVED_ALIASES[name], 0.)
                            for name, value in stars.items()}
        state["lya_blinding"] = self._blinding_config
        self.calls += 1

    def get_lya_cosmology(self):
        """Return the current immutable Cobaya cosmology snapshot.

        Returns
        -------
        CobayaCosmologySnapshot
            Point-specific linear cosmology product.
        """
        return self.current_state["lya_cosmology"]
