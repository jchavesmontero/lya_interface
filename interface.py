"""
IGMHub-Cobaya interface.

Initial version:
    - Cobaya handles the global cosmological/nuisance parameter space.
    - CAMB is used through Cobaya to compute cosmological quantities.
    - CUP1D, CUPIX and VEGA are optional likelihood components.
    - Each likelihood is configured through its own YAML file.
    - The total likelihood is the sum of the individual likelihoods.
    - Cross-correlations are not included yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
import numpy as np

from cobaya.model import get_model


class IGMHubCobaya:
    """
    Top-level interface between Cobaya and the IGMHub likelihood packages.

    Parameters
    ----------
    config_file
        YAML configuration file for the IGMHub-Cobaya interface.

    Notes
    -----
    The interface YAML controls:

        - Cobaya cosmological parameters
        - CUP1D configuration
        - CUPIX configuration
        - VEGA configuration

    A likelihood is enabled only if its configuration path is provided.
    """

    def __init__(self, config_file: str | Path):
        self.config_file = Path(config_file).resolve()

        if not self.config_file.exists():
            raise FileNotFoundError(
                f"Interface configuration not found: {self.config_file}"
            )

        # --------------------------------------------------------------
        # Read interface configuration
        # --------------------------------------------------------------
        self.config = self._read_yaml(self.config_file)

        self._validate_interface_config()

        # --------------------------------------------------------------
        # Initialize Cobaya
        # --------------------------------------------------------------
        self.cobaya_params = self.config["cobaya"].get("params", {})

        self.model = self._initialize_cobaya()

        # --------------------------------------------------------------
        # Initialize likelihoods
        # --------------------------------------------------------------
        self.cup1d = self._initialize_cup1d()
        self.cupix = self._initialize_cupix()
        self.vega = self._initialize_vega()

    # ==================================================================
    # Configuration
    # ==================================================================

    @staticmethod
    def _read_yaml(filename: Path) -> dict[str, Any]:
        """Read a YAML file."""
        with filename.open("r") as f:
            data = yaml.safe_load(f)

        if data is None:
            return {}

        if not isinstance(data, dict):
            raise TypeError(f"YAML file must contain a dictionary: {filename}")

        return data

    def _read_likelihood_config(
        self,
        config_path: str | Path,
    ) -> dict[str, Any]:
        """
        Read a likelihood configuration.

        Relative paths are interpreted relative to the interface
        configuration file.
        """
        path = Path(config_path)

        if not path.is_absolute():
            path = self.config_file.parent / path

        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(f"Likelihood configuration not found: {path}")

        return self._read_yaml(path)

    def _validate_interface_config(self) -> None:
        """Validate the top-level interface configuration."""

        if "cobaya" not in self.config:
            raise ValueError("Interface configuration must contain a 'cobaya' section.")

        if "params" not in self.config["cobaya"]:
            raise ValueError("The 'cobaya' section must contain a 'params' section.")

    # ==================================================================
    # Cobaya
    # ==================================================================

    def _initialize_cobaya(self):
        """
        Initialize a Cobaya model.

        For the initial implementation we use CAMB as the cosmological
        theory code.

        The parameters supplied in the interface YAML are added to the
        Cobaya parameter dictionary.
        """

        # --------------------------------------------------------------
        # Fixed baseline cosmology.
        #
        # Only As and ns are intended to vary in the first version.
        # --------------------------------------------------------------
        params = {
            "As": {
                "prior": {
                    "min": 1.0e-9,
                    "max": 3.0e-9,
                },
                "ref": {
                    "dist": "norm",
                    "loc": 2.1e-9,
                    "scale": 0.05e-9,
                },
                "proposal": 0.02e-9,
                "latex": r"A_s",
            },
            "ns": {
                "prior": {
                    "min": 0.8,
                    "max": 1.2,
                },
                "ref": {
                    "dist": "norm",
                    "loc": 0.965,
                    "scale": 0.01,
                },
                "proposal": 0.005,
                "latex": r"n_s",
            },
            # Fixed cosmology.
            "ombh2": 0.02237,
            "omch2": 0.1200,
            "H0": 67.36,
            "tau": 0.0544,
            "mnu": 0.06,
        }

        # --------------------------------------------------------------
        # Add parameters from the interface YAML.
        #
        # This allows the user to override the default Cobaya setup.
        # --------------------------------------------------------------
        params.update(self.cobaya_params)

        info = {
            "theory": {"camb": {}},
            "params": params,
        }

        return get_model(info)

    # ==================================================================
    # CUP1D
    # ==================================================================

    def _initialize_cup1d(self):
        """
        Initialize CUP1D if a configuration was provided.
        """

        path = self.config.get("likelihoods", {}).get("cup1d")

        if path is None:
            return None

        cup1d_config = self._read_likelihood_config(path)

        # --------------------------------------------------------------
        # Temporary placeholder.
        #
        # This is where the real CUP1D initialization will go.
        # For example:
        #
        # from cup1d.some_module import Cup1dLikelihood
        # return Cup1dLikelihood(cup1d_config)
        # --------------------------------------------------------------
        return _LikelihoodPlaceholder(
            name="cup1d",
            config=cup1d_config,
        )

    # ==================================================================
    # CUPIX
    # ==================================================================

    def _initialize_cupix(self):
        """
        Initialize CUPIX if a configuration was provided.
        """

        path = self.config.get("likelihoods", {}).get("cupix")

        if path is None:
            return None

        cupix_config = self._read_likelihood_config(path)

        return _LikelihoodPlaceholder(
            name="cupix",
            config=cupix_config,
        )

    # ==================================================================
    # VEGA
    # ==================================================================

    def _initialize_vega(self):
        """
        Initialize VEGA if a configuration was provided.
        """

        path = self.config.get("likelihoods", {}).get("vega")

        if path is None:
            return None

        vega_config = self._read_likelihood_config(path)

        return _LikelihoodPlaceholder(
            name="vega",
            config=vega_config,
        )

    # ==================================================================
    # Cosmology
    # ==================================================================

    def get_cosmology(self, As: float, ns: float) -> dict[str, Any]:
        """
        Evaluate the Cobaya cosmological theory for a given As and ns.

        This is mainly a debugging/convenience method at this stage.

        Returns
        -------
        dict
            Selected cosmological quantities computed by CAMB.
        """

        results = self.model.loglike({"As": As, "ns": ns})

        # The exact set of quantities exposed by CAMB/Cobaya will be
        # expanded as the interface develops.
        return results

    # ==================================================================
    # Likelihood evaluation
    # ==================================================================

    def loglike(self, **params: float) -> float:
        """
        Evaluate the complete IGMHub likelihood.

        Parameters
        ----------
        **params
            Global parameter dictionary.

        Returns
        -------
        float
            Total log-likelihood.

        Notes
        -----
        Currently:

            log L_total =
                log L_CUP1D
                + log L_CUPIX
                + log L_VEGA

        Cross-correlations are not included yet.
        """

        # --------------------------------------------------------------
        # Extract cosmological parameters.
        # --------------------------------------------------------------
        As = params["As"]
        ns = params["ns"]

        # --------------------------------------------------------------
        # Run Cobaya theory.
        # --------------------------------------------------------------
        cosmology = self._evaluate_cosmology(As, ns)

        loglike_total = 0.0

        # --------------------------------------------------------------
        # CUP1D
        # --------------------------------------------------------------
        if self.cup1d is not None:
            cup1d_params = self._select_params(
                params,
                self.cup1d,
            )

            loglike_total += self.cup1d.loglike(
                cosmology=cosmology,
                **cup1d_params,
            )

        # --------------------------------------------------------------
        # CUPIX
        # --------------------------------------------------------------
        if self.cupix is not None:
            cupix_params = self._select_params(
                params,
                self.cupix,
            )

            loglike_total += self.cupix.loglike(
                cosmology=cosmology,
                **cupix_params,
            )

        # --------------------------------------------------------------
        # VEGA
        # --------------------------------------------------------------
        if self.vega is not None:
            vega_params = self._select_params(
                params,
                self.vega,
            )

            loglike_total += self.vega.loglike(
                cosmology=cosmology,
                **vega_params,
            )

        return loglike_total

    def _evaluate_cosmology(
        self,
        As: float,
        ns: float,
    ) -> dict[str, Any]:
        """
        Evaluate the Cobaya cosmological theory.

        This method will eventually be expanded to return the specific
        cosmological products required by Lace, ForestFlow, CUP1D,
        CUPIX and VEGA.
        """

        # The exact API for extracting CAMB products should be isolated
        # here, rather than spread through the likelihood code.

        state = self.model.provider

        # At this early stage we simply return the parameter values.
        #
        # This method is deliberately isolated because this is where
        # we will later implement requests such as:
        #
        #   P(k, z)
        #   H(z)
        #   chi(z)
        #   sigma8
        #   growth(z)
        #   etc.
        #
        return {
            "As": As,
            "ns": ns,
            "provider": state,
        }

    # ==================================================================
    # Parameter routing
    # ==================================================================

    @staticmethod
    def _select_params(
        params: dict[str, float],
        likelihood,
    ) -> dict[str, float]:
        """
        Select parameters relevant for a likelihood.

        The current placeholder implementation removes the cosmological
        parameters because they are passed separately through `cosmology`.

        Later this should use the actual parameter declarations of the
        likelihood packages.
        """

        return {key: value for key, value in params.items() if key not in {"As", "ns"}}


class _LikelihoodPlaceholder:
    """
    Temporary adapter representing CUP1D/CUPIX/VEGA.

    This class should disappear once the real IGMHub likelihood
    interfaces are connected.
    """

    def __init__(
        self,
        name: str,
        config: dict[str, Any],
    ):
        self.name = name
        self.config = config

    def loglike(
        self,
        cosmology: dict[str, Any],
        **params: float,
    ) -> float:
        """
        Temporary likelihood evaluation.

        Replace this with the actual CUP1D/CUPIX/VEGA API.
        """

        raise NotImplementedError(
            f"The {self.name} likelihood interface has not been " "connected yet."
        )
