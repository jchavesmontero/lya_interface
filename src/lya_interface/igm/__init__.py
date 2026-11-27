"""IGM histories: cup1d's implementation, re-exported.

The models live in :mod:`cup1d.models.igm` (single source of truth, so a
Cobaya run evaluates exactly the native histories); this package only keeps
the ``lya_interface.igm`` import path.
"""

from cup1d.models.igm.base_igm import IGM_model
from cup1d.models.igm.mean_flux_class import MeanFlux
from cup1d.models.igm.model_igm import IGM
from cup1d.models.igm.pressure_class import Pressure
from cup1d.models.igm.thermal_class import Thermal

__all__ = ["IGM", "IGM_model", "MeanFlux", "Thermal", "Pressure"]
