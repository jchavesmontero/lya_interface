"""Route scalar Cobaya points through cup1d's columnar observation kernels."""
import numpy as np

from cup1d.likelihood.external import ExternalP1DLikelihood, apply_observation_model


class VectorizedP1DLikelihood(ExternalP1DLikelihood):
    """
    Preserve the scalar likelihood contract using native batch kernels.

    Cobaya supplies one parameter point at a time. A singleton parameter
    axis selects cup1d's columnar contaminant/resolution/rebinning path,
    without constructing native theory or duplicating any physical model.
    The inherited correlated likelihood contraction is already array based.
    """

    def apply_observation_model(self, p1d_lya_kms, context, nuisance_parameters):
        """Apply cup1d's columnar response kernels to one Cobaya point."""
        if context.request_id != self.request.identity or set(p1d_lya_kms) != set(self.data):
            raise ValueError("prediction request identity or dataset mismatch")
        # Configurations without nuisance parameters retain the native path.
        if not nuisance_parameters:
            return super().apply_observation_model(p1d_lya_kms, context, nuisance_parameters)
        columns = {name: np.array([value], dtype=float)
                   for name, value in nuisance_parameters.items()}
        index = {z: i for i, z in enumerate(context.redshifts)}
        result = {}
        for group in self.request.groups:
            rows = p1d_lya_kms[group.identifier]
            if len(rows) != len(group.redshifts):
                raise ValueError("redshift prediction count mismatch")
            for k, p in zip(group.k_ikms, rows):
                if np.shape(p) != (len(k),) or not np.all(np.isfinite(p)):
                    raise ValueError("malformed or non-finite external prediction")
            inds = [index[z] for z in group.redshifts]
            observed, _ = apply_observation_model(
                self.model_cont, self.model_syst, np.asarray(group.redshifts),
                [np.asarray(k) for k in group.k_ikms],
                [np.asarray(p)[None, :] for p in rows],
                np.asarray(context.mean_flux)[None, inds],
                np.asarray(context.dkms_diMpc)[None, inds], columns,
            )
            result[group.identifier] = [row[0] for row in
                self.Rebin_data.rebinning_batch(group.identifier, observed)]
        return result
