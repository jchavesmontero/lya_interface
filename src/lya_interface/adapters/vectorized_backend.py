"""Route scalar Cobaya points through cup1d's columnar observation kernels."""
import numpy as np

from cup1d.likelihood.external import (ExternalP1DLikelihood, apply_observation_model, apply_resolution,
                                      dataset_nuisance, without_resolution)


class VectorizedP1DLikelihood(ExternalP1DLikelihood):
    """
    Preserve the scalar likelihood contract using native batch kernels.

    Cobaya supplies one parameter point at a time. A singleton parameter
    axis selects cup1d's columnar contaminant/resolution/rebinning path,
    without constructing native theory or duplicating any physical model.
    The inherited correlated likelihood contraction is already array based.
    """

    def apply_observation_model(self, p1d_lya_kms, context, nuisance_parameters,
                                metal_moments=None):
        """Apply cup1d's columnar response kernels to one Cobaya point.

        The Kaiser metal model (which needs ``metal_moments``) has no columnar
        kernel and uses the native scalar path.
        """
        if context.request_id != self.request.identity or set(p1d_lya_kms) != set(self.data):
            raise ValueError("prediction request identity or dataset mismatch")
        # Configurations without nuisance parameters retain the native path.
        if not nuisance_parameters or metal_moments is not None:
            return super().apply_observation_model(p1d_lya_kms, context, nuisance_parameters,
                                                   metal_moments)
        columns = {name: np.array([value], dtype=float)
                   for name, value in nuisance_parameters.items()}
        index = {z: i for i, z in enumerate(context.redshifts)}
        groups = {group.identifier: group for group in self.request.groups}
        result = {}
        for keys in self.Rebin_data.groups:
            group = groups[keys[0]]  # data sets of a group share redshifts and fine grids
            for key in keys:
                if len(p1d_lya_kms[key]) != len(groups[key].redshifts):
                    raise ValueError("redshift prediction count mismatch")
            rows = p1d_lya_kms[group.identifier]
            for k, p in zip(group.k_ikms, rows):
                if np.shape(p) != (len(k),) or not np.all(np.isfinite(p)):
                    raise ValueError("malformed or non-finite external prediction")
            zs = np.asarray(group.redshifts)
            k_rows = [np.asarray(k) for k in group.k_ikms]
            inds = [index[z] for z in group.redshifts]
            # contaminants once per group; each data set adds its resolution correction
            base = columns if len(keys) == 1 else without_resolution(columns)
            observed, _ = apply_observation_model(
                self.model_cont, self.model_syst, zs, k_rows,
                [np.asarray(p)[None, :] for p in rows],
                np.asarray(context.mean_flux)[None, inds],
                np.asarray(context.dkms_diMpc)[None, inds],
                dataset_nuisance(keys[0], base, self.syst_data_labels),
            )
            for key in keys:
                own = observed if len(keys) == 1 else apply_resolution(
                    self.model_syst, zs, k_rows, observed,
                    dataset_nuisance(key, columns, self.syst_data_labels))
                result[key] = [row[0] for row in self.Rebin_data.rebinning_batch(key, own)]
        return result
