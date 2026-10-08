"""Small parameter-definition service used by shared IGM histories."""


def make_parameter(name, min_value, max_value, value=None,
                   Gauss_priors_width=None, fixed=False,
                   hessian_transform=None):
    """Return the legacy plain-dictionary parameter representation.

    This is intentionally compatible with cup1d's public parameter records,
    without importing a likelihood module into the shared IGM package.
    """
    return {"name": name, "value": value, "min_value": min_value,
            "max_value": max_value, "Gauss_priors_width": Gauss_priors_width,
            "fixed": fixed, "hessian_transform": hessian_transform}
