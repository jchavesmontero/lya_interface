from lya_interface.interface import Interface

interface = Interface("interface.yaml")

chi2 = interface.loglike(
    As=2.1e-9,
    ns=0.965,
    cup1d_parameter=1.0,
    cupix_parameter=0.5,
    vega_parameter=0.2,
)
