from lya_interface.interface import Interface

interface = Interface("examples/cup1d_evaluate.yaml")

loglike = interface.loglike(
    As=2.105209331337507e-9,
    ns=0.9665,
    igm_tau_eff_0=0.0,
    p1d_f_Lya_SiIII_0=-4.0,
)
print(loglike)
interface.close()
