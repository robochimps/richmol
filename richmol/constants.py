import numpy as np
from scipy import constants


class Constant:
    def __init__(self, val: float, doc=""):
        self.val = val
        self.__doc__ = doc

    def __repr__(self):
        return f"Constant({self.val}, doc='{self.__doc__}')"

    def __str__(self):
        return str(self.val)

    # Basic numeric behavior
    def __add__(self, other):
        return self.val + other

    def __radd__(self, other):
        return other + self.val

    def __sub__(self, other):
        return self.val - other

    def __rsub__(self, other):
        return other - self.val

    def __mul__(self, other):
        return self.val * other

    def __rmul__(self, other):
        return other * self.val

    def __truediv__(self, other):
        return self.val / other

    def __rtruediv__(self, other):
        return other / self.val

    def __neg__(self):
        return -self.val

    def __eq__(self, other):
        return self.val == other

    def __float__(self):
        return float(self.val)

    def __int__(self):
        return int(self.val)

    def __array_ufunc__(self, ufunc, method, *inpts, **kwargs):
        inpts_ = [x.val if isinstance(x, Constant) else x for x in inpts]
        res = getattr(ufunc, method)(*inpts_, **kwargs)
        return Constant(res, doc=self.__doc__) if np.isscalar(res) else res


DIP_X_FIELD_JOULE = Constant(
    val=constants.value("atomic unit of electric field")
    * constants.value("atomic unit of electric dipole mom."),
    doc="To convert from Dipole[a.u.] * Field[a.u.] to (Dipole * Field)[Joule]",
)

POL_X_FIELD2_JOULE = Constant(
    val=constants.value("atomic unit of electric field")
    * constants.value("atomic unit of electric field")
    * constants.value("atomic unit of electric polarizability"),
    doc="To convert from Polarizability[a.u.] * Field^2[a.u.^2] to (Polarizability * Field^2)[Joule]",
)

HYPERPOL_X_FIELD3_JOULE = Constant(
    val=constants.value("atomic unit of electric field")
    * constants.value("atomic unit of electric field")
    * constants.value("atomic unit of electric field")
    * constants.value("atomic unit of 1st hyperpolarizability"),
    doc="To convert from Hyperpolarizability[a.u.] * Field^3[a.u.^3] to (Hyperpolarizability * Field^3)[Joule]",
)

ENR_INVCM_JOULE = Constant(
    val=constants.value("Planck constant")
    * constants.value("speed of light in vacuum")
    * 1e2,
    doc="To convert from Energy[cm^-1] into Energy[Joule]",
)

ENR_INVCM_MHZ = Constant(
    val=constants.value("speed of light in vacuum") * 1e-4,
    doc="To convert from Energy[cm^-1] to Energy[MHz]",
)

DIP_X_FIELD_INVCM = Constant(
    val=DIP_X_FIELD_JOULE / ENR_INVCM_JOULE,
    doc="To convert from Dipole[a.u.] * Field[a.u.] to (Dipole * Field)[cm^-1]",
)

POL_X_FIELD2_INVCM = Constant(
    val=POL_X_FIELD2_JOULE / ENR_INVCM_JOULE,
    doc="To convert from Polarizability[a.u.] * Field^2[a.u.^2] to (Polarizability * Field^2)[cm^-1]",
)

HYPERPOL_X_FIELD3_INVCM = Constant(
    val=HYPERPOL_X_FIELD3_JOULE / ENR_INVCM_JOULE,
    doc="To convert from Hyperpolarizability[a.u.] * Field^3[a.u.^3] to (Hyperpolarizability * Field^3)[cm^-1]",
)
