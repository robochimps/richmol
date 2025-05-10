from typing import Callable, Optional

import numpy as np
from scipy import constants
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import LinearOperator, expm_multiply

from .asymtop import RotStates
from .cartens import CartTensor
from .constants import ENR_INVCM_JOULE


def propagate_rk(
    states: RotStates,
    tens_op: list[CartTensor],
    tens_prefac: list[float],
    field_func: Callable[[float], np.ndarray],
    t0: float,
    t1: float,
    time_unit: str,
    c0: np.ndarray,
    t_eval: Optional[np.ndarray] = None,
):
    time_units = {"ps": 1e-12, "fs": 1e-15, "ns": 1e-9}
    assert time_unit.lower() in time_units, (
        f"Unknown value for 'time_unit' = {time_unit}, "
        + f"accepted values = {list(time_units.keys())}"
    )

    h0 = states.mat()

    fac = (
        -1j
        * np.array([1] + tens_prefac)
        * ENR_INVCM_JOULE
        / constants.value("reduced Planck constant")
        * time_units[time_unit]
    )

    if t_eval is None:
        t_eval = np.linspace(t0, t1, 1000)

    def rhs(time, c_flat_real):
        c = c_flat_real.view(np.complex128)
        field = field_func(time)
        c2 = fac[0] * h0.dot(c)
        for f, tens in zip(fac[1:], tens_op):
            c2 += f * tens.mat_vec(field, c)
        return c2.view(np.float64)

    c0_real = c0.view(np.float64)
    sol = solve_ivp(rhs, (t0, t1), c0_real, t_eval=t_eval, method="DOP853")
    c_t = sol.y.T.copy().view(np.complex128)
    return sol.t, c_t
