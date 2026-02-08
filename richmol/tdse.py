from typing import Callable, Optional

import numpy as np
from scipy import constants
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import LinearOperator, expm_multiply, onenormest

from .asymtop import RotStates
from .cartens import CartTensor
from .constants import ENR_INVCM_JOULE, ENR_INVCM_MHZ
from .pyexpokit import zhexpv


def propagate_expokit_eb(
    states: RotStates,
    tens_op: tuple[list[CartTensor], list[CartTensor]],
    tens_prefac: tuple[list[float], list[float]],
    field_func: tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]],
    t0: float,
    t1: float,
    dt: float,
    time_unit: str,
    c0: np.ndarray,
    split_method: bool = True,
    field_tol: tuple[float, float] = [1e-12, 1e-12],
    no_krylov: int = 12,
    vec_tol: float = 0,
    timestep_func: Callable[[int, float, np.ndarray], None] = lambda *_: None,
):
    time_units = {"ps": 1e-12, "fs": 1e-15, "ns": 1e-9, "us": 1e-6}
    assert time_unit.lower() in time_units, (
        f"Unknown value for 'time_unit' = {time_unit}, "
        + f"accepted values: {list(time_units.keys())}"
    )

    try:
        enr_units = states.enr_units
    except AttributeError:
        raise AttributeError(f"States object has no attribute 'enr_units'") from None

    if enr_units == "invcm":
        enr_to_joule = ENR_INVCM_JOULE
    elif enr_units == "mhz":
        enr_to_joule = ENR_INVCM_JOULE / ENR_INVCM_MHZ
    else:
        raise ValueError(f"Unknown value for 'enr_units' = {enr_units}") from None

    print(f"energy units: {enr_units}")

    h0 = states.mat()

    fac0 = (
        -1j
        * dt
        * enr_to_joule
        / constants.value("reduced Planck constant")
        * time_units[time_unit]
    )

    fac = [
        -1j
        * dt
        * np.array(prefac)
        * enr_to_joule
        / constants.value("reduced Planck constant")
        * time_units[time_unit]
        for prefac in tens_prefac
    ]

    h0_exp = np.exp(0.5 * fac0 * h0.diagonal())

    def matvec(c, field):
        if c.ndim == 1:
            c_ = c
        else:
            c_ = c[:, 0]
        c2 = np.sum(
            [
                fc * tens.mat_vec(field1, c_)
                for fac1, tens1, field1 in zip(fac, tens_op, field)
                for fc, tens in zip(fac1, tens1)
            ],
            axis=0,
        )
        if not split_method:
            c2 += fac0 * h0.dot(c_)
        if c.ndim == 1:
            return c2
        else:
            return np.array([c2]).T

    nt = int((t1 - t0) / dt)
    time = np.linspace(t0, t1, num=nt, endpoint=False)
    time_dt = time + dt
    time_dt_c = time + dt / 2

    c = c0.copy()
    c_t = []

    for it, t in enumerate(time_dt_c):
        field = [fn(t) for fn in field_func]
        matvec_t = lambda c: matvec(c, field)

        if split_method:
            c = c * h0_exp
            if np.any(np.linalg.norm(field, axis=-1) > field_tol):
                norm = onenormest(
                    LinearOperator(
                        shape=h0.shape,
                        dtype=np.complex128,
                        matvec=matvec_t,
                        rmatvec=matvec_t,
                    )
                )
                c = zhexpv(c, no_krylov, norm, matvec_t, tol=vec_tol)
            c = c * h0_exp
        else:
            norm = onenormest(
                LinearOperator(
                    shape=h0.shape,
                    dtype=np.complex128,
                    matvec=matvec_t,
                    rmatvec=matvec_t,
                )
            )
            c = zhexpv(c, no_krylov, norm, matvec_t, tol=vec_tol)

        c_t.append(c)
        timestep_func(it, time_dt[it], c)

    c_t = np.array(c_t)
    return time_dt, c_t


def propagate_expokit(
    states: RotStates,
    tens_op: list[CartTensor],
    tens_prefac: list[float],
    field_func: Callable[[float], np.ndarray],
    t0: float,
    t1: float,
    dt: float,
    time_unit: str,
    c0: np.ndarray,
    split_method: bool = True,
    field_tol: float = 1e-12,
    no_krylov: int = 12,
    vec_tol: float = 0,
    timestep_func: Callable[[int, float, np.ndarray], None] = lambda *_: None,
):
    time_units = {"ps": 1e-12, "fs": 1e-15, "ns": 1e-9, "us": 1e-6}
    assert time_unit.lower() in time_units, (
        f"Unknown value for 'time_unit' = {time_unit}, "
        + f"accepted values: {list(time_units.keys())}"
    )

    try:
        enr_units = states.enr_units
    except AttributeError:
        raise AttributeError(f"States object has no attribute 'enr_units'") from None

    if enr_units == "invcm":
        enr_to_joule = ENR_INVCM_JOULE
    elif enr_units == "mhz":
        enr_to_joule = ENR_INVCM_JOULE / ENR_INVCM_MHZ
    else:
        raise ValueError(f"Unknown value for 'enr_units' = {enr_units}") from None

    print(f"energy units: {enr_units}")

    h0 = states.mat()

    fac = (
        -1j
        * dt
        * np.array([1] + tens_prefac)
        * enr_to_joule
        / constants.value("reduced Planck constant")
        * time_units[time_unit]
    )

    h0_exp = np.exp(0.5 * fac[0] * h0.diagonal())

    def matvec(c, field):
        if c.ndim == 1:
            c_ = c
        else:
            c_ = c[:, 0]
        c2 = np.sum(
            [f * tens.mat_vec(field, c_) for f, tens in zip(fac[1:], tens_op)],
            axis=0,
        )
        if not split_method:
            c2 += fac[0] * h0.dot(c_)
        if c.ndim == 1:
            return c2
        else:
            return np.array([c2]).T

    nt = int((t1 - t0) / dt)
    time = np.linspace(t0, t1, num=nt, endpoint=False)
    time_dt = time + dt
    time_dt_c = time + dt / 2

    c = c0.copy()
    c_t = []

    for it, t in enumerate(time_dt_c):
        field = field_func(t)
        matvec_t = lambda c: matvec(c, field)

        if split_method:
            c = c * h0_exp
            if np.linalg.norm(field) > field_tol:
                norm = onenormest(
                    LinearOperator(
                        shape=h0.shape,
                        dtype=np.complex128,
                        matvec=matvec_t,
                        rmatvec=matvec_t,
                    )
                )
                c = zhexpv(c, no_krylov, norm, matvec_t, tol=vec_tol)
            c = c * h0_exp
        else:
            norm = onenormest(
                LinearOperator(
                    shape=h0.shape,
                    dtype=np.complex128,
                    matvec=matvec_t,
                    rmatvec=matvec_t,
                )
            )
            c = zhexpv(c, no_krylov, norm, matvec_t, tol=vec_tol)

        c_t.append(c)
        timestep_func(it, time_dt[it], c)

    c_t = np.array(c_t)
    return time_dt, c_t


def propagate_expm(
    states: RotStates,
    tens_op: list[CartTensor],
    tens_prefac: list[float],
    field_func: Callable[[float], np.ndarray],
    t0: float,
    t1: float,
    dt: float,
    time_unit: str,
    c0: np.ndarray,
    split_method: bool = True,
    field_tol: float = 1e-12,
    timestep_func: Callable[[int, float, np.ndarray], None] = lambda *_: None,
):
    time_units = {"ps": 1e-12, "fs": 1e-15, "ns": 1e-9, "us": 1e-6}
    assert time_unit.lower() in time_units, (
        f"Unknown value for 'time_unit' = {time_unit}, "
        + f"accepted values: {list(time_units.keys())}"
    )

    try:
        enr_units = states.enr_units
    except AttributeError:
        raise AttributeError(f"States object has no attribute 'enr_units'") from None

    if enr_units == "invcm":
        enr_to_joule = ENR_INVCM_JOULE
    elif enr_units == "mhz":
        enr_to_joule = ENR_INVCM_JOULE / ENR_INVCM_MHZ
    else:
        raise ValueError(f"Unknown value for 'enr_units' = {enr_units}") from None

    print(f"energy units: {enr_units}")

    h0 = states.mat()

    fac = (
        -1j
        * dt
        * np.array([1] + tens_prefac)
        * enr_to_joule
        / constants.value("reduced Planck constant")
        * time_units[time_unit]
    )

    h0_exp = np.exp(0.5 * fac[0] * h0.diagonal())
    h0_tr = h0.diagonal().sum()

    def matvec(c, field):
        if c.ndim == 1:
            c_ = c
        else:
            c_ = c[:, 0]
        c2 = np.sum(
            [f * tens.mat_vec(field, c_) for f, tens in zip(fac[1:], tens_op)],
            axis=0,
        )
        if not split_method:
            c2 += fac[0] * h0.dot(c_)
        if c.ndim == 1:
            return c2
        else:
            return np.array([c2]).T

    def trace(field):
        tr = np.sum(
            [f * tens.mat_trace(field) for f, tens in zip(fac[1:], tens_op)],
            axis=0,
        )
        if not split_method:
            tr += fac[0] * h0_tr
        return tr

    nt = int((t1 - t0) / dt)
    time = np.linspace(t0, t1, num=nt, endpoint=False)
    time_dt = time + dt
    time_dt_c = time + dt / 2

    c = c0.copy()
    c_t = []

    for it, t in enumerate(time_dt_c):
        field = field_func(t)
        matvec_t = lambda c: matvec(c, field)
        H = LinearOperator(
            shape=h0.shape, dtype=complex, matvec=matvec_t, rmatvec=matvec_t
        )
        if split_method:
            c = c * h0_exp
            if np.linalg.norm(field) > field_tol:
                tr = trace(field)
                c = expm_multiply(H, c, traceA=tr)
            c = c * h0_exp
        else:
            tr = trace(field)
            c = expm_multiply(H, c, traceA=tr)

        c_t.append(c)
        timestep_func(it, time_dt[it], c)

    c_t = np.array(c_t)
    return time_dt, c_t


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
) -> tuple[np.ndarray, np.ndarray]:
    """Computes time-dependent wavepacket coefficients by solving the
    time-dependent Schrödinger equation for a molecule interacting
    with an external field, using the Runge-Kutta method
    from SciPy's `solve_ivp`.

    The total Hamiltonian includes:
        - A field-free component specified by rotational or rovibrational
        energy levels (`states`), in cm^{-1}.
        - A field-dependent component consisting of a linear combination of
        Cartesian tensor operators (`tens_op`) scaled by prefactors
        (`tens_prefac`) and the time-dependent external field (`field_func`).

    Args:
        states (RotStates):
            Rotational or rovibrational energy states forming the basis for
            the wavepacket. Energies must be in units of cm^{-1}.

        tens_op (list[CartTensor]):
            List of Cartesian tensor operators (e.g., dipole moment, polarizability)
            that couple to the external field.

        tens_prefac (list[float]):
            List of scaling factors corresponding to each tensor in `tens_op`.
            These should convert the products of tensors with field into energy
            units of cm^{-1}.

        field_func (Callable[[float], np.ndarray]):
            Function that returns a 1D array of shape (3,) representing the
            X, Y, and Z components of the external electric field at a given time.
            Ensure that the product of `field_func` and `tens_op` scaled by
            `tens_prefac` yields a Hamiltonian in cm^{-1}.
            The input time is in the unit specified by `time_unit`.

        t0 (float):
            Initial time of propagation.

        t1 (float):
            Final time of propagation.

        time_unit (str):
            Unit of time used for `t0`, `t1`, `t_eval` and values passed
            to `field_func`. Supported values  include, e.g., "fs" (femtoseconds),
            "ps" (picoseconds), or "ns" (nanoseconds).

        c0 (np.ndarray):
            Initial wavepacket coefficients at time `t0`.

        t_eval (Optional[np.ndarray]):
            Optional array of time points at which to evaluate and store the solution.
            If not provided, defaults to a uniform grid of 1000 points between
            `t0` and `t1`.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            A tuple containing:
            - `t_eval`: Array of time points.
            - `c_t`:Array of complex wavepacket coefficients at each time in
            `t_eval`, with shape (len(t_eval), len(c0)).
    """
    time_units = {"ps": 1e-12, "fs": 1e-15, "ns": 1e-9, "us": 1e-6}
    assert time_unit.lower() in time_units, (
        f"Unknown value for 'time_unit' = {time_unit}, "
        + f"accepted values: {list(time_units.keys())}"
    )

    try:
        enr_units = states.enr_units
    except AttributeError:
        raise AttributeError(f"States object has no attribute 'enr_units'") from None

    if enr_units == "invcm":
        enr_to_joule = ENR_INVCM_JOULE
    elif enr_units == "mhz":
        enr_to_joule = ENR_INVCM_JOULE / ENR_INVCM_MHZ
    else:
        raise ValueError(f"Unknown value for 'enr_units' = {enr_units}") from None

    print(f"energy units: {enr_units}")

    h0 = states.mat()

    fac = (
        -1j
        * np.array([1] + tens_prefac)
        * enr_to_joule
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
        # c2 = c2 / np.linalg.norm(c2)
        return c2.view(np.float64)

    c0_real = c0.view(np.float64)
    sol = solve_ivp(rhs, (t0, t1), c0_real, t_eval=t_eval, method="DOP853")
    c_t = sol.y.T.copy().view(np.complex128)
    return sol.t, c_t
