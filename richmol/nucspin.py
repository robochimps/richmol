import itertools
from abc import abstractmethod
from typing import Protocol, runtime_checkable
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
import py3nj


@runtime_checkable
class SpinOperator(Protocol):
    spin: float
    name: str
    rank: int

    @abstractmethod
    def reduced_me(self, tol: float = 1e-12) -> float:
        pass


@dataclass
class Spin(SpinOperator):
    """Defines nuclear spin operator I.

    Attributes:
        spin (float): Integer or half-integer value of nuclear spin.
    """

    spin: float
    name: str = "spin"
    rank: int = 1

    def reduced_me(self, tol: float = 1e-12) -> float:
        """Computes reduced matrix element of spin operator: <I || I^(1) || I>"""
        if self.spin == 0:
            return 0
        else:
            threej = py3nj.wigner3j(
                int(self.spin * 2),
                self.rank * 2,
                int(self.spin * 2),
                -int(self.spin * 2),
                0,
                int(self.spin * 2),
                ignore_invalid=True,
            )
            if abs(threej) < tol:
                raise ValueError(
                    f"Illegal division I/threej(I,1,I;-I,0,I): {self.spin} / {threej}"
                ) from None
            else:
                return self.spin / threej


@dataclass
class QuadMom(SpinOperator):
    """Defines nuclear quadrupole moment operator Q.

    Attributes:
        spin (float): Integer or half-integer value of nuclear spin.

        Q (float): Nuclear quadrupole constant in units of millibarns (mb).
    """

    spin: float
    Q: float
    name: str = "quad"
    rank: int = 2

    def reduced_me(self, tol: float = 1e-12) -> float:
        """Computes reduced matrix element of nuclear quadrupole moment operator: <I || Q^(2) || I>"""
        threej = py3nj.wigner3j(
            int(self.spin * 2),
            self.rank * 2,
            int(self.spin * 2),
            -int(self.spin * 2),
            0,
            int(self.spin * 2),
            ignore_invalid=True,
        )
        if abs(threej) < tol and abs(self.Q) < tol:
            return 0
        elif abs(threej) < tol and abs(self.Q) > tol:
            assert int(self.spin * 2) > 1, (
                f"For nucleus with spin = {self.spin}, the nuclear quadrupole moment must be zero, "
                + f"instead value of eQ = {self.Q} is provided"
            )
            raise ValueError(
                f"Illegal division eQ/threej(I,2,I;-I,0,I): {self.Q} / {threej}"
            ) from None
        else:
            return 0.5 * self.Q / threej


def _reduced_me_rec(
    I1: tuple[float], I2: tuple[float], opers: list[SpinOperator], ioper: int, n: int
) -> float:
    """Recursively computes reduced matrix element of a single-spin operator in coupled-spin basis"""

    I1_bra = I1[n - 1]
    I1_ket = I2[n - 1]
    I2_bra = opers[n].spin
    I2_ket = opers[n].spin
    I12_bra = I1[n]
    I12_ket = I2[n]
    rank = opers[ioper].rank

    if ioper == n:
        if n == 0:
            coef1 = 1
        elif n > 0 and I1_bra != I1_ket:
            return 0
        else:
            fac = I1_bra + I2_ket + I12_bra + rank
            assert float(fac).is_integer(), f"Non-integer power in (-1)**f: (-1)**{fac}"
            fac = int(fac)
            coef1 = (
                (-1) ** fac
                * np.sqrt((2 * I12_bra + 1) * (2 * I12_ket + 1))
                * py3nj.wigner6j(
                    int(I2_bra * 2),
                    int(I12_bra * 2),
                    int(I1_bra * 2),
                    int(I12_ket * 2),
                    int(I2_ket * 2),
                    int(rank * 2),
                )
            )
        coef2 = opers[ioper].reduced_me()

    else:
        if I2_bra != I2_ket:
            return 0
        else:
            fac = I1_bra + I2_bra + I12_ket + rank
            assert float(fac).is_integer(), f"Non-integer power in (-1)**g: (-1)**{fac}"
            fac = int(fac)
            coef1 = (
                (-1) ** fac
                * np.sqrt((2 * I12_bra + 1) * (2 * I12_ket + 1))
                * py3nj.wigner6j(
                    int(I1_bra * 2),
                    int(I12_bra * 2),
                    int(I2_bra * 2),
                    int(I12_ket * 2),
                    int(I1_ket * 2),
                    int(rank * 2),
                )
            )
            coef2 = _reduced_me_rec(I1, I2, opers, ioper, n - 1)

    return coef1 * coef2


def reduced_me(
    I1_list: list[tuple[float]],
    I2_list: list[tuple[float]],
    opers: list[SpinOperator],
    tol: float = 1e-14,
) -> dict[tuple[tuple[float], tuple[float]], np.ndarray]:
    """Computes reduced matrix elements of single-spin operators O(I_i) in a coupled-spin basis.

    Args:
        I1_list (list[tuple[float]]):
            List of coupled spin quantum numbers for the bra states.
            Each tuple corresponds to a basis vector and contains spins:
                (I_{1}, I_{12}, ..., I_{1N}),
            where:
                - I_{1} is the spin of operator 1,
                - I_{12} is the total spin of operators 1 and 2,
                - I_{1N} is the total spin of operators 1 through N.

        I2_list (list[tuple[float]]):
            List of coupled spin quantum numbers for the ket states, structured similarly to `I1_list`.

        opers (list[SpinOperator]):
            List of single-spin operators O(I_i) corresponding to the nuclear spins involved.
            Each operator defines an individual spin quantum number I_i.

        tol (float, optional):
            Numerical tolerance used to neglect near-zero matrix elements.
            Defaults to 1e-14.

    Returns:
        dict[tuple[tuple[float], tuple[float]], np.ndarray]:
            A dictionary mapping pairs of coupled spin states to arrays of reduced matrix elements:
                me[(I1_list[k], I2_list[l])][i] = <I1_list[k] || O(I_i) || I2_list[l]>
    """
    rme = {}
    for i1 in I1_list:
        for i2 in I2_list:
            me = np.array(
                [
                    _reduced_me_rec(i1, i2, opers, ioper, len(opers) - 1)
                    for ioper in range(len(opers))
                ]
            )
            if np.any(np.abs(me) > tol):
                rme[(i1, i2)] = me
    return rme


def reduced_me_IxI(
    I1_list: list[tuple[float]],
    I2_list: list[tuple[float]],
    I_list: list[tuple[float]],
    opers: list[SpinOperator],
    rank: Literal[0, 1, 2],
    tol: float = 1e-14,
) -> dict[tuple[tuple[float], tuple[float]], np.ndarray]:
    """Computes reduced matrix elements of tensor product of single-spin operators
    [O(I_i)^(1) x O(I_j)^(1)]^(rank) in a coupled-spin basis.

    Args:
        I1_list (list[tuple[float]]):
            List of coupled spin quantum numbers for the bra states.
            Each tuple corresponds to a basis vector and contains spins:
                (I_{1}, I_{12}, ..., I_{1N}),
            where:
                - I_{1} is the spin of operator 1,
                - I_{12} is the total spin of operators 1 and 2,
                - I_{1N} is the total spin of operators 1 through N.

        I2_list (list[tuple[float]]):
            List of coupled spin quantum numbers for the ket states, structured similarly to `I1_list`.

        I_list (list[list[float]]):
            Complete list of coupled spin quantum numbers spanned by spin basis,
            structured similarly to `I1_list`.

        opers (list[SpinOperator]):
            List of single-spin operators O(I_i) corresponding to the nuclear spins involved.
            Each operator defines an individual spin quantum number I_i.

        rank (Literal[0, 1, 2]):
            Desired rank of tensor product: 0, 1, or 2.

        tol (float, optional):
            Numerical tolerance used to neglect near-zero matrix elements.
            Defaults to 1e-14.

    Returns:
        dict[tuple[tuple[float], tuple[float]], np.ndarray]:
            A dictionary mapping pairs of coupled spin states to arrays of reduced matrix elements:
                me[(I1_list[k], I2_list[l])][i, j] = <I1_list[k] || [O(I_i)^(1) x O(I_j)^(1)]^(rank) || I2_list[l]>
    """
    assert all(
        elem in I_list for elem in I1_list
    ), f"Not all spin states in 'I1_list' = {I1_list} are contained in the basis 'I_list' = {I_list}"
    assert all(
        elem in I_list for elem in I2_list
    ), f"Not all spin states in 'I2_list' = {I2_list} are contained in the basis 'I_list' = {I_list}"

    rme1 = reduced_me(I1_list, I_list, opers)
    rme2 = reduced_me(I_list, I2_list, opers)

    rme = {}
    for i1 in I1_list:
        for i2 in I2_list:
            fac = i1[-1] + i2[-1] + rank
            assert float(fac).is_integer(), f"Non-integer power in (-1)**f: (-1)**{fac}"
            fac = int(fac)
            fac = (-1) ** fac * np.sqrt(2 * rank + 1)
            n = len(I_list)
            coef = fac * py3nj.wigner6j(
                [2] * n,
                [2] * n,
                [rank * 2] * n,
                [int(i2[-1] * 2)] * n,
                [int(i1[-1] * 2)] * n,
                [int(i[-1] * 2) for i in I_list],
            )
            try:
                me = np.sum(
                    [
                        rme1[(i1, i)][:, None] * c * rme2[(i, i2)][None, :]
                        for i, c in zip(I_list, coef)
                    ],
                    axis=0,
                )
            except KeyError:
                continue
            if np.any(np.abs(me) > tol):
                rme[(i1, i2)] = me
    return rme


def near_equal_coupling(opers: list[SpinOperator]) -> list[tuple[float]]:
    """Generates combinations of nuclear spin quanta following nearly-equal coupling scheme.

    Args:
        opers (list[SpinOperator]):
            List of single-spin operators O(I_i) corresponding to the nuclear spins involved.
            Each operator defines an individual spin quantum number I_i.

    Returns:
        (list[tuple[float]]):
            List of coupled nuclear spin quantum numbers for each basis state.
            Each tuple represents sequential coupling:
                (I_{1}, I_{12}, ..., I_{1N}),
            where:
                - I_{1} is the spin of operator 1,
                - I_{12} is the total spin of operators 1 and 2,
                - I_{1N} is the total spin of operators 1 through N.
    """
    queue = deque()
    queue.append([opers[0].spin])
    for oper in opers[1:]:
        I = oper.spin
        nelem = len(queue)
        for i in range(nelem):
            I0 = queue.popleft()
            queue += [
                I0 + [float(elem)]
                for elem in np.arange(np.abs(I0[-1] - I), I0[-1] + I + 1)
            ]
    return [tuple(elem) for elem in queue]


def near_equal_coupling_with_rotations(
    f_val: float, opers: list[SpinOperator]
) -> tuple[list[tuple[float]], list[int]]:
    """Generates combinations of nuclear spin quantum numbers and rotational quantum numbers (J)
    that couple to a given total spin-rotational angular momentum quantum number (F),
    following a nearly-equal coupling scheme.

    Args:
        f_val (float):
            Quantum number of the total angular momentum F = I + J,
            where I is the total nuclear spin and J is the rotational angular momentum.
            Combinations will be generated such that they satisfy this total F.

        opers (list[SpinOperator]):
            List of single-spin operators O(I_i) corresponding to the nuclear spins involved.
            Each operator defines an individual spin quantum number I_i.

    Returns:
        tuple:
            - spin_quanta (list[tuple[float]]):
                List of coupled nuclear spin quantum numbers for each basis state.
                Each tuple represents sequential coupling:
                    (I_{1}, I_{12}, ..., I_{1N}),
                where:
                    - I_{1} is the spin of operator 1,
                    - I_{12} is the total spin of operators 1 and 2,
                    - I_{1N} is the total spin of operators 1 through N.

            - rot_quanta (list[int]):
                List of corresponding rotational quantum numbers J
                coupled with the final total spin to achieve F.
    """
    I_list = near_equal_coupling(opers)
    J_list = list(
        set(
            [
                float(elem)
                for spin in I_list
                for elem in np.arange(abs(f_val - spin[-1]), f_val + spin[-1] + 1)
            ]
        )
    )

    if not all(float(elem).is_integer() for elem in J_list):
        spins = [oper.spin for oper in opers]
        raise ValueError(
            f"Invalid rotational quantum numbers J: {J_list}. "
            f"For spins {spins} and total F = {f_val}, non-integer values of J are produced."
        ) from None

    quanta = [
        (spin, j)
        for (spin, j) in itertools.product(I_list, J_list)
        if any(np.arange(np.abs(spin[-1] - j), spin[-1] + j + 1) == f_val)
    ]
    spin_quanta = [elem[0] for elem in quanta]
    rot_quanta = [int(elem[1]) for elem in quanta]
    return spin_quanta, rot_quanta


if __name__ == "__main__":

    # example of reduced matrix elements for two spins 1/2

    op1 = Spin(spin=1 / 2)
    op2 = Spin(spin=1 / 2)

    I_list = near_equal_coupling([op1, op2])

    me = reduced_me(I_list, I_list, [op1, op2])

    print("matrix elements < || I_i || >")
    for i1 in I_list:
        for i2 in I_list:
            try:
                print(i1, i2, me[(i1, i2)])
            except KeyError:
                continue

    me = reduced_me_IxI(I_list, I_list, I_list, [op1, op2], 2)

    print("matrix elements < || [I_i^(1) x I_j^(1)]^(2) || >")
    for i1 in I_list:
        for i2 in I_list:
            try:
                print(i1, i2, me[(i1, i2)])
            except KeyError:
                continue
