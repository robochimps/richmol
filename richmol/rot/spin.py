"""Matrix elements of nuclear spin operators"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import py3nj

_OPERATORS: dict[str, type] = {}


def register(cls):
    name = cls.name
    if name in _OPERATORS:
        raise ValueError(
            f"Class {_OPERATORS[name].__name__} and {cls.__name__} have the same name: {cls.name}"
        )
    _OPERATORS[name] = cls
    return cls


class SpinOperator(ABC):
    name: str
    rank: int

    @abstractmethod
    def reduced_me(self, tol: float = 1e-12) -> float:
        pass


@dataclass
@register
class Spin(SpinOperator):
    """Defines nuclear spin operator I

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
            )
            if abs(threej) < tol:
                raise ValueError(
                    f"Illegal division I/threej(I,1,I;-I,0,I): {self.spin} / {threej}"
                ) from None
            else:
                return self.spin / threej


@dataclass
@register
class QuadMom(SpinOperator):
    """Defines nuclear quadrupole moment operator Q

    Attributes:
        spin (float): Integer or half-integer value of nuclear spin.
        eQ (float): Nuclear quadrupole constant in units of millibarns (mb).
    """
    spin: float
    eQ: float
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
        )
        if abs(threej) < tol and abs(self.eQ) < tol:
            return 0
        elif abs(threej) < tol and abs(self.eQ) > tol:
            assert int(self.spin * 2) > 1, (
                f"For nucleus with spin = {self.spin}, the nuclear quadrupole moment must be zero, "
                + f"instead value of eQ = {self.eQ} is provided"
            )
            raise ValueError(
                f"Illegal division eQ/threej(I,2,I;-I,0,I): {self.eQ} / {threej}"
            ) from None
        else:
            return 0.5 * self.eQ / threej


class SpinOperBaseEnum(Enum):
    def __getattr__(self, attr):
        if attr in {"name", "value", "_name_", "_value_", "__class__"}:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{attr}'"
            )
        value = object.__getattribute__(self, "value")
        return getattr(value, attr)

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


SpinOperType = Enum(
    "SpinOperType",
    {name: cls for name, cls in _OPERATORS.items()},
    type=SpinOperBaseEnum,
)


def reduced_me_rec(
    I1: list[float], I2: list[float], opers: list[SpinOperator], ioper: int, n: int
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
            coef2 = reduced_me_rec(I1, I2, opers, ioper, n - 1)

    return coef1 * coef2


def reduced_me(
    I1_list: list[list[float]],
    I2_list: list[list[float]],
    opers: list[SpinOperator],
) -> np.ndarray:
    """Computes reduced matrix elements of single-spin operators O(I_i) in a coupled-spin basis.

    Args:
        I1_list (list[list[float]]):
            List of coupled spin quantum numbers for the bra states.
            Each sublist corresponds to a basis vector and contains spins
            [I_1, I_12, ..., I_1N], where:
            - I_1 is the spin of particle 1,
            - I_12 is the total spin of particles 1 and 2,
            - I_1N is the total spin of particles 1 through N.
        I2_list (list[list[float]]):
            List of coupled spin quantum numbers for the ket states, structured similarly to `I1_list`.
        opers (list[SpinOperator]):
            List of single-spin operators O(I_i) corresponding to each spin involved in the coupling.

    Returns:
        np.ndarray:
            A 3D array of shape (len(I1_list), len(I2_list), len(opers))
            containing the reduced matrix elements:
                me[k, l, i] = <I1_list[k] || O(I_i) || I2_list[l]>
    """
    me = np.zeros((len(I1_list), len(I2_list), len(opers)), dtype=np.float64)
    for i in range(len(I1_list)):
        for j in range(len(I2_list)):
            for ioper in range(len(opers)):
                me[i, j, ioper] = reduced_me_rec(
                    I1_list[i], I2_list[j], opers, ioper, len(opers) - 1
                )
    return me


def reduced_me_IxI(
    I1_list: list[list[float]],
    I2_list: list[list[float]],
    I_list: list[list[float]],
    opers: list[SpinOperator],
) -> np.ndarray:
    """Computes reduced matrix elements of tensor product of single-spin operators
    [O(I_i)^(1) x O(I_j)^(1)]^(rank) (where rank = 0, 1, 2) in a coupled-spin basis.

    Args:
        I1_list (list[list[float]]):
            List of coupled spin quantum numbers for the bra states.
            Each sublist corresponds to a basis vector and contains spins
            [I_1, I_12, ..., I_1N], where:
            - I_1 is the spin of particle 1,
            - I_12 is the total spin of particles 1 and 2,
            - I_1N is the total spin of particles 1 through N.
        I2_list (list[list[float]]):
            List of coupled spin quantum numbers for the ket states, structured similarly to `I1_list`.
        I_list (list[list[float]]):
            Complete list of coupled spin quantum numbers spanned by spin basis,
            structured similarly to `I1_list`.
        opers (list[SpinOperator]):
            List of single-spin operators O(I_i) corresponding to each spin involved in the coupling.

    Returns:
        np.ndarray:
            A dictionary of 4D arrays of shape (len(I1_list), len(I2_list), len(opers), len(opers))
            containing the reduced matrix elements:
                me[rank][k, l, i,  j] = <I1_list[k] || [O(I_i)^(1) x O(I_j)^(1)]^(rank) || I2_list[l]>
            The dictionary keys are integer product rank values 0, 1, 2.
    """
    assert all(
        elem in I_list for elem in I1_list
    ), f"Not all spin states in 'I1_list' = {I1_list} are contained in the basis 'I_list' = {I_list}"
    assert all(
        elem in I_list for elem in I2_list
    ), f"Not all spin states in 'I2_list' = {I2_list} are contained in the basis 'I_list' = {I_list}"

    rme1 = reduced_me(I1_list, I_list, opers)  # (n1, n, ioper)
    rme2 = reduced_me(I_list, I2_list, opers)  # (n, n2, ioper)

    me = {}
    for rank in (0, 1, 2):
        coef = np.zeros((len(I1_list), len(I2_list), len(I_list)), dtype=np.float64)

        for i, q1 in enumerate(I1_list):
            I1 = q1[-1]
            for j, q2 in enumerate(I2_list):
                I2 = q2[-1]
                fac = I1 + I2 + rank
                assert float(
                    fac
                ).is_integer(), f"Non-integer power in (-1)**f: (-1)**{fac}"
                fac = int(fac)
                fac = (-1) ** fac * np.sqrt(2 * rank + 1)
                n = len(I_list)
                coef[i, j, :] = fac * py3nj.wigner6j(
                    [2] * n,
                    [2] * n,
                    [rank * 2] * n,
                    [int(I2 * 2)] * n,
                    [int(I1 * 2)] * n,
                    [int(q[-1] * 2) for q in I_list],
                )
        me[rank] = np.einsum("kli,lnj,knl->knij", rme1, rme2, coef)
    return me


if __name__ == "__main__":

    # example of reduced matrix elements for two spins 1/2

    op1 = SpinOperType.spin(spin=1 / 2)
    op2 = SpinOperType.spin(spin=1 / 2)
    I_list = [[op1.spin, op1.spin - op2.spin], [op1.spin, op1.spin + op2.spin]]

    me = reduced_me(I_list, I_list, [op1, op2])

    print("matrix elements < || I_i || >")
    for i in range(len(I_list)):
        for j in range(len(I_list)):
            for iop in range(2):
                print(I_list[i], I_list[j], iop, me[i, j, iop])

    me = reduced_me_IxI(I_list, I_list, I_list, [op1, op2])

    print("matrix elements < || [I_i^(1) x I_j^(1)]^(2) || >")
    for i in range(len(I_list)):
        for j in range(len(I_list)):
            for iop in range(2):
                for jop in range(2):
                    print(I_list[i], I_list[j], iop, jop, me[2][i, j, iop, jop])
