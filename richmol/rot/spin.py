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
    def reduced_me(self, I1: float, I2: float, tol: float = 1e-12) -> float:
        pass


@dataclass
@register
class Spin(SpinOperator):
    name: str = "spin"
    rank: int = 1

    def reduced_me(self, I1: float, I2: float, tol: float = 1e-12) -> float:
        """Computes reduced matrix element of spin operator <I1 || I^(1) || I2>"""
        if I1 == I2:
            if I1 == 0:
                return 0
            else:
                threej = py3nj.wigner3j(
                    int(I1 * 2), 2, int(I1 * 2), -int(I1 * 2), 0, int(I1 * 2)
                )
                if abs(threej) < tol:
                    raise ValueError(
                        f"Illegal division I/threej(I,1,I;-I,0,I): {I1} / {threej}"
                    ) from None
                else:
                    return I1 / threej
        else:
            return 0


@dataclass
@register
class QuadMom(SpinOperator):
    eQ: float
    name: str = "quad"
    rank: int = 2

    def reduced_me(self, I1: float, I2: float, tol: float = 1e-12) -> float:
        """Computes reduced matrix element of nuclear quadrupole moment operator:
        <I1 || Q^(2) || I2>
        """
        if I1 == I2:
            threej = py3nj.wigner3j(
                int(I1 * 2), 4, int(I1 * 2), -int(I1 * 2), 0, int(I1 * 2)
            )
            if abs(threej) < tol and abs(self.eQ) < tol:
                return 0
            elif abs(threej) < tol and abs(self.eQ) > tol:
                assert int(I1 * 2) > 1, (
                    f"For nuclei with spin = {I1} the nuclear quadrupole moment must be zero, "
                    + f"instead values of eQ = {self.eQ} is provided"
                )
                raise ValueError(
                    f"Illegal division eQ/threej(I,2,I;-I,0,I): {I1} / {threej}"
                ) from None
            else:
                return 0.5 * self.eQ / threej
        return 0


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
    I1: list[float],
    I2: list[float],
    ispin: int,
    no_spins: int,
    spins: list[float],
    oper: SpinOperator,
) -> float:
    """Recursively computes reduced matrix element of a single-spin operator in coupled-spin basis"""

    I1_bra = I1[no_spins - 1]
    I1_ket = I2[no_spins - 1]
    I2_bra = spins[no_spins]
    I2_ket = spins[no_spins]
    I12_bra = I1[no_spins]
    I12_ket = I2[no_spins]
    rank = oper.rank

    if ispin == no_spins:
        if no_spins == 0:
            coef1 = 1
        elif no_spins > 0 and I1_bra != I1_ket:
            return 0
        else:
            fac = I1_bra + I2_ket + I12_bra + rank
            assert float(
                fac
            ).is_integer(), f"Non-integer power in (-1)**f: '(-1)**{fac}'"
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
        coef2 = oper.reduced_me(spins[ispin], spins[ispin])

    else:
        if I2_bra != I2_ket:
            return 0
        else:
            fac = I1_bra + I2_bra + I12_ket + rank
            assert float(
                fac
            ).is_integer(), f"Non-integer power in (-1)**g: '(-1)**{fac}'"
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
            coef2 = reduced_me_rec(I1, I2, ispin, no_spins - 1, spins, oper)

    return coef1 * coef2


def reduced_me(
    I1_list: list[list[float]],
    I2_list: list[list[float]],
    spins: list[float],
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
        spins (list[float]): 
            List of individual particle spins [I_1, I_2, ..., I_N] involved in the coupling.
        opers (list[SpinOperator]): 
            List of single-spin operators O(I_i) corresponding to each spin in `spins`.

    Returns:
        np.ndarray:
            A 3D array of shape (len(I1_list), len(I2_list), len(spins)) containing the reduced matrix elements:
                me[k, l, i] = <I1_list[k] || O(I_i) || I2_list[l]>
    """
    me = np.zeros((len(I1_list), len(I2_list), len(spins)), dtype=np.float64)
    for i in range(len(I1_list)):
        for j in range(len(I2_list)):
            for ispin in range(len(spins)):
                n = len(spins) - 1
                me[i, j, ispin] = reduced_me_rec(
                    I1_list[i], I2_list[i], ispin, n, spins, opers[ispin]
                )


if __name__ == "__main__":
    q = SpinOperType.quad(eQ=1)
    i = SpinOperType.spin
    print(q.eQ)
    print(q.rank)
    print(q.name)
    print(i.rank)
    print(i.name)
    pass
