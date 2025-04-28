from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Literal, Type

_SYMMETRIES: dict[str, Type["MetaSymmetry"]] = {}


def register(cls: Type["MetaSymmetry"]) -> Type["MetaSymmetry"]:
    _SYMMETRIES[cls.__name__.lower()] = cls
    return cls


class MetaSymmetry(ABC):
    irreps: List[str]

    @staticmethod
    @abstractmethod
    def irrep_from_k_tau(k: int, tau: Literal[0, 1]) -> str:
        pass

    def __str__(self):
        return self.__class__.__name__


@register
class C1(MetaSymmetry):
    irreps = ["A"]

    @staticmethod
    def irrep_from_k_tau(k: int, tau: Literal[0, 1]) -> str:
        return "A"


@register
class C2v(MetaSymmetry):
    irreps = ["A1", "A2", "B1", "B2"]

    @staticmethod
    def irrep_from_k_tau(k: int, tau: Literal[0, 1]) -> str:
        k_mod_2 = k % 2
        return {(0, 0): "A1", (0, 1): "B1", (1, 0): "B2", (1, 1): "A2"}[(k_mod_2, tau)]


@register
class D2h(MetaSymmetry):
    # ! TODO
    @staticmethod
    def irrep_from_k_tau(k: int, tau: Literal[0, 1]) -> str:
        return "A"


class SymmetryBaseEnum(Enum):
    def __getattr__(self, attr):
        if attr in {"name", "value", "_name_", "_value_", "__class__"}:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{attr}'"
            )
        value = object.__getattribute__(self, "value")
        return getattr(value, attr)


SymmetryType = Enum(
    "SymmetryType",
    {name.lower(): cls for name, cls in _SYMMETRIES.items()},
    type=SymmetryBaseEnum,
)
