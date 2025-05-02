from enum import Enum

from scipy import constants

# _UNITS: dict[str, type] = {}


# def register(cls):
#     _UNITS[cls.__name__.lower()] = cls
#     return cls


# @register
class angstrom:
    @staticmethod
    def to(unit: str):
        if unit.lower() == "bohr":  # Bohr
            return 1 / bohr.to("angstrom")
        elif unit.lower() == "pm":  # pm
            return 100
        elif unit.lower() == "angstrom":  # Angstrom
            return 1
        else:
            raise ValueError(f"Unit '{unit}' is unknown")


# @register
class bohr:
    @staticmethod
    def to(unit: str):
        if unit.lower() == "angstrom":  # Angstrom
            return constants.value("Bohr radius") * 1e10
        elif unit.lower() == "pm":  # pm
            return constants.value("Bohr radius") * 1e10 * 100
        elif unit.lower() == "bohr":  # Bohr
            return 1
        else:
            raise ValueError(f"Unit '{unit}' is unknown")


class ToProxy:
    def __init__(self, unit_class):
        self.unit_class = unit_class

    def __getattr__(self, unit_name: str):
        # e.g., for .to.angstrom, call unit_class.to('angstrom')
        return self.unit_class.to(unit_name)


class UnitBaseEnum(Enum):
    @property
    def to(self):
        return ToProxy(self.value)


# UnitType = Enum(
#     "UnitType", {name: cls for name, cls in _UNITS.items()}, type=UnitBaseEnum
# )


class UnitType(UnitBaseEnum):
    angstrom = angstrom
    bohr = bohr
