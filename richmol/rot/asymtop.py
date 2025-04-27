import re
from collections import defaultdict
from dataclasses import dataclass

import jax
import numpy as np
from jax import numpy as jnp
from mendeleev import element
from scipy import constants

from .symmetry import SymmetryType
from .units import UnitType
from .symtop import rotme_rot, rotme_rot_diag

jax.config.update("jax_enable_x64", True)


G_to_invcm = (
    constants.value("Planck constant")
    * constants.value("Avogadro constant")
    * 1e16
    / (4.0 * np.pi**2 * constants.value("speed of light in vacuum"))
    * 1e5
)

EPS = jnp.array(
    [
        [[int((i - j) * (j - k) * (k - i) * 0.5) for k in range(3)] for j in range(3)]
        for i in range(3)
    ],
    dtype=jnp.float64,
)


@dataclass
class RotStates:
    atom_masses: np.ndarray
    atom_xyz: np.ndarray
    j_list: list[int]
    sym_list: dict[int, list[str]]
    enr: defaultdict[int, defaultdict[str, np.ndarray]]
    vec: defaultdict[int, defaultdict[str, np.ndarray]]
    r_ind: defaultdict[int, defaultdict[str, np.ndarray]]
    v_ind: defaultdict[int, defaultdict[str, np.ndarray]]

    @classmethod
    def from_geometry(
        cls,
        max_j: int,
        inp,
    ):
        """Generates rigid-rotor rotational states of a molecule using its geometry as input.
        Args:
            max_j (int): max value of rotational J quantum number. States for J=0..`max_j` will be generated.
            inp (list): input list
        """
        print("\nCompute rigid-rotor solutions using molecular geometry as input")

        atom_labels, atom_xyz, atom_masses, units, sym_label = _parse_input(inp)

        atom_masses = [
            mass if mass != None else _atom_mass(labes)
            for mass, labes in zip(atom_masses, atom_labels)
        ]

        try:
            i = [elem.name for elem in UnitType].index(units)
            units = [elem for elem in UnitType][i]
        except ValueError:
            raise ValueError(
                f"No units '{units}' found {[elem.name for name in UnitType]}"
            ) from None

        print(
            f"Cartesian units: {units.name}, conversion to Angstrom: {units.to.angstrom}"
        )

        atom_xyz = np.array(atom_xyz) * units.to.angstrom

        print(f"{'Atom':>6} {'Mass (u)':>18} {'X (Å)':>18} {'Y (Å)':>18} {'Z (Å)':>18}")
        for label, mass, coords in zip(atom_labels, atom_masses, atom_xyz):
            x, y, z = coords
            print(
                f"{label:>6} "
                f"{mass:18.12f} "
                f"{x:18.12f} "
                f"{y:18.12f} "
                f"{z:18.12f}"
            )

        try:
            i = [elem.name for elem in SymmetryType].index(sym_label)
            sym = [elem for elem in SymmetryType][i]
        except ValueError:
            raise ValueError(f"No symmetry '{sym_label}' found") from None

        print("Symmetry group:", sym.name)

        # rotational kinetic energy G-matrix
        masses = np.array(atom_masses)
        xyz = np.array(atom_xyz)
        com = masses @ xyz / jnp.sum(masses)
        xyz -= com[None, :]
        gmat = _gmat(masses, xyz)[:3, :3]
        gmat = np.linalg.inv(gmat) * G_to_invcm

        # matrix elements of JxJy
        linear = _check_linear(xyz)

        nested_dict = lambda: defaultdict(nested_dict)
        enr = nested_dict()
        vec = nested_dict()
        v_ind = nested_dict()
        r_ind = nested_dict()
        sym_list = {}
        jktau_list = {}
        k_list = {}

        j_list = [j for j in range(max_j + 1)]
        sym_list = {j: [] for j in j_list}

        for j in j_list:
            print(f"solve for J = {j}")
            me, k_list[j], jktau_list[j] = rotme_rot(j=j, linear=linear, sym=sym)

            # matrix elements of KEO
            ham = 0.5 * np.einsum("ab,abij->ij", gmat, me, optimize="optimal")

            for irrep in sym.irreps:
                ind = np.where(np.array([elem[-1] for elem in jktau_list[j]]) == irrep)[
                    0
                ]
                if len(ind) > 0:
                    h = ham[np.ix_(ind, ind)]
                    e, v = np.linalg.eigh(h)
                    enr[j][irrep] = e
                    vec[j][irrep] = v
                    r_ind[j][irrep] = ind
                    v_ind[j][irrep] = np.array([0])
                    sym_list[j].append(irrep)

        # print solutions

        print(
            f"{'J':>3} {'Irrep':>5} {'i':>4} {'Energy (E)':>18} {'(J,k,tau,Irrep)':>20} {'c_max²':>10}"
        )

        for j in j_list:
            for irrep in sym_list[j]:
                e = enr[j][irrep]
                v = vec[j][irrep]
                jktau = [jktau_list[j][i] for i in r_ind[j][irrep]]
                for i in range(len(e)):
                    ind = np.argmax(v[:, i] ** 2)
                    print(
                        f"{j:3d} "
                        f"{irrep:>5} "
                        f"{i:4d} "
                        f"{e[i]:18.8f} "
                        f"{str(jktau[ind]):>20} "
                        f"{v[ind, i] ** 2:10.5f}"
                    )

        return cls(masses, xyz, j_list, sym_list, enr, vec, r_ind, v_ind)


def _gmat(masses, xyz):
    t_rot = np.transpose(EPS @ np.asarray(xyz).T, (2, 0, 1))
    t_tra = np.array([np.eye(3, dtype=np.float64) for _ in range(len(xyz))])
    t = np.concatenate((t_rot, t_tra), axis=2)
    masses_sq = np.sqrt(np.asarray(masses))
    t = t * masses_sq[:, None, None]
    t = np.reshape(t, (len(xyz) * 3, 6))
    return t.T @ t


def _check_linear(xyz, tol=1e-8):
    if len(xyz) < 2:
        return True
    v0 = xyz[1] - xyz[0]
    for i in range(2, len(xyz)):
        if np.linalg.norm(np.linalg.cross(v0, xyz[i] - xyz[0])) > tol:
            return False
    return True


def _moment_of_inertia(masses, xyz):
    cm = np.sum([x * m for x, m in zip(xyz, masses)], axis=0) / np.sum(masses)
    xyz = xyz - cm[None, :]
    imat = np.zeros((3, 3), dtype=np.float64)

    # off-diagonals
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            imat[i, j] = -np.sum(
                [xyz[iat, i] * xyz[iat, j] * masses[iat] for iat in range(len(xyz))]
            )

    # diagonals
    imat[0, 0] = np.sum(
        [(xyz[iat, 1] ** 2 + xyz[iat, 2] ** 2) * masses[iat] for iat in range(len(xyz))]
    )
    imat[1, 1] = np.sum(
        [(xyz[iat, 0] ** 2 + xyz[iat, 2] ** 2) * masses[iat] for iat in range(len(xyz))]
    )
    imat[2, 2] = np.sum(
        [(xyz[iat, 0] ** 2 + xyz[iat, 1] ** 2) * masses[iat] for iat in range(len(xyz))]
    )
    return imat


def _rotation_constants(masses, xyz):
    imat = _moment_of_inertia(masses, xyz)
    d, v = np.linalg.eigh(imat)
    a, b, c = 0.5 / d * G_to_invcm
    return (a, b, c), v


def _atom_mass(atom_label: str) -> float:
    match = re.match(r"^([A-Z][a-z]*)(\d*)$", atom_label)
    if not match:
        raise ValueError(f"Invalid atom label: {atom_label}")

    symbol, mass_number = match.groups()
    el = element(symbol)

    if mass_number:
        # specific isotope
        mass_number = int(mass_number)
        for iso in el.isotopes:
            if iso.mass_number == mass_number:
                return iso.mass
        raise ValueError(f"No isotope {symbol}{mass_number} found")
    else:
        # no specific isotope
        main_iso = max(
            (iso for iso in el.isotopes if iso.abundance is not None),
            key=lambda iso: iso.abundance,
        )
        return main_iso.mass


def _parse_input(inp):
    unit = "angstrom"
    symmetry = None
    atom_labels = []
    atom_coords = []
    atom_masses = []

    ielem = 0
    while ielem < len(inp):
        item = inp[ielem]

        if isinstance(item, str):
            lower_item = item.lower()

            # unit
            if lower_item in {"bohr", "angstrom", "pm"}:
                unit = lower_item
                ielem += 1
                continue

            # symmetry, "sym="
            if lower_item.startswith("sym="):
                symmetry = lower_item.split("=")[1]
                ielem += 1
                continue

            # atom label
            label = item
            x = inp[ielem + 1]
            y = inp[ielem + 2]
            z = inp[ielem + 3]
            atom_labels.append(label)
            atom_coords.append((x, y, z))
            ielem += 4

            # mass override
            if (
                ielem < len(inp)
                and isinstance(inp[ielem], str)
                and inp[ielem].startswith("m=")
            ):
                mass = float(inp[ielem][2:])
                atom_masses.append(mass)
                ielem += 1
            else:
                atom_masses.append(None)
        else:
            raise ValueError(f"Unexpected item at position {ielem}: {item}")

    return atom_labels, atom_coords, atom_masses, unit, symmetry
