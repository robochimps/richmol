import re
from collections import defaultdict
from dataclasses import dataclass

import jax
import numpy as np
from jax import numpy as jnp
from mendeleev import element
from scipy import constants
from scipy.sparse import csr_array, diags

from .symmetry import SymmetryType
from .symtop import rotme_rot, rotme_rot_diag, symtop_on_grid_split_angles
from .units import UnitType

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
    linear: bool
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
        """Computes rigid rotor states from a given molecular geometry.

        Args:
            max_j (int):
                Maximum rotational angular momentum quantum number (J).
                Rotational states will be computed for all values from J = 0 up to J = max_j.
            inp (list or tuple):
                A specification of the molecular geometry, including atomic positions, units,
                optional mass assignments, and symmetry labels. Examples are shown below.

        Examples of 'inp':

        1. Standard geometry specification:

            >>> xyz = (
            ...     "bohr", "c2v",
            ...     "O", 0.00000000, 0.00000000, 0.12395915,
            ...     "H", 0.00000000, -1.43102686, -0.98366080,
            ...     "H", 0.00000000,  1.43102686, -0.98366080,
            ... )

        2. Custom atomic mass specification to override default atomic masses:

            >>> xyz = (
            ...     "bohr", "c2v",
            ...     "O", 0.00000000, 0.00000000, 0.12395915, "m=13.024815",  # custom mass for "O"
            ...     "H", 0.00000000, -1.43102686, -0.98366080,
            ...     "H", 0.00000000,  1.43102686, -0.98366080,
            ... )

        3. Isotope specification by element label:

            >>> xyz = (
            ...     "bohr", "c2v",
            ...     "O13", 0.00000000, 0.00000000, 0.12395915,  # oxygen-13 isotope
            ...     "H", 0.00000000, -1.43102686, -0.98366080,
            ...     "H", 0.00000000,  1.43102686, -0.98366080,
            ... )

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
                f"No units '{units}' found {[elem.name for elem in UnitType]}"
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
            me, k_list[j], jktau_list[j] = rotme_rot(j=j, linear=linear, sym=sym)

            # matrix elements of KEO
            ham = 0.5 * np.einsum("ab,abij->ij", gmat, me, optimize="optimal")

            for irrep in sym.irreps:
                print(f"solve for J = {j} and symmetry {irrep} ...")
                ind = np.where(np.array([elem[-1] for elem in jktau_list[j]]) == irrep)[
                    0
                ]
                if len(ind) > 0:
                    h = ham[np.ix_(ind, ind)]
                    e, v = np.linalg.eigh(h)
                    enr[j][irrep] = e
                    vec[j][irrep] = v
                    r_ind[j][irrep] = ind
                    v_ind[j][irrep] = np.array([0] * len(ind))
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

        return cls(masses, xyz, linear, j_list, sym_list, enr, vec, r_ind, v_ind)

    def mat(self):
        e0 = []
        for j in self.j_list:
            for sym in self.sym_list[j]:
                for m in range(-j, j + 1):
                    e0.append(self.enr[j][sym])
        return csr_array(diags(np.concatenate(e0)))

    def dens_on_grid(
        self,
        j1: int,
        sym1: str,
        istate1: int,
        j2: int,
        sym2: str,
        istate2: int,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
    ) -> np.ndarray:
        """Computes the reduced rotational probability density between two states
        as a function of the three Euler angles (α, β, γ).
        The density is obtained by integrating over vibrational coordinates,
        assuming an orthonormal vibrational basis.

        Args:
            j1 (int):
                Quantum number of the rotational angular momentum J for the bra-state.

            sym1 (str):
                Symmetry label of the bra-state.

            istate1 (int):
                Index of the bra-state within states of the same J and symmetry.

            j2 (int):
                Quantum number of the rotational angular momentum J for the ket-state.

            sym2 (str):
                Symmetry label of the ket-state.

            istate2 (int):
                Index of the ket-state within states of the same J and symmetry.

            alpha (np.ndarray):
                1D array of alpha Euler angles in radians.

            beta (np.ndarray):
                1D array of beta Euler angles in radians.

            gamma (np.ndarray):
                1D array of gamma Euler angles in radians.

        Returns:
            np.ndarray:
                3D array of shape (len(alpha), len(beta), len(gamma)) representing
                the reduced probability density for the specified states
                over the Euler angle grid.
        """
        rot_kv1, rot_m1, rot_l1, vib_ind1 = self._psi_on_grid(
            j1, sym1, istate1, alpha, beta, gamma
        )

        rot_kv2, rot_m2, rot_l2, vib_ind2 = self._psi_on_grid(
            j2, sym2, istate2, alpha, beta, gamma
        )

        vib_ind12 = list(set(vib_ind1) & set(vib_ind2))
        vi1 = [vib_ind1.index(v) for v in vib_ind12]
        vi2 = [vib_ind2.index(v) for v in vib_ind12]
        diff = list(set(vib_ind1) - set(vib_ind2))
        assert len(diff) == 0, (
            f"States (j1, sym1, istate1) = {(j1, sym1, istate1)} and (j2, sym2, istate2) = {(j2, sym2, istate2)}\n"
            + f"have non-overlapping sets of unique vibrational quanta: {list(set(vib_ind1))} != {list(set(vib_ind2))},\n"
            + f"difference: {diff}"
        )

        den_kv = np.einsum(
            "vlg,vng->lng", np.conj(rot_kv1[vi1]), rot_kv2[vi2], optimize="optimal"
        )
        den_m = np.einsum("mlg,mng->lng", np.conj(rot_m1), rot_m2, optimize="optimal")
        den_l = np.einsum("lg,ng->lng", np.conj(rot_l1), rot_l2, optimize="optimal")

        dens = np.einsum(
            "lng,lna,lnb,b->abg", den_kv, den_m, den_l, np.sin(beta), optimize="optimal"
        )
        return dens

    def _psi_on_grid(
        self,
        j: int,
        sym: str,
        istate: int,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        rot_k, rot_m, rot_l, k_list, jktau_list = symtop_on_grid_split_angles(
            j, alpha, beta, gamma
        )

        rot_ind = self.r_ind[j][sym]
        vib_ind = self.v_ind[j][sym]
        coefs = self.vec[j][sym][:, istate]

        rot_k = rot_k[rot_ind]
        vib_ind_unique = list(set(vib_ind))
        v_ind = [np.where(vib_ind == ind)[0] for ind in vib_ind_unique]
        unique_vec = np.zeros((len(vib_ind), len(vib_ind_unique)))
        for i, v in enumerate(v_ind):
            unique_vec[v, i] = 1
        pass
        rot_kv = np.einsum(
            "k,klg,kv->vlg", coefs, rot_k, unique_vec, optimize="optimal"
        )
        return rot_kv, rot_m, rot_l, vib_ind_unique


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
    symmetry = "c1"
    atom_labels = []
    atom_coords = []
    atom_masses = []

    avail_units = [elem.name for elem in UnitType]
    avail_sym = [elem.name for elem in SymmetryType]
    assert unit in avail_units, f"Default unit '{unit}' is not supported by {UnitType}"
    assert (
        symmetry in avail_sym
    ), f"Default symmetry '{symmetry}' is not supported by {SymmetryType}"

    ielem = 0
    while ielem < len(inp):
        item = inp[ielem]

        if isinstance(item, str):
            lower_item = item.lower()

            # unit
            if lower_item in avail_units:
                unit = lower_item
                ielem += 1
                continue

            # symmetry
            if lower_item in avail_sym:
                symmetry = lower_item
                ielem += 1
                continue

            # symmetry, "sym="
            # if lower_item.startswith("sym="):
            #     symmetry = lower_item.split("=")[1]
            #     ielem += 1
            #     continue

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
