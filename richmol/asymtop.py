import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import jax
import numpy as np
from jax import numpy as jnp
from mendeleev import element
from scipy import constants
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_array, diags
from scipy.spatial.transform import Rotation

from .rotsym import R0, RalphaPi, RzBeta, wang_symmetry_by_sampling
from .symtop import rotme_rot, rotme_watson, symtop_on_grid_split_angles
from .units import UnitType

jax.config.update("jax_enable_x64", True)


G_TO_INVCM = (
    constants.value("Planck constant")
    * constants.value("Avogadro constant")
    * 1e16
    / (4.0 * np.pi**2 * constants.value("speed of light in vacuum"))
    * 1e5
)

G_SING_TOL = 1e-12

EPS = jnp.array(
    [
        [[int((i - j) * (j - k) * (k - i) * 0.5) for k in range(3)] for j in range(3)]
        for i in range(3)
    ],
    dtype=jnp.float64,
)

ENERGY_UNITS = ("mhz", "invcm")
Energy_units = Literal[*ENERGY_UNITS]


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
    jktau_list: dict[int, list[tuple[int, int, int, str]]]
    m_list: dict[int, list[int]]
    enr_units: Energy_units

    _id: str = field(init=False)
    dim_k: dict[int, dict[str, int]] = field(init=False)
    dim_m: dict[int, int] = field(init=False)
    mk_ind: dict[int, dict[str, list[tuple[int, int]]]] = field(init=False)

    # quanta_dict[j][sym][n] = (j, m, k, tau, sym, c),
    #   where n runs across dim_m[j] -> dim_k[j][sym]
    quanta_dict: dict[int, dict[str, list[tuple[float]]]] = field(init=False)

    # quanta_dict_k[j][sym][n] = (j, k, tau, sym, c),
    #   where n runs across dim_k[j][sym]
    quanta_dict_k: dict[int, dict[str, list[tuple[float]]]] = field(init=False)

    # quanta[n] = (j, m, k, tau, sym, c),
    #   where n runs across j -> sym -> dim_m[j] -> dim_k[j][sym]
    quanta: np.ndarray = field(init=False)

    def __post_init__(self):
        self._id = str(uuid.uuid4())
        self.dim_m = {j: len(m) for j, m in self.m_list.items()}
        self.dim_k = {
            j: {sym: len(self.enr[j][sym]) for sym in self.sym_list[j]}
            for j in self.j_list
        }

        self.mk_ind = {
            j: {
                sym: np.array(
                    [
                        (im, ik)
                        for im in range(self.dim_m[j])
                        for ik in range(self.dim_k[j][sym])
                    ]
                )
                for sym in self.sym_list[j]
            }
            for j in self.j_list
        }

        # state assignment

        self.quanta_dict = {}
        self.quanta_dict_k = {}
        for j in self.j_list:
            quanta_sym = {}
            quanta_sym_k = {}
            for sym in self.sym_list[j]:
                e = self.enr[j][sym]
                v = self.vec[j][sym]
                jktau = [self.jktau_list[j][i] for i in self.r_ind[j][sym]]
                qua_k = []
                for i in range(len(e)):
                    ind = np.argmax(v[:, i] ** 2)
                    qua_k.append((*jktau[ind], v[ind, i]))
                qua_mk = [(q[0], m, *q[1:]) for m in self.m_list[j] for q in qua_k]
                quanta_sym[sym] = qua_mk
                quanta_sym_k[sym] = qua_k
            self.quanta_dict[j] = quanta_sym
            self.quanta_dict_k[j] = quanta_sym_k
        self.quanta = self._dict_to_vec(self.quanta_dict)

    @classmethod
    def watson(
        cls,
        max_j: int,
        inp,
        print_enr: bool = False,
        rotations: list[RzBeta | RalphaPi | R0] = [R0()],
        irreps: dict[str, list[int]] = {"A": [1]},
        m_list: list[int] | None = None,
    ):
        print(
            "\nCompute rotational solutions using Watson's effective Hamiltonian approach"
        )

        # constants used by symtop.rotme_watson
        avail_const = [
            "A",
            "B",
            "C",
            "DeltaJ",
            "DeltaJK",
            "DeltaK",
            "d1",
            "d2",
            "deltaJ",
            "deltaK",
            "HJ",
            "HJK",
            "HKJ",
            "HK",
            "h1",
            "h2",
            "h3",
            "phiJ",
            "phiJK",
            "phiK",
        ]
        avail_units = ["MHz", "kHz", "Hz"]
        const = _parse_rotconst_input(inp, avail_const, avail_units)

        # convert input constants to MHz
        mhz_units = {"mhz": 1, "khz": 1e-3, "hz": 1e-6}  # working units are MHz
        rot_const = {
            name: val * mhz_units[units.lower()] for (name, units), val in const.items()
        }

        print("Input rotational constants (MHz):")
        for name, val in rot_const.items():
            print(f" {name:>10} " f"{val:18.12f} ")

        # determine type of Watson Hamiltonian, A or S
        if any(
            elem in rot_const for elem in ("deltaJ", "deltaK", "phiJ", "phiJK", "phiK")
        ):
            watson_form = "A"
        elif any(elem in rot_const for elem in ("d1", "d2", "h1", "h2", "h3")):
            watson_form = "S"
        else:
            raise ValueError(
                "Cannot infer type of Watson Hamiltonian (S or A) from input rotational constants."
            )
        print(f"Watson reduction form: {watson_form}")

        # read A, B, C constants
        if {"A", "B", "C"}.issubset(rot_const.keys()):
            rot_a = rot_const["A"]
            rot_b = rot_const["B"]
            rot_c = rot_const["C"]
            linear = False
            if not (rot_a >= rot_b >= rot_c):
                raise ValueError(
                    f"Expected A >= B >= C, but got: A={rot_a}, B={rot_b}, C={rot_c} (in MHz)"
                )
        elif {"B"}.issubset(rot_const.keys()):
            rot_a = None
            rot_b = rot_const["B"]
            rot_c = None
            linear = True
            print("Molecule is linear")
        else:
            raise ValueError(
                f"Input for A, B, and C rotational constants is not provided"
            )

        # define Hamiltonian as function of J
        ham_func = lambda j: rotme_watson(
            j, rot_a, rot_b, rot_c, rot_const, watson_form
        )

        enr_units: Energy_units = "mhz"

        # solve Schrödinger equation
        return cls._solve(
            max_j,
            ham_func,
            masses=np.full(1, None),
            xyz=np.full(1, None),
            linear=linear,
            print_enr=print_enr,
            enr_units=enr_units,
            rotations=rotations,
            irreps=irreps,
            m_list=m_list,
        )

    @classmethod
    def from_geometry(
        cls,
        max_j: int,
        inp,
        print_enr: bool = False,
        rotations: list[RzBeta | RalphaPi | R0] = [R0()],
        irreps: dict[str, list[int]] = {"A": [1]},
        m_list: list[int] | None = None,
    ):
        """Computes rigid rotor states from a given molecular geometry.

        Args:
            max_j (int):
                Maximum rotational angular momentum quantum number (J).
                Rotational states will be computed for all values from J = 0 up to J = `max_j`.

            inp (list or tuple):
                A specification of the molecular geometry, including atomic positions, units,
                and optional mass assignments. Examples are shown below.

            print_enr (bool, optional):
                If True, prints a table of computed rotational energy levels and
                their corresponding quantum state assignments.
                Default is False.

        Examples of 'inp':

        1. Standard geometry specification:

            >>> xyz = (
            ...     "bohr",
            ...     "O", 0.00000000, 0.00000000, 0.12395915,
            ...     "H", 0.00000000, -1.43102686, -0.98366080,
            ...     "H", 0.00000000,  1.43102686, -0.98366080,
            ... )

        2. Custom atomic mass specification to override default atomic masses:

            >>> xyz = (
            ...     "bohr",
            ...     "O", 0.00000000, 0.00000000, 0.12395915, "m=13.024815",  # custom mass for "O"
            ...     "H", 0.00000000, -1.43102686, -0.98366080,
            ...     "H", 0.00000000,  1.43102686, -0.98366080,
            ... )

        3. Isotope specification by element label:

            >>> xyz = (
            ...     "bohr",
            ...     "O13", 0.00000000, 0.00000000, 0.12395915,  # oxygen-13 isotope
            ...     "H", 0.00000000, -1.43102686, -0.98366080,
            ...     "H", 0.00000000,  1.43102686, -0.98366080,
            ... )

        """
        print("\nCompute rigid-rotor solutions using molecular geometry as input")

        atom_labels, atom_xyz, atom_masses, units = _parse_geom_input(inp)

        atom_masses = [
            mass if mass != None else atom_mass(labes)
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

        # rotational kinetic energy G-matrix

        masses = np.array(atom_masses)
        xyz = np.array(atom_xyz)
        com = masses @ xyz / jnp.sum(masses)
        xyz -= com[None, :]

        linear = check_linear(xyz)
        if linear:
            print("Molecule is linear")

        g = gmat(masses, xyz)[:3, :3]
        u, sv, v = np.linalg.svd(g, full_matrices=True)
        d = np.where(sv > G_SING_TOL, 1 / sv, 0)

        nnz = np.count_nonzero(d == 0)
        if (nnz == 1 and not linear) or nnz > 1:
            raise ValueError(f"Kinetic energy g-matrix is singular") from None

        G = (u @ np.diag(d) @ v) * G_TO_INVCM

        print("G-matrix from input Cartesian coordinates (cm^-1):\n", G)

        # define Hamiltonian as function of J
        def ham_func(j):
            me, k_list, jktau_list = rotme_rot(j=j, linear=linear)
            ham = 0.5 * np.einsum("ab,abij->ij", G, me, optimize="optimal")
            return ham, k_list, jktau_list

        enr_units: Energy_units = "invcm"

        # solve Schrödinger equation
        return cls._solve(
            max_j,
            ham_func,
            masses=masses,
            xyz=xyz,
            linear=linear,
            print_enr=print_enr,
            enr_units=enr_units,
            rotations=rotations,
            irreps=irreps,
            m_list=m_list,
        )

    @classmethod
    def _solve(
        cls,
        max_j,
        ham_func,
        masses,
        xyz,
        linear: bool,
        print_enr: bool,
        enr_units: Energy_units,
        rotations: list[RzBeta | RalphaPi | R0],
        irreps: dict[str, list[int]],
        m_list: list[int] | None,
    ):
        # identify symmetry of symmetric-top function in Wang representation

        sym_table = {}
        for j in range(max_j + 1):
            sym_table[j] = wang_symmetry_by_sampling(j, linear, rotations, irreps)

        sym_labels = list(
            set([sym for sym_j in sym_table.values() for sym in sym_j.values()])
        )

        # solve for J = 0 .. max_j

        nested_dict = lambda: defaultdict(nested_dict)
        enr = nested_dict()
        vec = nested_dict()
        v_ind = nested_dict()
        r_ind = nested_dict()
        sym_list = {}
        jktau_list = {}
        k_list = {}

        j_list = [j for j in range(max_j + 1)]

        # consider m quanta, truncate j_list if necessary
        if m_list is None:
            m_list_j = {j: list(range(-j, j + 1)) for j in j_list}
        else:
            m_list_j = {j: [m for m in m_list if abs(m) <= j] for j in j_list}
        j_list = [j for j, m in m_list_j.items() if len(m) > 0]
        m_list_j = {j: m_list_j[j] for j in j_list}

        sym_list = {j: [] for j in j_list}

        for j in j_list:
            ham, k_list[j], jktau_list[j] = ham_func(j=j)

            for sym in sym_labels:
                print(f"solve for J = {j} and symmetry {sym} ...")

                ind = [
                    jktau_list[j].index(jktau)
                    for jktau, sym_ in sym_table[j].items()
                    if sym_ == sym
                ]

                print(f"number of functions:", len(ind))

                if len(ind) > 0:
                    h = ham[np.ix_(ind, ind)]
                    e, v = np.linalg.eigh(h)
                    enr[j][sym] = e
                    vec[j][sym] = v
                    r_ind[j][sym] = ind
                    v_ind[j][sym] = np.array([0] * len(ind))
                    sym_list[j].append(sym)

        # print solutions

        print("Energy units:", enr_units)

        if print_enr:
            print(
                f"{'J':>3} {'Irrep':>5} {'i':>4} {'Energy':>18} {'(J,k,tau,Irrep)':>20} {'c_max²':>16}"
            )
            for j in j_list:
                for sym in sym_list[j]:
                    e = enr[j][sym]
                    v = vec[j][sym]
                    jktau = [jktau_list[j][i] for i in r_ind[j][sym]]
                    for i in range(len(e)):
                        v2 = v[:, i] ** 2
                        largest_ind = np.argsort(v2)[-3:][::-1]
                        for ii, ind in enumerate(largest_ind):
                            if ii == 0:
                                print(
                                    f"{j:3d} "
                                    f"{sym:>5} "
                                    f"{i:4d} "
                                    f"{e[i]:18.8f} "
                                    f"{str(jktau[ind]):>20} "
                                    f"{v2[ind]:16.12f}"
                                )
                            else:
                                print(
                                    " " * 33,
                                    f"{str(jktau[ind]):>20} " f"{v2[ind]:16.12f}",
                                )

        return cls(
            masses,
            xyz,
            linear,
            j_list,
            sym_list,
            enr,
            vec,
            r_ind,
            v_ind,
            jktau_list,
            m_list_j,
            enr_units,
        )

    def mat(self):
        e0 = []
        for j in self.j_list:
            for sym in self.sym_list[j]:
                for m in self.m_list[j]:
                    e0.append(self.enr[j][sym])
        return csr_array(diags(np.concatenate(e0)))

    def mc_costheta(
        self,
        coefs: np.ndarray,
        mol_axis: np.ndarray = np.array([0, 0, 1]),
        lab_axis: np.ndarray = np.array([0, 0, 1]),
        lab_plane: np.ndarray = np.array([[0, 1, 0], [0, 0, 1]]),
        alpha: np.ndarray = np.linspace(0, 2 * np.pi, 30),
        beta: np.ndarray = np.linspace(0, np.pi, 30),
        gamma: np.ndarray = np.linspace(0, 2 * np.pi, 30),
        npoints: int = 1000000,
        thresh: float = 1e-8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculates expectation values of angular observables for a wavepacket
        using Metropolis sampling approach.

        Computes the expectation values of cos(theta), cos^2(theta), cos(theta_2D), and
        cos^2(theta_2D), where theta is the angle between a molecular-frame
        vector (`mol_axis`) and a laboratory-frame vector (`lab_axis`), and
        theta_2D is the projection of theta onto a plane defined by `lab_plane`.

        Args:
            coefs (np.ndarray):
                2D array of wavepacket coefficients, where the first
                dimension indexes rovibrational states and the second
                may index time steps or other parameters.

            mol_axis (np.ndarray, optional):
                3-element array representing molecular-frame axis.
                Default is [0.0, 0.0, 1.0], i.e., z-axis.

            lab_axis (np.ndarray, optional):
                3-element array representing laboratory-frame axis.
                Default is [0.0, 0.0, 1.0], i.e., Z-axis.

            lab_plane (np.ndarray, optional):
                2x3 array where each row defines a vector spanning the laboratory plane
                used for theta_2D projection.
                Default is [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], i.e., YZ-plane.

            alpha (np.ndarray, optional):
                1D array of alpha Euler angles in radians, used for computing
                of rotational density.
                Default is np.linspace(0, 2 * np.pi, 30).

            beta (np.ndarray, optional):
                1D array of beta Euler angles in radians, used for computing
                of rotational density.
                Default is np.linspace(0, np.pi, 30).

            gamma (np.ndarray, optional):
                1D array of gamma Euler angles in radians, used for computing
                of rotational density.
                Default is np.linspace(0, 2 * np.pi, 30).

            npoints (int, optional):
                Number of points used for Metropolis rejection sampling.
                Default is 1,000,000

            thresh (float, optional):
                Coefficients with magnitudes below this threshold are ignored.
                Default is 1e-8.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing the following arrays:
                - costheta: cos(theta) expectation values along the second dimension
                    of wavepacket coefficients `coefs`.
                - cos2theta: cos^2(theta) expectation values along the second dimension of `coefs`.
                - costheta2d: cos(theta_2D) expectation values along the second dimension of `coefs`.
                - cos2theta2d: cos^2(theta_2D) expectation values along the second dimension of `coefs`.
        """
        dens = self.rot_dens_wp(
            coefs, alpha, beta, gamma, thresh=thresh, method="wp_dens"
        )
        fdens = RegularGridInterpolator((alpha, beta, gamma), dens)
        max_dens = np.max(dens, axis=(0, 1, 2))
        pts = np.random.uniform(
            low=[0, 0, 0], high=[2 * np.pi, np.pi, 2 * np.pi], size=(npoints, 3)
        )
        w = fdens(pts) / max_dens
        eta = np.random.uniform(0.0, 1.0, size=len(w))
        points = [pts[np.where(w_ > eta)] for w_ in w.T]
        rot_mat = [Rotation.from_euler("ZYZ", pts).as_matrix() for pts in points]
        # TODO!!! check "ZYZ" or "zyz"?

        # distribution of the molecular-frame axis `mol_axis` in laboratory frame
        mol_axis = np.array(mol_axis) / np.linalg.norm(mol_axis)
        mol_axis_distr = [np.dot(mat, mol_axis) for mat in rot_mat]

        # normalize lab axis and lab plane
        lab_axis = np.array(lab_axis) / np.linalg.norm(lab_axis)
        lab_plane = np.array(lab_plane) / np.linalg.norm(lab_plane, axis=-1)[:, None]

        # vector perpendicular to the laboratory plane `lab_plane`
        lab_plane_norm = np.cross(*lab_plane)
        lab_plane_norm = lab_plane_norm / np.linalg.norm(lab_plane_norm)

        # projection of the laboratory axis `lab_axis` onto the laboratory plane `lab_plane`
        lab_axis_lab_plane = np.cross(
            lab_plane_norm, np.cross(lab_axis, lab_plane_norm)
        )

        # projection of the molecular axis distribution onto the laboratory axis `lab_axis`
        mol_axis_lab_axis = [np.dot(ax, lab_axis) for ax in mol_axis_distr]

        # projection of the molecular axis distribution onto the laboratory plane `lab_plane`
        mol_axis_lab_plane = [
            np.cross(lab_plane_norm, np.cross(ax, lab_plane_norm))
            for ax in mol_axis_distr
        ]
        mol_axis_lab_plane = [
            ax / np.linalg.norm(ax, axis=-1)[:, None] for ax in mol_axis_lab_plane
        ]

        # projection of `mol_axis_lab_plane` onto the `lab_axis_lab_plane`
        mol_axis_lab_axis_plane = [
            np.dot(ax, lab_axis_lab_plane) for ax in mol_axis_lab_plane
        ]

        # cos of theta between molecular axis `mol_axis` and laboratory axis `lab_axis`
        # and its projection onto laboratory plane `lab_plane`
        costheta = np.array([np.mean(elem) for elem in mol_axis_lab_axis])
        cos2theta = np.array([np.mean(elem**2) for elem in mol_axis_lab_axis])
        costheta2d = np.array([np.mean(elem) for elem in mol_axis_lab_axis_plane])
        cos2theta2d = np.array([np.mean(elem**2) for elem in mol_axis_lab_axis_plane])

        return costheta, cos2theta, costheta2d, cos2theta2d

    def mc_rot_dens_wp(
        self,
        coefs: np.ndarray,
        alpha: np.ndarray = np.linspace(0, 2 * np.pi, 30),
        beta: np.ndarray = np.linspace(0, np.pi, 30),
        gamma: np.ndarray = np.linspace(0, 2 * np.pi, 30),
        npoints: int = 1000000,
        thresh: float = 1e-8,
    ) -> list[np.ndarray]:
        dens = self.rot_dens_wp(
            coefs, alpha, beta, gamma, thresh=thresh, method="wp_dens"
        )
        fdens = RegularGridInterpolator((alpha, beta, gamma), dens)
        max_dens = np.max(dens, axis=(0, 1, 2))
        pts = np.random.uniform(
            low=[0, 0, 0], high=[2 * np.pi, np.pi, 2 * np.pi], size=(npoints, 3)
        )
        w = fdens(pts) / max_dens
        eta = np.random.uniform(0.0, 1.0, size=len(w))
        points = [pts[np.where(w_ > eta)] for w_ in w.T]
        rot_mat = [Rotation.from_euler("ZYZ", pts).as_matrix() for pts in points]
        # TODO!!! check "ZYZ" or "zyz"?
        return rot_mat

    def rot_dens_wp(
        self,
        coefs: np.ndarray,
        alpha: np.ndarray = np.linspace(0, 2 * np.pi, 30),
        beta: np.ndarray = np.linspace(0, np.pi, 30),
        gamma: np.ndarray = np.linspace(0, 2 * np.pi, 30),
        thresh: float = 1e-8,
        method: str = "wp_dens",
    ) -> np.ndarray:
        """Computes the reduced rotational probability density of a wavepacket
        as a function of Euler angles α, β, γ.
        The density is obtained by integrating over vibrational coordinates,
        assuming an orthonormal vibrational basis.

        Args:
            coefs (np.ndarray):
                Array of wavepacket coefficients, where the first
                dimension indexes rovibrational states.

            alpha (np.ndarray, optional):
                1D array of alpha Euler angles in radians.
                Default is np.linspace(0, 2 * np.pi, 30).

            beta (np.ndarray, optional):
                1D array of beta Euler angles in radians.
                Default is np.linspace(0, np.pi, 30).

            gamma (np.ndarray, optional):
                1D array of gamma Euler angles in radians.
                Default is np.linspace(0, 2 * np.pi, 30).

            thresh (float, optional):
                Coefficients with magnitudes below this threshold are ignored.
                Default is 1e-8.

        Returns:
            np.ndarray:
                An array of shape (len(alpha), len(beta), len(gamma), ...) representing
                the reduced rotational probability density on the Euler angle grid.
                The trailing dimensions match `coefs.shape[1:]` or equal to 1
                if `coefs` has only a single dimension.
        """
        if method == "wp_dens":
            return self._rot_dens_wp1(coefs, alpha, beta, gamma, thresh)
        elif method == "prim_dens":
            return self._rot_dens_wp2(coefs, alpha, beta, gamma, thresh)
        else:
            raise ValueError(f"Unknown value of parameter 'method' = {method}")

    def _rot_dens_wp1(
        self,
        coefs: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        thresh: float,
    ) -> np.ndarray:
        print("remark: compute reduced rotational density using _rot_dens_wp1")

        if coefs.ndim == 1:
            coefs_ = np.array([coefs]).T
        else:
            coefs_ = coefs

        # convert wavepacket coefficients coefs[...]
        #   into dictionary coefs[j][sym][...]
        #   and identify indices for coefs > thresh
        coefs_dict = self._vec_to_dict(coefs_)
        coefs_ind = {
            j: {
                sym: np.any(np.abs(coefs_dict[j][sym]) > thresh, axis=1).nonzero()[0]
                for sym in self.sym_list[j]
            }
            for j in self.j_list
        }

        # set of unique vibrational state indices
        #   across all J and symmetries
        vib_ind = list(
            set(
                [
                    v
                    for j in self.j_list
                    for sym in self.sym_list[j]
                    for v in self.v_ind[j][sym]
                ]
            )
        )

        psi = np.zeros(
            (len(vib_ind), len(alpha), len(beta), len(gamma), *coefs_.shape[1:]),
            dtype=np.complex128,
        )

        for j in self.j_list:
            for sym in self.sym_list[j]:
                ind = coefs_ind[j][sym]
                dim = len(ind)
                if dim == 0:
                    continue
                c = coefs_dict[j][sym][ind]  # wavepacket coefficients

                rot_kv, rot_m, rot_l, v_ind = self._rot_psi_grid(
                    j, sym, alpha, beta, gamma
                )
                im, ik = self.mk_ind[j][sym][ind].T
                iv = [vib_ind.index(v) for v in v_ind]

                psi[iv] += np.einsum(
                    "i...,ivlg,ila,lb->vabg...",
                    c,
                    rot_kv[ik],
                    rot_m[im],
                    rot_l,
                    optimize="optimal",
                )

        return np.einsum(
            "vabg...,vabg...,b->abg...",
            np.conj(psi),
            psi,
            np.sin(beta),
            optimize="optimal",
        )

    def _rot_dens_wp2(
        self,
        coefs: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        thresh: float,
    ) -> np.ndarray:
        print("remark: compute reduced rotational density using _rot_dens_wp2")

        na = len(alpha)
        nb = len(beta)
        ng = len(gamma)

        if coefs.ndim == 1:
            coefs_ = np.array([coefs]).T
        else:
            coefs_ = coefs

        coefs_dict = self._vec_to_dict(coefs_)
        coefs_ind = {
            j: {
                sym: np.any(np.abs(coefs_dict[j][sym]) > thresh, axis=1).nonzero()[0]
                for sym in self.sym_list[j]
            }
            for j in self.j_list
        }

        dens = np.zeros((na, nb, ng, *coefs_.shape[1:]), dtype=np.complex128)

        for j1 in self.j_list:
            for sym1 in self.sym_list[j1]:
                ind1 = coefs_ind[j1][sym1]
                dim1 = len(ind1)
                if dim1 == 0:
                    continue
                c1 = coefs_dict[j1][sym1][ind1]

                for j2 in self.j_list:
                    for sym2 in self.sym_list[j2]:
                        ind2 = coefs_ind[j2][sym2]
                        dim2 = len(ind2)
                        if dim2 == 0:
                            continue
                        c2 = coefs_dict[j2][sym2][ind2]

                        d = np.zeros((dim1, dim2, na, nb, ng), dtype=np.complex128)

                        for i, istate1 in enumerate(ind1):
                            for j, istate2 in enumerate(ind2):
                                d[i, j] = self.rot_dens(
                                    j1,
                                    sym1,
                                    istate1,
                                    j2,
                                    sym2,
                                    istate2,
                                    alpha,
                                    beta,
                                    gamma,
                                )
                        dens += np.einsum(
                            "i...,ijabg,j...->abg...",
                            np.conj(c1),
                            d,
                            c2,
                            optimize="optimal",
                        )
        return dens

    def rot_dens(
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
        """Computes the reduced rotational probability density for two selected states
        as a function of Euler angles α, β, γ.
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
                the reduced rotational probability density for the specified states
                on the Euler angle grid.
        """
        im1, ik1 = self.mk_ind[j1][sym1][istate1]
        rot_kv1, rot_m1, rot_l1, vib_ind1 = self._rot_psi_grid(
            j1, sym1, alpha, beta, gamma, ik1
        )

        im2, ik2 = self.mk_ind[j2][sym2][istate2]
        rot_kv2, rot_m2, rot_l2, vib_ind2 = self._rot_psi_grid(
            j2, sym2, alpha, beta, gamma, ik2
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
        den_m = np.einsum(
            "lg,ng->lng", np.conj(rot_m1[im1]), rot_m2[im2], optimize="optimal"
        )
        den_l = np.einsum("lg,ng->lng", np.conj(rot_l1), rot_l2, optimize="optimal")

        dens = np.einsum(
            "lng,lna,lnb,b->abg", den_kv, den_m, den_l, np.sin(beta), optimize="optimal"
        )
        return dens

    def _rot_psi_grid(
        self,
        j: int,
        sym: str,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        istate: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:

        rot_k, rot_m, rot_l, k_list, jktau_list = symtop_on_grid_split_angles(
            j, alpha, beta, gamma, self.linear
        )

        rot_ind = self.r_ind[j][sym]
        vib_ind = self.v_ind[j][sym]
        coefs = self.vec[j][sym]
        if istate is not None:
            coefs = coefs[:, istate]

        rot_k = rot_k[rot_ind]
        vib_ind_unique = list(set(vib_ind))
        v_ind = [np.where(vib_ind == ind)[0] for ind in vib_ind_unique]
        unique_vec = np.zeros((len(vib_ind), len(vib_ind_unique)))
        for i, v in enumerate(v_ind):
            unique_vec[v, i] = 1
        rot_kv = np.einsum(
            "k...,klg,kv->...vlg", coefs, rot_k, unique_vec, optimize="optimal"
        )
        return rot_kv, rot_m, rot_l, vib_ind_unique

    def _vec_to_dict(self, vec: np.ndarray):
        """Converts vector vec[n] to vec[j][sym][k] where n runs across j -> sym -> k"""
        vec_dict = {}
        offset = 0
        for j in self.j_list:
            vec_dict[j] = {}
            for sym in self.sym_list[j]:
                d = self.dim_k[j][sym] * self.dim_m[j]
                vec_dict[j][sym] = vec[offset : offset + d]
                offset += d
        return vec_dict

    def _dict_to_vec(self, vec_dict) -> np.ndarray:
        """Converts vector vec[j][sym][k] to vec[n] where n runs across j -> sym -> k"""
        blocks = []
        for j in self.j_list:
            for sym in self.sym_list[j]:
                blocks.append(vec_dict[j][sym])
        return np.concatenate(blocks)


def gmat(masses, xyz):
    t_rot = np.transpose(EPS @ np.asarray(xyz).T, (2, 0, 1))
    t_tra = np.array([np.eye(3, dtype=np.float64) for _ in range(len(xyz))])
    t = np.concatenate((t_rot, t_tra), axis=2)
    masses_sq = np.sqrt(np.asarray(masses))
    t = t * masses_sq[:, None, None]
    t = np.reshape(t, (len(xyz) * 3, 6))
    return t.T @ t


def check_linear(xyz, tol=1e-8):
    if len(xyz) < 2:
        return True
    v0 = xyz[1] - xyz[0]
    for i in range(2, len(xyz)):
        if np.linalg.norm(np.linalg.cross(v0, xyz[i] - xyz[0])) > tol:
            return False
    return True


def com(masses, xyz):
    return np.sum([x * m for x, m in zip(xyz, masses)], axis=0) / np.sum(masses)


def inertia_tensor(masses, xyz):
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


def write_xyz(
    filename, atom_labels: list[str], atom_xyz: np.ndarray, comment: str = ""
):
    if len(atom_labels) != len(atom_xyz):
        raise ValueError("Length of 'atom_labels' and 'atom_xyz' must match.")

    with open(filename, "w") as f:
        f.write(f"{len(atom_labels)}\n")
        f.write(f"{comment}\n")
        for atom, (x, y, z) in zip(atom_labels, atom_xyz):
            f.write(f"{atom:2} {x:15.8f} {y:15.8f} {z:15.8f}\n")


def rotational_constants(masses, xyz):
    imat = inertia_tensor(masses, xyz)
    d, v = np.linalg.eigh(imat)
    a, b, c = 0.5 / d * G_TO_INVCM
    return (a, b, c), v


def atom_mass(atom_label: str) -> float:
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


def _parse_geom_input(inp):
    unit = "angstrom"
    atom_labels = []
    atom_coords = []
    atom_masses = []

    avail_units = [elem.name for elem in UnitType]

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

    return atom_labels, atom_coords, atom_masses, unit


def _parse_rotconst_input(inp, avail_const, avail_units):
    const = {}

    ielem = 0
    while ielem < len(inp):
        item = inp[ielem]

        if isinstance(item, str):
            lower_item = item.lower()

            # constant label
            label = item
            parts = label.split("/")
            if len(parts) == 2 and all(parts):
                name, units = parts
                if name.lower() not in (elem.lower() for elem in avail_const):
                    raise ValueError(
                        f"Unknown input constant '{name}', supported values: {avail_const}"
                    )
                if units.lower() not in (elem.lower() for elem in avail_units):
                    raise ValueError(
                        f"Unknown input units '{units}', supported values: {avail_units}"
                    )
                const[(name, units)] = inp[ielem + 1]
            else:
                raise ValueError(
                    f"Unexpected input for rotational constant: '{label}', expected format: name/units, e.g. 'A/MHz'"
                )
            ielem += 2

        else:
            raise ValueError(f"Unexpected item at position {ielem}: {item}")

    return const
