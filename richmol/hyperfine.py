import numpy as np
import py3nj
from scipy import constants

from .asymtop import RotStates
from .cartens import CartTensor, Rank2Tensor
from .nucspin import *

# Conversion from EFG[a.u.]*Q[mb] to EFG*Q[cm^-1]
quad_mb_to_meter2 = 1e-31
efg_au_to_volts_per_meter2 = constants.value("atomic unit of electric field gradient")
efg_x_quad_joul = (
    constants.elementary_charge * quad_mb_to_meter2 * efg_au_to_volts_per_meter2
)
joul_to_invcm = 1 / (
    constants.value("Planck constant")
    * 1e2
    * constants.value("speed of light in vacuum")
)
EFG_X_QUAD_INVCM = efg_x_quad_joul * joul_to_invcm


def _symmetry_spin_rotation_placeholder(
    j_list: list[int],
    j_sym_list: dict[int, list[str]],
    spin_list: list[tuple[float]],
):
    sym_list = list(
        set([sym for sym_list in list(j_sym_list.values()) for sym in sym_list])
    )
    assert sym_list == ["A"], (
        "The placeholder spin-rotation symmetrization function works only for C1 spatial symmetry group,\n"
        + f"i.e., no symmetry, instead got spatial symmetries: {sym_list}"
    )
    j_spin_list = {}
    for j, spin in zip(j_list, spin_list):
        for j_sym in j_sym_list[j]:
            ##########
            # f_sym = ... # code symmetry spin-spatial symmetry products
            f_sym = j_sym
            spin_sym = "A"
            ##########
            try:
                j_spin_list[f_sym].append((j, spin, j_sym, spin_sym))
            except KeyError:
                j_spin_list[f_sym] = [(j, spin, j_sym, spin_sym)]
    return j_spin_list


class HyperStates:
    """Computes hyperfine states using molecular rovibrational states
    and rovibrational matrix elements of electric and magnetic spin
    interaction tensors.

    Args:
        min_f (float):
            Minimal total angular momentum quantum number F = I + J,
            where I is the total nuclear spin and J is the rotational
            angular momentum.

        max_f (float):
            Maximal total angular momentum quantum number F = I + J.

        states (RotStates):
            Molecular rotational or rovibrational energy states
            used as the basis for hyperfine calculations.

        spin_op (list[SpinOperator]):
            List of nuclear spin operators for each coupled nucleus.

        efg_op (list[CartTensor | None], optional):
            List of electric field gradient tensors corresponding
            to each spin operator in `spin_op`.
            For non-quadrupolar nuclei, set the corresponding entry to `None`.
            Each EFG tensor must be in atomic units.
            Defaults to an empty list, meaning quadrupole interactions are excluded.
    """

    f_list: list[float]
    f_sym_list: dict[float, list[str]]
    j_spin_list: dict[float, dict[str, list[tuple[int, tuple[float], str, str, int]]]]
    enr0: dict[float, dict[str, np.ndarray]]
    enr: dict[float, dict[str, np.ndarray]]
    vec: dict[float, dict[str, np.ndarray]]

    def __init__(
        self,
        min_f: float,
        max_f: float,
        states: RotStates,
        spin_op: list[SpinOperator],
        efg_op: list[CartTensor | None] = [],
    ):
        print("\nCompute hyperfine states")

        assert min_f <= max_f, f"'min_f'> 'max_f': {min_f} > {max_f}"
        assert len(spin_op) > 0, f"'spin_op' is empty"

        # check input for quadrupole

        quad_op_ind = []  # elements of `spin_op` and `efg_op` for quadrupole coupling

        if len(efg_op) > 0:
            if len(efg_op) != len(spin_op):
                raise ValueError(
                    f"'efg_op' and 'spin_op' must have the same number of elements,"
                    f"len(efg_op) = {len(efg_op)}, len(spin_op) = {len(spin_op)}"
                )

            for i, efg in enumerate(efg_op):
                if efg is not None:
                    if not hasattr(spin_op[i], "Q"):
                        raise ValueError(
                            f"EFG tensor in 'efg_op[{i}]' is provided (non-None),\n"
                            f"but the corresponding spin operator in 'spin_op[{i}]' "
                            + "lacks a quadrupole moment ('Q' attribute).\n"
                        )

                    if not efg.spher_ind == Rank2Tensor().spher_ind:
                        raise ValueError(
                            f"EFG tensor in 'efg_op[{i}]' has invalid attribute 'spher_ind': {efg.spher_ind}, "
                            + f"expected: {Rank2Tensor.spher_ind}"
                        ) from None

                    quad_op_ind.append(i)

        # generate combinations of rovibrational and spin angular momentum quantum numbers

        self.f_list = [
            round(f) for f in np.linspace(min_f, max_f, int(max_f - min_f) + 1)
        ]
        print(f"List of F quanta: {self.f_list}")

        self.j_spin_list = {}
        self.f_sym_list = {}

        for f in self.f_list:
            spin_list, j_list = near_equal_coupling_with_rotations(f, spin_op)
            j_sym_list = states.sym_list
            j_spin_list = _symmetry_spin_rotation_placeholder(
                j_list, j_sym_list, spin_list
            )
            self.f_sym_list[f] = list(j_spin_list.keys())

            # add dimension of the rovibrational basis to the list
            # i.e., (J, spin, J sym, spin sym) -> (J, spin, J sym, spin sym, J dim)
            self.j_spin_list[f] = {
                f_sym: [
                    (j, spin, j_sym, spin_sym, len(states.enr[j][j_sym]))
                    for (j, spin, j_sym, spin_sym) in j_spin_list[f_sym]
                ]
                for f_sym in self.f_sym_list[f]
            }

        # print quantum numbers

        print(
            f"{'F':<3} {'tot.sym.':<10} {'J':<3} {'(I_1, I_12, ... I_1N)':<25} "
            + f"{'rovib.sym.':<12} {'spin.sym.':<10} {'rovib.dim':<10}"
        )
        print("-" * 76)

        for f in self.f_list:
            for f_sym in self.f_sym_list[f]:
                for j, spin, j_sym, spin_sym, j_dim in self.j_spin_list[f][f_sym]:
                    spin_str = (
                        "("
                        + ", ".join(f"{s:.1f}" if s % 1 else f"{int(s)}" for s in spin)
                        + ")"
                    )
                    print(
                        f"{f:<3} {f_sym:<10} {j:<3} {spin_str:<25} "
                        + f"{j_sym:<12} {spin_sym:<10} {j_dim:<10}"
                    )

        # solve hyperfine problem for different F and symmetries

        self.enr0 = {}
        self.enr = {}
        self.vec = {}

        for f in self.f_list:

            enr0 = {}
            enr = {}
            vec = {}

            for sym in self.f_sym_list[f]:

                print(f"solve for F = {f} and symmetry {sym} ...")

                # pure rovibrational Hamiltonian

                enr0[sym] = np.concatenate(
                    [
                        states.enr[j][j_sym]
                        for (j, spin, j_sym, *_) in self.j_spin_list[f][sym]
                    ]
                )

                h = np.diag(enr0[sym])

                # add quadrupole interaction Hamiltonian

                if len(quad_op_ind) > 0:
                    print("add quadrupole")
                    h = h + _quadrupole_me(
                        f,
                        self.j_spin_list[f][sym],
                        [spin_op[i] for i in quad_op_ind],
                        [efg_op[i] for i in quad_op_ind],
                    )

                # diagonalization

                enr[sym], vec[sym] = np.linalg.eigh(h)

            self.enr0[f] = enr0
            self.enr[f] = enr
            self.vec[f] = vec


def _quadrupole_me(
    f_val: float,
    j_spin_list: list[tuple[int, tuple[float], str, str, int]],
    quad_op: list[SpinOperator],
    efg_op: list[CartTensor],
):
    spin_list = [spin for (_, spin, *_) in j_spin_list]

    # <I' || Q(i) || I>
    quad_me = reduced_me(spin_list, spin_list, quad_op)

    h = []
    for j1, spin1, j_sym1, spin_sym1, j_dim1 in j_spin_list:
        h_ = []
        for j2, spin2, j_sym2, spin_sym2, j_dim2 in j_spin_list:

            zero_block = np.zeros((j_dim1, j_dim2))

            try:
                quad = quad_me[(spin1, spin2)]
            except KeyError:
                h_.append(zero_block)
                continue

            me = 0
            for efg, q in zip(efg_op, quad):
                try:
                    kmat = efg.kmat[(j1, j2)][(j_sym1, j_sym2)][2]  # for omega = 2
                except KeyError:
                    continue
                me += kmat * q

            if isinstance(me, int) and me == 0:
                h_.append(zero_block)
                continue

            fac = spin1[-1] + f_val
            assert float(fac).is_integer(), f"Non-integer power in (-1)**f: (-1)**{fac}"
            fac = int(fac)
            prefac = (
                (-1) ** fac
                / np.sqrt(6)
                * np.sqrt((2 * j1 + 1) * (2 * j2 + 1))
                * py3nj.wigner6j(
                    int(f_val * 2),
                    int(spin1[-1] * 2),
                    j1 * 2,
                    2 * 2,
                    j2 * 2,
                    int(spin2[-1] * 2),
                    ignore_invalid=True,
                )
            )

            me_cm = prefac * me * EFG_X_QUAD_INVCM
            h_.append(me_cm)

        h.append(h_)

    h = np.block(h)
    return h


class HyperCartTensor:
    rank: int
    kmat: dict[tuple[float, float], dict[tuple[str, str], np.ndarray]]
    mmat: dict[tuple[float, float], np.ndarray]
    f_list: list[float]
    f_sym_list: dict[float, list[str]]
    j_spin_list: dict[float, dict[str, list[tuple[int, tuple[float], str, str, int]]]]

    def __init__(self, states: HyperStates, cart_tens: CartTensor, tol: float = 1e-12):
        self.kmat = self._k_tens(states, cart_tens, tol)
        self.mmat = self._m_tens(states, cart_tens, tol)
        self.rank = cart_tens.rank
        self.f_list = states.f_list
        self.f_sym_list = states.f_sym_list
        self.j_spin_list = states.j_spin_list

    def _k_tens(self, states, cart_tens, tol: float):
        omega_list = list(cart_tens.umat_cart_to_spher.keys())

        k_me = {}

        for f1 in states.f_list:
            for f2 in states.f_list:

                k_me_sym = {}

                for sym1 in states.f_sym_list[f1]:
                    for sym2 in states.f_sym_list[f2]:
                        v1 = states.vec[f1][sym1]
                        v2 = states.vec[f2][sym2]

                        k_me_omega = {}

                        for omega in omega_list:
                            me = self._k_me_omega(
                                f1,
                                f2,
                                sym1,
                                sym2,
                                states.j_spin_list,
                                omega,
                                cart_tens.kmat,
                            )

                            if np.any(np.abs(me) > tol):
                                k_me_omega[omega] = np.einsum(
                                    "ik,ij,jl->kl",
                                    np.conj(v1),
                                    me,
                                    v2,
                                    optimize="optimal",
                                )

                        if k_me_omega:
                            k_me_sym[(sym1, sym2)] = k_me_omega

                if k_me_sym:
                    k_me[(f1, f2)] = k_me_sym
        return k_me

    def _m_tens(self, states, cart_tens, tol: float):
        m_me = {}
        for f1 in states.f_list:
            for f2 in states.f_list:
                m_list1, m_list2, threej_u = cart_tens._threej_umat_spher_to_cart(
                    f1, f2, hyperfine=True
                )
                fac = np.sqrt((2 * f1 + 1) * (2 * f2 + 1))

                me = {
                    omega: val * fac
                    for omega, val in threej_u.items()
                    if np.any(np.abs(val) > tol)
                }

                if me:
                    m_me[(f1, f2)] = me
        return m_me

    def _k_me_omega(self, f1, f2, sym1, sym2, j_spin_list, omega, kmat):
        k_me = []
        for j1, spin1, j_sym1, spin_sym1, j_dim1 in j_spin_list[f1][sym1]:
            k_me_ = []
            for j2, spin2, j_sym2, spin_sym2, j_dim2 in j_spin_list[f2][sym2]:

                if (
                    spin1 != spin2
                    or (j1, j2) not in kmat
                    or (j_sym1, j_sym2) not in kmat[(j1, j2)]
                    or omega not in kmat[(j1, j2)][(j_sym1, j_sym2)]
                ):
                    k_me_.append(np.zeros((j_dim1, j_dim2)))
                    continue

                fac = j1 + spin2[-1] + f2 + j2
                assert float(
                    fac
                ).is_integer(), f"Non-integer power in (-1)**f: (-1)**{fac}"
                fac = int(fac)

                prefac = (
                    (-1) ** fac
                    * np.sqrt((2 * j1 + 1) * (2 * j2 + 1))
                    * py3nj.wigner6j(
                        j1 * 2,
                        int(f1 * 2),
                        int(spin2[-1] * 2),
                        int(f2 * 2),
                        j2 * 2,
                        int(omega * 2),
                        ignore_invalid=True,
                    )
                )

                me = kmat[(j1, j2)][(j_sym1, j_sym2)][omega] * prefac

                k_me_.append(me)

            k_me.append(k_me_)

        k_me = np.block(k_me)
        return k_me
