import numpy as np
import py3nj
from scipy import constants
from scipy.sparse import block_array, csr_array, diags

from .asymtop import RotStates
from .cartens import CartTensor, Rank2Tensor, MKTensor
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

    # assert sym_list == ["A"], (
    #     "The placeholder spin-rotation symmetrization function works only for C1 spatial symmetry group,\n"
    #     + f"i.e., no symmetry, instead got spatial symmetries: {sym_list}"
    # )

    j_spin_list = {}
    for j, spin in zip(j_list, spin_list):
        for j_sym in j_sym_list[j]:
            ##########
            # f_sym = ... # code symmetry spin-spatial symmetry products
            f_sym = "A"  # j_sym
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
    j_spin_list: dict[
        float, dict[str, list[tuple[int, tuple[float], str, str, int]]]
    ]  # j_spin_list[f][sym](J, spin, rovib_sym, spin_sym, rovib_dim)
    enr0: dict[float, dict[str, np.ndarray]]
    enr: dict[float, dict[str, np.ndarray]]
    vec: dict[float, dict[str, np.ndarray]]

    dim_k: dict[float, dict[str, int]]  # dim_k[f][sym]
    dim_m: dict[float, int]  # dim_m[f]
    mk_ind: dict[float, dict[str, list[tuple[int, int]]]]

    # quanta_dict[j][sym][n] = (f, m, j, *spin, rovib_sym, spin_sym, *rot_qua, c),
    #   where n runs across dim_m[f] -> dim_k[f][sym]
    quanta_dict: dict[float, dict[str, list[tuple[float]]]]

    # quanta_dict[j][sym][n] = (f, j, *spin, rovib_sym, spin_sym, *rot_qua, c),
    #   where n runs across dim_k[f][sym]
    quanta_dict_k: dict[float, dict[str, list[tuple[float]]]]

    # quanta[n] = (f, m, j, *spin, rovib_sym, spin_sym, *rot_qua, c),
    #   where n runs across f -> sym -> dim_m[f] -> dim_k[f][sym]
    quanta: np.ndarray

    def __init__(
        self,
        min_f: float,
        max_f: float,
        states: RotStates,
        spin_op: list[SpinOperator],
        efg_op: list[CartTensor | None] = [],
        symmetry_rules = _symmetry_spin_rotation_placeholder,
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
            round(f, 1) for f in np.linspace(min_f, max_f, int(max_f - min_f) + 1)
        ]
        print(f"List of F quanta: {self.f_list}")

        self.j_spin_list = {}
        self.f_sym_list = {}

        for f in self.f_list:
            spin_list, j_list = near_equal_coupling_with_rotations(f, spin_op)
            j_sym_list = states.sym_list
            j_spin_list = symmetry_rules(
                j_list, j_sym_list, spin_list
            )
            self.f_sym_list[f] = list(j_spin_list.keys())

            # add dimension of the rovibrational basis to the list
            # i.e., (J, spin, rovib_sym, spin_sym) -> (J, spin, rovib_sym, spin_sym, rovib_dim)
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
                    h = (
                        h
                        + _quadrupole_me(
                            f,
                            self.j_spin_list[f][sym],
                            [spin_op[i] for i in quad_op_ind],
                            [efg_op[i] for i in quad_op_ind],
                        ).toarray()
                    )

                # diagonalization

                enr[sym], vec[sym] = np.linalg.eigh(h)

            self.enr0[f] = enr0
            self.enr[f] = enr
            self.vec[f] = vec

        # add dimensions and state assignment

        self.dim_m = {f: int(2 * f) + 1 for f in self.f_list}
        self.dim_k = {
            f: {sym: len(self.enr[f][sym]) for sym in self.f_sym_list[f]}
            for f in self.f_list
        }

        self.mk_ind = {
            f: {
                sym: [
                    (im, ik)
                    for im in range(self.dim_m[f])
                    for ik in range(self.dim_k[f][sym])
                ]
                for sym in self.f_sym_list[f]
            }
            for f in self.f_list
        }

        self.quanta_dict = {}
        self.quanta_dict_k = {}
        for f in self.f_list:
            quanta_sym = {}
            quanta_sym_k = {}
            for sym in self.f_sym_list[f]:
                e = self.enr[f][sym]
                v = self.vec[f][sym]

                qua = [
                    (f, j, *spin, j_sym, spin_sym, *rot_qua)
                    for (j, spin, j_sym, spin_sym, *_) in self.j_spin_list[f][sym]
                    for rot_qua in states.quanta_dict_k[j][j_sym]
                ]

                if len(qua) != len(e):
                    raise ValueError(
                        f"Number of elements in 'qua' = {len(qua)} does not match the number"
                        f"of hyperfine states = {len(e)} for F = {f} and symmetry = {sym}"
                    )

                qua_k = []
                for i in range(len(e)):
                    ind = np.argmax(v[:, i] ** 2)
                    qua_k.append((*qua[ind], v[ind, i]))

                qua_mk = [
                    (q[0], m, *q[1:]) for m in np.arange(-f, f + 1) for q in qua_k
                ]

                quanta_sym[sym] = qua_mk
                quanta_sym_k[sym] = qua_k
            self.quanta_dict[f] = quanta_sym
            self.quanta_dict_k[f] = quanta_sym_k
        self.quanta = self._dict_to_vec(self.quanta_dict)

    def mat(self):
        e0 = []
        for f in self.f_list:
            for sym in self.f_sym_list[f]:
                for m in np.arange(-f, f + 1):
                    e0.append(self.enr[f][sym])
        return csr_array(diags(np.concatenate(e0)))

    def _vec_to_dict(self, vec: np.ndarray):
        """Converts vector vec[n] to vec[f][sym][k] where n runs across f -> sym -> k"""
        vec_dict = {}
        offset = 0
        for f in self.f_list:
            vec_dict[f] = {}
            for sym in self.f_sym_list[f]:
                d = self.dim_k[f][sym] * self.dim_m[f]
                vec_dict[f][sym] = vec[offset : offset + d]
                offset += d
        return vec_dict

    def _dict_to_vec(self, vec_dict) -> np.ndarray:
        """Converts vector vec[f][sym][k] to vec[n] where n runs across f -> sym -> k"""
        blocks = []
        for f in self.f_list:
            for sym in self.f_sym_list[f]:
                blocks.append(vec_dict[f][sym])
        return np.concatenate(blocks)


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

            zero_block = csr_array(np.zeros((j_dim1, j_dim2)))

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

    h = block_array(h)
    return h


class HyperCartTensor(MKTensor):
    def __init__(
        self, states: HyperStates, cart_tens: CartTensor, thresh: float = 1e-12
    ):
        self.rank = cart_tens.rank
        self.cart_ind = cart_tens.cart_ind
        self.spher_ind = cart_tens.spher_ind
        self.umat_cart_to_spher = cart_tens.umat_cart_to_spher
        self.umat_spher_to_cart = cart_tens.umat_spher_to_cart

        self.kmat = self._k_tens(states, cart_tens, thresh)
        self.mmat = self._m_tens(states, thresh)
        self.j_list = states.f_list
        self.sym_list = states.f_sym_list
        self.dim_m = {f: int(2 * f) + 1 for f in self.j_list}
        self.dim_k = {
            f: {sym: len(states.enr[f][sym]) for sym in self.sym_list[f]}
            for f in self.j_list
        }

    def _k_tens(self, states, cart_tens, thresh: float):
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
                            ).toarray()

                            if np.any(np.abs(me) > thresh):
                                me = np.einsum(
                                    "ik,ij,jl->kl",
                                    np.conj(v1),
                                    me,
                                    v2,
                                    optimize="optimal",
                                )

                                me[np.abs(me) < thresh] = 0
                                me = csr_array(me)
                                if me.nnz > 0:
                                    k_me_omega[omega] = me

                        if k_me_omega:
                            k_me_sym[(sym1, sym2)] = k_me_omega

                if k_me_sym:
                    k_me[(f1, f2)] = k_me_sym

        return k_me

    def _m_tens(self, states, thresh: float):
        m_me = {}
        for f1 in states.f_list:
            for f2 in states.f_list:
                m_list1, m_list2, threej_u = self._threej_umat_spher_to_cart(
                    f1, f2, hyperfine=True
                )

                fac = np.sqrt((2 * f1 + 1) * (2 * f2 + 1))

                me_cart = {}
                for icart, cart in enumerate(self.cart_ind):

                    me_o = {}
                    for omega in threej_u.keys():
                        me = threej_u[omega][:, :, icart] * fac
                        me[np.abs(me) < thresh] = 0
                        me = csr_array(me)

                        if me.nnz > 0:
                            me_o[omega] = me

                    if me_o:
                        me_cart[cart] = me_o

                if me_cart:
                    m_me[(f1, f2)] = me_cart

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

        k_me = block_array(k_me)

        return k_me
