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


def quadrupole_me(
    f_val: float,
    states: RotStates,
    quad_op: list[QuadMom],
    efg_op: list[CartTensor],
) -> tuple[list[tuple[float]], list[int], np.ndarray, np.ndarray]:
    """Compute matrix elements of the nuclear quadrupole Hamiltonian
    for a given total angular momentum F.

    Args:
        f_val (float):
            Quantum number of the total angular momentum F = I + J,
            where I is the total nuclear spin and J is the rotational angular momentum.

        states (RotStates):
            Molecular rotational or rovibrational states of the molecule.

        quad_op (list[QuadMom]):
            List of nuclear quadruple moment operators.
            Each operator must have its quadrupole moment specified in millibarns (mb).

        efg_op (list[CartTensor]):
            List of electric field gradient tensors corresponding to each nucleus with
            quadrupoles listed in `quad_op`. Each tensor must be expressed in atomic units.

    Returns:
        tuple:
            - spin_list (list[tuple[float]]):
                List of coupled nuclear spin quantum numbers for each basis state.
                Each tuple represents sequential coupling:
                    (I_{1}, I_{12}, ..., I_{1N}),
                where:
                    - I_{1} is the spin of operator 1,
                    - I_{12} is the total spin of operators 1 and 2,
                    - I_{1N} is the total spin of operators 1 through N.

            - j_list (list[int]):
                List of corresponding rotational quantum numbers J
                coupled with the final total spin to achieve F.

            - h0 (np.ndarray):
                Diagonal matrix of pure rovibrational energies (in cm⁻¹),
                without quadrupole interaction.

            - h (np.ndarray):
                Matrix of quadrupole Hamiltonian matrix elements (in cm⁻¹),
                without pure rovibrational part.
    """
    for i, efg in enumerate(efg_op):
        if not efg.spher_ind == Rank2Tensor().spher_ind:
            raise ValueError(
                f"{i}th element of 'efg_op' has invalid 'spher_ind': {efg.spher_ind}, "
                + f"expected: {Rank2Tensor.spher_ind}"
            ) from None

    for i, quad in enumerate(quad_op):
        if quad.name != "quad":
            raise ValueError(
                f"{i}th element of 'quad_op' has invalid 'name': {quad.name}, "
                + f"expected: {QuadMom.name}"
            )

    assert len(quad_op) == len(efg_op), (
        "'quad_op' and 'efg_op' must have the same number of elements, "
        + f"found len(quad_op) = {len(quad_op)}, len(efg_op) = {len(efg_op)}"
    )

    # coupling combinations of J and I for given F

    spin_list, j_list = near_equal_coupling_with_rotations(f_val, quad_op)

    # check if all necessary J are spanned in input states and EFG tensors
    missing = [j for j in j_list if j not in states.j_list]
    if missing:
        raise ValueError(
            f"'states' is missing required J quantum numbers at 'f_val' = {f_val}\n"
            f"expected J values: {sorted(set(j_list))}\n"
            f"present in 'states': {sorted(states.j_list)}\n"
            f"missing values: {sorted(set(missing))}"
        )
    for i, efg in enumerate(efg_op):
        missing = [j for j in j_list if j not in efg.j_list]
        if missing:
            raise ValueError(
                f"{i}th element of 'efg_op' is missing required J quantum numbers at 'f_val' = {f_val}\n"
                f"expected J values: {sorted(set(j_list))}\n"
                f"present in tensor: {sorted(efg.j_list)}\n"
                f"missing values: {sorted(set(missing))}"
            )

    # quantum numbers

    # spin_j_k_list = [
    #     (spin, j, k)
    #     for j, spin in zip(j_list, spin_list)
    #     for sym in states.sym_list[j]
    #     for k in range(len(states.enr[j][sym]))
    # ]

    # <I' || Q(i) || I>
    quad_me = reduced_me(spin_list, spin_list, quad_op)

    # build quadrupole interaction Hamiltonian

    h = []
    for j1, spin1 in zip(j_list, spin_list):
        h_ = []
        for j2, spin2 in zip(j_list, spin_list):

            sym_list1 = states.sym_list[j1]
            sym_list2 = states.sym_list[j2]
            dim1_sym = {sym: len(states.enr[j1][sym]) for sym in sym_list1}
            dim2_sym = {sym: len(states.enr[j2][sym]) for sym in sym_list2}
            dim1 = sum(dim1_sym.values())
            dim2 = sum(dim2_sym.values())

            try:
                quad = quad_me[(spin1, spin2)]
            except KeyError:
                h_.append(np.zeros((dim1, dim2)))
                continue

            me = 0
            for efg, q in zip(efg_op, quad):
                try:
                    kmat = efg.kmat[(j1, j2)]
                except KeyError:
                    continue
                me += (
                    np.block(
                        [
                            [
                                (
                                    kmat[(sym1, sym2)][2]
                                    if (sym1, sym2) in kmat
                                    else np.zeros((dim1_sym[sym1], dim2_sym[sym2]))
                                )
                                for sym2 in sym_list2
                            ]
                            for sym1 in sym_list1
                        ]
                    )
                    * q
                )

            if isinstance(me, int) and me == 0:
                h_.append(np.zeros((dim1, dim2)))
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

    # field-free part

    h0 = np.diag(
        np.concatenate(
            [states.enr[j][sym] for j in j_list for sym in states.sym_list[j]]
        )
    )

    return spin_list, j_list, h0, h
