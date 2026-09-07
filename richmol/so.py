import numpy as np

from .asymtop import ENERGY_UNITS, Energy_units, RotStates
from .cartens_ms import CartTensorMS
from .nucspin import SpinOperator, near_equal_coupling_with_rotations, Spin
import py3nj
from scipy.sparse import block_array, csr_array


class SpinOrbitSingletTriplet:
    f_list: list[float]
    f_sym_list: dict[float, list[str]]
    j_spin_list: dict[
        float, dict[str, list[tuple[int, tuple[float], str, str, int]]]
    ]  # j_spin_list[f][sym](J, spin, rovib_sym, spin_sym, rovib_dim)
    enr0: dict[float, dict[str, np.ndarray]]
    enr: dict[float, dict[str, np.ndarray]]
    vec: dict[float, dict[str, np.ndarray]]
    enr_units: Energy_units
    _rot_states_id: str

    dim_k: dict[float, dict[str, int]]  # dim_k[f][sym]
    dim_m: dict[float, int]  # dim_m[f]
    mk_ind: dict[float, dict[str, list[tuple[int, int]]]]

    # quanta_dict[j][sym][n] = (f, m, j, *spin, rovib_sym, spin_sym, *rot_qua, c),
    #   where n runs across dim_m[f] -> dim_k[f][sym]
    quanta_dict: dict[float, dict[str, list[tuple[float]]]]

    # quanta_dict_k[j][sym][n] = (f, j, *spin, rovib_sym, spin_sym, *rot_qua, c),
    #   where n runs across dim_k[f][sym]
    quanta_dict_k: dict[float, dict[str, list[tuple[float]]]]

    # quanta[n] = (f, m, j, *spin, rovib_sym, spin_sym, *rot_qua, c),
    #   where n runs across f -> sym -> dim_m[f] -> dim_k[f][sym]
    quanta: np.ndarray

    def __init__(
        self,
        min_f: float,
        max_f: float,
        singlet_states: RotStates,
        triplet_states: RotStates,
        soc_op: CartTensorMS,
    ):
        spin1 = 0
        spin2 = 1

        states1 = singlet_states
        states2 = triplet_states

        f_list = [
            round(f, 1) for f in np.linspace(min_f, max_f, int(max_f - min_f) + 1)
        ]

        j_list1 = {}
        j_list2 = {}

        for f in f_list:
            spin_list, j_list = near_equal_coupling_with_rotations(f, [Spin(spin1)])
            j_list1[f] = j_list

            spin_list, j_list = near_equal_coupling_with_rotations(f, [Spin(spin2)])
            j_list2[f] = j_list

        for f_val in f_list:
            h11 = np.diag(
                np.concatenate(
                    [
                        states1.enr[j][sym]
                        for j in j_list1[f_val]
                        for sym in states1.sym_list[j]
                    ]
                )
            )

            h22 = np.diag(
                np.concatenate(
                    [
                        states2.enr[j][sym]
                        for j in j_list2[f_val]
                        for sym in states2.sym_list[j]
                    ]
                )
            )

            mat = []

            for j1 in j_list1[f_val]:
                row = []

                for j2 in j_list2[f_val]:
                    prefac = (
                        (-1) ** (f_val + spin1 + 1)
                        * np.sqrt((2 * j1 + 1) * (2 * j2 + 1))
                        * py3nj.wigner6j(
                            int(f_val * 2),
                            spin1 * 2,
                            j1 * 2,
                            2,
                            j2 * 2,
                            spin2 * 2,
                            ignore_invalid=True,
                        )
                    )

                    dim1 = soc_op.dim_k1[j1]
                    dim2 = soc_op.dim_k2[j2]

                    submat = []

                    for sym1 in states1.sym_list[j1]:
                        subrow = []

                        for sym2 in states2.sym_list[j2]:
                            try:
                                kmat = soc_op.kmat[(j1, j2)][(sym1, sym2)][1]
                                subrow.append(kmat * prefac)
                            except KeyError:
                                subrow.append(csr_array((dim1[sym1], dim2[sym2])))

                        submat.append(subrow)

                    row.append(block_array(submat))

                mat.append(row)

            h12 = block_array(mat)
            print(f_val, h11.shape, h22.shape, h12.shape)
            h = block_array([[h11, h12], [h12.T, h22]])
