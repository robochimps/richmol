import numpy as np
import py3nj
from scipy.sparse import block_array, csr_array

from .asymtop import RotStates
from .cartens_ms import CartTensorMS
from .nucspin import Spin, near_equal_coupling_with_rotations


class SpinOrbitSingletTriplet:
    @classmethod
    def hmat(
        cls,
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
            float(round(f, 1))
            for f in np.linspace(min_f, max_f, int(max_f - min_f) + 1)
        ]

        j_list1 = {}
        j_list2 = {}

        h11 = {}
        h22 = {}
        h12 = {}
        quanta1 = {}
        quanta2 = {}

        for f in f_list:
            # generate combinations of J and S=0 for F=J+S
            # for single spin, spin_list = [(S,)]
            spin_list, j_list = near_equal_coupling_with_rotations(f, [Spin(spin1)])
            j_list1[f] = j_list

            # generate combinations of J and S=1 for F=J+S
            # for single spin, spin_list = [(S,)]
            spin_list, j_list = near_equal_coupling_with_rotations(f, [Spin(spin2)])
            j_list2[f] = j_list

        for f_val in f_list:

            # singlet-state block: diagonal, given by rovibrational energies
            h11[f_val] = np.diag(
                np.concatenate(
                    [
                        states1.enr[j][sym]
                        for j in j_list1[f_val]
                        for sym in states1.sym_list[j]
                    ]
                )
            )

            # triplet-state block: diagonal, given by rovibrational energies
            h22[f_val] = np.diag(
                np.concatenate(
                    [
                        states2.enr[j][sym]
                        for j in j_list2[f_val]
                        for sym in states2.sym_list[j]
                    ]
                )
            )

            # singlet-triplet block
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

            h12[f_val] = block_array(mat)

            # assign quantum numbers
            quanta1[f_val] = [
                (f_val, j, spin1, sym, float(e), k, tau, float(c))
                for j in j_list1[f_val]
                for sym in states1.sym_list[j]
                for e, (j, k, tau, c) in zip(
                    states1.enr[j][sym], states1.quanta_dict_k[j][sym]
                )
            ]

            quanta2[f_val] = [
                (f_val, j, spin2, sym, float(e), k, tau, float(c))
                for j in j_list2[f_val]
                for sym in states2.sym_list[j]
                for e, (j, k, tau, c) in zip(
                    states2.enr[j][sym], states2.quanta_dict_k[j][sym]
                )
            ]

        return h11, h22, h12, quanta1, quanta2
