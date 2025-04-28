from .asymtop import RotStates
from .cartens import CartTensor
import numpy as np

if __name__ == "__main__":
    j = 4

    xyz = (
        "bohr",
        "O",
        0.00000000,
        0.00000000,
        0.12395915,
        "H",
        0.00000000,
        -1.43102686,
        -0.98366080,
        "H",
        0.00000000,
        1.43102686,
        -0.98366080,
    )

    states = RotStates.from_geometry(j, xyz)

    # molecular-frame dipole moment in au
    dip_mol = [0, 0, -0.7288]

    # molecular-frame polarizability in au
    pol_mol = [[9.1369, 0, 0], [0, 9.8701, 0], [0, 0, 9.4486]]

    dip_lab = CartTensor(states, dip_mol)
    pol_lab = CartTensor(states, pol_mol)

    # j = (3, 1)
    # s = ("A", "A")
    # me = pol_lab.kmat[j][s]
    # for i in range(me.shape[0]):
    #     for j in range(me.shape[1]):
    #         if np.abs(me[i, j, 2]) < 1e-10:
    #             continue
    #         print(i, j, np.round(np.real(me[i, j]), 8), np.round(np.imag(me[i, j]), 8))

    for (j1, j2), mmat in dip_lab.mmat.items():
        print(
            j1,
            j2,
            mmat.shape,
            np.sum(np.linalg.norm(mmat, axis=(2, 3)) ** 2),
            (2 * j1 + 1) * (2 * j2 + 1) if abs(j1 - j2) <= 1 else 0,
        )

    for (j1, j2), kmat in dip_lab.kmat.items():
        for (sym1, sym2), kmat_s in kmat.items():
            print(j1, j2, sym1, sym2, kmat_s.shape)