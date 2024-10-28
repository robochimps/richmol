from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from symmetry import C1, C2v
from symtop import wang_coefs


def _j_plus(j, k, c=1):
    return (j, k - 1, np.sqrt(j * (j + 1) - k * (k - 1)) * c if abs(k - 1) <= j else 0)


def _j_minus(j, k, c=1):
    return (j, k + 1, np.sqrt(j * (j + 1) - k * (k + 1)) * c if abs(k + 1) <= j else 0)


def _j_z(j, k, c=1):
    return (j, k, k * c)


def _j_square(j, k, c=1):
    return (j, k, j * (j + 1) * c)


def _delta(x, y):
    return 1 if x == y else 0


def _overlap(j1, k1, c1, j2, k2, c2):
    return c1 * c2 * _delta(j1, j2) * _delta(k1, k2)


def from_geometry(j: int, cart_coords: NDArray[np.float64]):
    pass


def from_abc(
    j: int, rot_a: float, rot_b: float, rot_c: float, symmetry: C1 = C1
) -> Tuple[
    str,
    NDArray[np.float64],
    NDArray[np.float64],
    List[Tuple[int, int]],
    List[Tuple[int, int]],
]:

    if abs(rot_b - rot_c) < (rot_a - rot_b):
        near_prolate = True
        abc_axes = "zyx"
    else:
        near_prolate = False  # near-oblate
        abc_axes = "xyz"

    k_list, ktau_list, wang_c = wang_coefs(j, linear=False, symmetry=symmetry)

    # matrix elements of operators J+^2, J_^2, Jz^2, and J^2

    j_plus = np.array([_j_plus(*_j_plus(j, k)) for k in k_list])
    j_plus_matelem = np.array(
        [[_overlap(j, k, 1, *elem) for elem in j_plus] for k in k_list]
    )

    j_minus = np.array([_j_minus(*_j_minus(j, k)) for k in k_list])
    j_minus_matelem = np.array(
        [[_overlap(j, k, 1, *elem) for elem in j_minus] for k in k_list]
    )

    j_z_square = np.array([_j_z(*_j_z(j, k)) for k in k_list])
    j_z_square_matelem = np.array(
        [[_overlap(j, k, 1, *elem) for elem in j_z_square] for k in k_list]
    )

    j_square = np.array([_j_square(j, k) for k in k_list])
    j_square_matelem = np.array(
        [[_overlap(j, k, 1, *elem) for elem in j_square] for k in k_list]
    )

    # Hamiltonian for near-prolate and near-oblate tops

    ham_near_prolate = (
        (j_plus_matelem + j_minus_matelem) * (rot_b - rot_c) / 4
        + j_z_square_matelem * (2 * rot_a - rot_b - rot_c) / 2
        + (rot_b + rot_c) / 2 * j_square_matelem
    )

    ham_near_oblate = (
        (j_plus_matelem + j_minus_matelem) * (rot_a - rot_b) / 4
        + j_z_square_matelem * (2 * rot_c - rot_a - rot_b) / 2
        + (rot_a + rot_b) / 2 * j_square_matelem
    )

    # transform to Wang basis

    ham_near_prolate = np.dot(np.conj(wang_c.T), np.dot(ham_near_prolate, wang_c))
    ham_near_oblate = np.dot(np.conj(wang_c.T), np.dot(ham_near_oblate, wang_c))

    # energies and wave functions

    enr_near_prolate, vec_near_prolate = np.linalg.eigh(ham_near_prolate)
    enr_near_oblate, vec_near_oblate = np.linalg.eigh(ham_near_oblate)

    # energies and assignments by k_a and k_c quantum numbers

    ka_kc_assignment = []
    k_tau_assignment = []
    for istate in range(len(k_list)):
        ind_a = np.argmax(vec_near_prolate[:, istate] ** 2)
        ind_c = np.argmax(vec_near_oblate[:, istate] ** 2)
        ka_kc_assignment.append((abs(k_list[ind_a]), abs(k_list[ind_c])))
        if near_prolate:
            k_tau_assignment.append(ktau_list[ind_a])
        else:
            k_tau_assignment.append(ktau_list[ind_c])

    if near_prolate:
        enr = enr_near_prolate
        vec = vec_near_prolate
    else:
        enr = enr_near_oblate
        vec = vec_near_oblate

    return abc_axes, enr, vec, ka_kc_assignment, k_tau_assignment


if __name__ == "__main__":

    # Camphor molecule, rotational constants (in MHz)
    # from supersonic expansion FTMW spectra, Kisiel, et al., PCCP 5, 820 (2003), https://doi.org/10.1039/B212029A
    rot_a, rot_b, rot_c = (1446.968977, 1183.367110, 1097.101031)

    max_j = 10
    for j in range(max_j + 1):
        abc_axes, enr, vec, ka_kc, k_tau = from_abc(j, rot_a, rot_b, rot_c, symmetry=C2v)
        print(j, k_tau)