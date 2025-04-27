"""Rotational matrix elements of laboratory-frame
Cartesian tensor operators
"""

from collections import defaultdict

import numpy as np
from py3nj import wigner3j
from symmetry import Symmetry
from symtop import wang_coefs


class CartTensor:
    rank: int
    cart_ind: list[str]
    spher_ind: list[tuple[int, int]]
    umat_cart_to_spher: dict[int, np.ndarray]
    umat_spher_to_cart: dict[int, np.ndarray]
    mol_tens: np.ndarray
    linear: bool
    sym: Symmetry

    def k_tens(self, states):
        """Computes K-tensor matrix elements:

        (-1)**k' sum_{alpha} sum_{sigma=-omega^omega}
            threej(J, omega, J', k, sigma, -k')
            × U_{omega,sigma,alpha}^{(omega)}
            × T_{alpha},

        """
        # flatten molecular-frame tensor matrix elements such that the order
        # of Cartesian indices matches that in `self.cart_ind`
        cart_ind = [["xyz".index(x) for x in elem] for elem in self.cart_ind]
        if self.rank == 1:
            mol_tens = np.moveaxis(
                np.array([self.mol_tens[..., i] for i in cart_ind]), 0, -1
            )
        elif self.rank == 2:
            mol_tens = np.moveaxis(
                np.array([self.mol_tens[..., i, j] for (i, j) in cart_ind]), 0, -1
            )
        else:
            raise NotImplementedError(
                f"Flattening molecular-frame tensor of rank = {self.rank} is not implemented"
            )

        nested_dict = lambda: defaultdict(nested_dict)
        k_me = nested_dict()

        for j1 in states.j_list:
            for j2 in states.j_list:
                jktau_list1, jktau_list2, rot_me = self._threej_umat(j1, j2)

                for sym1 in states.sym_list[j1]:
                    coefs1 = states.coefs[j1][sym1]
                    v_ind1 = states.v_ind[j1][sym1]
                    r_ind1 = states.r_ind[j1][sym1]

                    for sym2 in states.sym_list[j2]:
                        coefs2 = states.coefs[j2][sym2]
                        v_ind2 = states.v_ind[j2][sym2]
                        r_ind2 = states.r_ind[j2][sym2]

                        vib_me = mol_tens[np.ix_(v_ind1, v_ind2)]
                        me = []
                        for omega in rot_me.keys():
                            me_ = np.einsum(
                                "ijc,ijc->ij",
                                vib_me,
                                rot_me[omega][np.ix_(r_ind1, r_ind2)],
                                optimize="optimal",
                            )
                            me.append(
                                np.einsum(
                                    "ik,ij,jl->kl",
                                    np.conj(coefs1),
                                    me_,
                                    coefs2,
                                    optimize="optimal",
                                )
                            )
                        k_me[(j1, j2)][(sym1, sym2)] = np.moveaxis(np.array(me), 0, -1)

    def _threej_umat(self, j1: int, j2: int):
        """Computes three-j symbol contracted with tensor's Cartesian-to-spherical transformation,

        (-1)**k' sum_{sigma=-omega^omega}
            threej(J, omega, J', k, sigma, -k')
            × U_{omega,sigma,alpha}^{(omega)},

        and transforms the results to the Wang's symmetrized representation of |J,k,tau> functions.

        Here, J'=`j1`, J=`j2`, and omega=`self.rank`
        """
        k_list1, jktau_list1, wang_coefs1 = wang_coefs(j1, self.linear, self.sym)
        k_list2, jktau_list2, wang_coefs2 = wang_coefs(j2, self.linear, self.sym)

        k1 = np.array(k_list1)
        k2 = np.array(k_list2)
        k12 = np.concatenate(
            (
                k1[:, None, None].repeat(len(k2), axis=1),
                k2[None, :, None].repeat(len(k1), axis=0),
            ),
            axis=-1,
        ).reshape(-1, 2)
        n = len(k12)
        k12_1 = k12[:, 0]
        k12_2 = k12[:, 1]

        threej = {
            omega: np.zeros((2 * omega + 1, len(k1), len(k2)), dtype=np.complex128)
            for omega in self.umat_cart_to_spher.keys()
        }
        for omega, sigma in self.spher_ind:
            thrj = (-1) ** np.abs(k12_1) * wigner3j(
                [j2 * 2] * n,
                [omega * 2] * n,
                [j1 * 2] * n,
                k12_2 * 2,
                [sigma * 2] * n,
                -k12_1 * 2,
                ignore_invalid=True,
            )
            threej[omega][sigma + omega] = thrj.reshape(len(k1), len(k2))

        threej_wang = {}
        for omega in self.umat_cart_to_spher.keys():
            threej_wang[omega] = np.einsum(
                "ki,skl,lj,sc->ijc",
                np.conj(wang_coefs1),
                threej[omega],
                wang_coefs2,
                self.umat_cart_to_spher[omega],
                optimize="optimal",
            )
        return jktau_list1, jktau_list2, threej_wang


class Rank1Tensor(CartTensor):
    rank = 1
    cart_ind = ["x", "y", "z"]
    spher_ind = [(1, -1), (1, 0), (1, 1)]
    umat_cart_to_spher = {
        1: np.array(
            [
                [np.sqrt(2) / 2, -np.sqrt(2) * 1j / 2, 0],
                [0, 0, 1],
                [-np.sqrt(2) / 2, -np.sqrt(2) * 1j / 2, 0],
            ],
            dtype=np.complex128,
        ),
    }
    umat_spher_to_cart = {
        key: np.linalg.pinv(val) for key, val in umat_cart_to_spher.items()
    }

    def __init__(self, mol_tens: list[float] | np.ndarray, linear: bool, sym: Symmetry):
        self.linear = linear
        self.sym = sym
        self.mol_tens = np.array(mol_tens)[np.newaxis, np.newaxis, ...]


class Rank2Tensor(CartTensor):
    rank = 2
    cart_ind = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]
    spher_ind = [(o, s) for o in range(3) for s in range(-o, o + 1)]
    umat_cart_to_spher = {
        0: np.array(
            [[-1 / np.sqrt(3), 0, 0, 0, -1 / np.sqrt(3), 0, 0, 0, -1 / np.sqrt(3)]]
        ),
        1: np.array(
            [
                [0, 0, -0.5, 0, 0, 0.5 * 1j, 0.5, -0.5 * 1j, 0],
                [0, 1.0 / np.sqrt(2) * 1j, 0, -1.0 / np.sqrt(2) * 1j, 0, 0, 0, 0, 0],
                [0, 0, -0.5, 0, 0, -0.5 * 1j, 0.5, 0.5 * 1j, 0],
            ],
            dtype=np.complex128,
        ),
        2: np.array(
            [
                [0.5, -0.5 * 1j, 0, -0.5 * 1j, -0.5, 0, 0, 0, 0],
                [0, 0, 0.5, 0, 0, -0.5 * 1j, 0.5, -0.5 * 1j, 0],
                [
                    -1 / np.sqrt(6),
                    0,
                    0,
                    0,
                    -1 / np.sqrt(6),
                    0,
                    0,
                    0,
                    (1 / 3) * np.sqrt(6),
                ],
                [0, 0, -0.5, 0, 0, -0.5 * 1j, -0.5, -0.5 * 1j, 0],
                [0.5, 0.5 * 1j, 0, 0.5 * 1j, -0.5, 0, 0, 0, 0],
            ],
            dtype=np.complex128,
        ),
    }
    umat_spher_to_cart = {
        key: np.linalg.pinv(val) for key, val in umat_cart_to_spher.items()
    }

    def __init__(
        self, mol_tens: list[list[float]] | np.ndarray, linear: bool, sym: Symmetry
    ):
        self.linear = linear
        self.sym = sym
        self.mol_tens = np.array(mol_tens)[np.newaxis, np.newaxis, ...]
