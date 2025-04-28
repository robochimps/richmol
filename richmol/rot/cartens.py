"""Rotational matrix elements of laboratory-frame Cartesian tensor operators"""

from collections import defaultdict
from dataclasses import dataclass, fields, field

import numpy as np
from py3nj import wigner3j

from .asymtop import RotStates
from .symtop import wang_coefs


_TENSORS: dict[int, type] = {}


def register(cls):
    rank = cls.rank
    if rank in _TENSORS:
        raise ValueError(
            f"Class {_TENSORS[rank].__name__} and {cls.__name__} have the same rank: {cls.rank}"
        )
    _TENSORS[rank] = cls
    return cls


@dataclass
@register
class Rank1Tensor:
    rank: int = 1
    cart_ind: list[str] = field(default_factory=lambda: ["x", "y", "z"])
    spher_ind: list[tuple[int, int]] = field(
        default_factory=lambda: [(1, -1), (1, 0), (1, 1)]
    )
    # Cartesian-to-spherical tensor transformation
    # rows: spherical indices {1: [(1,-1), (1,0), (1,1)]}
    # columns: Cartesian indices x, y, z
    umat_cart_to_spher: dict[int, np.ndarray] = field(
        default_factory=lambda: {
            1: np.array(
                [
                    [np.sqrt(2) / 2, -np.sqrt(2) * 1j / 2, 0],
                    [0, 0, 1],
                    [-np.sqrt(2) / 2, -np.sqrt(2) * 1j / 2, 0],
                ],
                dtype=np.complex128,
            ),
        }
    )
    umat_spher_to_cart: dict[int, np.ndarray] = field(init=False)

    def __post_init__(self):
        omega = sorted(list(self.umat_cart_to_spher.keys()))
        umat_spher_to_cart = np.linalg.pinv(
            np.concatenate([self.umat_cart_to_spher[o] for o in omega], axis=0)
        )
        # spherical-to-Cartesian tensor transformation
        # rows: Cartesian indices x, y, z
        # columns: spherical indices {1: [(1,-1), (1,0), (1,1)]}
        self.umat_spher_to_cart = {}
        for om in omega:
            sigma_ind = [
                self.spher_ind.index((o, s)) for (o, s) in self.spher_ind if o == om
            ]
            self.umat_spher_to_cart[om] = umat_spher_to_cart[:, sigma_ind]


@dataclass
@register
class Rank2Tensor:
    rank: int = 2
    cart_ind: list[str] = field(
        default_factory=lambda: ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]
    )
    spher_ind: list[tuple[int, int]] = field(
        default_factory=lambda: [(o, s) for o in range(3) for s in range(-o, o + 1)]
    )
    # Cartesian-to-spherical tensor transformation
    # rows: spherical indices {0: [(0,0)], 1: [(1,-1), (1,0), (1,1)], 2: [(2,-2), (2,-1), ..., (2,2)]}
    # columns: Cartesian indices [xx, xy, xz, yz, yy, ..., zz]
    umat_cart_to_spher: dict[int, np.ndarray] = field(
        default_factory=lambda: {
            0: np.array(
                [[-1 / np.sqrt(3), 0, 0, 0, -1 / np.sqrt(3), 0, 0, 0, -1 / np.sqrt(3)]]
            ),
            1: np.array(
                [
                    [0, 0, -0.5, 0, 0, 0.5 * 1j, 0.5, -0.5 * 1j, 0],
                    [
                        0,
                        1.0 / np.sqrt(2) * 1j,
                        0,
                        -1.0 / np.sqrt(2) * 1j,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
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
    )
    umat_spher_to_cart: dict[int, np.ndarray] = field(init=False)

    def __post_init__(self):
        omega = sorted(list(self.umat_cart_to_spher.keys()))
        umat_spher_to_cart = np.linalg.pinv(
            np.concatenate([self.umat_cart_to_spher[o] for o in omega], axis=0)
        )
        # spherical-to-Cartesian tensor transformation
        # rows: Cartesian indices [xx, xy, xz, yz, yy, ..., zz]
        # columns: spherical indices {0: [(0,0)], 1: [(1,-1), (1,0), (1,1)], 2: [(2,-2), (2,-1), ..., (2,2)]}
        self.umat_spher_to_cart = {}
        for om in omega:
            sigma_ind = [
                self.spher_ind.index((o, s)) for (o, s) in self.spher_ind if o == om
            ]
            self.umat_spher_to_cart[om] = umat_spher_to_cart[:, sigma_ind]


class CartTensor:

    def __init__(self, states: RotStates, mol_tens: np.ndarray, vib: bool = False):
        if not isinstance(mol_tens, np.ndarray):
            mol_tens = np.array(mol_tens)
        if vib:
            if mol_tens.ndim < 3:
                raise ValueError(
                    f"Since 'vib' = {vib}, tensor 'mol_tens' must have at least 3 dimensions "
                    + f"(vib basis, vib basis, 3 ...), got shape = {mol_tens.shape}"
                )
            if not all(dim == 3 for dim in mol_tens.shape[2:]):
                raise ValueError(
                    f"Since 'vib' = {vib}, all dimensions of 'mol_tens' starting from the third "
                    + f"must be size 3, got shape = {mol_tens.shape}"
                )
            rank = mol_tens.ndim - 2
            self.mol_tens = mol_tens
        else:
            if not all(dim == 3 for dim in mol_tens.shape):
                raise ValueError(
                    f"Since 'vib' = {vib}, all dimensions of 'mol_tens' must be size 3, "
                    + f"got shape = {mol_tens.shape}"
                )
            rank = mol_tens.ndim
            self.mol_tens = np.array(mol_tens)[np.newaxis, np.newaxis, ...]

        if rank in _TENSORS:
            tens = _TENSORS[rank]()
            for f in fields(tens):
                setattr(self, f.name, getattr(tens, f.name))
        else:
            raise ValueError(f"Cartesian tensor for rank = {rank} is not implemented")

        self.kmat = self.k_tens(states)
        self.mmat = self.m_tens(states)

    def m_tens(self, states):
        r"""Computes M-tensor matrix elements:"""
        m_me = {}
        for j1 in states.j_list:
            for j2 in states.j_list:
                fac = np.sqrt(2 * j1 + 1) * np.sqrt(2 * j2 + 1)
                m_list1, m_list2, rot_me = self._threej_umat_spher_to_cart(j1, j2)
                m_me[(j1, j2)] = np.moveaxis(
                    fac * np.array([rot_me[omega] for omega in rot_me.keys()]), 0, -1
                )
        return m_me

    def k_tens(self, states):
        r"""Computes K-tensor matrix elements:

        K_{\omega}^{(J',l',J,l)} = \sum_{k',v'} \sum_{k,v} [c_{k',v'}^{(l')}]^* c_{k,v}^{(l)} (-1)^{k'}
            \sum_{\alpha} \sum_{\sigma=-\omega}^{\omega}
            threej(J, \omega, J', k, \sigma, -k')
            U_{\omega\sigma,\alpha}^{(\Omega)}
            \langle v' |T_{\alpha}|v\rangle

        Here,
        - c_{k,v}^{(l)} are expansion coefficients of rovibrational wavefunctions,
        - \omega = 0..rank of tensor,
        - \alpha = x, y, z for rank=1, xx, xy, xz, yx, yy, ... zz for rank=2,
        - U is Cartesian-to-spherical tensor transformation matrix,
        - T_{\alpha} are elements of tensor in molecular frame.
        """
        # flatten molecular-frame tensor matrix elements such that the order
        # of Cartesian indices matches that in `self.cart_ind`
        cart_ind = [["xyz".index(x) for x in elem] for elem in self.cart_ind]
        if self.rank == 1:
            mol_tens = np.moveaxis(
                np.array([self.mol_tens[..., i] for (i,) in cart_ind]), 0, -1
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
                jktau_list1, jktau_list2, rot_me = self._threej_umat_cart_to_spher(
                    j1, j2, states.linear
                )

                for sym1 in states.sym_list[j1]:
                    vec1 = states.vec[j1][sym1]
                    v_ind1 = states.v_ind[j1][sym1]
                    r_ind1 = states.r_ind[j1][sym1]

                    for sym2 in states.sym_list[j2]:
                        vec2 = states.vec[j2][sym2]
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
                                    np.conj(vec1),
                                    me_,
                                    vec2,
                                    optimize="optimal",
                                )
                            )
                        k_me[(j1, j2)][(sym1, sym2)] = np.moveaxis(np.array(me), 0, -1)
        return k_me

    def _threej_umat_cart_to_spher(self, j1: int, j2: int, linear: bool):
        r"""Computes three-j symbol contracted with tensor's Cartesian-to-spherical transformation,

        (-1)^{k'} \sum_{\sigma=-\omega}^{\omega}
            threej(J, \omega, J', k, \sigma, -k')
            U_{\omega\sigma,\alpha}^{(\Omega)},

        and transforms the results to Wang's symmetrized representation of |J,k,\tau\rangle functions.
        Here, |j1, k'=-j1..j1\rangle are bra states, and |j2, k=-j2..j2\rangle are ket states.
        """
        k_list1, jktau_list1, wang_coefs1 = wang_coefs(j1, linear)
        k_list2, jktau_list2, wang_coefs2 = wang_coefs(j2, linear)

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

    def _threej_umat_spher_to_cart(self, j1: int, j2: int):
        r"""Computes three-j symbol contracted with tensor's spherical-to-Cartesian transformation,

        (-1)^{m'} \sum_{\sigma=-\omega}^{\omega}
            threej(J, \omega, J', m, \sigma, -m')
            [U^{(\Omega)}]^{-1}_{A,\omega\sigma}

        Here, |j1, m'=-j1..j1\rangle are bra states, and |j2, m=-j2..j2\rangle are ket states.
        """
        m_list1 = np.arange(-j1, j1 + 1)
        m_list2 = np.arange(-j2, j2 + 1)
        m1 = np.array(m_list1)
        m2 = np.array(m_list2)
        m12 = np.concatenate(
            (
                m1[:, None, None].repeat(len(m2), axis=1),
                m2[None, :, None].repeat(len(m1), axis=0),
            ),
            axis=-1,
        ).reshape(-1, 2)
        n = len(m12)
        m12_1 = m12[:, 0]
        m12_2 = m12[:, 1]

        threej = {
            omega: np.zeros((2 * omega + 1, len(m1), len(m2)), dtype=np.complex128)
            for omega in self.umat_spher_to_cart.keys()
        }
        for omega, sigma in self.spher_ind:
            thrj = (-1) ** np.abs(m12_1) * wigner3j(
                [j2 * 2] * n,
                [omega * 2] * n,
                [j1 * 2] * n,
                m12_2 * 2,
                [sigma * 2] * n,
                -m12_1 * 2,
                ignore_invalid=True,
            )
            threej[omega][sigma + omega] = thrj.reshape(len(m1), len(m2))

        threej_u = {}
        for omega in self.umat_spher_to_cart.keys():
            threej_u[omega] = np.einsum(
                "sij,cs->ijc",
                threej[omega],
                self.umat_spher_to_cart[omega],
                optimize="optimal",
            )
        return m_list1, m_list2, threej_u
