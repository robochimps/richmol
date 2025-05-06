"""Rotational matrix elements of laboratory-frame Cartesian tensor operators"""

from dataclasses import dataclass, field, fields

import numpy as np
from py3nj import wigner3j
from scipy.sparse import block_array, csr_array, kron

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
    rank: int
    cart_ind: list[str]
    spher_ind: list[tuple[int, int]]
    umat_cart_to_spher: dict[int, np.ndarray]
    umat_spher_to_cart: dict[int, np.ndarray]
    kmat: dict[
        tuple[int, int], dict[tuple[str, str], dict[int, csr_array]]
    ]  # [(j1, j2)][(sym1, sym2)][omega][{kv1, kv2}]
    mmat: dict[
        tuple[int, int], dict[str, dict[int, csr_array]]
    ]  # [(j1, j2)][A][omega][{m1, m2}]
    j_list: list[int]
    sym_list: dict[int, list[str]]
    dim_k: dict[int, dict[str, int]]
    dim_m: dict[int, int]

    def __init__(
        self,
        states: RotStates,
        mol_tens: np.ndarray,
        vib: bool = False,
        tol: float = 1e-12,
    ):
        if not isinstance(mol_tens, np.ndarray):
            mol_tens = np.array(mol_tens)

        if vib:  # first two dimensions are vibrational quanta
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

        else:
            if not all(dim == 3 for dim in mol_tens.shape):
                raise ValueError(
                    f"Since 'vib' = {vib}, all dimensions of 'mol_tens' must be size 3, "
                    + f"got shape = {mol_tens.shape}"
                )

            rank = mol_tens.ndim
            mol_tens = np.array(mol_tens)[np.newaxis, np.newaxis, ...]

        if rank in _TENSORS:
            tens = _TENSORS[rank]()
            for f in fields(tens):
                setattr(self, f.name, getattr(tens, f.name))
        else:
            raise ValueError(f"Cartesian tensor for rank = {rank} is not implemented")

        self.kmat = self._k_tens(states, mol_tens, tol)
        self.mmat = self._m_tens(states, tol)
        self.j_list = states.j_list
        self.sym_list = states.sym_list
        self.dim_m = {j: 2 * j + 1 for j in self.j_list}
        self.dim_k = {
            j: {sym: len(states.enr[j][sym]) for sym in self.sym_list[j]}
            for j in self.j_list
        }

    def mat(self, cart: str) -> csr_array:
        """Assembles full matrix corresponding to a specific Cartesian component
        of the tensor operator.

        Args:
            cart (str):
                A string indicating the Cartesian component of tensor for which
                the matrix should be assembled.
                For example, `cart` can be 'x', 'y', or 'z' for rank-1 tensor,
                'xx', 'xy', 'xz', 'yz', ..., 'zz' for rank-2 tensor.

        Returns:
            csr_array:
                A sparse matrix in CSR format representing the matrix elements
                of a specific Cartesian component of the tensor.
        """
        assert cart in self.cart_ind, (
            f"Invalid Cartesian component 'cart' = {cart} for tensor of rank = {self.rank}\n"
            + f"valid components: {self.cart_ind}"
        )
        me_j = {}
        for j1, j2 in list(set(self.mmat.keys()) & set(self.kmat.keys())):
            try:
                mmat = self.mmat[(j1, j2)][cart]
            except KeyError:
                continue

            kmat_j = self.kmat[(j1, j2)]

            me_sym = {}
            for (sym1, sym2), kmat in kmat_j.items():

                # M \otimes K
                me = sum(
                    kron(mmat[omega], kmat[omega])
                    for omega in list(set(mmat.keys()) & set(kmat.keys()))
                )

                if me.nnz > 0:
                    me_sym[(sym1, sym2)] = me

            if me_sym:
                me_j[(j1, j2)] = me_sym

        mat = self._dict_to_mat(me_j)
        return mat

    def mat_field(self, field: np.ndarray) -> csr_array:
        r"""Constructs the matrix resulting from contracting the Cartesian
        tensor operator with the given external field.

        For example:
            - \sum_{A=X, Y, Z} mu_A F_A, for dipole tensor (where F = `field`),
            - \sum_{A,B=X, Y, Z} \alpha_{A,B} F_A F_B, for polarizability tensor.

        Args:
            field (np.ndarray):
                A 1D array of shape (3,) representing the Cartesian components
                of the external electric field.

        Returns:
            csr_array:
                A sparse matrix in CSR format representing the field-contracted
                operator.
        """
        mf = self._mf_tens(field)

        me_j = {}

        for j1, j2 in list(set(mf.keys()) & set(self.kmat.keys())):
            mfmat = mf[(j1, j2)]
            kmat_j = self.kmat[(j1, j2)]

            me_sym = {}
            for (sym1, sym2), kmat in kmat_j.items():

                # M \otimes K
                me = sum(
                    kron(mfmat[omega], kmat[omega])
                    for omega in list(set(mfmat.keys()) & set(kmat.keys()))
                )

                if me.nnz > 0:
                    me_sym[(sym1, sym2)] = me

            if me_sym:
                me_j[(j1, j2)] = me_sym

        mat = self._dict_to_mat(me_j)
        return mat

    def mat_vec(self, field: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Computes the action of a Cartesian tensor operator on a vector,
        contracted with a given external field.

        Args:
            field (np.ndarray):
                A 1D array of shape (3,) representing the Cartesian components
                of the external field.

            vec (np.ndarray):
                The input vector to which the tensor operator is applied.

        Returns:
            np.ndarray:
                The resulting vector after contraction with the field.
        """
        vec_dict = self._vec_to_dict(vec)

        # multiply M-tensor with field
        mf = self._mf_tens(field)

        # build (M \otimes K) \cdot vec

        vec2 = {}

        for j1, j2 in list(set(mf) & set(self.kmat.keys())):
            mfmat = mf[(j1, j2)]
            kmat_j = self.kmat[(j1, j2)]
            dim_m1 = self.dim_m[j1]
            dim_m2 = self.dim_m[j2]

            if j1 not in vec2.keys():
                vec2[j1] = {}

            for (sym1, sym2), kmat in kmat_j.items():
                dim_k1 = self.dim_k[j1][sym1]
                dim_k2 = self.dim_k[j2][sym2]
                dim1 = dim_m1 * dim_k1
                vec_tr = np.transpose(vec_dict[j2][sym2].reshape(dim_m2, dim_k2))

                res = []
                for omega in list(set(mfmat.keys()) & set(kmat.keys())):
                    kv = kmat[omega].dot(vec_tr)
                    mkv = mfmat[omega].dot(kv.T)
                    res.append(mkv.reshape(dim1))

                try:
                    vec2[j1][sym1] += sum(res)
                except KeyError:
                    vec2[j1][sym1] = sum(res)

        vec2 = self._dict_to_vec(vec2)
        return vec2

    def _mf_tens(self, field: np.ndarray):
        """Multiplies the M-tensor matrix elements with the input field
        and sums over all Cartesian components.
        """
        field_arr = np.array(field)
        assert field.shape == (3,), (
            f"Invalid 'field': expected 3 Cartesian components (shape = (3,)), "
            f"but got shape = {field_arr.shape}"
        )

        field_tens = {
            elem: np.prod(field_arr[["xyz".index(x) for x in elem]])
            for elem in self.cart_ind
        }

        mf = {}
        for (j1, j2), m_j in self.mmat.items():
            mf_o = {}
            for cart, m_cart in m_j.items():
                for omega, m in m_cart.items():
                    try:
                        mf_o[omega] += m * field_tens[cart]
                    except KeyError:
                        mf_o[omega] = m * field_tens[cart]

            if mf_o:
                mf[(j1, j2)] = mf_o
        return mf

    def _m_tens(self, states, tol: float):
        r"""Computes M-tensor matrix elements:

        M_{A,\omega}^{(J',m',J,m)} = \sqrt{(2J'+1)(2J+1)} (-1)^{m'}
            \sum_{\sigma=-\omega}^{\omega}
            threej(J, \omega, J', m, \sigma, -m')
            [U^{(\Omega)}]^{-1}_{A,\omega\sigma}

        Here:
            - \omega = 0..\Omega (rank of tensor)
            - A = X, Y, Z for \Omega=1, XX, XY, XZ, YX, YY, ... ZZ for \Omega=2, etc.
            - [U^{(\Omega)}]^{-1} is spherical-to-Cartesian tensor transformation matrix
        """
        m_me = {}
        for j1 in states.j_list:
            for j2 in states.j_list:
                fac = np.sqrt(2 * j1 + 1) * np.sqrt(2 * j2 + 1)

                m_list1, m_list2, threej_u = self._threej_umat_spher_to_cart(j1, j2)

                me_cart = {}
                for icart, cart in enumerate(self.cart_ind):

                    me_o = {}
                    for omega in threej_u.keys():
                        me = threej_u[omega][:, :, icart] * fac
                        me[np.abs(me) < tol] = 0
                        me = csr_array(me)

                        if me.nnz > 0:
                            me_o[omega] = me

                    if me_o:
                        me_cart[cart] = me_o

                if me_cart:
                    m_me[(j1, j2)] = me_cart
        return m_me

    def _k_tens(self, states, mol_tens, tol: float):
        r"""Computes K-tensor matrix elements

        K_{\omega}^{(J',l',J,l)} = \sum_{k',v'} \sum_{k,v}
            [c_{k',v'}^{(l')}]^* c_{k,v}^{(l)} (-1)^{k'}
            \sum_{\alpha} \sum_{\sigma=-\omega}^{\omega}
            threej(J, \omega, J', k, \sigma, -k')
            U_{\omega\sigma,\alpha}^{(\Omega)}
            \langle v' |T_{\alpha}|v\rangle

        Here:
            - c_{k,v}^{(l)} are expansion coefficients of rovibrational wavefunctions
            - \omega = 0..\Omega (rank of tensor)
            - \alpha = x, y, z for \Omega=1, xx, xy, xz, yx, yy, ... zz for \Omega=2
            - U is Cartesian-to-spherical tensor transformation matrix
            - T_{\alpha} are elements of tensor in molecular frame
        """
        # flatten molecular-frame tensor matrix elements such that the order
        # of Cartesian indices matches that in `self.cart_ind`
        cart_ind = [["xyz".index(x) for x in elem] for elem in self.cart_ind]
        if self.rank == 1:
            mol_tens = np.moveaxis(
                np.array([mol_tens[..., i] for (i,) in cart_ind]), 0, -1
            )
        elif self.rank == 2:
            mol_tens = np.moveaxis(
                np.array([mol_tens[..., i, j] for (i, j) in cart_ind]), 0, -1
            )
        else:
            raise NotImplementedError(
                f"Flattening molecular-frame tensor of rank = {self.rank} is not implemented"
            )

        k_me = {}

        for j1 in states.j_list:
            for j2 in states.j_list:
                jktau_list1, jktau_list2, threej_u = self._threej_umat_cart_to_spher(
                    j1, j2, states.linear
                )

                k_me_sym = {}

                for sym1 in states.sym_list[j1]:
                    vec1 = states.vec[j1][sym1]
                    v_ind1 = states.v_ind[j1][sym1]
                    r_ind1 = states.r_ind[j1][sym1]

                    for sym2 in states.sym_list[j2]:
                        vec2 = states.vec[j2][sym2]
                        v_ind2 = states.v_ind[j2][sym2]
                        r_ind2 = states.r_ind[j2][sym2]

                        vib_me = mol_tens[np.ix_(v_ind1, v_ind2)]

                        k_me_omega = {}

                        for omega in threej_u.keys():
                            me = np.einsum(
                                "ijc,ijc->ij",
                                vib_me,
                                threej_u[omega][np.ix_(r_ind1, r_ind2)],
                                optimize="optimal",
                            )

                            if np.any(np.abs(me) > tol):
                                me = np.einsum(
                                    "ik,ij,jl->kl",
                                    np.conj(vec1),
                                    me,
                                    vec2,
                                    optimize="optimal",
                                )

                                me[np.abs(me) < tol] = 0
                                me = csr_array(me)
                                if me.nnz > 0:
                                    k_me_omega[omega] = me

                        if k_me_omega:
                            k_me_sym[(sym1, sym2)] = k_me_omega

                if k_me_sym:
                    k_me[(j1, j2)] = k_me_sym
        return k_me

    def _threej_umat_cart_to_spher(self, j1: int, j2: int, linear: bool):
        r"""Computes three-j symbol contracted with tensor's Cartesian-to-spherical
        transformation

        (-1)^{k'} \sum_{\sigma=-\omega}^{\omega}
            threej(J, \omega, J', k, \sigma, -k')
            U_{\omega\sigma,\alpha}^{(\Omega)},

        and transforms the results to Wang's symmetrized representation
        of |J,k,\tau\rangle functions.

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

    def _threej_umat_spher_to_cart(
        self, j1: int | float, j2: int | float, hyperfine: bool = False
    ):
        r"""Computes three-j symbol contracted with tensor's spherical-to-Cartesian
        transformation

        (-1)^{m'} \sum_{\sigma=-\omega}^{\omega}
            threej(J, \omega, J', m, \sigma, -m')
            [U^{(\Omega)}]^{-1}_{A,\omega\sigma}

        Here, |j1, m'=-j1..j1> are bra states, and |j2, m=-j2..j2> are ket states.
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
        two_m12_1 = m12_1 * 2
        two_m12_2 = m12_2 * 2

        threej = {
            omega: np.zeros((2 * omega + 1, len(m1), len(m2)), dtype=np.complex128)
            for omega in self.umat_spher_to_cart.keys()
        }
        for omega, sigma in self.spher_ind:
            if hyperfine:
                fac = np.abs(m12_1 - j2 - omega)
            else:
                fac = np.abs(m12_1)
            assert np.all(
                np.isclose(fac % 1, 0)
            ), f"Non-integer power in (-1)**f: (-1)**{fac}"
            fac = fac.astype(int)

            thrj = (-1) ** fac * wigner3j(
                [int(j2 * 2)] * n,
                [omega * 2] * n,
                [int(j1 * 2)] * n,
                two_m12_2.astype(int),
                [sigma * 2] * n,
                -two_m12_1.astype(int),
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

    def _vec_to_dict(self, vec: np.ndarray):
        """Converts vector vec[...] to vec[j][sym][:]"""
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
        """Converts vector vec[j][sym][:] to vec[...]"""
        blocks = []
        for j in self.j_list:
            for sym in self.sym_list[j]:
                blocks.append(vec_dict[j][sym])
        return np.concatenate(blocks)

    def _dict_to_mat(self, mat_dict) -> csr_array:
        """Converts matrix mat[(j1, j2)][(sym1, sym2)][:, :] to mat[...]"""
        mat = block_array(
            [
                [
                    (
                        mat_dict[(j1, j2)][(sym1, sym2)]
                        if (j1, j2) in mat_dict.keys()
                        and (sym1, sym2) in mat_dict[(j1, j2)].keys()
                        else csr_array(
                            np.zeros(
                                (
                                    self.dim_m[j1] * self.dim_k[j1][sym1],
                                    self.dim_m[j2] * self.dim_k[j2][sym2],
                                )
                            )
                        )
                    )
                    for j2 in self.j_list
                    for sym2 in self.sym_list[j2]
                ]
                for j1 in self.j_list
                for sym1 in self.sym_list[j1]
            ]
        )
        return mat
