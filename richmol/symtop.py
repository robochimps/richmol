from collections import defaultdict
from collections.abc import Iterable
from typing import List, Literal, Tuple, cast

import numpy as np
from jax import config
from jax import numpy as jnp

from .wigner import _jy_eig, wigner_D

config.update("jax_enable_x64", True)


def wang_coefs(j: int, linear: bool):
    k_list = [k for k in range(-j, j + 1)]
    if linear:
        k = 0
        t = cast(Literal[0, 1], j % 2)
        jktau_list = [(j, k, t)]
    else:
        jktau_list = []
        for k in range(0, j + 1):
            if k == 0:
                tau = [cast(Literal[0, 1], j % 2)]
            else:
                tau = [0, 1]
            for t in tau:
                jktau_list.append((j, k, t))

    coefs = np.zeros((len(k_list), len(jktau_list)), dtype=np.complex128)
    for i, (j, k, tau) in enumerate(jktau_list):
        c, k_pair = wang_coefs_jktau(j, k, tau)
        for kk, cc in zip(k_pair, c):
            i_k = k_list.index(kk)
            coefs[i_k, i] = cc
    return k_list, jktau_list, coefs


def wang_coefs_jktau(j: int, k: int, tau: Literal[0, 1]):
    sigma = jnp.fmod(k, 3) * tau
    fac1 = pow(-1.0, sigma) / jnp.sqrt(2.0)
    fac2 = fac1 * pow(-1.0, (j + k))
    kval = [k, -k]
    if tau == 0:
        if k == 0:
            coefs = [1.0]
        elif k > 0:
            coefs = [fac1, fac2]
    elif tau == 1:
        if k == 0:
            coefs = [1j]
        elif k > 0:
            coefs = [fac1 * 1j, -fac2 * 1j]
    return coefs, kval


def _jplus(j, k, c=1):
    return (j, k - 1, jnp.sqrt(j * (j + 1) - k * (k - 1)) * c if abs(k - 1) <= j else 0)


def _jminus(j, k, c=1):
    return (j, k + 1, jnp.sqrt(j * (j + 1) - k * (k + 1)) * c if abs(k + 1) <= j else 0)


def _jx(j, k, c=1):
    return _sum_oper(
        [
            _jminus(j, k, c),
            _jplus(j, k, c),
        ],
        [0.5, 0.5],
    )


def _jy(j, k, c=1):
    return _sum_oper(
        [
            _jminus(j, k, c),
            _jplus(j, k, c),
        ],
        [0.5j, -0.5j],
    )


def _jz(j, k, c=1):
    return (j, k, k * c)


def _jj(j, k, c=1):
    return (j, k, j * (j + 1) * c)


def _jminus_jminus(j, k, c=1):
    return _jminus(*_jminus(j, k, c))


def _jminus_jplus(j, k, c=1):
    return _jminus(*_jplus(j, k, c))


def _jplus_jminus(j, k, c=1):
    return _jplus(*_jminus(j, k, c))


def _jplus_jplus(j, k, c=1):
    return _jplus(*_jplus(j, k, c))


def _jminus_jz(j, k, c=1):
    return _jminus(*_jz(j, k, c))


def _jplus_jz(j, k, c=1):
    return _jplus(*_jz(j, k, c))


def _jz_jminus(j, k, c=1):
    return _jz(*_jminus(j, k, c))


def _jz_jplus(j, k, c=1):
    return _jz(*_jplus(j, k, c))


def _sum_oper(oper: List[Tuple[int, int, float]], prefac: List[complex]):
    res = defaultdict(complex)
    for (j, k, c), fac in zip(oper, prefac):
        res[(j, k)] += c * fac
    return [(j, k, c) for (j, k), c in res.items()]


def _jx_jx(j, k, c=1):
    return _sum_oper(
        [
            _jminus_jminus(j, k, c),
            _jminus_jplus(j, k, c),
            _jplus_jminus(j, k, c),
            _jplus_jplus(j, k, c),
        ],
        [0.25, 0.25, 0.25, 0.25],
    )


def _jx_jy(j, k, c=1):
    return _sum_oper(
        [
            _jminus_jminus(j, k, c),
            _jminus_jplus(j, k, c),
            _jplus_jminus(j, k, c),
            _jplus_jplus(j, k, c),
        ],
        [0.25j, -0.25j, 0.25j, -0.25j],
    )


def _jx_jz(j, k, c=1):
    return _sum_oper(
        [
            _jminus_jz(j, k, c),
            _jplus_jz(j, k, c),
        ],
        [0.5, 0.5],
    )


def _jy_jx(j, k, c=1):
    return _sum_oper(
        [
            _jminus_jminus(j, k, c),
            _jminus_jplus(j, k, c),
            _jplus_jminus(j, k, c),
            _jplus_jplus(j, k, c),
        ],
        [0.25j, 0.25j, -0.25j, -0.25j],
    )


def _jy_jy(j, k, c=1):
    return _sum_oper(
        [
            _jminus_jminus(j, k, c),
            _jminus_jplus(j, k, c),
            _jplus_jminus(j, k, c),
            _jplus_jplus(j, k, c),
        ],
        [-0.25, 0.25, 0.25, -0.25],
    )


def _jy_jz(j, k, c=1):
    return _sum_oper(
        [
            _jminus_jz(j, k, c),
            _jplus_jz(j, k, c),
        ],
        [0.5j, -0.5j],
    )


def _jz_jx(j, k, c=1):
    return _sum_oper(
        [
            _jz_jminus(j, k, c),
            _jz_jplus(j, k, c),
        ],
        [0.5, 0.5],
    )


def _jz_jy(j, k, c=1):
    return _sum_oper(
        [
            _jz_jminus(j, k, c),
            _jz_jplus(j, k, c),
        ],
        [0.5j, -0.5j],
    )


def _jz_jz(j, k, c=1):
    return _jz(*_jz(j, k, c))


def _j4(j, k, c=1):
    return _jj(*_jj(j, k, c))


def _j6(j, k, c=1):
    return _jj(*_j4(j, k, c))


def _jz4(j, k, c=1):
    return _jz_jz(*_jz_jz(j, k, c))


def _jz6(j, k, c=1):
    return _jz_jz(*_jz4(j, k, c))


def _j2_jz2(j, k, c=1):
    return _jj(*_jz_jz(j, k, c))


def _j4_jz2(j, k, c=1):
    return _jj(*_j2_jz2(j, k, c))


def _j2_jz4(j, k, c=1):
    return _jj(*_jz4(j, k, c))


def _jplus4(j, k, c=1):
    return _jplus_jplus(*_jplus_jplus(j, k, c))


def _jplus6(j, k, c=1):
    return _jplus_jplus(*_jplus4(j, k, c))


def _jminus4(j, k, c=1):
    return _jminus_jminus(*_jminus_jminus(j, k, c))


def _jminus6(j, k, c=1):
    return _jminus_jminus(*_jminus4(j, k, c))


def _delta(x, y):
    return 1 if x == y else 0


def _overlap(jkc1, jkc2):
    if all(isinstance(elem, Iterable) for elem in jkc1):
        jkc1_ = jkc1
    else:
        jkc1_ = [jkc1]
    if all(isinstance(elem, Iterable) for elem in jkc2):
        jkc2_ = jkc2
    else:
        jkc2_ = [jkc2]
    return jnp.sum(
        jnp.array(
            [
                jnp.conj(c1) * c2 * _delta(j1, j2) * _delta(k1, k2)
                for (j1, k1, c1) in jkc1_
                for (j2, k2, c2) in jkc2_
            ]
        )
    )


def rotme_ovlp(j: int, linear: bool):
    k_list, jktau_list, coefs = wang_coefs(j, linear)
    s = jnp.array(
        [[_overlap((j, k1, 1), (j, k2, 1)) for k2 in k_list] for k1 in k_list]
    )
    res = jnp.einsum("ki,kl,lj->ij", jnp.conj(coefs), s, coefs)
    max_imag = jnp.max(jnp.abs(jnp.imag(res)))
    assert (
        max_imag < 1e-12
    ), f"<J',k',tau'|J,k,tau> matrix elements are not real-valued, max imaginary component: {max_imag}"
    return jnp.real(res), k_list, jktau_list


def rotme_rot(j: int, linear: bool):
    k_list, jktau_list, coefs = wang_coefs(j, linear)
    jxx = jnp.array(
        [[_overlap((j, k1, 1), _jx_jx(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jxy = jnp.array(
        [[_overlap((j, k1, 1), _jx_jy(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jxz = jnp.array(
        [[_overlap((j, k1, 1), _jx_jz(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jyx = jnp.array(
        [[_overlap((j, k1, 1), _jy_jx(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jyy = jnp.array(
        [[_overlap((j, k1, 1), _jy_jy(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jyz = jnp.array(
        [[_overlap((j, k1, 1), _jy_jz(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jzx = jnp.array(
        [[_overlap((j, k1, 1), _jz_jx(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jzy = jnp.array(
        [[_overlap((j, k1, 1), _jz_jy(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jzz = jnp.array(
        [[_overlap((j, k1, 1), _jz_jz(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jmat = jnp.array([[jxx, jxy, jxz], [jyx, jyy, jyz], [jzx, jzy, jzz]])
    res = jnp.einsum("ki,abkl,lj->abij", jnp.conj(coefs), jmat, coefs)
    max_imag = jnp.max(jnp.abs(jnp.imag(res)))
    assert (
        max_imag < 1e-12
    ), f"<J',k',tau'|Ja*Jb|J,k,tau> matrix elements are not real-valued, max imaginary component: {max_imag}"
    return jnp.real(res), k_list, jktau_list


def rotme_rot_diag(j: int, linear: bool):
    k_list, jktau_list, coefs = wang_coefs(j, linear)
    jxx = jnp.array(
        [[_overlap((j, k1, 1), _jx_jx(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jyy = jnp.array(
        [[_overlap((j, k1, 1), _jy_jy(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jzz = jnp.array(
        [[_overlap((j, k1, 1), _jz_jz(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jmat = jnp.array([jxx, jyy, jzz])
    res = jnp.einsum("ki,akl,lj->aij", jnp.conj(coefs), jmat, coefs)
    max_imag = jnp.max(jnp.abs(jnp.imag(res)))
    assert (
        max_imag < 1e-12
    ), f"<J',k',tau'|Ja*Ja|J,k,tau> matrix elements are not real-valued, max imaginary component: {max_imag}"
    return jnp.real(res), k_list, jktau_list


def rotme_cor(j: int, linear: bool):
    k_list, jktau_list, coefs = wang_coefs(j, linear)
    jx = jnp.array(
        [[_overlap((j, k1, 1), _jx(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jy = jnp.array(
        [[_overlap((j, k1, 1), _jy(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jz = jnp.array(
        [[_overlap((j, k1, 1), _jz(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jvec = jnp.array([jx, jy, jz])
    res = 1j * jnp.einsum("ki,akl,lj->aij", jnp.conj(coefs), jvec, coefs)
    max_imag = jnp.max(jnp.abs(jnp.imag(res)))
    assert (
        max_imag < 1e-12
    ), f"i*<J',k',tau'|Ja|J,k,tau> matrix elements are not real-valued, max imaginary component: {max_imag}"
    return jnp.real(res), k_list, jktau_list


def rotme_watson(
    j: int,
    rot_a: float | None,
    rot_b: float,
    rot_c: float | None,
    rot_const: dict[str, float],
    form: Literal["S", "A"],
):
    r"""Watson-type asymmetric top Hamiltonian in S or A standard reduced form
    (J. K. G. Watson in "Vibrational Spectra and Structure" (Ed: J. Durig) Vol 6 p 1, Elsevier, Amsterdam, 1977).

    S-reduced form:

        :math:`H = H_{rr} - \Delta_{J} * J^{4} - \Delta_{JK} * J^{2} * J_{z}^{2} - \Delta_{K} * J_{z}^{4}`
        :math:`+ d_{1} * J^{2} * (J_{+}^{2} + J_{-}^{2}) + d_{2} * (J_{+}^{4} + J_{-}^{4})`
        :math:`+ H_{J} * J^{6} + H_{JK} * J^{4} * J_{z}^{2} + H_{KJ} * J^{2} * J_{z}^{4} + H_{K} * J_{z}^{6}`
        :math:`+ h_{1} * J^{4} * (J_{+}^{2} + J_{-}^{2}) + h_{2} * J^{2} * (J_{+}^{4} + J_{-}^{4})`
        :math:`+ h_{3} * (J_{+}^{6} + J_{-}^{6})`

    A-reduced form:

        :math:`H = H_{rr} - \Delta_{J} * J^{4} - \Delta_{JK} * J^{2} * J_{z}^{2} - \Delta_{K} * J_{z}^{4}`
        :math:`-\\frac{1}{2} * [ \delta_J_{1} * J^{2} + \delta_{k} * J_{z}^{2}, J_{+}^{2} + J_{-}^{2} ]_{+}`
        :math:`+ H_{J} * J^{6} + H_{JK} * J^{4} * J_{z}^{2} + H_{KJ} * J^{2} * J_{z}^{4} + H_{K} * J_{z}^{6}`
        :math:`+ \\frac{1}{2} * [ \phi_{J} * J^{4} + \phi_{JK} * J^{2} * J_{z}^{2} + \phi_{K} * J_{z}^{4}, J_{+}^{2} + J_{-}^{2} ]_{+}`
    """
    if (rot_a is None) != (rot_c is None):
        raise ValueError(
            "Parameters 'rot_a' and 'rot_c' must both be provided "
            + "or both left as None (for linear molecule).",
        )

    linear = False
    if rot_a is None or rot_b is None:
        linear = True

    k_list, jktau_list, coefs = wang_coefs(j, linear)
    j2 = jnp.array(
        [[_overlap((j, k1, 1), _jj(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    j4 = jnp.array(
        [[_overlap((j, k1, 1), _j4(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    j6 = jnp.array(
        [[_overlap((j, k1, 1), _j6(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jz2 = jnp.array(
        [[_overlap((j, k1, 1), _jz_jz(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jz4 = jnp.array(
        [[_overlap((j, k1, 1), _jz4(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jz6 = jnp.array(
        [[_overlap((j, k1, 1), _jz6(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    j2jz2 = jnp.array(
        [[_overlap((j, k1, 1), _j2_jz2(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    j4jz2 = jnp.array(
        [[_overlap((j, k1, 1), _j4_jz2(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    j2jz4 = jnp.array(
        [[_overlap((j, k1, 1), _j2_jz4(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jp2 = jnp.array(
        [[_overlap((j, k1, 1), _jplus_jplus(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jm2 = jnp.array(
        [
            [_overlap((j, k1, 1), _jminus_jminus(j, k2)) for k2 in k_list]
            for k1 in k_list
        ]
    )
    jp4 = jnp.array(
        [[_overlap((j, k1, 1), _jplus4(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jm4 = jnp.array(
        [[_overlap((j, k1, 1), _jminus4(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jp6 = jnp.array(
        [[_overlap((j, k1, 1), _jplus6(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jm6 = jnp.array(
        [[_overlap((j, k1, 1), _jminus6(j, k2)) for k2 in k_list] for k1 in k_list]
    )

    # rigid-rotor Hamiltonian

    if rot_a is None or rot_c is None:
        # linear molecule
        ham = rot_b * j2
    elif abs(rot_c - rot_b) < abs(rot_a - rot_b):
        # near-prolate top
        print("Axes convention for near-prolate: I(r), xyz = bca")
        print(
            "Remark: manually apply permutation (132) to other input "
            + "Cartesian tensors (like dipole moment) in the PAS frame,\n"
            + "   e.g., using permutation matrix np.array([[0,1,0],[0,0,1],[1,0,0]])"
        )
        ham = (
            (jp2 + jm2) * (rot_b - rot_c) / 4
            + jz2 * (2 * rot_a - rot_b - rot_c) / 2
            + (rot_b + rot_c) / 2 * j2
        )
    else:
        # near-oblate top
        ham = (
            (jp2 + jm2) * (rot_a - rot_b) / 4
            + jz2 * (2 * rot_c - rot_a - rot_b) / 2
            + (rot_a + rot_b) / 2 * j2
        )
        print("Axes convention for near-oblate: III(r), xyz = abc")

    # effective Hamiltonian

    expr = {}

    if form == "S":
        expr["DeltaJ"] = -j4
        expr["DeltaJK"] = -j2jz2
        expr["DeltaK"] = -jz4
        expr["d1"] = j2 * (jp2 + jm2)
        expr["d2"] = jp4 + jm4
        expr["HJ"] = j6
        expr["HJK"] = j4jz2
        expr["HKJ"] = j2jz4
        expr["HK"] = jz6
        expr["h1"] = j4 * (jp2 + jm2)
        expr["h2"] = j2 * (jp4 + jm4)
        expr["h3"] = jp6 + jm6

    elif form == "A":
        expr["DeltaJ"] = -j4
        expr["DeltaJK"] = -j2jz2
        expr["DeltaK"] = -jz4
        expr["deltaJ"] = -0.5 * (j2 * (jp2 + jm2) + jp2 * j2 + jm2 * j2)
        expr["deltaK"] = -0.5 * (jz2 * (jp2 + jm2) + jp2 * jz2 + jm2 * jz2)
        expr["HJ"] = j6
        expr["HJK"] = j4jz2
        expr["HKJ"] = j2jz4
        expr["HK"] = jz6
        expr["phiJ"] = 0.5 * (j4 * (jp2 + jm2) + jp2 * j4 + jm2 * j4)
        expr["phiJK"] = 0.5 * (j2jz2 * (jp2 + jm2) + jp2 * j2jz2 + jm2 * j2jz2)
        expr["phiK"] = 0.5 * (jz4 * (jp2 + jm2) + jp2 * jz4 + jm2 * jz4)

    else:
        raise ValueError(
            f"Unknown form of Watson reduced Hamiltonian = '{form}' (available 'S' or 'A')"
        )

    for name, const in rot_const.items():
        if name.upper() in ("A", "B", "C"):
            continue
        # print(f"add Watson term '{name}' = {const}")
        try:
            ham = ham + const * expr[name]
        except KeyError:
            raise ValueError(
                f"Uknown keys in rotational constants input 'rot_const': {name}.\n"
                f"Valid keys: {list(expr.keys())}"
            )

    # transform from |j,k> to Wang |j,|k|,tau> basis
    res = jnp.einsum("ki,kl,lj->ij", jnp.conj(coefs), ham, coefs)

    max_imag = jnp.max(jnp.abs(jnp.imag(res)))
    assert (
        max_imag < 1e-8
    ), f"<J',k',tau'|Watson|J,k,tau> matrix elements are not real-valued, max imaginary component: {max_imag}"

    return jnp.real(res), k_list, jktau_list


def symtop_on_grid(j: int, grid, linear: bool):
    k_list, jktau_list, coefs = wang_coefs(j, linear)
    # psi[J+m, J+k, ipoint] for k,m = -J..J
    psi = np.sqrt((2 * j + 1) / (8 * np.pi**2)) * np.conj(wigner_D(j, grid))
    ind = [k for k in range(-j, j + 1)]
    map_ind = [ind.index(k) for k in k_list]
    res = jnp.einsum("ki,mkg->mig", coefs, psi[:, map_ind, :])
    return res, k_list, jktau_list


def symtop_on_grid_split_angles(j: int, alpha, beta, gamma, linear: bool):
    k_list, jktau_list, coefs = wang_coefs(j, linear)
    ind = list(range(-j, j + 1))
    wang_map = [ind.index(k) for k in k_list]
    ind = np.array(ind)

    v = _jy_eig(j)
    em = np.exp(1j * alpha[None, :] * ind[:, None])
    ek = np.exp(1j * gamma[None, :] * ind[:, None])

    rot_k = np.einsum(
        "kt,kl,kg->tlg", coefs, v[wang_map], ek[wang_map], optimize="optimal"
    )  # (ktau, l, gamma)

    rot_m = np.sqrt((2 * j + 1) / (8 * np.pi**2)) * np.einsum(
        "ml,mg->mlg", np.conj(v), em, optimize="optimal"
    )  # (m, l, alpha)

    rot_l = np.exp(-1j * beta[None, :] * ind[:, None])  # (l, beta)
    return rot_k, rot_m, rot_l, k_list, jktau_list
