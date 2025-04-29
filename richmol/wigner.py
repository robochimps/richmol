"""Wigner D-functions"""

import numpy as np
from scipy.sparse import diags
from numpy.typing import NDArray


def _jy_eig(j: float):
    assert j >= 0, f"j < 0: {j} < 0"
    x = lambda m: np.sqrt((j + m) * (j - m + 1))
    d = lambda m, n: 1 if m == n else 0
    ld = np.array([x(n) for n in np.linspace(-j, j, int(2 * j + 1))])
    ud = np.array([x(-n) for n in np.linspace(-j, j, int(2 * j + 1))])
    jmat = diags([ld[1:], np.zeros(2 * j + 1), ud[:-1]], [-1, 0, 1]).toarray() * 0.5j
    e, v = np.linalg.eigh(jmat)
    return v


def wigner_d(j: float, beta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Computes the Wigner d-function, d_{m,k}^{j}(β).

    Args:
        j (float): The total angular momentum quantum number. It must be a non-negative integer
            or half-integer value.
        beta (numpy.ndarray): A 1D array containing a grid of β angles in radians.

    Returns:
        (numpy.ndarray): A 3D numpy array containing the computed values of the Wigner d-function
            for the specified angles β. The dimensions of the array are as follows:
            - The first dimension corresponds to the quantum number m, ranging from -j to j.
            - The second dimension corresponds to the quantum number k, ranging from -j to j.
            - The third dimension corresponds to the index of the grid point in the input array `beta`.
    """
    v = _jy_eig(j)
    e = np.exp(-1j * np.linspace(-j, j, int(2 * j + 1))[None, :] * beta[:, None])
    res = np.einsum("gi,mi,ki->mkg", e, np.conj(v), v, optimize="optimal")
    return res


def wigner_D(
    j: float,
    grid: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Computes the Wigner D-function, D_{m,k}^{j}(α, β, γ).

    Args:
        j (float): The total angular momentum quantum number. It must be a non-negative integer
            or half-integer value.
        grid (numpy.ndarray): A 2D array representing a grid of Euler angles (α, β, γ),
            where each row corresponds to a set of angles with indices 0, 1, and 2 denoting
            α, β, and γ, respectively.

    Returns:
        (numpy.ndarray): A 3D numpy array containing the computed values of the Wigner D-function
            for the specified Euler angles. The dimensions of the array are as follows:
            - The first dimension corresponds to the quantum number m, ranging from -j to j.
            - The second dimension corresponds to the quantum number k, ranging from -j to j.
            - The third dimension corresponds to the index of the grid point in the input array `grid`.
    """
    alpha, beta, gamma = grid.T
    d = wigner_d(j, beta)
    m = np.linspace(-j, j, int(2 * j + 1))
    k = np.linspace(-j, j, int(2 * j + 1))
    em = np.exp(1j * alpha[None, None, :] * m[:, None, None])
    ek = np.exp(1j * gamma[None, None, :] * k[None, :, None])
    return em * ek * d