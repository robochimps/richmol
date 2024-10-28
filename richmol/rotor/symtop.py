import numpy as np
from symmetry import C1


def wang_coefs(j: int, linear: bool, symmetry: C1 = C1):

    def _wang_coefs(j, k, tau):
        sigma = np.fmod(k, 3) * tau
        fac1 = pow(-1.0, sigma) / np.sqrt(2.0)
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

    k_list = [k for k in range(-j, j + 1)]
    if linear:
        k = 0
        t = j % 2
        sym = symmetry.irrep_from_k_tau(k, t)
        ktau_list = [(j, k, t, sym)]
    else:
        ktau_list = []
        for k in range(0, j + 1):
            if k == 0:
                tau = [j % 2]
            else:
                tau = [0, 1]
            for t in tau:
                sym = symmetry.irrep_from_k_tau(k, t)
                ktau_list.append((k, t, sym))

    coefs = np.zeros((len(k_list), len(ktau_list)), dtype=np.complex128)
    for i, (k, tau, sym) in enumerate(ktau_list):
        c, k_pair = _wang_coefs(j, k, tau)
        for kk, cc in zip(k_pair, c):
            i_k = k_list.index(kk)
            coefs[i_k, i] = cc

    return k_list, ktau_list, coefs