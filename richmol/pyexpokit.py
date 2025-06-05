import numpy as np

from richmol import expokit


class ExpokitError(Exception):
    pass


def check_exit_flag(iflag):
    if iflag < 0:
        raise ExpokitError("Bad input arguments")
    elif iflag > 0:
        raise ExpokitError(
            {
                1: "Maximum number of steps reached without convergence",
                2: "Requested tolerance was too high",
            }[iflag]
        )


def dsexpv(vec, no_krylov, norm, matvec, tol: float = 0):
    n = vec.shape[0]
    m = min(no_krylov, n-1)
    v = vec.astype(np.float64, casting="safe").ravel()
    workspace = np.zeros(n * (m + 2) + 5 * (m + 2) * (m + 2) + 7, dtype=np.float64)
    iworkspace = np.zeros(m + 2, dtype=np.int32)
    t = 1.0
    u, tol0, iflag = expokit.dsexpv(
        m, t, v, tol, norm, workspace, iworkspace, matvec, 0
    )
    check_exit_flag(iflag)
    return u


def zhexpv(vec, no_krylov, norm, matvec, tol: float = 0):
    n = vec.shape[0]
    m = min(no_krylov, n-1)
    v = vec.astype(np.complex128, casting="safe").ravel()
    workspace = np.zeros(
        n * (m + 1) + n + (m + 2) ** 2 + 4 * (m + 2) ** 2 + 7, dtype=np.complex128
    )
    iworkspace = np.zeros(m + 2, dtype=np.int32)
    t = 1.0
    u, tol0, iflag = expokit.zhexpv(
        m, t, v, tol, norm, workspace, iworkspace, matvec, 0
    )
    check_exit_flag(iflag)
    return u
