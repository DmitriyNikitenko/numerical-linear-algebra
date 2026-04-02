import numpy as np
from .utils import (
    norm_inf_vec,
    norm_inf_matrix,
    is_zero,
    forward_substitution,
)


def is_diagonally_dominant(A, strict=True, eps=1e-12):
    A = np.asarray(A, dtype=float)
    n, m = A.shape

    if n != m:
        return False

    for i in range(n):
        diag = abs(A[i, i])
        rest = np.sum(np.abs(A[i])) - diag

        if strict:
            if not (diag > rest + eps):
                return False
        else:
            if not (diag >= rest - eps):
                return False

    return True


def _jacobi_step(A, b, x, eps=1e-12):
    # One step of Jacobi's method.
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)

    n = A.shape[0]
    x_new = np.zeros(n, dtype=float)

    for i in range(n):
        if is_zero(A[i, i], eps):
            raise np.linalg.LinAlgError("Zero diagonal element in Jacobi method.")

        s = 0.0
        for j in range(n):
            if j != i:
                s += A[i, j] * x[j]

        x_new[i] = (b[i] - s) / A[i, i]

    return x_new


def _seidel_step(A, b, x, eps=1e-12):
    # One step of the Gauss–Seidel method.
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)

    n = A.shape[0]
    x_new = x.copy()

    for i in range(n):
        if is_zero(A[i, i], eps):
            raise np.linalg.LinAlgError("Zero diagonal element in Gauss-Seidel method.")

        s1 = 0.0
        for j in range(i):
            s1 += A[i, j] * x_new[j]

        s2 = 0.0
        for j in range(i + 1, n):
            s2 += A[i, j] * x[j]

        x_new[i] = (b[i] - s1 - s2) / A[i, i]

    return x_new


def jacobi(A, b, x0=None, eps=1e-10, max_iter=1000, return_history=False):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square.")

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).reshape(-1).copy()

    history = []

    q = norm_inf_matrix(jacobi_iteration_matrix(A, eps=1e-10))
    for it in range(1, max_iter + 1):
        x_new = _jacobi_step(A, b, x, eps=eps)
        diff = norm_inf_vec(x_new - x)
        if q < 1:
            diff = diff * (q/(1-q))
        history.append(diff)

        if diff < eps:
            if return_history:
                return x_new, it, history
            return x_new, it

        x = x_new

    if return_history:
        return x, max_iter, history
    return x, max_iter


def gauss_seidel(A, b, x0=None, eps=1e-10, max_iter=1000, return_history=False):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square.")

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).reshape(-1).copy()

    history = []

    q = norm_inf_matrix(gauss_seidel_iteration_matrix(A, eps=1e-10))
    for it in range(1, max_iter + 1):
        x_new = _seidel_step(A, b, x, eps=eps)
        diff = norm_inf_vec(x_new - x)
        if q < 1:
            diff = diff * (q/(1-q))
        history.append(diff)

        if diff < eps:
            if return_history:
                return x_new, it, history
            return x_new, it

        x = x_new

    if return_history:
        return x, max_iter, history
    return x, max_iter


def jacobi_iteration_matrix(A, eps=1e-12):
    # Jacobi iteration matrix: B = -D^{-1}(L + U)
    A = np.asarray(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square.")

    d = np.diag(A)
    if np.any(np.abs(d) <= eps):
        raise np.linalg.LinAlgError("Zero diagonal element in Jacobi iteration matrix.")

    D = np.diag(d)
    return -np.diag(1.0 / d) @ (A - D)


def gauss_seidel_iteration_matrix(A, eps=1e-12):
    # Seidel iteration matrix: B = -(D + L)^{-1} U
    A = np.asarray(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square.")

    d = np.diag(A)
    if np.any(np.abs(d) <= eps):
        raise np.linalg.LinAlgError("Zero diagonal element in Gauss-Seidel iteration matrix.")

    D = np.diag(d)
    L = np.tril(A, -1)
    U = np.triu(A, 1)

    return -np.linalg.inv(D + L) @ U


def apriori_iterations(A, b, x0=None, eps=1e-10, method="jacobi", tol=1e-12):
    # A priori estimate of the number of iterations.
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square.")

    if x0 is None:
        x0 = np.zeros(n, dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float).reshape(-1)

    if method == "jacobi":
        B = jacobi_iteration_matrix(A, eps=tol)
        x1 = _jacobi_step(A, b, x0, eps=tol)
    elif method in ("seidel", "gauss_seidel"):
        B = gauss_seidel_iteration_matrix(A, eps=tol)
        x1 = _seidel_step(A, b, x0, eps=tol)
    else:
        raise ValueError("method must be 'jacobi' or 'seidel'.")

    q = norm_inf_matrix(B)
    if q >= 1:
        return np.inf

    d0 = norm_inf_vec(x1 - x0)
    if d0 <= tol:
        return 1

    k = np.log(eps * (1 - q) / d0) / np.log(q)
    return int(np.ceil(k))


def a_posteriori_bound(x_prev, x_curr, q, eps=1e-12):
    # Post hoc estimate: ||x* - x_k|| <= q/(1-q) * ||x_k - x_{k-1}||
    if q >= 1:
        return np.inf

    x_prev = np.asarray(x_prev, dtype=float).reshape(-1)
    x_curr = np.asarray(x_curr, dtype=float).reshape(-1)

    return (q / (1 - q)) * norm_inf_vec(x_curr - x_prev)