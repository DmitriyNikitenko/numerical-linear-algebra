import numpy as np
from .utils import is_zero, norm_2_vec, backward_substitution

def qr_decomposition(A, eps=1e-12):
    # QR decomposition by the Householder reflection method.
    # Returns Q, R such that A = Q R.
    A = np.asarray(A, dtype=float)
    n, m = A.shape

    if n != m:
        raise ValueError("QR decomposition in this project is implemented for square matrices only.")

    R = A.copy()
    Q = np.eye(n, dtype=float)

    for k in range(n - 1):
        x = R[k:, k]
        norm_x = norm_2_vec(x)

        if is_zero(norm_x, eps):
            continue

        # v = x + sign(x1) * ||x|| * e1
        v = x.copy()
        sign = 1.0 if x[0] >= 0 else -1.0
        v[0] += sign * norm_x

        v_norm = norm_2_vec(v) 
        if is_zero(v_norm, eps):
            continue

        v = v / v_norm

        H_small = np.eye(n - k, dtype=float) - 2.0 * np.outer(v, v)

        # Apply reflection to R
        R[k:, k:] = H_small @ R[k:, k:]

        # Expand the reflection to size n x n
        H = np.eye(n, dtype=float)
        H[k:, k:] = H_small

        # Q = H1 H2 ... Hk
        Q = Q @ H

    # Creates a mask (True/False table) where True is set and assigns the value 0.0 to these elements.
    R[np.abs(R) < eps] = 0.0
    return Q, R


def qr_solve(A, b, eps=1e-12):
    # Solves Ax = b via QR factorization.
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    n, m = A.shape
    if n != m:
        raise ValueError("QR solve is implemented for square matrices only.")

    Q, R = qr_decomposition(A, eps=eps)
    if np.any(np.abs(np.diag(R)) <= eps):
        raise np.linalg.LinAlgError("Matrix is singular or nearly singular.")

    y = Q.T @ b
    x = backward_substitution(R, y)
    return x

def check_qr_decomposition(A, eps=1e-10):
    # Decomposition check: A = Q R
    Q, R = qr_decomposition(A, eps=eps)
    return np.allclose(Q @ R, np.asarray(A, dtype=float), atol=eps)