import numpy as np
from .utils import (
    swap_rows,
    swap_cols,
    forward_substitution,
    backward_substitution,
    is_zero,
    norm_inf_matrix
)

def lu_decomposition(A, full_pivot = True):
    U = A.copy().astype(float)
    n = U.shape[0]
    L = np.eye(n)
    P = np.eye(n)
    Q = np.eye(n)

    for k in range(n):
        
        if full_pivot:
            sub = np.abs(U[k:, k:])
            i, j = np.unravel_index(
                np.argmax(sub),
                sub.shape
            )
            i += k
            j += k
        else:
            sub = np.abs(U[k:,k])
            i = k + np.argmax(sub)
            j = k

        if i != k:
            swap_rows(U, k, i)
            swap_rows(P, k, i)
            # Select from L two rows (with indices k and i) and the first k columns,
            # and replace them with the same two rows taken in reverse order (i, k),
            # effectively swapping the values between rows k and i in the first k columns.
            L[[k, i], :k] = L[[i, k], :k]

        if j != k and full_pivot:
            swap_cols(U, k, j)
            swap_cols(Q, k, j)

        if is_zero(U[k, k]):
            continue

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    
    return L, U, P, Q

def lu_solve(A, b):
    L, U, P, Q = lu_decomposition(A)
    b1 = P @ b
    z = forward_substitution(L, b1)
    y = backward_substitution(U, z)
    x = Q @ y
    return x

def determinant(A):
    # det(A) = det(P^-1)* det(L)* det(U)*det(Q^-1)
    # det(A) = det(U) / (det(P) * det(Q))
    L, U, P, Q = lu_decomposition(A)
    detU = np.prod(np.diag(U))
    detP = np.linalg.det(P)
    detQ = np.linalg.det(Q)
    detA = detU / (detP * detQ)
    return detA

def inverse(A):
    # A^{-1} = (x1 x2 ... xn)
    # Ax_i = e_i
    n = A.shape[0]
    A_inv = np.zeros((n, n))
    for i in range(n):
        # Create e_i
        e = np.zeros(n)
        e[i] = 1
        # Solve Ax = e_i
        x = lu_solve(A, e)
        # Put x in column
        A_inv[:, i] = x
    return A_inv

def condition(A):
    A_inv = inverse(A)
    return norm_inf_matrix(A) * norm_inf_matrix(A_inv)

def check_lu_decomposition(A, full_pivot=True, eps=1e-10):
    # Decomposition check: P A Q = L U
    L, U, P, Q = lu_decomposition(A, full_pivot=full_pivot)
    return np.allclose(P @ np.asarray(A, dtype=float) @ Q, L @ U, atol=eps)