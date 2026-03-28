import numpy as np

def is_zero(x, eps=1e-10):
    return abs(x) < eps

def swap_rows(A, i, j):
    A[[i, j]] = A[[j, i]]

def swap_cols(A, i, j):
    A[:, [i, j]] = A[:, [j, i]]

def norm_inf_vec(x):
    return np.max(np.abs(x))

def norm_1_vec(x):
    return np.sum(np.abs(x))

def norm_2_vec(x):
    return np.sqrt(np.sum(x**2))

# Maximum amount per line
def norm_inf_matrix(A):
    return np.max(np.sum(np.abs(A), axis=1))

# Maximum amount per column
def norm_one_matrix(A):
    return np.max(np.sum(np.abs(A), axis=0))

def norm_frobenius(A):
    return np.sqrt(np.sum(A**2))

def forward_substitution(L, b):
    # Solve L z = b
    n = len(b)
    z = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i, j] * z[j]
        z[i] = (b[i] - s) / L[i, i]
    return z

def backward_substitution(U, b):
    # Solve U x = b
    n = len(b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U[i, j] * x[j]
        x[i] = (b[i] - s) / U[i, i]
    return x

def random_matrix(n, m = None, scale = 1.0):
    if m is None:
        m = n
    return scale * np.random.randn(n, m)

def random_vector(n, scale = 1.0):
    return np.random.randn(n)

def diagonal_dominant(n):
    A = np.random.randn(n, n)
    for i in range(n):
        A[i, i] += n
    return A

def residual(A, x, b):
    #r = A x - b
    return A @ x - b

def residual_norm(A, x, b):
    r = residual(A, x, b)
    return norm_inf_vec(r)

def relative_error(x, x_true):
    return norm_inf_vec(x - x_true) / norm_inf_vec(x_true)