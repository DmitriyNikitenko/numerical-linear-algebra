import numpy as np
from src.iterative_methods import (
    jacobi,
    gauss_seidel,
    is_diagonally_dominant,
    apriori_iterations,
    a_posteriori_bound,
    jacobi_iteration_matrix,
)
from src.utils import residual_norm, norm_inf_matrix, norm_inf_vec


def _jacobi_step(A, b, x):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.asarray(x, dtype=float)

    n = A.shape[0]
    x_new = np.zeros(n, dtype=float)

    for i in range(n):
        s = 0.0
        for j in range(n):
            if j != i:
                s += A[i, j] * x[j]
        x_new[i] = (b[i] - s) / A[i, i]

    return x_new


def test_diagonal_dominance():
    A = np.array([
        [10.0, 1.0, 1.0],
        [2.0, 10.0, 1.0],
        [2.0, 2.0, 10.0],
    ])
    assert is_diagonally_dominant(A)


def test_compare_dd_and_spd_without_diagonal_dominance():
    # Comparison on a matrix with diagonal dominance
    # and on a well-defined matrix without diagonal dominance.

    # Diagonally dominant matrix
    A_dd = np.array([
        [10.0, 1.0, 1.0],
        [2.0, 10.0, 1.0],
        [2.0, 2.0, 10.0],
    ])
    b_dd = np.array([12.0, 13.0, 14.0])
    x_true_dd = np.linalg.solve(A_dd, b_dd)

    x_j, it_j = jacobi(A_dd, b_dd, eps=1e-10, max_iter=5000)
    x_s, it_s = gauss_seidel(A_dd, b_dd, eps=1e-10, max_iter=5000)

    assert it_j > 0
    assert it_s > 0
    assert residual_norm(A_dd, x_j, b_dd) < 1e-8
    assert residual_norm(A_dd, x_s, b_dd) < 1e-8
    assert norm_inf_vec(x_j - x_true_dd) < 1e-6
    assert norm_inf_vec(x_s - x_true_dd) < 1e-6

    # Positive definite matrix without diagonal dominance
    A_spd = np.array([
        [3.0, 2.0, 2.0],
        [2.0, 3.0, 2.0],
        [2.0, 2.0, 3.0],
    ])
    b_spd = np.array([1.0, 2.0, 3.0])
    x_true_spd = np.linalg.solve(A_spd, b_spd)

    assert not is_diagonally_dominant(A_spd)

    x_s_spd, it_s_spd = gauss_seidel(A_spd, b_spd, eps=1e-10, max_iter=5000)

    assert it_s_spd > 0
    assert residual_norm(A_spd, x_s_spd, b_spd) < 1e-8
    assert norm_inf_vec(x_s_spd - x_true_spd) < 1e-6


def test_apriori_and_aposteriori_estimates():
    A = np.array([
        [10.0, 1.0, 1.0],
        [1.0, 10.0, 1.0],
        [1.0, 1.0, 10.0],
    ])
    b = np.array([12.0, 13.0, 14.0])
    x0 = np.zeros(3)

    x, it = jacobi(A, b, x0=x0, eps=1e-10, max_iter=5000)
    x_true = np.linalg.solve(A, b)

    # A priori estimate
    k_apr = apriori_iterations(A, b, x0=x0, eps=1e-10, method="jacobi")

    # Iteration matrix and q
    B = jacobi_iteration_matrix(A)
    q = norm_inf_matrix(B)

    # Restore two adjacent iterations to check the posterior estimate
    x_prev = x0.copy()  # x_0
    x_curr = _jacobi_step(A, b, x_prev)  # x_1

    if it > 1:
        for _ in range(1, it):
            x_prev, x_curr = x_curr, _jacobi_step(A, b, x_curr)  # x_{k+1}

    bound = a_posteriori_bound(x_prev, x_curr, q)

    assert residual_norm(A, x, b) < 1e-8
    assert norm_inf_vec(x - x_true) < 1e-6
    assert it <= k_apr or np.isinf(k_apr)
    assert norm_inf_vec(x_curr - x_true) <= bound + 1e-8