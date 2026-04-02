import numpy as np
from src.lu import lu_decomposition, lu_solve, determinant, inverse, condition
from src.utils import residual_norm, norm_inf_matrix


def test_lu_reconstruction():
    A = np.array([
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0],
    ])

    L, U, P, Q = lu_decomposition(A)
    assert norm_inf_matrix(P @ A @ Q - L @ U) < 1e-8


def test_lu_solve():
    A = np.array([
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0],
    ])
    b = np.array([5.0, -2.0, 9.0])

    x = lu_solve(A, b)

    assert residual_norm(A, x, b) < 1e-8


def test_determinant_and_inverse():
    A = np.array([
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0],
    ])

    det_a = determinant(A)
    det_np = np.linalg.det(A)
    assert abs(det_a - det_np) < 1e-8

    A_inv = inverse(A)
    I = np.eye(A.shape[0])

    assert norm_inf_matrix(A @ A_inv - I) < 1e-8
    assert norm_inf_matrix(A_inv @ A - I) < 1e-8


def test_condition_number():
    A = np.array([
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0],
    ])

    A_inv = inverse(A)
    cond_a = condition(A)
    cond_manual = norm_inf_matrix(A) * norm_inf_matrix(A_inv)

    assert np.isfinite(cond_a)
    assert cond_a >= 1.0
    assert abs(cond_a - cond_manual) < 1e-8