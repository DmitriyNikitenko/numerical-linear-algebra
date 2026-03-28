import numpy as np
from src.lu import lu_decomposition, lu_solve, determinant, inverse, condition

def test_lu_reconstruction():
    A = np.array([
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0],
    ])
    L, U, P, Q = lu_decomposition(A)
    assert np.allclose(P @ A @ Q, L @ U)

def test_lu_solve():
    A = np.array([
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0],
    ])
    b = np.array([5.0, -2.0, 9.0])
    x = lu_solve(A, b)
    assert np.allclose(A @ x, b)

def test_determinant_and_inverse():
    A = np.array([
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0],
    ])
    det_a = determinant(A)
    det_np = np.linalg.det(A)
    assert np.allclose(det_a, det_np)

    A_inv = inverse(A)
    I = np.eye(A.shape[0])
    assert np.allclose(A @ A_inv, I)
    assert np.allclose(A_inv @ A, I)

def test_condition_number():
    A = np.array([
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0],
    ])
    cond_a = condition(A)
    assert cond_a >= 1.0
