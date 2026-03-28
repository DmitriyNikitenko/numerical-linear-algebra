import numpy as np
from src.iterative_methods import (
    jacobi,
    gauss_seidel,
    is_diagonally_dominant,
)

def test_diagonal_dominance():
    A = np.array([
        [10.0, 1.0, 1.0],
        [2.0, 10.0, 1.0],
        [2.0, 2.0, 10.0],
    ])
    assert is_diagonally_dominant(A)

def test_jacobi_and_seidel_converge():
    A = np.array([
        [10.0, 1.0, 1.0],
        [2.0, 10.0, 1.0],
        [2.0, 2.0, 10.0],
    ])
    b = np.array([12.0, 13.0, 14.0])

    x_true = np.linalg.solve(A, b)

    x_j, it_j = jacobi(A, b, eps=1e-10, max_iter=5000)
    x_s, it_s = gauss_seidel(A, b, eps=1e-10, max_iter=5000)

    assert it_j > 0
    assert it_s > 0
    assert np.allclose(x_j, x_true, atol=1e-6)
    assert np.allclose(x_s, x_true, atol=1e-6)
