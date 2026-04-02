import numpy as np
from src.qr import qr_decomposition, qr_solve
from src.utils import residual_norm, norm_inf_matrix


def test_qr_decomposition():
    A = np.array([
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0],
    ])

    Q, R = qr_decomposition(A)
    I = np.eye(A.shape[0])

    assert norm_inf_matrix(Q @ R - A) < 1e-8
    assert norm_inf_matrix(Q.T @ Q - I) < 1e-8


def test_qr_solve():
    A = np.array([
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0],
    ])
    b = np.array([5.0, -2.0, 9.0])

    x = qr_solve(A, b)

    assert residual_norm(A, x, b) < 1e-8