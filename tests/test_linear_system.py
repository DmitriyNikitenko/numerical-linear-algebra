import numpy as np
from src.linear_system import (
    gauss,
    rank,
    is_consistent,
    is_degenerate,
    solve_singular,
)
from src.utils import residual_norm


def test_rank_basic():
    A = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [1, 1, 1]
    ], dtype=float)

    assert rank(A) == 2


def test_rank_full_pivot():
    A = np.array([
        [0, 0, 1],
        [0, 2, 3],
        [4, 5, 6]
    ], dtype=float)

    U1 = gauss(A, full_pivot=False)
    U2 = gauss(A, full_pivot=True)

    assert rank(U1) == rank(U2) == 3


def test_consistent_system():
    A = np.array([
        [1, 2],
        [2, 4]
    ], dtype=float)

    b = np.array([3, 6], dtype=float)

    assert is_consistent(A, b)


def test_inconsistent_system():
    A = np.array([
        [1, 2],
        [2, 4]
    ], dtype=float)

    b = np.array([3, 5], dtype=float)

    assert not is_consistent(A, b)


def test_is_degenerate():
    A = np.array([
        [1, 2],
        [2, 4]
    ], dtype=float)

    assert is_degenerate(A)


def test_non_degenerate():
    A = np.array([
        [1, 2],
        [3, 4]
    ], dtype=float)

    assert not is_degenerate(A)


def test_gauss_shape():
    A = np.random.randn(5, 3)
    U = gauss(A)

    assert U.shape == A.shape


def test_solve_singular():
    A = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [1, 1, 1]
    ], dtype=float)

    b = np.array([6, 12, 3], dtype=float)

    x = solve_singular(A, b)

    assert is_consistent(A, b)
    assert residual_norm(A, x, b) < 1e-8


def test_solve_singular_full_pivot():
    A = np.array([
        [0, 1, 2],
        [0, 2, 4],
        [1, 1, 1]
    ], dtype=float)

    b = np.array([3, 6, 3], dtype=float)

    x = solve_singular(A, b, full_pivot=True)

    assert is_consistent(A, b)
    assert residual_norm(A, x, b) < 1e-8


def test_random_consistent_system():
    np.random.seed(0)

    A = np.random.randn(5, 3)
    x_true = np.random.randn(3)
    b = A @ x_true

    assert is_consistent(A, b)

    x = solve_singular(A, b)

    assert residual_norm(A, x, b) < 1e-8


def test_rank_rectangular():
    A = np.random.randn(4, 6)

    r = rank(A)

    assert r <= min(A.shape)