import argparse
import numpy as np
from lu import (
    lu_decomposition,
    lu_solve,
    determinant,
    inverse,
    condition,
)
from linear_system import rank, is_consistent, is_degenerate, solve_singular
from qr import qr_decomposition, qr_solve
from iterative_methods import jacobi, gauss_seidel, is_diagonally_dominant
from utils import residual_norm


def demo_lu():
    A = np.array([
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0],
    ])
    b = np.array([5.0, -2.0, 9.0])

    L, U, P, Q = lu_decomposition(A)
    print("=== LU demo ===")
    print("L:\n", L)
    print("U:\n", U)
    print("Check P A Q ≈ L U:", np.allclose(P @ A @ Q, L @ U))
    x = lu_solve(A, b)
    print("Solution x:", x)
    print("Residual norm:", residual_norm(A, x, b))
    print("det(A):", determinant(A))
    A_inv = inverse(A)
    print("A @ A_inv ≈ I:", np.allclose(A @ A_inv, np.eye(A.shape[0])))
    print("cond_inf(A):", condition(A))
    print()


def demo_systems():
    A = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [1.0, 1.0, 1.0],
    ])
    b = np.array([6.0, 12.0, 3.0])

    print("=== Singular systems demo ===")
    print("rank(A):", rank(A))
    print("degenerate:", is_degenerate(A))
    print("consistent:", is_consistent(A, b))
    if is_consistent(A, b):
        x = solve_singular(A, b)
        print("One particular solution:", x)
        print("A x:", A @ x)
    print()


def demo_qr():
    A = np.array([
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0],
    ])
    b = np.array([5.0, -2.0, 9.0])

    Q, R = qr_decomposition(A)
    print("=== QR demo ===")
    print("Q:\n", Q)
    print("R:\n", R)
    print("Check Q R ≈ A:", np.allclose(Q @ R, A))
    print("Check Q^T Q ≈ I:", np.allclose(Q.T @ Q, np.eye(A.shape[0])))
    x = qr_solve(A, b)
    print("Solution x:", x)
    print("Residual norm:", residual_norm(A, x, b))
    print()


def demo_iterative():
    A = np.array([
        [10.0, 1.0, 1.0],
        [2.0, 10.0, 1.0],
        [2.0, 2.0, 10.0],
    ])
    b = np.array([12.0, 13.0, 14.0])

    print("=== Iterative methods demo ===")
    print("Diagonal dominance:", is_diagonally_dominant(A))

    x_j, it_j = jacobi(A, b, eps=1e-10, max_iter=1000)
    x_s, it_s = gauss_seidel(A, b, eps=1e-10, max_iter=1000)

    print("Jacobi solution:", x_j)
    print("Jacobi iterations:", it_j)
    print("Jacobi residual:", residual_norm(A, x_j, b))

    print("Gauss-Seidel solution:", x_s)
    print("Gauss-Seidel iterations:", it_s)
    print("Gauss-Seidel residual:", residual_norm(A, x_s, b))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Numerical linear algebra demos"
    )
    parser.add_argument(
        "--demo",
        choices=["lu", "systems", "qr", "iterative", "all"],
        default="all",
        help="Choose which demo to run",
    )
    args = parser.parse_args()

    if args.demo in ("lu", "all"):
        demo_lu()
    if args.demo in ("systems", "all"):
        demo_systems()
    if args.demo in ("qr", "all"):
        demo_qr()
    if args.demo in ("iterative", "all"):
        demo_iterative()


if __name__ == "__main__":
    main()
