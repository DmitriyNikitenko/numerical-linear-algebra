import numpy as np
from src.lu import lu_decomposition, lu_solve, determinant, inverse, condition
from src.utils import residual_norm

def main():
    A = np.array([
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0],
    ])
    b = np.array([5.0, -2.0, 9.0])

    L, U, P, Q = lu_decomposition(A)

    print("A:")
    print(A)
    print("\nL:")
    print(L)
    print("\nU:")
    print(U)
    print("\nCheck P @ A @ Q ≈ L @ U:", np.allclose(P @ A @ Q, L @ U))

    x = lu_solve(A, b)
    print("\nSolution x:")
    print(x)
    print("Residual norm:", residual_norm(A, x, b))

    print("\ndet(A):", determinant(A))

    A_inv = inverse(A)
    print("\nA^{-1}:")
    print(A_inv)
    print("A @ A^{-1} ≈ I:", np.allclose(A @ A_inv, np.eye(A.shape[0]), atol=1e-8))
    print("A^{-1} @ A ≈ I:", np.allclose(A_inv @ A, np.eye(A.shape[0]), atol=1e-8))

    print("\ncond_inf(A):", condition(A))


if __name__ == "__main__":
    main()
