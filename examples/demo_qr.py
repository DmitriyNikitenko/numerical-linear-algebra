import numpy as np
from src.qr import qr_decomposition, qr_solve
from src.utils import residual_norm

def main():
    A = np.array([
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0],
    ])
    b = np.array([5.0, -2.0, 9.0])

    Q, R = qr_decomposition(A)

    print("A:")
    print(A)
    print("\nQ:")
    print(Q)
    print("\nR:")
    print(R)
    print("\nCheck Q @ R ≈ A:", np.allclose(Q @ R, A, atol=1e-8))
    print("Check Q.T @ Q ≈ I:", np.allclose(Q.T @ Q, np.eye(A.shape[0]), atol=1e-8))

    x = qr_solve(A, b)
    print("\nSolution x:")
    print(x)
    print("Residual norm:", residual_norm(A, x, b))


if __name__ == "__main__":
    main()
