import numpy as np
from src.iterative_methods import (
    jacobi,
    gauss_seidel,
    is_diagonally_dominant,
    apriori_iterations,
)
from src.utils import residual_norm

def main():
    A = np.array([
        [10.0, 1.0, 1.0],
        [2.0, 10.0, 1.0],
        [2.0, 2.0, 10.0],
    ])
    b = np.array([12.0, 13.0, 14.0])

    print("A:")
    print(A)
    print("\nb:")
    print(b)

    print("\nStrict diagonal dominance:", is_diagonally_dominant(A, strict=True))
    print("A priori iterations (Jacobi):", apriori_iterations(A, b, method="jacobi"))
    print("A priori iterations (Seidel):", apriori_iterations(A, b, method="seidel"))

    x_j, it_j, hist_j = jacobi(A, b, return_history=True)
    x_s, it_s, hist_s = gauss_seidel(A, b, return_history=True)

    print("\nJacobi solution:", x_j)
    print("Jacobi iterations:", it_j)
    print("Jacobi residual norm:", residual_norm(A, x_j, b))

    print("\nGauss-Seidel solution:", x_s)
    print("Gauss-Seidel iterations:", it_s)
    print("Gauss-Seidel residual norm:", residual_norm(A, x_s, b))

    print("\nFirst 5 Jacobi differences:", hist_j[:5])
    print("First 5 Seidel differences:", hist_s[:5])


if __name__ == "__main__":
    main()
