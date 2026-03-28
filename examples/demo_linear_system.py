import numpy as np
from src.linear_system import rank, is_consistent, solve_singular, is_degenerate

def main():
    A = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [1.0, 1.0, 1.0],
    ])
    b = np.array([6.0, 12.0, 3.0])

    print("A:")
    print(A)
    print("\nb:")
    print(b)

    print("\nrank(A):", rank(A))
    print("degenerate:", is_degenerate(A))
    print("consistent:", is_consistent(A, b))

    if is_consistent(A, b):
        x = solve_singular(A, b)
        print("\nOne particular solution:")
        print(x)
        print("A @ x:")
        print(A @ x)


if __name__ == "__main__":
    main()
