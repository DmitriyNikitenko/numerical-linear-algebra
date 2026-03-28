import numpy as np
from .utils import swap_rows, swap_cols, is_zero


def gauss(A, b=None, full_pivot=False, return_perm=False, eps=1e-12):
    U = A.copy().astype(float)

    if b is not None:
        b = b.copy().astype(float).reshape(-1)

    n, m = U.shape
    row = 0

    # colsindex[j] - the number of the original variable that is currently in the column j
    colsindex = np.arange(m)

    for col in range(m):
        if row >= n:
            break

        # Find pivot
        if full_pivot:
            sub = np.abs(U[row:, col:])
            if sub.size == 0:
                break

            i_rel, j_rel = np.unravel_index(np.argmax(sub), sub.shape)
            i = row + i_rel
            j = col + j_rel
        else:
            i = row + int(np.argmax(np.abs(U[row:, col])))
            j = col

        if is_zero(U[i, j], eps):
            continue

        swap_rows(U, row, i)
        if b is not None:
            swap_rows(b, row, i)

        if full_pivot and j != col:
            swap_cols(U, col, j)
            colsindex[col], colsindex[j] = colsindex[j], colsindex[col]

        # Zeroing
        pivot = U[row, col]
        for r in range(row + 1, n):
            factor = U[r, col] / pivot
            U[r, col:] -= factor * U[row, col:]
            U[r, col] = 0.0

            if b is not None:
                b[r] -= factor * b[row]

        row += 1

    if return_perm:
        return U, b, colsindex

    return (U, b) if b is not None else U


def rank(A):
    U = gauss(A)
    n, m = U.shape
    r = 0

    for i in range(n):
        if not all(is_zero(U[i, j]) for j in range(m)):
            r += 1

    return r

def is_consistent(A,b):
    X, Y = gauss(A,b)
    n, m = X.shape
    for i in range(n):
        if all(is_zero(X[i,j]) for j in range(m)) and not is_zero(Y[i]):
            return False
    return True

def is_degenerate(A):
    return rank(A) < min(A.shape)


def solve_singular(A, b, full_pivot=False, eps=1e-12):
    # Finds one particular solution of the joint degenerate system Ax = b.
    U, Y, colsindex = gauss(A, b, full_pivot=full_pivot, return_perm=True, eps=eps)
    n, m = U.shape

    # Compatibility check already in step form
    for i in range(n):
        if np.all(np.abs(U[i]) <= eps) and not is_zero(Y[i], eps):
            raise np.linalg.LinAlgError("System is inconsistent.")

    x_perm = np.zeros(m, dtype=float)

    pivot_rows = []
    pivot_cols = []

    for i in range(n):
        nz = np.where(np.abs(U[i]) > eps)[0]
        if nz.size > 0:
            pivot_rows.append(i)
            pivot_cols.append(int(nz[0]))

    for idx in range(len(pivot_rows) - 1, -1, -1):
        i = pivot_rows[idx]
        j = pivot_cols[idx]

        s = np.dot(U[i, j + 1:], x_perm[j + 1:])
        x_perm[j] = (Y[i] - s) / U[i, j]

    x = np.zeros(m, dtype=float)
    x[colsindex] = x_perm
    return x
