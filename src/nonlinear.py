import numpy as np
import time
from .lu import lu_solve
from .utils import norm_inf_vec


def newton_system(F, J, x0, eps=1e-10, max_iter=100, mode="full", m=1, k=np.inf):
    """
    mode:
        "full"       - classical Newton
        "modified"   - Jacobian fixed at x0
        "hybrid"     - first k full steps, then reuse Jacobian every m steps
    """

    x = np.asarray(x0, dtype=float).copy()

    t0 = time.perf_counter()

    jacobian_recomputations = 0
    iterations = 0

    J_current = None

    for it in range(max_iter):
        Fx = F(x)

        if norm_inf_vec(Fx) < eps:
            break

        recompute = False

        if mode == "full":
            recompute = True

        elif mode == "modified":
            if it == 0:
                recompute = True

        elif mode == "hybrid":
            if it < k:
                recompute = True
            else:
                if (it - k) % m == 0:
                    recompute = True

        else:
            raise ValueError("Unknown mode")

        if recompute:
            J_current = J(x)
            jacobian_recomputations += 1

        delta = lu_solve(J_current, -Fx)

        x = x + delta
        iterations += 1

        if norm_inf_vec(delta) < eps:
            break

    t1 = time.perf_counter()

    return {
        "x": x,
        "iterations": iterations,
        "jacobian_recomputations": jacobian_recomputations,
        "residual_norm": norm_inf_vec(F(x)),
        "time": t1 - t0,
    }