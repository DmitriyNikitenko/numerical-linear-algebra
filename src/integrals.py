import time
from typing import Callable, Dict, List, Optional
import numpy as np

_GAUSS3_NODES = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)], dtype=float)
_GAUSS3_WEIGHTS = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=float)
_WEIGHT_TRANSFORM_POWER = 16


def _validate_inputs(a: float, b: float, N0: int, eps: float, method: str) -> None:
    if b <= a:
        raise ValueError("Interval must satisfy a < b.")
    if N0 < 1:
        raise ValueError("N0 must be >= 1.")
    if eps <= 0:
        raise ValueError("eps must be > 0.")
    if method not in {"midpoint", "trapezoid", "simpson", "gauss"}:
        raise ValueError("Unknown method. Use 'midpoint', 'trapezoid', 'simpson', or 'gauss'.")


def _validate_weighted_inputs(a: float, b: float, alpha: float, beta: float, N0: int, eps: float, method: str) -> None:
    """Validate arguments for the weighted integral."""
    if b <= a:
        raise ValueError("Interval must satisfy a < b.")
    if N0 < 1:
        raise ValueError("N0 must be >= 1.")
    if eps <= 0:
        raise ValueError("eps must be > 0.")
    if not (0.0 <= alpha < 1.0):
        raise ValueError("alpha must satisfy 0 <= alpha < 1.")
    if not (0.0 <= beta < 1.0):
        raise ValueError("beta must satisfy 0 <= beta < 1.")
    if method not in {"midpoint", "trapezoid", "simpson", "gauss"}:
        raise ValueError("Unknown method. Use 'midpoint', 'trapezoid', 'simpson', or 'gauss'.")


def _evaluate_function(f: Callable, x):
    x_arr = np.asarray(x, dtype=float)
    y = np.asarray(f(x_arr), dtype=float)

    if y.shape == ():
        return np.full_like(x_arr, float(y), dtype=float)

    if y.shape != x_arr.shape:
        y = np.broadcast_to(y, x_arr.shape).astype(float, copy=False)

    return y


def _estimate_order_from_three(S_h2: float, S_h1: float, S_h: float) -> Optional[float]:
    num = S_h2 - S_h1
    den = S_h1 - S_h
    if num == 0 or den == 0:
        return None

    ratio = num / den
    if ratio <= 0:
        return None

    return float(-np.log2(abs(ratio)))

# Standard composite quadrature rules
def _midpoint_composite(f: Callable, a: float, b: float, N: int) -> float:
    h = (b - a) / N
    x = a + (np.arange(N, dtype=float) + 0.5) * h
    return float(h * np.sum(_evaluate_function(f, x)))


def _trapezoid_composite(f: Callable, a: float, b: float, N: int) -> float:
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = _evaluate_function(f, x)
    return float(h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1]))


def _simpson_composite(f: Callable, a: float, b: float, N: int) -> float:
    if N % 2 == 1:
        N += 1

    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = _evaluate_function(f, x)

    return float((h / 3.0) * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])))


def _gauss3_composite(f: Callable, a: float, b: float, N: int) -> float:
    h = (b - a) / N
    total = 0.0

    for i in range(N):
        left = a + i * h
        right = left + h
        mid = 0.5 * (left + right)
        half = 0.5 * h

        x = mid + half * _GAUSS3_NODES
        y = _evaluate_function(f, x)
        total += half * np.dot(_GAUSS3_WEIGHTS, y)

    return float(total)


def _eval_rule(f: Callable, a: float, b: float, N: int, method: str) -> float:
    if method == "midpoint":
        return _midpoint_composite(f, a, b, N)
    if method == "trapezoid":
        return _trapezoid_composite(f, a, b, N)
    if method == "simpson":
        return _simpson_composite(f, a, b, N)
    if method == "gauss":
        return _gauss3_composite(f, a, b, N)
    raise ValueError("Unknown method.")


def _integrate_core(
    f: Callable,
    a: float,
    b: float,
    method: str,
    N0: int,
    eps: float,
    max_iter: int,
    exact: Optional[float],
) -> Dict[str, object]:

    _validate_inputs(a, b, N0, eps, method)

    # Simpson needs an even number of panels.
    if method == "simpson" and N0 % 2 == 1:
        N0 += 1

    # Asymptotic orders for Runge correction.
    p_order = {
        "midpoint": 2,
        "trapezoid": 2,
        "simpson": 4,
        "gauss": 6,
    }[method]

    t0 = time.perf_counter()

    table: List[Dict[str, object]] = [] # A list where the dictionary with data will be added at each iteration
    N = int(N0)
    S_prev = None
    S_prev2 = None

    for _ in range(max_iter):
        h = (b - a) / N
        S = _eval_rule(f, a, b, N, method)

        if exact is not None:
            E = abs(S - exact) / abs(exact) if exact != 0 else abs(S - exact)
        else:
            E = None

        # First iteration
        if S_prev is None:
            row = {
                "N": N,
                "h": h,
                "S": S,
                "E": E,
                "R": None,
                "p": None,
                "S_prime": None,
                "E_prime": None,
                "p_prime": None,
            }
            table.append(row)
        else:
            # Richardson's extrapolation
            R_signed = (S - S_prev) / (2.0**p_order - 1.0)
            S_prime = S + R_signed

            if exact is not None:
                E_prime = abs(S_prime - exact) / abs(exact) if exact != 0 else abs(S_prime - exact)
            else:
                E_prime = None
            
            # Estimation of the actual order of convergence
            p_est = None
            if S_prev2 is not None:
                p_est = _estimate_order_from_three(S_prev2, S_prev, S)

            row = {
                "N": N,
                "h": h,
                "S": S,
                "E": E,
                "R": abs(R_signed),
                "p": p_est,
                "S_prime": S_prime,
                "E_prime": E_prime,
                "p_prime": None,
            }
            table.append(row)

            # Stop condition
            if exact is not None:
                if method == "trapezoid":
                    if E is not None and E <= eps:
                        break
                else:
                    if E_prime is not None and E_prime <= eps:
                        break

            if exact is None and abs(R_signed) / max(1.0, abs(S_prime)) <= eps:
                break
        
        # Updating variables before the next iteration
        S_prev2 = S_prev
        S_prev = S
        N *= 2

    t1 = time.perf_counter()

    result = table[-1]["S_prime"] if table[-1]["S_prime"] is not None else table[-1]["S"]

    return {
        "result": result,
        "N": table[-1]["N"],
        "iterations": len(table),
        "table": table,
        "time": t1 - t0,
    }

# Weighted integral support
def _weighted_transform(f: Callable, a: float, b: float, alpha: float, beta: float):
    L = b - a
    m = _WEIGHT_TRANSFORM_POWER

    def g(t):
        t_arr = np.asarray(t, dtype=float)
        out = np.zeros_like(t_arr, dtype=float)


        # Mask for internal points (excluding ends)
        mask = (t_arr > 0.0) & (t_arr < 1.0)

        # If at least one array element lies within (0,1), perform the calculation. Otherwise (all t are 0 or 1), return an array of zeros.
        if np.any(mask):
            # Extracting internal points
            tt = t_arr[mask]

            # Calculation of the main terms of the transformation
            tm = tt**m
            om = (1.0 - tt)**m
            denom = tm + om

            # Mapping t → x(t)
            phi = tm / denom
            x = a + L * phi

            # Calculating the multiplier
            factor = (
                (L ** (1.0 - alpha - beta))
                * m
                * (tt ** (m - 1 - m * alpha))
                * ((1.0 - tt) ** (m - 1 - m * beta))
                * (denom ** (alpha + beta - 2.0))
            )

            fx = _evaluate_function(f, x)
            out[mask] = fx * factor

        return out if out.shape != () else float(out)

    return g


# Public
def integrate(
    f: Callable,
    a: float,
    b: float,
    method: str = "midpoint",
    N0: int = 1,
    eps: float = 1e-10,
    max_iter: int = 20,
    exact: Optional[float] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
) -> Dict[str, object]:
    if (alpha is None) != (beta is None):
        raise ValueError("alpha and beta must be given together.")

    if alpha is not None and beta is not None:
        _validate_weighted_inputs(a, b, alpha, beta, N0, eps, method)
        g = _weighted_transform(f, a, b, alpha, beta)
        return _integrate_core(g, 0.0, 1.0, method, N0, eps, max_iter, exact)

    return _integrate_core(f, a, b, method, N0, eps, max_iter, exact)


def integrate_weighted(
    f: Callable,
    a: float,
    b: float,
    alpha: float,
    beta: float,
    method: str = "gauss",
    N0: int = 1,
    eps: float = 1e-10,
    max_iter: int = 20,
    exact: Optional[float] = None,
) -> Dict[str, object]:
    return integrate(
        f,
        a,
        b,
        method=method,
        N0=N0,
        eps=eps,
        max_iter=max_iter,
        exact=exact,
        alpha=alpha,
        beta=beta,
    )