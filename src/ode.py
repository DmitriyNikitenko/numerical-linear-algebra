import numpy as np


# -----------------------------------------------------------------------------
# Butcher tableaus:
# 26 — midpoint
# 27 — Heun (explicit trapezoid)
# 28 — third-order Heun
# 29 — third-order Simpson-type RK
# 30 — classical RK4
# 31 — RK4 3/8-rule
# -----------------------------------------------------------------------------
"""
For each method, the matrix A (stage coefficients), vectors b (weights) and c (nodes), and the order of accuracy are specified.
"""
_METHODS = {
    "midpoint": {
        "A": np.array([[0.0, 0.0],
                       [0.5, 0.0]], dtype=float),
        "b": np.array([0.0, 1.0], dtype=float),
        "c": np.array([0.0, 0.5], dtype=float),
        "order": 2,
    },
    "heun": {
        "A": np.array([[0.0, 0.0],
                       [1.0, 0.0]], dtype=float),
        "b": np.array([0.5, 0.5], dtype=float),
        "c": np.array([0.0, 1.0], dtype=float),
        "order": 2,
    },
    "rk3_heun": {
        "A": np.array([[0.0, 0.0, 0.0],
                       [1.0 / 3.0, 0.0, 0.0],
                       [0.0, 2.0 / 3.0, 0.0]], dtype=float),
        "b": np.array([1.0 / 4.0, 0.0, 3.0 / 4.0], dtype=float),
        "c": np.array([0.0, 1.0 / 3.0, 2.0 / 3.0], dtype=float),
        "order": 3,
    },
    "rk3_simpson": {
        "A": np.array([[0.0, 0.0, 0.0],
                       [0.5, 0.0, 0.0],
                       [-1.0, 2.0, 0.0]], dtype=float),
        "b": np.array([1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0], dtype=float),
        "c": np.array([0.0, 0.5, 1.0], dtype=float),
        "order": 3,
    },
    "rk4_classic": {
        "A": np.array([[0.0, 0.0, 0.0, 0.0],
                       [0.5, 0.0, 0.0, 0.0],
                       [0.0, 0.5, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0]], dtype=float),
        "b": np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0], dtype=float),
        "c": np.array([0.0, 0.5, 0.5, 1.0], dtype=float),
        "order": 4,
    },
    "rk4_38": {
        "A": np.array([[0.0, 0.0, 0.0, 0.0],
                       [1.0 / 3.0, 0.0, 0.0, 0.0],
                       [-1.0 / 3.0, 1.0, 0.0, 0.0],
                       [1.0, -1.0, 1.0, 0.0]], dtype=float),
        "b": np.array([1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0], dtype=float),
        "c": np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=float),
        "order": 4,
    },
}

"""
The functions rk_step, solve_ivp_fixed, and solve_ivp_adaptive take the 'method' argument,
 which can be any of these keys, and automatically use the corresponding coefficients
"""

# Dictionary for convenient access by numbers (26-31) or short names
_ALIASES = {
    "26": "midpoint",
    "27": "heun",
    "28": "rk3_heun",
    "29": "rk3_simpson",
    "30": "rk4_classic",
    "31": "rk4_38",
    "mid": "midpoint",
    "midpoint": "midpoint",
    "heun": "heun",
    "trapezoid": "heun",
    "rk3_heun": "rk3_heun",
    "heun3": "rk3_heun",
    "rk3_simpson": "rk3_simpson",
    "simpson3": "rk3_simpson",
    "rk4_classic": "rk4_classic",
    "classic": "rk4_classic",
    "rk4_38": "rk4_38",
    "three_eighths": "rk4_38",
}

def _method_data(method):
    key = str(method).strip().lower()  # Convert to string, strip spaces, convert to lowercase
    key = _ALIASES.get(key, key)  # If key is in the aliases dictionary, use the method name, otherwise leave it as is
    if key not in _METHODS:
        raise ValueError(
            "Unknown method. Use 26-31 or one of: midpoint, heun, rk3_heun, rk3_simpson, rk4_classic, rk4_38."
        )
    return key, _METHODS[key] # Return the method name and its data (matrices A, b, c, and order)

# Converts the initial condition y0 to a uniform format (1D float array)
def _as_state(y):
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 0: # scalar is converted to a single-element array
        return arr.reshape(1), True # returns the array and a flag indicating whether y0 is a scalar
    return arr.reshape(-1), False  # vector or matrix is converted to a 1D array

# Restores the original format of y0
def _restore_state(y, scalar):
    y = np.asarray(y, dtype=float)
    return float(y[0]) if scalar else y

# Wrapper around the right-hand side function
def _call_f(f, x, y):
    arg = float(y[0]) if y.size == 1 else y
    out = np.asarray(f(float(x), arg), dtype=float) # Call f, the result is a float array

    if out.ndim == 0: # If scalar
        out = out.reshape(1) # Convert to a single-element array
    else:
        out = out.reshape(-1) # Force it to be 1D

    if out.size == 1 and y.size > 1: # If f returned a scalar but the system is vector-based
        out = np.full(y.shape, float(out[0]), dtype=float) # Broadcast the scalar to all components

    return out  # Return a correct 1D array

# Calculates the infinity norm of a vector
def _norm_inf(v):
    v = np.asarray(v, dtype=float).reshape(-1) # Convert input data to a 1D float array
    return float(np.max(np.abs(v))) if v.size else 0.0 # If the array is not empty - max absolute value, else 0.0

# The rk_step function performs one step of numerical integration of ODEs
def rk_step(f, x, y, h, method="rk4_classic"):
    """
    k_i = f( x + c_i·h ,  y + h·Σ_{j=1}^{i-1} a_{ij}·k_j ),   i = 1..s
    y_next = y + h·Σ_{i=1}^{s} b_i·k_i
    """

    _, tab = _method_data(method) # Get the Butcher tableau (A, b, c, order)
    y_vec, scalar = _as_state(y)  # Convert y to a 1D array and remember if it was a scalar

    s = len(tab["b"]) # Number of stages
    k = np.zeros((s, y_vec.size), dtype=float) # Array to store vectors k_i

    for i in range(s): # Loop over stages
        stage = y_vec.copy() # Start with the current value of y
        if i:
            # Calculate h * Σ(A[i][j] * k_j)
            stage += h * np.sum(tab["A"][i, :i, None] * k[:i], axis=0)  # For i>0, add the contribution of previous stages
        # Calculate k_i = f(x + c_i*h, stage)
        k[i] = _call_f(f, x + tab["c"][i] * h, stage)

    # Compute the new value: y_next = y + h * Σ(b_i * k_i)
    y_next = y_vec + h * np.sum(tab["b"][:, None] * k, axis=0)
    return _restore_state(y_next, scalar) # Return in the original form

# ODE integration on a uniform grid using a fixed-step Runge-Kutta method
# sequentially applies rk_step n_steps times
def solve_ivp_fixed(f, x0, xf, y0, n_steps, method="rk4_classic"):
    # Input data validation
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")
    if xf <= x0:
        raise ValueError("Require xf > x0.")

    name, tab = _method_data(method) # Get the Butcher tableau and method name
    y, scalar = _as_state(y0) # Convert y0 to a 1D array, remember the scalar flag

    x = float(x0) # Current value of x
    h = (float(xf) - float(x0)) / n_steps # Constant grid step size

    xs = [x] # List to store x nodes
    ys = [y.copy()] # List to store y solutions
    rhs_calls = 0 # Counter for right-hand side f calls

    for _ in range(n_steps): # Loop over integration steps
        s = len(tab["b"]) # Number of stages
        k = np.zeros((s, y.size), dtype=float) # Array to store stages k_i

        # Calculation of all stages (as in rk_step)
        for i in range(s):
            stage = y.copy() # Initial stage approximation
            if i:
                # Add the contribution of previous stages: h * Σ(A[i][j] * k_j)
                stage += h * np.sum(tab["A"][i, :i, None] * k[:i], axis=0)
            k[i] = _call_f(f, x + tab["c"][i] * h, stage)  # Calculate k_i

        rhs_calls += s # Account for all f calls in this step
        # y_{next} = y + h * Σ(b_i * k_i)
        y = y + h * np.sum(tab["b"][:, None] * k, axis=0)
        x += h # Move to the next node

        xs.append(x) # Store x
        ys.append(y.copy()) # Store y

    # Convert history to numpy arrays
    x_hist = np.asarray(xs, dtype=float)
    y_hist = np.asarray(ys, dtype=float)
    if scalar: # If the original problem was scalar
        y_hist = y_hist[:, 0] # Extract the single column

    # Return a dictionary with results
    return {
        "x": x_hist,
        "y": y_hist,
        "method": name,
        "order": tab["order"],
        "n_steps": n_steps,
        "rhs_calls": rhs_calls,
    }

# Estimation of the initial step size for adaptive integration
def estimate_initial_step(f, x0, xf, y0, method="rk4_classic", tol=1e-6, max_step=None):
    _, tab = _method_data(method) # Get method data
    y_vec, _ = _as_state(y0) # Convert y0 to a 1D array
    f0 = _call_f(f, float(x0), y_vec) # Calculate f(x0, y0) - the derivative at the initial point

    interval = abs(float(xf) - float(x0)) # Integration interval length
    if interval == 0:
        raise ValueError("xf must differ from x0.")

    # Derivative scale (prevent it from becoming zero to avoid division by zero)
    scale = max(_norm_inf(f0), 1e-14) # Infinity norm of f0, but not less than 1e-14

    # Main formula: h = (tol / scale)^{1/(p+1)}
    h = (tol / scale) ** (1.0 / (tab["order"] + 1))

    # If the result is non-numeric or non-positive - use a fallback step size
    if not np.isfinite(h) or h <= 0:
        h = interval / 100.0

    # Bound from above if max_step is specified
    if max_step is not None:
        h = min(h, float(max_step))

    # The step size cannot be greater than the entire interval length
    return min(h, interval)

# Adaptive ODE integration with step size control using step doubling
# Integration with automatic step size selection so that the local error at each step does not exceed the specified tolerances rtol (relative) and atol (absolute).
def solve_ivp_adaptive(
    f,
    x0,
    xf,
    y0,
    method="rk4_classic",
    rtol=1e-6,
    atol=1e-12,
    h0=None,
    h_min=1e-14,
    h_max=None,
    safety=0.9,
    max_steps=100000,
):
    #   rtol - relative local error tolerance (>0)
    #   atol - absolute local error tolerance (>=0)
    #   h0 - initial step size
    #   h_min - minimum allowed step size
    #   h_max - maximum allowed step size (None means the entire interval)
    # safety - safety factor (0 < safety < 1), reduces the step to decrease the number of rejections


    # Input data validation
    if xf <= x0:
        raise ValueError("Require xf > x0.")
    if rtol <= 0 or atol < 0:
        raise ValueError("rtol must be > 0 and atol must be >= 0.")

    name, tab = _method_data(method) # Method data
    y, scalar = _as_state(y0) # y - 1D array, scalar - flag
    x = float(x0)
    interval = float(xf - x0)

    # Initial step size, either passed or estimated
    if h0 is None:
        h = estimate_initial_step(f, x0, xf, y0, method=name, tol=max(rtol, atol), max_step=interval)
    else:
        h = float(h0)

    if h_max is None:
        h_max = interval

    h = min(max(h, h_min), h_max, interval) # Enforced limits

    # Solution history
    xs = [x]
    ys = [y.copy()]
    accepted_steps = [] # Store all accepted steps
    rejected_steps = [] # Store all rejected steps
    rhs_calls = 0 # Number of f calls
    n_accept = 0 # Number of accepted steps
    n_reject = 0 # Number of rejected steps

    p = tab["order"] # Method's order of accuracy
    corr = 2**p - 1 # Correction factor for Richardson extrapolation

    # Main loop for adaptive integration
    while x < xf and (n_accept + n_reject) < max_steps:
        h = min(h, xf - x) # Do not overshoot the final point
        if h < h_min:
            raise RuntimeError("Step size underflow.")

        # Calculate two approximations:
        # y_full - one large step h
        # y_two_half - two small steps h/2
        y_full = np.asarray(rk_step(f, x, y, h, method=name), dtype=float).reshape(-1)
        y_half = np.asarray(rk_step(f, x, y, h / 2.0, method=name), dtype=float).reshape(-1)
        y_two_half = np.asarray(rk_step(f, x + h / 2.0, y_half, h / 2.0, method=name), dtype=float).reshape(-1)
        rhs_calls += 3 * len(tab["b"]) # 3 sets of stages (one step h and two h/2 steps)

        # Local error estimation
        diff = y_two_half - y_full # Difference between the two approaches
        scale = atol + rtol * np.maximum(np.abs(y_full), np.abs(y_two_half)) # Scale for normalization
        err = _norm_inf(diff / scale) # Relative error (in infinity norm)
        accepted = err <= 1.0 # Is the step accepted

        # Calculate the step size modification factor (standard formula for step doubling)
        if err == 0.0:
            factor = 2.0
        else:
            factor = safety * err ** (-1.0 / (p + 1))
        factor = float(np.clip(factor, 0.2, 5.0)) # Constrain so the step size does not change too abruptly

        if accepted:
            # Refine the solution using Richardson extrapolation
            """
            Richardson extrapolation is a method of increasing the accuracy of a numerical result by combining two approximations obtained with different step sizes.
            """
            y = y_two_half + diff / corr
            x += h
            xs.append(x)
            ys.append(y.copy())
            accepted_steps.append(h)
            n_accept += 1
            h = min(h * factor, h_max) # Increase the step size for the next iteration
        else:
            rejected_steps.append(h)
            n_reject += 1
            # Decrease the step size, but not below h_min
            h = max(h * max(0.2, min(0.8, factor)), h_min)

    if x < xf:
        raise RuntimeError("Maximum number of steps exceeded before reaching xf.")

    # Convert history to numpy arrays
    x_hist = np.asarray(xs, dtype=float)
    y_hist = np.asarray(ys, dtype=float)
    if scalar:
        y_hist = y_hist[:, 0]

    # Return a detailed results dictionary
    return {
        "x": x_hist,
        "y": y_hist,
        "method": name,
        "order": p,
        "accepted_steps": np.asarray(accepted_steps, dtype=float),
        "rejected_steps": np.asarray(rejected_steps, dtype=float),
        "rhs_calls": rhs_calls,
        "iterations": n_accept + n_reject,
        "accepted_count": n_accept,
        "rejected_count": n_reject,
        "rtol": rtol,
        "atol": atol,
    }