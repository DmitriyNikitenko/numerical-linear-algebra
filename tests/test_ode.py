import numpy as np
from src.numerical_lib.ode import solve_ivp_fixed, solve_ivp_adaptive


# Variant parameters
A, B, C = 3.0, 3.0, -3.0
METHOD = "heun"

# Right-hand side of the system and the exact solution
def f(x, y):
    y1, y2, y3, y4 = y
    dy1 = 2.0 * x * (y2 ** (1.0 / B)) * y4
    dy2 = 2.0 * B * x * np.exp((B / C) * (y3 - A)) * y4
    dy3 = 2.0 * C * x * y4
    dy4 = -2.0 * x * np.log(y1)
    return np.array([dy1, dy2, dy3, dy4])

def exact(x):
    y1 = np.exp(np.sin(x**2))
    y2 = np.exp(B * np.sin(x**2))
    y3 = C * np.sin(x**2) + A
    y4 = np.cos(x**2)
    return np.array([y1, y2, y3, y4])

def test_solve_ivp_fixed_convergence():
    """Test convergence: error should decrease when the step size is halved."""
    x0, xf = 0.0, 1.0
    y0 = exact(x0)
    
    # Solve with h = 0.01 and h = 0.005
    sol1 = solve_ivp_fixed(f, x0, xf, y0, int(np.ceil(xf / 0.01)), method=METHOD)
    sol2 = solve_ivp_fixed(f, x0, xf, y0, int(np.ceil(xf / 0.005)), method=METHOD)

    err1 = np.linalg.norm(sol1["y"][-1] - exact(xf), np.inf)
    err2 = np.linalg.norm(sol2["y"][-1] - exact(xf), np.inf)

    # Error must decrease and remain within acceptable bounds
    assert err2 < err1
    assert err2 < 1e-3

def test_runge_rule_optimal_step():
    """Test optimal step size selection using Runge's rule."""
    x0, xf, y0 = 0.0, 1.0, exact(0.0)

    def err_est(h):
        # Estimate error using Runge's rule for a 2nd-order method (2^p - 1 = 3)
        y_h = solve_ivp_fixed(f, x0, xf, y0, int(np.ceil(xf / h)), method=METHOD)["y"][-1]
        y_half = solve_ivp_fixed(f, x0, xf, y0, int(np.ceil(xf / (h / 2))), method=METHOD)["y"][-1]
        return np.linalg.norm(y_h - y_half, np.inf) / 3.0

    h_cands = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    # Find the first candidate step size that satisfies the tolerance (1e-5)
    hopt = next((h for h in h_cands if err_est(h) <= 1e-5), None)
    assert hopt is not None
    
    # Ensure the actual error is acceptable for the found optimal step
    sol_opt = solve_ivp_fixed(f, x0, xf, y0, int(np.ceil(xf / hopt)), method=METHOD)
    assert np.linalg.norm(sol_opt["y"][-1] - exact(xf), np.inf) < 5e-4

def test_solve_ivp_adaptive():
    """Test the adaptive step size solver for correctness and tolerance adherence."""
    x0, xf, y0 = 0.0, 1.0, exact(0.0)
    sol = solve_ivp_adaptive(f, x0, xf, y0, method=METHOD, rtol=1e-6, atol=1e-12, h0=0.1)

    # Check if the solver reached the final point successfully without failing
    assert sol["x"][-1] >= xf - 1e-12
    # Ensure the global error is bounded within reasonable expectations
    assert np.linalg.norm(sol["y"][-1] - exact(xf), np.inf) < 1e-4
    # The solver should have accepted multiple steps
    assert len(sol["accepted_steps"]) > 0

def test_adaptive_efficiency_comparison():
    """Verify that a stricter tolerance requires more computational work."""
    x0, xf, y0 = 0.0, 1.0, exact(0.0)
    kwargs = dict(f=f, x0=x0, xf=xf, y0=y0, method=METHOD, atol=1e-12, h0=0.1, max_steps=50000)
    
    # Solve the system twice with different relative tolerances
    sol_loose = solve_ivp_adaptive(**kwargs, rtol=1e-4)
    sol_strict = solve_ivp_adaptive(**kwargs, rtol=1e-7)
    
    assert sol_strict["rhs_calls"] > sol_loose["rhs_calls"]