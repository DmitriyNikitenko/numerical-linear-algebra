import math
import src.numerical_lib.methods_eq_one_var as nm
from src.numerical_lib.methods_eq_one_var import (
    bisection, trisection, false_position, fp_mse,
    ridders, brent, secant, modified_secant,
    steffensen, modified_steffensen, newton, halley
)


# F(x) = e^x * sin(x) - 1 = 0
def f(x):
    return math.exp(x) * math.sin(x) - 1.0

def df(F, x):
    return math.exp(x) * (math.sin(x) + math.cos(x))

def ddf(F, x):
    return 2.0 * math.exp(x) * math.cos(x)

def test_all_12_methods():
    nm.dF = df
    nm.ddF = ddf

    TRUE_ROOT = 0.588532743981861
    TOL = 1e-9
    ERR = 1e-7  # Permissible error for assert

    # Initial conditions
    a, b = 0.0, 1.0  
    x0 = 1.0         
    x1 = 0.0         

    # Bisection
    root_bis, _ = bisection(f, a, b, tol=TOL)
    assert abs(root_bis - TRUE_ROOT) < ERR
    assert abs(f(root_bis)) < ERR

    # Trisection
    root_tri, _ = trisection(f, a, b, tol=TOL)
    assert abs(root_tri - TRUE_ROOT) < ERR
    assert abs(f(root_tri)) < ERR

    # False Position
    root_fp, _ = false_position(f, a, b, tol=TOL)
    assert abs(root_fp - TRUE_ROOT) < ERR
    assert abs(f(root_fp)) < ERR

    # FP-MSe
    root_fpmse, _ = fp_mse(f, a, b, tol=TOL)
    assert abs(root_fpmse - TRUE_ROOT) < ERR
    assert abs(f(root_fpmse)) < ERR

    # Ridders
    root_rid, _ = ridders(f, a, b, tol=TOL)
    assert abs(root_rid - TRUE_ROOT) < ERR
    assert abs(f(root_rid)) < ERR

    # VW-Brent
    root_brent, _ = brent(f, a, b, tol=TOL)
    assert abs(root_brent - TRUE_ROOT) < ERR
    assert abs(f(root_brent)) < ERR

    #  Secant
    root_sec, _ = secant(f, x0, x1, tol=TOL)
    assert abs(root_sec - TRUE_ROOT) < ERR
    assert abs(f(root_sec)) < ERR

    # Modified Secant
    root_msec, _ = modified_secant(f, x0, tol=TOL)
    assert abs(root_msec - TRUE_ROOT) < ERR
    assert abs(f(root_msec)) < ERR

    # Steffensen
    root_stef, _ = steffensen(f, x0, tol=TOL)
    assert abs(root_stef - TRUE_ROOT) < ERR
    assert abs(f(root_stef)) < ERR

    # Modified Steffensen
    root_mstef, _ = modified_steffensen(f, x0, tol=TOL)
    assert abs(root_mstef - TRUE_ROOT) < ERR
    assert abs(f(root_mstef)) < ERR

    # Newton 
    root_newt, _ = newton(f, x0, tol=TOL)
    assert abs(root_newt - TRUE_ROOT) < ERR
    assert abs(f(root_newt)) < ERR

    # Halley
    root_hal, _ = halley(f, x0, tol=TOL)
    assert abs(root_hal - TRUE_ROOT) < ERR
    assert abs(f(root_hal)) < ERR