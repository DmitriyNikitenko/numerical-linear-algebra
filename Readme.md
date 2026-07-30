# Numerical Methods Library

This project is a comprehensive Python library designed for numerically solving a wide range of mathematical problems. Built on `numpy`, it provides implementations of algorithms for linear algebra, numerical integration, ordinary differential equations (ODE), nonlinear equations with one variable, and system of nonlinear equations.

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [How to Use](#how-to-use)
- [Installation](#installation)
- [Requirements](#requirements)


## Project Structure

```text
numerical-library/
├── src/
│   └─ numerical_lib/
│      ├─ integrals.py                  # Numerical integration
│      ├─ iterative_methods.py      # Iterative methods of SLAE
│      ├─ linear_system.py             # Direct methods of SLAE
│      ├─ lu.py                  # LU decomposition of matrices
│      ├─ methods_eq_one_var.py     # Roots of scalar equations
│      ├─ nonliniar.py         # Systems of nonlinear equations
│      ├─ ode.py                              # Solution of ODE
│      ├─ qr.py                  # QR decomposition of matrices
│      └─ utils.py                       # Additional functions
├─ tests/                                          # Unit tests
├─ setup.py              # Package build and dependencies setup
├─ README.md
└─ LICENSE
```


## Features
### Linear Algebra

This module provides tools for solving linear systems and analyzing matrices across several specialized components.

#### QR Decomposition
*   `qr_decomposition`: Computes QR decomposition of a square matrix via Householder reflection.
*   `qr_solve`: Solves $Ax = b$ using QR factorization.
*   `check_qr_decomposition`: Validates decomposition by verifying $A = QR$.

#### LU Decomposition
*   `lu_decomposition`: Computes LU decomposition with optional full pivoting.
*   `lu_solve`: Solves linear systems using LU decomposition.
*   `determinant`: Calculates matrix determinant from LU factors.
*   `inverse`: Computes matrix inverse.
*   `condition`: Calculates matrix condition number using the infinity norm.
*   `check_lu_decomposition`: Validates decomposition by verifying $PAQ = LU$.

#### Gaussian Elimination & Matrix Analysis
*   `gauss`: Reduces a system to stepwise form via Gaussian elimination.
*   `rank`: Determines matrix rank using Gaussian elimination.
*   `is_consistent`: Checks if a linear system is consistent.
*   `is_degenerate`: Checks if a matrix is degenerate.
*   `solve_singular`: Solves singular systems by setting free variables to zero.

#### Iterative Solvers & Error Estimation
*   `is_diagonally_dominant`: Verifies matrix diagonal dominance.
*   `jacobi` & `gauss_seidel`: Solves systems iteratively with optional convergence history.
*   `jacobi_iteration_matrix` & `gauss_seidel_iteration_matrix`: Constructs iteration matrices.
*   `apriori_iterations`: Estimates required iterations a priori.
*   `a_posteriori_bound`: Calculates an a posteriori error bound.

---

### Numerical Integration

This module provides the primary interfaces for computing definite integrals, including robust support for functions with boundary singularities.


*   `integrate`: The core function for computing numerical integrals with iterative precision control.
    *   `f` (Callable): The integrand function to be evaluated.
    *   `a` (float): The lower limit of integration.
    *   `b` (float): The upper limit of integration.
    *   `method` (str, default="midpoint"): The quadrature rule to apply. Supported options are `"midpoint"`, `"trapezoid"`, `"simpson"`, `"gauss"`.
    *   `N0` (int, default=1): The initial number of subintervals to divide the integration domain into.
    *   `eps` (float, default=1e-10): The target error tolerance for the stopping condition.
    *   `max_iter` (int, default=20): The maximum number of grid refinement iterations. In each iteration, the number of intervals is doubled.
    *   `exact` (Optional[float], default=None): The exact analytical value of the integral, if known. Used solely to compute and track the true relative error during iterations.
    *   `alpha` (Optional[float], default=None): The weight parameter characterizing a singularity at the lower limit `a`. Must be $0 \le \alpha < 1$.
    *   `beta` (Optional[float], default=None): The weight parameter characterizing a singularity at the upper limit `b`. Must be $0 \le \beta < 1$.

*   `integrate_weighted`: A specialized wrapper specifically designed for integrating functions with known boundary singularities.
    *   `f` (Callable): The integrand function to be evaluated.
    *   `a` (float): The lower limit of integration.
    *   `b` (float): The upper limit of integration.
    *   `alpha` (float): The weight parameter for the singularity at `a` <br> ($0 \le \alpha < 1$).
    *   `beta` (float): The weight parameter for the singularity at `b` <br>  ($0 \le \beta < 1$).
    *   `method` (str, default="gauss"): The quadrature rule to apply. Supported options are `"midpoint"`, `"trapezoid"`, `"simpson"`, `"gauss"`.
    *   `N0` (int, default=1): The initial number of subintervals to divide the integration domain into.
    *   `eps` (float, default=1e-10): The target error tolerance for the stopping condition.
    *   `max_iter` (int, default=20): The maximum number of grid refinement iterations.
    *   `exact` (Optional[float], default=None): The exact value of the integral for true error calculation.

---

### Differential Equations (ODE)

This module provides tools for numerically solving Initial Value Problems (IVPs) for ordinary differential equations (ODEs), featuring both fixed-step and adaptive-step integrators based on the Runge-Kutta family of methods.

#### Supported Integrators
The module supports several explicit Runge-Kutta methods defined by their Butcher tableaus, which can be selected via numerical aliases or short names:
*   **Midpoint** (`26`, `midpoint`): Second-order method.
*   **Heun** (`27`, `heun`): Explicit trapezoidal second-order method.
*   **Third-order Heun** (`28`, `rk3_heun`): Third-order method.
*   **Third-order Simpson-type RK** (`29`, `rk3_simpson`): Third-order method.
*   **Classical RK4** (`30`, `rk4_classic`): The standard fourth-order Runge-Kutta method.
*   **RK4 3/8-rule** (`31`, `rk4_38`): A fourth-order variant.

#### Core Functions

*   `rk_step`: Computes a single numerical integration step.
    *   `f` (Callable): The right-hand side function of the ODE system.
    *   `x` (float): The current independent variable value.
    *   `y` (float or array): The current state vector or scalar value.
    *   `h` (float): The step size.
    *   `method` (str): The solver method to utilize (defaults to `"rk4_classic"`).

*   `solve_ivp_fixed`: Solves an ODE over a defined interval using a uniform grid with a fixed step size.
    *   `f` (Callable): The right-hand side function.
    *   `x0` (float): The initial point of integration.
    *   `xf` (float): The final point of integration.
    *   `y0` (float or array): The initial conditions.
    *   `n_steps` (int): The exact number of sequential integration steps to take.
    *   `method` (str): The solver method to utilize.

*   `estimate_initial_step`: Estimates an optimal starting step size for adaptive solvers based on the derivative at the starting point and the method's order of accuracy.
    *   `tol` (float): The target error tolerance used for estimation.
    *   `max_step` (Optional[float]): An optional upper bound for the returned step size.

*   `solve_ivp_adaptive`: Integrates an ODE using automatic step size control (via step doubling) and Richardson extrapolation to refine the solution and ensure local error stays within limits.
    *   `rtol` (float): The relative local error tolerance (must be > 0).
    *   `atol` (float): The absolute local error tolerance (must be >= 0).
    *   `h0` (Optional[float]): The initial step size; if omitted, it is estimated automatically.
    *   `h_min` (float): The minimum allowed step size to prevent underflow.
    *   `h_max` (Optional[float]): The maximum allowed step size limit.
    *   `safety` (float): A safety multiplier (typically 0.9) applied when resizing the step to minimize the chance of future rejections.
    *   `max_steps` (int): A safeguard parameter that caps the maximum number of allowed iterations to prevent infinite loops.


---

### Nonlinear Equations

This module provides a suite of numerical solvers for root-finding in single-variable scalar nonlinear equations as well as systems of multivariate nonlinear equations.


#### 1. Scalar Root-Finding Algorithms

All scalar root-finding functions return a tuple `(root, iterations)`.

 Methods are categorized by their convergence guarantees:
*   **Bracketing methods** are guaranteed to converge given an interval `[a, b]` where $f(a) \cdot f(b) < 0$. Parameters: `(F, a, b, tol, maxiter)`.
*   **Open methods** offer no global convergence guarantee and require initial estimates near the root. Parameters: `(F, x0, tol, maxiter)`. *(Exception: `secant` requires `x0, x1`)*.

<br>

*   `bisection`: Bracketing, **Linear (order 1)**.
*   `trisection`: Bracketing, **Linear (order 1)**.
*   `false_position`: Bracketing, **Linear (order 1)**.
*   `newton`: Open, **Quadratic (order 2)**.
*   `secant`: Open, **Superlinear (order $\approx 1.618$)**.
*   `modified_secant`: Open, **Superlinear**.
*   `fp_mse`: Hybrid bracketing, **Superlinear**.
*   `halley`: Open, **Cubic (order 3)**.
*   `ridders`: Bracketing, **Superlinear (order $\approx 1.414$)**.
*   `brent`: Hybrid bracketing, **Superlinear**.
*   `steffensen`: Open (derivative-free), **Quadratic (order 2)**.
*   `modified_steffensen`: Open (derivative-free), **Quadratic (order 2)**.


#### 2. Multivariate Systems Solver

*   `newton_system`: Solves a system of non-linear equations $F(\mathbf{x}) = \mathbf{0}$ using LU decomposition. Returns a dictionary with metrics like the solution, iterations, and Jacobian recomputations.
    <br>`newton_system(F, J, x0, eps, max_iter, mode, m, k)`
   *   `F` (Callable): The non-linear system of equations.
   *   `J` (Callable): The Jacobian matrix.
   *   `x0` (array-like): The initial guess for the solution vector.
   *   `eps` (float), `max_iter` (int): Convergence tolerance.
   *   `mode` (str): Jacobian evaluation strategy: `"full"` (update every step), `"modified"` (compute once at $x_0$), or `"hybrid"`.
   *   `m`, `k` (int): Parameters for `"hybrid"` mode: perform `k` full Newton steps initially, then recompute the Jacobian every `m` steps.

---


## How to Use

Once installed, you can import and use functions from `numerical_lib` in any Python script:

### Example: Solving a Linear System (LU Decomposition)
```python
import numpy as np
from numerical_lib.lu import lu_solve 

# Define matrix A and vector b
A = np.array([[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]])
b = np.array([1, -2, 0])

# Solve the system
x = lu_solve(A, b)
print("Solution:", x)
```

### Example: Finding a Root
```python
from numerical_lib.roots import brent

def f(x):
    return x**3 - x - 2

# Search for the root in the interval [1, 2]
root, iterations = brent(f, 1.0, 2.0, tol=1e-10)
print(f"Root: {root}, iterations: {iterations}")
```


## Installation

You can install the library directly from GitHub or clone it for local development.

### Option 1: Install directly via pip for users
If you just want to use the library without modifying the code, you can install it directly from the repository:
```bash
pip install git+https://github.com/DmitriyNikitenko/numerical-library.git
```

### Option 2: Clone for development
If you want to contribute, run tests, or modify the source code:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DmitriyNikitenko/numerical-library
   cd numerical-library
   ```

2. **Install the library in editable mode:**
   ```bash
   pip install -e .
   ```



## Requirements

This library was developed and tested using **Python 3.10.12** and **NumPy 2.2.6**. To ensure correct execution, please install at least these versions:
   *   **Python**: `>= 3.10.12`
   *   **NumPy**: `>= 2.2.6`