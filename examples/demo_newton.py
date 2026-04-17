import numpy as np
from src.nonlinear import newton_system

# System F(x)
def F(x):
    x1,x2,x3,x4,x5,x6,x7,x8,x9,x10 = x

    return np.array([
        np.cos(x2 * x1) - np.exp(-3 * x3) + x4 * x5**2 - x6 - np.sinh(2 * x8) * x9 + 2 * x10 + 2.000433974165385440,
        np.sin(x2 * x1) + x3 * x9 * x7 - np.exp(-x10 + x6) + 3 * x5**2 - x6 * (x8 + 1) + 10.886272036407019994,
        x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904,
        2 * np.cos(-x9 + x4) + x5 / (x3 + x1) - np.sin(x2**2) + np.cos(x7 * x10)**2 - x8 - 0.1707472705022304757,
        np.sin(x5) + 2 * x8 * (x3 + x1) - np.exp(-x7 * (-x10 + x6)) + 2 * np.cos(x2) - 1.0 / (-x9 + x4) - 0.3685896273101277862,
        np.exp(x1 - x4 - x9) + x5**2 / x8 + 0.5 * np.cos(3 * x10 * x2) - x6 * x3 + 2.0491086016771875115,
        x2**3 * x7 - np.sin(x10 / x5 + x8) + (x1 - x6) * np.cos(x4) + x3 - 0.7380430076202798014,
        x5 * (x1 - 2 * x6)**2 - 2 * np.sin(-x9 + x3) + 1.5 * x4 - np.exp(x2 * x7 + x10) + 3.5668321989693809040,
        7 / x6 + np.exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499,
        x10 * x1 + x9 * x2 - x8 * x3 + np.sin(x4 + x5 + x6) * x7 - 0.78238095238095238096
    ])


# Jacobian J(x)
def J(x):
    x1,x2,x3,x4,x5,x6,x7,x8,x9,x10 = x

    return np.array([
        [-x2*np.sin(x2*x1), -x1*np.sin(x2*x1), 3*np.exp(-3*x3), x5**2, 2*x4*x5,
         -1, 0, -2*np.cosh(2*x8)*x9, -np.sinh(2*x8), 2],

        [x2*np.cos(x2*x1), x1*np.cos(x2*x1), x9*x7, 0, 6*x5,
         -np.exp(-x10+x6)-x8-1, x3*x9, -x6, x3*x7, np.exp(-x10+x6)],

        [1,-1,1,-1,1,-1,1,-1,1,-1],

        [-x5/(x3+x1)**2, -2*x2*np.cos(x2**2), -x5/(x3+x1)**2, -2*np.sin(-x9+x4),
         1/(x3+x1), 0, -2*np.cos(x7*x10)*x10*np.sin(x7*x10), -1,
         2*np.sin(-x9+x4), -2*np.cos(x7*x10)*x7*np.sin(x7*x10)],

        [2*x8, -2*np.sin(x2), 2*x8, 1/(-x9+x4)**2, np.cos(x5),
         x7*np.exp(-x7*(-x10+x6)), -(x10-x6)*np.exp(-x7*(-x10+x6)), 2*x3+2*x1,
         -1/(-x9+x4)**2, -x7*np.exp(-x7*(-x10+x6))],

        [np.exp(x1-x4-x9), -1.5*x10*np.sin(3*x10*x2), -x6, -np.exp(x1-x4-x9),
         2*x5/x8, -x3, 0, -x5**2/x8**2, -np.exp(x1-x4-x9), -1.5*x2*np.sin(3*x10*x2)],

        [np.cos(x4), 3*x2**2*x7, 1, -(x1-x6)*np.sin(x4),
         x10/x5**2*np.cos(x10/x5+x8), -np.cos(x4), x2**3,
         -np.cos(x10/x5+x8), 0, -1/x5*np.cos(x10/x5+x8)],

        [2*x5*(x1-2*x6), -x7*np.exp(x2*x7+x10), -2*np.cos(-x9+x3), 1.5,
         (x1-2*x6)**2, -4*x5*(x1-2*x6), -x2*np.exp(x2*x7+x10), 0,
         2*np.cos(-x9+x3), -np.exp(x2*x7+x10)],

        [-3, -2*x8*x10*x7, 0, np.exp(x5+x4), np.exp(x5+x4),
         -7/x6**2, -2*x2*x8*x10, -2*x2*x10*x7, 3, -2*x2*x8*x7],

        [x10, x9, -x8, np.cos(x4+x5+x6)*x7, np.cos(x4+x5+x6)*x7,
         np.cos(x4+x5+x6)*x7, np.sin(x4+x5+x6), -x3, x2, x1]
    ])


# Initial points
def x0_default():
    return np.array([0.5,0.5,1.5,-1.0,-0.5,1.5,0.5,-0.5,1.5,-1.5])


def x0_variant():
    x = x0_default()
    x[4] = -0.2
    return x


# Runner
def run_case(name, x0, **kwargs):
    res = newton_system(F, J, x0, **kwargs)

    print(f"\n{name}")
    print("iterations:", res["iterations"])
    print("jacobian recomputations:", res["jacobian_recomputations"])
    print("residual:", res["residual_norm"])
    print("time:", res["time"])
    print("x:", res["x"])


if __name__ == "__main__":
    x0 = x0_default()

    run_case("a) full Newton", x0, mode="full")
    run_case("b) modified Newton", x0, mode="modified")
    run_case("c) k=5", x0, mode="hybrid", k=5, m=np.inf)
    run_case("d) m=3", x0, mode="hybrid", k=0, m=3)
    run_case("e) m=3, k=5", x0, mode="hybrid", k=5, m=3)
    run_case("f) x5=-0.2", x0_variant(), mode="full")