import numpy as np
from src.nonlinear import newton_system
from src.utils import norm_inf_vec


# Small scalar equation
def F1(x):
    return np.array([
        x[0] - np.sin(x[0]) - 0.25
    ])

def J1(x):
    return np.array([
        [1 - np.cos(x[0])]
    ])

def test_scalar_equation():
    x0 = np.array([1.0])
    res = newton_system(F1, J1, x0)

    x = res["x"]
    assert abs(F1(x)[0]) < 1e-6

# Small nonlinear system
def F2(x):
    return np.array([
        np.sin(x[0] + 1) - x[1] - 1.2,
        2 * x[0] + np.cos(x[1]) - 2
    ])

def J2(x):
    return np.array([
        [np.cos(x[0] + 1), -1],
        [2, -np.sin(x[1])]
    ])

def test_system_solution():
    x0 = np.array([0.5, 0.5])
    res = newton_system(F2, J2, x0)

    assert norm_inf_vec(F2(res["x"])) < 1e-6

def test_modes_consistency_system():
    x0 = np.array([0.5, 0.5])

    res_full = newton_system(F2, J2, x0, mode="full")
    res_mod = newton_system(F2, J2, x0, mode="modified")

    assert norm_inf_vec(res_full["x"] - res_mod["x"]) < 1e-6

# Large 10x10 nonlinear system
def F(x):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x

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

def J(x):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x

    return np.array([
        [-x2 * np.sin(x2 * x1), -x1 * np.sin(x2 * x1), 3 * np.exp(-3 * x3), x5**2, 2 * x4 * x5,
         -1, 0, -2 * np.cosh(2 * x8) * x9, -np.sinh(2 * x8), 2],

        [x2 * np.cos(x2 * x1), x1 * np.cos(x2 * x1), x9 * x7, 0, 6 * x5,
         -np.exp(-x10 + x6) - x8 - 1, x3 * x9, -x6, x3 * x7, np.exp(-x10 + x6)],

        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],

        [-x5 / (x3 + x1)**2, -2 * x2 * np.cos(x2**2), -x5 / (x3 + x1)**2, -2 * np.sin(-x9 + x4),
         1 / (x3 + x1), 0, -2 * np.cos(x7 * x10) * x10 * np.sin(x7 * x10), -1,
         2 * np.sin(-x9 + x4), -2 * np.cos(x7 * x10) * x7 * np.sin(x7 * x10)],

        [2 * x8, -2 * np.sin(x2), 2 * x8, 1 / (-x9 + x4)**2, np.cos(x5),
         x7 * np.exp(-x7 * (-x10 + x6)), -(x10 - x6) * np.exp(-x7 * (-x10 + x6)), 2 * x3 + 2 * x1,
         -1 / (-x9 + x4)**2, -x7 * np.exp(-x7 * (-x10 + x6))],

        [np.exp(x1 - x4 - x9), -1.5 * x10 * np.sin(3 * x10 * x2), -x6, -np.exp(x1 - x4 - x9),
         2 * x5 / x8, -x3, 0, -x5**2 / x8**2, -np.exp(x1 - x4 - x9), -1.5 * x2 * np.sin(3 * x10 * x2)],

        [np.cos(x4), 3 * x2**2 * x7, 1, -(x1 - x6) * np.sin(x4),
         x10 / x5**2 * np.cos(x10 / x5 + x8), -np.cos(x4), x2**3,
         -np.cos(x10 / x5 + x8), 0, -1 / x5 * np.cos(x10 / x5 + x8)],

        [2 * x5 * (x1 - 2 * x6), -x7 * np.exp(x2 * x7 + x10), -2 * np.cos(-x9 + x3), 1.5,
         (x1 - 2 * x6)**2, -4 * x5 * (x1 - 2 * x6), -x2 * np.exp(x2 * x7 + x10), 0,
         2 * np.cos(-x9 + x3), -np.exp(x2 * x7 + x10)],

        [-3, -2 * x8 * x10 * x7, 0, np.exp(x5 + x4), np.exp(x5 + x4),
         -7 / x6**2, -2 * x2 * x8 * x10, -2 * x2 * x10 * x7, 3, -2 * x2 * x8 * x7],

        [x10, x9, -x8, np.cos(x4 + x5 + x6) * x7, np.cos(x4 + x5 + x6) * x7,
         np.cos(x4 + x5 + x6) * x7, np.sin(x4 + x5 + x6), -x3, x2, x1]
    ])

def x0_default():
    return np.array([0.5, 0.5, 1.5, -1.0, -0.5, 1.5, 0.5, -0.5, 1.5, -1.5])

def x0_variant():
    x = x0_default()
    x[4] = -0.2
    return x

def test_large_system_full_newton():
    x0 = x0_variant()
    res = newton_system(F, J, x0, mode="full")

    assert norm_inf_vec(F(res["x"])) < 1e-6

def test_large_system_hybrid_newton():
    x0 = x0_variant()
    res = newton_system(F, J, x0, mode="hybrid", k=5, m=3)

    assert norm_inf_vec(F(res["x"])) < 1e-6

def test_large_system_modes_consistency():
    x0 = x0_variant()

    res_full = newton_system(F, J, x0, mode="full")
    res_mod = newton_system(F, J, x0, mode="hybrid")

    assert norm_inf_vec(res_full["x"] - res_mod["x"]) < 1e-5