import numpy as np
from src.nonlinear import newton_system
from src.utils import norm_inf_vec


# x - sin(x) = 0.25
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

# System
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