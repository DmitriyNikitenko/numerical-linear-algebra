import numpy as np
from src.nonlinear import newton_system
from src.utils import norm_inf_vec


def simple_F(x):
    return np.array([x[0]**2 - 2])


def simple_J(x):
    return np.array([[2 * x[0]]])


def test_newton_scalar():
    x0 = np.array([1.0])

    res = newton_system(simple_F, simple_J, x0)

    x = res["x"]
    assert abs(x[0] - np.sqrt(2)) < 1e-6


def test_modes_consistency():
    x0 = np.array([1.0])

    res_full = newton_system(simple_F, simple_J, x0, mode="full")
    res_mod = newton_system(simple_F, simple_J, x0, mode="modified")

    assert abs(res_full["x"][0] - res_mod["x"][0]) < 1e-6


def test_convergence():
    def F(x):
        return np.array([
            x[0] + x[1] - 2,
            x[0] - x[1]
        ])

    def J(x):
        return np.array([
            [1, 1],
            [1, -1]
        ])

    x0 = np.array([0.0, 0.0])

    res = newton_system(F, J, x0)

    assert norm_inf_vec(F(res["x"])) < 1e-8