import numpy as np
import pytest

from src.integrals import integrate


# Ordinary integrals: smoke tests

def test_midpoint_polynomial():
    # ∫_0^1 x^2 dx = 1/3
    f = lambda x: x**2
    res = integrate(f, 0, 1, method="midpoint", N0=1, eps=1e-6, exact=1 / 3)
    assert abs(res["result"] - 1 / 3) < 1e-6


def test_trapezoid_sin():
    # ∫_0^π sin(x) dx = 2
    f = np.sin
    res = integrate(f, 0, np.pi, method="trapezoid", N0=4, eps=1e-6, exact=2.0)
    assert abs(res["result"] - 2.0) < 1e-6


def test_simpson_known():
    # ∫_0^2 (3x^2 + 2x + 1) dx = 14
    f = lambda x: 3 * x**2 + 2 * x + 1
    res = integrate(f, 0, 2, method="simpson", N0=2, eps=1e-8, exact=14.0)
    assert abs(res["result"] - 14.0) < 1e-8


def test_gauss_known():
    # ∫_0^1 x^4 dx = 1/5
    f = lambda x: x**4
    res = integrate(f, 0, 1, method="gauss", N0=1, eps=1e-10, exact=1 / 5)
    assert abs(res["result"] - 1 / 5) < 1e-10


def test_convergence_and_runge():
    f = np.exp
    exact = np.e - 1.0

    res1 = integrate(f, 0, 1, method="trapezoid", N0=1, eps=1e-2, exact=exact)
    res2 = integrate(f, 0, 1, method="trapezoid", N0=1, eps=1e-6, exact=exact)

    assert res2["N"] >= res1["N"]
    assert abs(res2["result"] - exact) <= abs(res1["result"] - exact)


def test_invalid_inputs():
    f = lambda x: x

    with pytest.raises(ValueError):
        integrate(f, 1, 0, method="midpoint")

    with pytest.raises(ValueError):
        integrate(f, 0, 1, method="unknown")

    with pytest.raises(ValueError):
        integrate(f, 0, 1, method="simpson", N0=0)

    with pytest.raises(ValueError):
        integrate(f, 0, 1, method="trapezoid", eps=0)


# J = ∫_a^b f(x) / ((x-a)^alpha (b-x)^beta) dx

def f1(x):
    return 2 * np.cos(2.5 * x) * np.exp(x / 3) + 4 * np.sin(3.5 * x) * np.exp(-3 * x) + x


def f2(x):
    return 3 * np.cos(0.5 * x) * np.exp(x / 4) + 5 * np.sin(2.5 * x) * np.exp(-x / 3) + 2 * x


def f3(x):
    return 2.5 * np.cos(2 * x) * np.exp(2 * x / 3) + 4 * np.sin(3.5 * x) * np.exp(-3 * x) + 3 * x


def f4(x):
    return 3 * np.cos(3.5 * x) * np.exp(4 * x / 3) + 2 * np.sin(3.5 * x) * np.exp(-2 * x / 3) + 4 * x


def f5(x):
    return np.cos(1.5 * x) * np.exp(2 * x / 3) + 3 * np.sin(5.5 * x) * np.exp(-2 * x) + 2


def f6(x):
    return 4 * np.cos(0.5 * x) * np.exp(-5 * x / 4) + 2 * np.sin(4.5 * x) * np.exp(x / 8) + 2


def f7(x):
    return 4.5 * np.cos(7 * x) * np.exp(-2 * x / 3) + 1.4 * np.sin(1.5 * x) * np.exp(-x / 3) + 3


def f8(x):
    return 3.7 * np.cos(1.5 * x) * np.exp(-4 * x / 3) + 2.4 * np.sin(4.5 * x) * np.exp(2 * x / 3) + 4


def f9(x):
    return 3 * np.cos(1.5 * x) * np.exp(x / 4) + 4 * np.sin(3.5 * x) * np.exp(-3 * x) + 4 * x


def f10(x):
    return 1.3 * np.cos(3.5 * x) * np.exp(2 * x / 3) + 6 * np.sin(4.5 * x) * np.exp(-x / 8) + 5 * x


def f11(x):
    return 0.5 * np.cos(2 * x) * np.exp(2 * x / 5) + 2.4 * np.sin(1.5 * x) * np.exp(-6 * x) + 6 * x


def f12(x):
    return 4 * np.cos(2.5 * x) * np.exp(4 * x / 7) + 2.5 * np.sin(5.5 * x) * np.exp(-3 * x / 5) + 4.3 * x


def f13(x):
    return 2 * np.cos(3.5 * x) * np.exp(5 * x / 3) + 3 * np.sin(1.5 * x) * np.exp(-4 * x) + 3


def f14(x):
    return 3 * np.cos(2.5 * x) * np.exp(7 * x / 4) + 5 * np.sin(0.5 * x) * np.exp(3 * x / 8) + 4


def f15(x):
    return 3.5 * np.cos(0.7 * x) * np.exp(-5 * x / 3) + 2.4 * np.sin(5.5 * x) * np.exp(-3 * x / 4) + 5


def f16(x):
    return 2.7 * np.cos(3.5 * x) * np.exp(-7 * x / 3) + 4.4 * np.sin(2.5 * x) * np.exp(5 * x / 3) + 2


def f17(x):
    return 6 * np.cos(1.5 * x) * np.exp(5 * x / 3) + 2 * np.sin(0.5 * x) * np.exp(-1.3 * x) + 5.4 * x


def f18(x):
    return 4 * np.cos(2.5 * x) * np.exp(5 * x / 4) + 2.5 * np.sin(1.5 * x) * np.exp(-2 * x / 7) + 5 * x


def f19(x):
    return 0.5 * np.cos(3 * x) * np.exp(2 * x / 5) + 4 * np.sin(3.5 * x) * np.exp(-3 * x) + 3 * x


def f20(x):
    return 1.5 * np.cos(3.7 * x) * np.exp(4 * x / 7) + 3 * np.sin(2.5 * x) * np.exp(3 * x / 4) + 3 * x


def f21(x):
    return 3 * np.cos(2.5 * x) * np.exp(4 * x / 3) + 4 * np.sin(5.5 * x) * np.exp(-3.5 * x) + 3


def f22(x):
    return 5 * np.cos(0.3 * x) * np.exp(-7 * x / 4) + 7 * np.sin(0.5 * x) * np.exp(2 * x / 3) + 4


def f23(x):
    return 2.5 * np.cos(5.7 * x) * np.exp(-4 * x / 3) + 2.4 * np.sin(2.5 * x) * np.exp(-x) + 7


def f24(x):
    return 5.7 * np.cos(2.5 * x) * np.exp(-4 * x / 7) + 4.4 * np.sin(4.3 * x) * np.exp(2 * x / 7) + 5


REFERENCE_VALUES = {
    1: 7.07703143799579361026,
    2: 11.83933565874812191865,
    3: 3.57886153604053991544,
    4: -41.88816344003630606891,
    5: 10.65722906811476196545,
    6: 10.83954510946909397741,
    7: 4.46151270533119411284,
    8: 1.18514197495624182491,
    9: 20.73027110955223102602,
    10: 24.14209267859915860831,
    11: 18.60294785731848208627,
    12: 57.48462064655285571821,
    13: 32.21951452884234295709,
    14: 348.81813442539113603636,
    15: 27.56649553650691538578,
    16: -3246.87592632732889436788,
    17: 2308.28752445280943613281,
    18: 78.38144689028315358350,
    19: 8.56553422240763400676,
    20: 161.78429047482359453211,
    21: -262.76276057047037253923,
    22: 69.34894027882668315183,
    23: 28.98579534502018413363,
    24: -4.24939310114503594176,
}

CASES = [
    (1, f1, 1.5, 3.3, 1 / 3, 0, REFERENCE_VALUES[1]),
    (2, f2, 1.7, 3.2, 0, 1 / 4, REFERENCE_VALUES[2]),
    (3, f3, 0.1, 2.3, 1 / 5, 0, REFERENCE_VALUES[3]),
    (4, f4, 1.0, 3.0, 0, 1 / 6, REFERENCE_VALUES[4]),
    (5, f5, 2.5, 4.3, 2 / 7, 0, REFERENCE_VALUES[5]),
    (6, f6, 1.3, 2.2, 0, 5 / 6, REFERENCE_VALUES[6]),
    (7, f7, 2.1, 3.3, 2 / 5, 0, REFERENCE_VALUES[7]),
    (8, f8, 1.8, 2.3, 0, 3 / 5, REFERENCE_VALUES[8]),
    (9, f9, 2.5, 3.3, 2 / 3, 0, REFERENCE_VALUES[9]),
    (10, f10, 0.7, 3.2, 0, 1 / 4, REFERENCE_VALUES[10]),
    (11, f11, 1.1, 2.5, 2 / 5, 0, REFERENCE_VALUES[11]),
    (12, f12, 1.8, 2.9, 0, 4 / 7, REFERENCE_VALUES[12]),
    (13, f13, 1.5, 2.3, 1 / 5, 0, REFERENCE_VALUES[13]),
    (14, f14, 2.3, 2.9, 0, 2 / 5, REFERENCE_VALUES[14]),
    (15, f15, 1.1, 2.3, 4 / 5, 0, REFERENCE_VALUES[15]),
    (16, f16, 2.8, 4.3, 0, 3 / 7, REFERENCE_VALUES[16]),
    (17, f17, 3.5, 3.7, 2 / 3, 0, REFERENCE_VALUES[17]),
    (18, f18, 2.7, 3.2, 0, 3 / 4, REFERENCE_VALUES[18]),
    (19, f19, 1.1, 2.3, 2 / 5, 0, REFERENCE_VALUES[19]),
    (20, f20, 1.5, 3.0, 0, 5 / 6, REFERENCE_VALUES[20]),
    (21, f21, 2.5, 4.3, 2 / 7, 0, REFERENCE_VALUES[21]),
    (22, f22, 0.5, 2.2, 0, 3 / 5, REFERENCE_VALUES[22]),
    (23, f23, 0.2, 3.1, 3 / 5, 0, REFERENCE_VALUES[23]),
    (24, f24, 0.8, 1.3, 0, 4 / 7, REFERENCE_VALUES[24]),
]

# This is a parameterization decorator from the pytest library. It turns a single test function into a suite of tests.
@pytest.mark.parametrize("case_id, func, a, b, alpha, beta, exact", CASES)  
def test_weighted_reference_integrals(case_id, func, a, b, alpha, beta, exact):
    res = integrate(
        func,
        a,
        b,
        method="gauss",
        N0=16,
        eps=1e-12,
        max_iter=30,
        exact=exact,
        alpha=alpha,
        beta=beta,
    )

    assert np.isclose(
        res["result"],
        exact,
        rtol=1e-8,
        atol=1e-8,
    ), f"Integral #{case_id} failed: got {res['result']}, expected {exact}"


def test_weighted_invalid_inputs():
    with pytest.raises(ValueError):
        integrate(f1, 1.5, 3.3, alpha=1 / 3)

    with pytest.raises(ValueError):
        integrate(f1, 3.3, 1.5, alpha=1 / 3, beta=0)

    with pytest.raises(ValueError):
        integrate(f1, 1.5, 3.3, method="unknown", alpha=1 / 3, beta=0)

    with pytest.raises(ValueError):
        integrate(f1, 1.5, 3.3, method="gauss", alpha=-0.1, beta=0)

    with pytest.raises(ValueError):
        integrate(f1, 1.5, 3.3, method="gauss", alpha=0, beta=1.2)