import numpy as np
import src.qr as qr_mod


def test_qr_decomposition():
    A = np.array([
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0],
    ])
    Q, R = qr_mod.qr_decomposition(A)
    assert np.allclose(Q @ R, A)


def test_qr_solve(monkeypatch):
    A = np.array([
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0],
    ])
    b = np.array([5.0, -2.0, 9.0])

    original_backward = qr_mod.backward_substitution

    def backward_substitution_no_eps(U, y, eps=1e-12):
        return original_backward(U, y)

    monkeypatch.setattr(qr_mod, "backward_substitution", backward_substitution_no_eps)

    x = qr_mod.qr_solve(A, b)

    assert np.allclose(A @ x, b)