import math
import numpy as np


# Bisection
def bisection(F, a, b, tol=1e-12, maxiter=100):
    fa, fb = F(a), F(b)
    assert fa * fb < 0, "Bisection: no sign change on the interval"
    it = 0 # Iteration counter
    while (b - a) / 2 > tol and it < maxiter:
        # Calculate the midpoint of the interval
        m = a + (b - a) / 2 # (a + b) / 2
        fm = F(m)

        # Update interval boundaries and iteration counter
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
        it += 1
    return (a + (b - a) / 2), it

# Trisection
def trisection(F, a, b, tol=1e-12, maxiter=100):
    fa, fb = F(a), F(b)
    assert fa * fb < 0, "Trisection: no sign change on the interval"
    it = 0 # Iteration counter
    while (b - a) > tol and it < maxiter:
        # Calculate reference points
        x1 = a + (b - a) / 3
        x2 = a + 2 * (b - a) / 3
        f1, f2 = F(x1), F(x2)

        # Update interval boundaries and iteration counter
        if fa * f1 <= 0:
            b, fb = x1, f1
        elif f1 * f2 <= 0:
            a, fa = x1, f1
            b, fb = x2, f2
        else:
            a, fa = x2, f2
        it += 1
    return (a + (b - a) / 2), it

# False Position
def false_position(F, a, b, tol=1e-12, maxiter=100):
    fa, fb = F(a), F(b)
    assert fa * fb < 0, "False position: no sign change on the interval"
    it = 0 # Iteration counter
    c = a
    while it < maxiter:
        c = b - fb * (b - a) / (fb - fa) # Calculate new approximation
        fc = F(c)
        # Check stopping criterion by residual
        if abs(fc) < tol:
            return c, it + 1
        # Update interval boundaries and iteration counter
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        it += 1
    return c, it

# Newton
def newton(F, x0, tol=1e-12, maxiter=100):
    x = x0 # Initial approximation
    it = 0 # Iteration counter
    while it < maxiter:
        fx = F(x)
        # Check stopping criterion by residual
        if abs(fx) < tol:
            return x, it
        dfx = dF(F, x)
        xnew = x - fx / dfx # Calculate new approximation
        # Check stopping criterion by argument increment
        if abs(xnew - x) < tol:
            return xnew, it + 1
        # Update current approximation and iteration counter
        x = xnew
        it += 1
    return x, it

# Secant
def secant(F, x0, x1, tol=1e-12, maxiter=100):
    f0, f1 = F(x0), F(x1)
    it = 0 # Iteration counter
    while it < maxiter:
        denom = f1 - f0 # Difference of function values
        assert denom != 0.0, "Secant: zero denominator"
        x2 = x1 - f1 * (x1 - x0) / denom # Calculate new approximation
        # Check stopping criteria
        if abs(x2 - x1) < tol or abs(F(x2)) < tol:
            return x2, it + 1
        # Update current approximation and iteration counter
        x0, f0 = x1, f1
        x1, f1 = x2, F(x2)
        it += 1
    return x1, it

# Modified Secant
def modified_secant(F, x0, tol=1e-12, maxiter=100):
    x = x0 # Initial approximation
    it = 0 # Iteration counter 
    while it < maxiter:
        fx = F(x) 
        # Check stopping criterion by residual 
        if abs(fx) < tol:
            return x, it
        d = fx
        denom = F(x + d) - fx # Difference of function values
        assert denom != 0.0, "Modified Secant: zero denominator"
        xnew = x - d * fx / denom # Calculate new approximation
        # Check stopping criterion by argument increment
        if abs(xnew - x) < tol:
            return xnew, it + 1
        # Update current approximation and iteration counter
        x = xnew
        it += 1
    return x, it

# FP-MSe 
def fp_mse(F, a, b, drel=1e-6, tol=1e-12, maxiter=100):
    fa, fb = F(a), F(b)
    assert fa * fb < 0, "FP-MSe: no sign change on the interval"
    it = 0
    while it < maxiter:
        it += 1

        # False position step
        x_fp = a - fa * (b - a) / (fb - fa)
        fx_fp = F(x_fp)

        # Stopping criterion by residual
        if abs(fx_fp) < tol:
            return x_fp, it
        else:        
            # Modified secant step
            d = fx_fp
            denom = F(x_fp + d) - fx_fp
            assert denom != 0.0, "FP-MSe: zero denominator"
            x_mse = x_fp - d * fx_fp / denom
            fx_mse = F(x_mse)
            
            # Select the best new approximation
            if a < x_mse < b and abs(fx_mse) < abs(fx_fp):
                if fa * fx_mse < 0:
                    b, fb = x_mse, fx_mse
                else: 
                    a, fa = x_mse, fx_mse
            else:
                if fa * fx_fp < 0:
                    b, fb = x_fp, fx_fp
                else:
                    a, fa = x_fp, fx_fp
                    
    return (a + b) / 2, maxiter

# Halley
def halley(F, x0, tol=1e-12, maxiter=100):
    x = x0 # Initial approximation
    it = 0 # Iteration counter
    while it < maxiter:
        fx = F(x)
        # Check stopping criterion by residual
        if abs(fx) < tol:
            return x, it
        # Calculate 1st and 2nd derivatives
        dfx = dF(F, x)
        ddfx = ddF(F, x)
        denom = 2 * dfx**2 - fx * ddfx
        assert denom != 0.0, "Halley: zero denominator"
        xnew = x - (2 * fx * dfx) / denom # Calculate new approximation
        # Check stopping criterion by argument increment
        if abs(xnew - x) < tol:
            return xnew, it + 1
        # Update current approximation and iteration counter
        x = xnew
        it += 1
    return x, it

# Ridders
def ridders(F, a, b, tol=1e-12, maxiter=100):
    fa, fb = F(a), F(b)
    assert fa * fb < 0, "Ridders: no sign change on the interval"
    it = 0 # Iteration counter
    while it < maxiter:
        # Calculate values required for the new approximation
        m = a + (b - a) / 2 # (a + b) / 2
        fm = F(m)
        s2 = fm**2 - fa * fb
        if s2 <= 0:
            return m, it + 1
        s = math.sqrt(s2)
        x = m + (m - a) * np.sign(fa - fb) * fm / s # Calculate new approximation
        fx = F(x)

        # Check stopping criteria
        if abs(fx) < tol or abs(b - a) < tol:
            return x, it + 1

        # Select the best new approximation
        if fm * fx < 0:
            a, fa = m, fm
            b, fb = x, fx
        elif fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx

        it += 1
    return (a + (b - a) / 2), it

# VW-Brent
def brent(F, a, b, tol=1e-12, maxiter=100):
    fa, fb = F(a), F(b)
    assert fa * fb < 0, "Brent: no sign change on the interval"

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    d_prev = a # Variable to track the previous step
    mflag = True # The algorithm used the bisection method in the previous iteration
    s = b

    for it in range(1, maxiter + 1):
        # Calculate parameters R, S, T
        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            R = fb / fc
            S = fb / fa
            T = fa / fc
            
            P = S * (T * (R - T) * (c - b) - (1.0 - R) * (b - a))
            Q = (T - 1.0) * (R - 1.0) * (S - 1.0)
        else:
            # Secant method
            S = fb / fa
            P = S * (c - b)
            Q = S - 1.0

        # Potential new approximation (x = b + P/Q)
        d = P / Q
        s = b + d

        # Conditions for switching to the bisection method
        low = min(a, b)
        high = max(a, b)

        cond1 = not (low <= s <= high)                      # Out of bounds
        cond2 = mflag and abs(d) >= abs(b - c) / 2          # Interpolation step did not halve the interval (after bisection)
        cond3 = not mflag and abs(d) >= abs(c - d_prev) / 2 # Interpolation step did not halve the interval (after interpolation)
        cond4 = mflag and abs(b - c) < tol                  # Current interval is already too small (at the error level)
        cond5 = not mflag and abs(c - d_prev) < tol         # Previous step was too small (protection against infinite loop)

        if cond1 or cond2 or cond3 or cond4 or cond5:
            # Bisection method
            s = (a + (b - a) / 2)
            mflag = True
        else:
            mflag = False

        fs = F(s)
        d_prev, c = c, b
        fc = fb

        # Update localization interval
        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        # Point b must always be the best approximation
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        # Check stopping criteria
        if abs(fb) < tol or abs(b - a) < tol:
            return b, it

    return b, maxiter

# Steffensen
def steffensen(F, x0, tol=1e-12, maxiter=100):
    x = x0 # Initial approximation
    it = 0 # Iteration counter
    while it < maxiter:
        fx = F(x)
        # Check stopping criterion by residual
        if abs(fx) < tol:
            return x, it
        denom = F(x + fx) - fx
        assert denom != 0.0, "Steffensen: zero denominator"
        xnew = x - fx**2 / denom
        # Check stopping criterion by argument increment
        if abs(xnew - x) < tol:
            return xnew, it + 1
        # Update current approximation and iteration counter
        x = xnew
        it += 1
    return x, it

# Modified Steffensen
def modified_steffensen(F, x0, tol=1e-12, maxiter=100, gamma0=1.0):
    x = x0
    gamma = gamma0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    it = 0
    
    # Variables to store memory from the previous iteration
    x_prev = 0.0
    w_prev = 0.0
    fx_prev = 0.0
    fw_prev = 0.0

    while it < maxiter:
        fx = F(x)
        if abs(fx) < tol:
            return x, it

        # Starting from the second iteration (k >= 1), calculate a new gamma_k
        if it > 0:
            # 1st order divided differences
            f_x_xprev = (fx - fx_prev) / (x - x_prev)
            f_xprev_wprev = (fx_prev - fw_prev) / (x_prev - w_prev)
            
            # 2nd order divided difference
            f_x_xprev_wprev = (f_x_xprev - f_xprev_wprev) / (x - w_prev)
            
            # Derivative approximation: N2'(x_k)
            n2_prime = f_x_xprev + f_x_xprev_wprev * (x - x_prev)
            
            assert n2_prime != 0.0, "Modified Steffensen: zero derivative approximation (N2'(x_k) = 0)"
            gamma = -1.0 / n2_prime

        # Calculate node w_k and the function value at it
        w = x + gamma * fx
        fw = F(w)

        denom = fw - fx
        assert denom != 0.0, "Modified Steffensen: zero denominator"

        # Iteration step
        xnew = x - gamma * fx**2 / denom

        if abs(xnew - x) < tol:
            return xnew, it + 1

        # Save current nodes to memory for the next iteration
        x_prev = x
        w_prev = w
        fx_prev = fx
        fw_prev = fw

        # Update current approximation and iteration counter
        x = xnew
        it += 1

    return x, it