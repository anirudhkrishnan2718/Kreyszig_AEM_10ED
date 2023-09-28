import numpy as np
import pandas as pd
import math

from sympy import (
    Array,
    simplify,
    latex,
    log,
    diff,
    factorial,
    Interval,
    minimum,
    maximum,
    E,
    sin,
    pi,
    erf,
    besselj,
    binomial,
    Sum,
    expand_func,
    lambdify,
)
from sympy.abc import x, t, s, k


def lagrange_interpolation(xval, yval):
    factors = [(x - item) for item in xval]
    l_k = np.array(
        [(factors[:idx] + factors[idx + 1 :]) for idx, val in enumerate(xval)]
    )
    p_n = 0
    for idx, val in enumerate(l_k):
        l = np.prod(val)
        print(f"Single polynomial\t\t {(l / (l.subs(x, xval[idx]))) * yval[idx]}")
        p_n = p_n + (l / (l.subs(x, xval[idx]))) * yval[idx]

    return simplify(p_n)


def lagrange_error(xval, yval, f, point):
    err = (
        np.prod([x - item for item in xval])
        * diff(f.subs(x, t), t, len(xval))
        / factorial(len(xval))
    )
    print(f"error exp\t\t{err.subs(x, point)}")
    return minimum((err.subs(x, point)), t, Interval(xval[0], xval[-1])), maximum(
        err.subs(x, point), t, Interval(xval[0], xval[-1])
    )


def newtons_forward_diff(xval, yval, point):
    if np.all(np.isclose(np.diff(xval), np.diff(xval)[0])):
        h = np.diff(xval)[0]
        r = (x - xval[0]) / h
        n = len(yval) - 1
        f_vals = Array([np.diff(yval, item)[0] for item in range(0, n + 1)])
        p_n = Sum((binomial(r, s) * f_vals[s]), (s, 0, n))
        print(p_n.doit().expand(func=True).evalf(7))
        return p_n.subs(x, point)


def newtons_backward_diff(xval, yval, point):
    if np.all(np.isclose(np.diff(xval), np.diff(xval)[0])):
        h = np.diff(xval)[0]
        r = (x - xval[-1]) / h
        n = len(yval) - 1
        f_vals = Array([np.diff(yval, item)[-1] for item in range(0, n + 1)])
        p_n = Sum((binomial(r + s - 1, s) * f_vals[s]), (s, 0, n))
        print(p_n.doit().expand(func=True).evalf(7))
        return p_n.subs(x, point)


def newtons_divided_diff(xval, yval, point):
    div_dif = np.array([yval])

    def f(i, j):
        if i == j:
            return yval[i]
        elif j == i + 1:
            return (yval[j] - yval[i]) / (xval[j] - xval[i])
        else:
            return (f(i + 1, j) - f(i, j - 1)) / (xval[j] - xval[i])

    diffs_list = Array([f(0, p) for p in range(len(xval))])
    p_n = diffs_list[0]
    for l in range(1, len(xval)):
        term = np.prod([x - xval[:l]])
        p_n = p_n + diffs_list[l] * (term)
    print(f"poly\t\t{p_n.simplify().evalf(6)}")
    return p_n.simplify().subs(x, point)


def spline_eq(xval, yval, k_0, k_n):
    n = len(xval)
    h = xval[1] - xval[0]
    b = [k_0] + [(3 / h) * (yval[j + 1] - yval[j - 1]) for j in range(1, n - 1)] + [k_n]
    a = np.identity(n)
    for q in range(1, n - 1):
        a[q, q - 1 : q + 2] = [1, 4, 1]
    k = np.linalg.solve(a, np.array(b))
    result = []
    for j in range(n - 1):
        a_0 = yval[j]
        a_1 = k[j]
        a_2 = (3 / h**2) * (yval[j + 1] - yval[j]) - (1 / h) * (k[j + 1] + 2 * k[j])
        a_3 = (2 / h**3) * (yval[j] - yval[j + 1]) + (1 / h**2) * (k[j + 1] + k[j])
        p_x = (
            a_0
            + a_1 * (x - xval[j])
            + a_2 * (x - xval[j]) ** 2
            + a_3 * (x - xval[j]) ** 3
        )
        result = result + [
            [
                a_0
                + a_1 * (x - xval[j])
                + a_2 * (x - xval[j]) ** 2
                + a_3 * (x - xval[j]) ** 3
            ]
        ]
    return result


def derivative_approx(f, xval, point):
    if np.all(np.isclose(np.diff(xval), np.diff(xval)[0])):
        yval = f(xval)
        h = np.diff(xval)[0]
        n = len(yval) - 1
        f_vals = np.array([np.diff(yval, item)[0] for item in range(0, n + 1)])
        result = 0
        for count in range(1, n + 1):
            result = (
                result + (1 / h) * (1 / count) * ((-1) ** (count + 1)) * f_vals[count]
            )
        return result


# g = x**4
# f = lambdify(x, g)
# xvals = np.array([0.4, 0.6, 0.8, 1.0, 1.2])
# print(derivative_approx(f, xvals[:2], 0.4))
# print(derivative_approx(f, xvals[:3], 0.4))
# print(derivative_approx(f, xvals[:4], 0.4))
# print(derivative_approx(f, xvals[:5], 0.4))

# xin = Array([9.0, 9.5, 11.0])
# yin = Array([2.1972, 2.2513, 2.3979])
# g = log(x)
# x_interp = 9.2
# out = lagrange_interpolation(xin, yin)
# err_min, err_max = lagrange_error(xin, yin, g, x_interp)
# print(f"polynomial\t\t{latex(out.evalf(4))}")
# print(f"value\t\t\t{latex(out.subs(x, x_interp).evalf(5))}")
# print(f"error min\t\t{err_min.evalf(5)}")
# print(f"error max\t\t{err_max.evalf(5)}")

# xin = Array([0, 0.5, 1])
# yin = Array([1, 0.6065, 0.3679])
# g = E ** (-x)
# x_interp = 0.75
# out = lagrange_interpolation(xin, yin)
# err_min, err_max = lagrange_error(xin, yin, g, x_interp)
# print(f"polynomial\t\t{latex(out.evalf(4))}")
# print(f"value\t\t\t{latex(out.subs(x, x_interp).evalf(5))}")
# print(f"error min\t\t{err_min.evalf(5)}")
# print(f"error max\t\t{err_max.evalf(5)}")

# xin = Array([9.0, 9.5, 11.0])
# yin = Array([2.1972, 2.2513, 2.3979])
# g = log(x)
# x_interp = 9.2
# out = lagrange_interpolation(xin, yin)
# err_min, err_max = lagrange_error(xin, yin, g, x_interp)
# print(f"polynomial\t\t{latex(out.evalf(4))}")
# print(f"value\t\t\t{latex(out.subs(x, x_interp).evalf(5))}")
# print(f"error min\t\t{err_min.evalf(5)}")
# print(f"error max\t\t{err_max.evalf(5)}")

# xin = Array([0, pi / 4, pi / 2])
# yin = Array([0.0, 0.70711, 1.0])
# g = sin(x)
# x_interp = 5 * pi / 8
# out = lagrange_interpolation(xin, yin)
# err_min, err_max = lagrange_error(xin, yin, g, x_interp)
# print(f"polynomial\t\t{latex(out.evalf(4))}")
# print(f"value\t\t\t{out.subs(x, x_interp).evalf(5)}")
# print(f"error min\t\t{err_min.evalf(5)}")
# print(f"error max\t\t{err_max.evalf(5)}")
# print(
#     f"accurate error\t\t{g.subs(x, x_interp).evalf(4) - out.subs(x, x_interp).evalf(4)}"
# )

# xin = Array([0.25, 0.5, 1.0])
# yin = Array([0.27633, 0.52050, 0.84270])
# g = erf(x)
# x_interp = 0.75
# out = lagrange_interpolation(xin, yin)
# err_min, err_max = lagrange_error(xin, yin, g, x_interp)
# print(f"polynomial\t\t{latex(out.evalf(6))}")
# print(f"value\t\t\t{latex(out.subs(x, x_interp).evalf(5))}")
# print(f"error min\t\t{err_min.evalf(5)}")
# print(f"error max\t\t{err_max.evalf(5)}")

# xin = Array([0.0, 1.0, 2.0, 3.0])
# yin = Array([1.0, 0.765198, 0.223891, -0.260052])
# g = besselj(0, x)
# x_interp = 2.5
# out = lagrange_interpolation(xin, yin)
# err_min, err_max = lagrange_error(xin, yin, g, x_interp)
# print(f"polynomial\t\t{latex(out.evalf(6))}")
# print(f"value\t\t\t{latex(out.subs(x, x_interp).evalf(6))}")
# print(f"error min\t\t{err_min.evalf(5)}")
# print(f"error max\t\t{err_max.evalf(5)}")

# xin = np.array([1, 1.5, 2.0, 2.5])
# yin = np.array([0.94608, 1.32468, 1.60541, 1.77852])
# x_interp = 1.25
# out = newtons_forward_diff(xin[:2], yin[:2], x_interp)
# print(f"value\t\t\t{out.evalf(7)}")
# out = newtons_forward_diff(xin[:3], yin[:3], x_interp)
# print(f"value\t\t\t{out.evalf(7)}")
# out = newtons_forward_diff(xin, yin, x_interp)
# print(f"value\t\t\t{out.evalf(7)}")

# xin = np.array([-4, -2, 0, 2, 4])
# yin = np.array([50, 18, 2, 2, 18])
# x_interp = 1
# out = newtons_forward_diff(xin, yin, x_interp)
# print(f"value\t\t\t{out.evalf(7)}")

# xin = np.array([1, 1.02, 1.04])
# yin = np.array([1, 0.9888, 0.9784])
# x_interp = 1.05
# out = newtons_forward_diff(xin, yin, x_interp)
# print(f"value\t\t\t{out.evalf(4)}")

xin = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
yin = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
x_interp = 0.1
out = newtons_divided_diff(xin, yin, x_interp)
print(f"value\t\t{out.evalf(5)}")

# xin = np.array([0.25, 0.5, 1.0])
# yin = np.array([0.27633, 0.52050, 0.84270])
# x_interp = 0.75
# out = newtons_divided_diff(xin, yin, x_interp)
# print(f"value\t\t{out.evalf(6)}")

# xin = np.array([0.2, 0.4, 0.6])
# yin = np.array([0.2227, 0.4284, 0.6039])
# x_interp = 0.3
# out = newtons_backward_diff(xin, yin, x_interp)
# print(f"value\t\t\t{out.evalf(4)}")

# xin = np.array([0.5, 0.6, 0.7, 0.8])
# yin = np.array([1.127626, 1.185465, 1.255169, 1.337435])
# x_interp = 0.56
# out = newtons_backward_diff(xin, yin, x_interp)
# print(f"value\t\t\t{out.evalf(7)}")

# xin = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
# yin = np.array([0.2227, 0.4284, 0.6039, 0.7421, 0.8427])
# x_interp = 0.3
# out = newtons_backward_diff(xin[:2], yin[:2], x_interp)
# print(f"value\t\t\t{out.evalf(6)}")
# out = newtons_backward_diff(xin[:3], yin[:3], x_interp)
# print(f"value\t\t\t{out.evalf(6)}")
# out = newtons_backward_diff(xin[:4], yin[:4], x_interp)
# print(f"value\t\t\t{out.evalf(6)}")
# out = newtons_backward_diff(xin, yin, x_interp)
# print(f"value\t\t\t{out.evalf(6)}")

# xin = np.array([9.0, 9.5, 11.0])
# yin = np.array([2.19722, 2.25129, 2.39789])
# g = log(x)
# x_interp = 9.2
# out = lagrange_interpolation(xin[:2], yin[:2])
# print(f"value\t\t\t{latex(out.subs(x, x_interp).evalf(7))}")
# out = lagrange_interpolation(xin, yin)
# print(f"value\t\t\t{latex(out.subs(x, x_interp).evalf(7))}")

# xin = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
# yin = np.array([0.9980, 0.9686, 0.8443, 0.5358, 0])
# x_interp = 0.7
# out = newtons_forward_diff(xin[2:], yin[2:], x_interp)
# print(f"value\t\t\t{out.evalf(4)}")
# out = newtons_forward_diff(xin[1:4], yin[1:4], x_interp)
# print(f"value\t\t\t{out.evalf(4)}")
# out = newtons_forward_diff(xin[:3], yin[:3], x_interp)
# print(f"value\t\t\t{out.evalf(4)}")

# xin = np.array([-1, 0, 1])
# yin = np.array([1, 0, 1])
# f = x**4
# out = spline_eq(xin, yin, -4, 4)
# for item in out:
#     print(item)

# xin = np.array([-1, 0, 1])
# yin = np.array([1, 0, 1])
# x_interp = 0.5
# out = newtons_forward_diff(xin, yin, x_interp)
# print(f"value\t\t\t{out.evalf(4)}")

# xin = np.array([-2, -1, 0, 1, 2])
# yin = np.array([0, 0, 1, 0, 0])
# out = spline_eq(xin, yin, 0, 0)
# for item in out:
#     print(item)

# xin = np.array([-2, -1, 0, 1, 2])
# yin = np.array([0, 0, 1, 0, 0])
# x_interp = 0.5
# out = newtons_forward_diff(xin, yin, x_interp)
# print(f"value\t\t\t{out.evalf(4)}")

# xin = np.array([0, 2, 4, 6])
# yin = np.array([2, -2, 2, 78])
# out = spline_eq(xin, yin, 0, 0)
# for item in out:
#     print(item)
