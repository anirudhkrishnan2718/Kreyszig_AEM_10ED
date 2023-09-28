import numpy as np
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
    cos,
    pi,
    erf,
    besselj,
    binomial,
    Sum,
    expand_func,
    lambdify,
    integrate,
    sinc,
)
from sympy.integrals.quadrature import gauss_legendre
from sympy.abc import x, t, s


def rectangular_rule(f, a, b, h):
    n = int((b - a) / h) + 1
    xval = np.linspace(a, b, n)[:-1] + (h / 2)
    yval = [f.subs(x, item) for item in xval]
    return h * np.sum(yval)


def trapezoidal_rule(f, a, b, h):
    n = int((b - a) / h) + 1
    xval = np.linspace(a, b, n)
    yval = f(xval)
    heights_summed = yval[:-1] + yval[1:]
    return 0.5 * h * np.sum(heights_summed)


def trapezoidal_rule_direct(f, a, b, n):
    xval = np.linspace(a, b, n + 1)
    yval = f(xval)
    heights_summed = yval[:-1] + yval[1:]
    return 0.5 * ((b - a) / n) * np.sum(heights_summed)


def simpson_rule(f, a, b, n):
    xval = np.linspace(a, b, n + 1)
    yval = f(xval)
    return ((b - a) / (3 * n)) * (
        np.sum(2 * yval[::2]) + np.sum(4 * yval[1::2]) - yval[0] - yval[-1]
    )


def gaussian_integration(f, a, b, order):
    x, w = gauss_legendre(5, 15)
    xval = np.array(x).astype("float64")
    weights = np.array(w).astype("float64")
    yval = f(xval)
    return np.sum(weights * yval)


# f = E ** (-(x**2))
# a, b = 0, 1
# h = 0.1
# out = rectangular_rule(f, a, b, h)
# print(f"Result \t\t {out.evalf(7)}")

# f = lambdify(x, sin(pi * 0.5 * x))
# a, b = 0, 1
# for h in [1, 0.5, 0.25, 0.125]:
#     out = trapezoidal_rule(f, a, b, h)
#     out_better = trapezoidal_rule(f, a, b, 2 * h)
#     print(out, "\t\t", (out - out_better) / 3, "\t\t", (2 / pi).evalf() - out)

# g = 1 / (1 + x**2)
# f = lambdify(x, g)
# a, b = 0, 1
# for n in [4, 8]:
#     out = simpson_rule(f, a, b, n)
#     exact = integrate(g, (x, a, b)).evalf()
#     print(f"Result \t {out}")
#     print(f"Error \t {(exact - out).evalf(8)}")

# g = E ** (-x)
# f = lambdify(x, g)
# a, b = 0, 2
# for n in [2, 4]:
#     out = simpson_rule(f, a, b, n)
#     exact = integrate(g, (x, a, b)).evalf()
#     print(f"Result \t {out}")
#     print(f"Error \t {(exact - out).evalf(8)}")

# g = sinc(x)
# f = lambdify(x, g)
# a, b = 0, 1
# for h in [0.2, 0.1]:
#     out = trapezoidal_rule(f, a, b, h)
#     out_better = trapezoidal_rule(f, a, b, 2 * h)
#     print(
#         out,
#         "\t\t",
#         (out - out_better) / 3,
#         "\t\t",
#         integrate(g, (x, a, b)).evalf() - out,
#     )

# g = sinc(x)
# f = lambdify(x, g)
# a, b = 0, 1
# for n in [10]:
#     out = simpson_rule(f, a, b, n)
#     out_better = simpson_rule(f, a, b, int(n / 2))
#     exact = integrate(g, (x, a, b)).evalf()
#     print(f"Result \t {out}")
#     print(f"Practical \t {(out - out_better)/15}")
#     print(f"Error \t {(exact - out).evalf()}")

# g = cos(x**2)
# f = lambdify(x, g)
# a, b = 0, 1.25
# for n in [10]:
#     out = simpson_rule(f, a, b, n)
#     out_better = simpson_rule(f, a, b, int(n / 2))
#     exact = integrate(g, (x, a, b)).evalf()
#     print(f"Result \t {out}")
#     print(f"Practical \t {(out - out_better)/15}")
#     print(f"Error \t {(exact - out).evalf()}")

# g = cos(x)
# h = cos((pi / 4) * (t + 1))
# f = lambdify(t, h)
# a, b = 0, pi / 2
# factor = (pi / 4).evalf()
# for order in [5]:
#     out = gaussian_integration(f, a, b, order) * factor
#     exact = integrate(g, (x, a, b)).evalf()
#     print(f"Result \t {out}")
#     print(f"Exact \t {exact}")
#     print(f"Error \t {(exact - out).evalf()}")

# g = x * E ** (-x)
# h = ((t + 1) / 2) * E ** ((-t - 1) / 2)
# f = lambdify(t, h)
# a, b = 0, 1
# factor = 0.5
# for order in [5]:
#     out = gaussian_integration(f, a, b, order) * factor
#     exact = integrate(g, (x, a, b)).evalf()
#     print(f"Result \t {out}")
#     print(f"Exact \t {exact}")
#     print(f"Error \t {(exact - out).evalf()}")

# g = sin(x**2)
# h = sin(((5 / 8) * (1 + t)) ** 2)
# f = lambdify(t, h)
# a, b = 0, 1.25
# factor = 5 / 8
# for order in [5]:
#     out = gaussian_integration(f, a, b, order) * factor
#     exact = integrate(g, (x, a, b)).evalf()
#     print(f"Result \t {out}")
#     print(f"Exact \t {exact}")
#     print(f"Error \t {(exact - out).evalf()}")

# g = E ** (-(x**2))
# h = E ** (-0.25 * (1 + t) ** 2)
# f = lambdify(t, h)
# a, b = 0, 1
# factor = 1 / 2
# for order in [5]:
#     out = gaussian_integration(f, a, b, order) * factor
#     exact = integrate(g, (x, a, b)).evalf()
#     print(f"Result \t {out}")
#     print(f"Exact \t {exact}")
#     print(f"Error \t {(exact - out).evalf()}")
