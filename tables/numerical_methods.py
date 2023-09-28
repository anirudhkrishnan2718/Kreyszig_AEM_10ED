import numpy as np
import pandas as pd
import math
from sympy import (
    integrate,
    sin,
    cos,
    pi,
    exp,
    Si,
    Sum,
    factorial,
    sqrt,
    E,
    erf,
    besselj,
    floor,
    I,
    E,
    rootof,
    diff,
    Float,
    N,
    asin,
    cosh,
    log,
    tan,
    cot,
)

from sympy.abc import x, y, f, g, h, k, i, w, m, n, z

# from scipy.special import jn_zeros

# xs = 1.5 * np.array([(16.0) ** (-k) for k in range(1, 30)])
# print(xs)
# yval = np.empty([])
# for k in range(len(xs)):
#     print(np.sum(xs[:k]))
# print(xs[k])

# result = np.transpose(np.vstack((xs, yval[1:])))
# print(yval)
# np.savetxt("./tables/table_15_04_16.csv", result, delimiter=",", fmt="%d")

# from decimal import *

# getcontext().prec = 50

# xs = [Decimal(3) / (Decimal(2) * Decimal(16**k)) for k in range(1, 31)]
# getcontext().prec = 35
# for i in range(len(xs)):
#     print(i + 1, (Decimal(1) / Decimal(10)) - sum(xs[:i]))


# acc = 40
# xvals = np.array([])
# yvals = np.array([])
# for acc in range(4, 50):
#     val = (E - 1).evalf(acc)
#     i = 1
#     while True:
#         val = E.evalf(acc) - (i * val.evalf(acc)).evalf(acc)
#         if val < 0:
#             xvals = np.append(xvals, acc)
#             yvals = np.append(yvals, i)
#             # print(f" {i=} and {acc=} and {val=}")
#             break
#         else:
#             i += 1
# print(xvals, yvals)
# print(np.polyfit(xvals, yvals, 1))


# acc = 4
# xvals = np.array([])
# yvals = np.array([])
# val = N(0, 4)
# for acc in range(15, 0, -1):
#     val = ((E.evalf(acc) - val.evalf(acc)) / acc).evalf(4)
#     xvals = np.append(xvals, acc)
#     yvals = np.append(yvals, val)
# print(xvals, yvals)


# g = 1 - x**3
# xvals = np.array([])
# yvals = np.array([])
# curr = -0.51
# for i in range(10):
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, g.subs(x, curr))
#     curr = yvals[-1]
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, curr)
# result = np.transpose(np.vstack((xvals, yvals)))
# np.savetxt("./tables/table_18_02_02_alt.csv", result, delimiter=",", fmt="%.4f")

# g = 0.5 * cos(x)
# xvals = np.array([])
# yvals = np.array([])
# curr = 1
# for i in range(10):
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, g.subs(x, curr))
#     curr = yvals[-1]
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, curr)
# result = np.transpose(np.vstack((xvals, yvals)))
# np.savetxt("./tables/table_18_02_03.csv", result, delimiter=",", fmt="%.6f")

# g = (5 * x**2 - 1.01 * x - 1.88) / (x**2)

# curr = -1
# xvals = np.array([curr])
# yvals = np.array([curr])
# for i in range(10):
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, g.subs(x, curr))
#     curr = yvals[-1]
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, curr)
# result = np.transpose(np.vstack((xvals, yvals)))
# np.savetxt("./tables/table_18_02_05_d.csv", result, delimiter=",", fmt="%.6f")

# g = asin(E ** (-x))

# curr = 0.7
# xvals = np.array([curr])
# yvals = np.array([curr])
# for i in range(30):
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, g.subs(x, curr))
#     curr = yvals[-1]
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, curr)
# result = np.transpose(np.vstack((xvals, yvals)))
# np.savetxt("./tables/table_18_02_07_b.csv", result, delimiter=",", fmt="%.6f")

# g = (x + 0.12) ** (1 / 4)

# curr = 1
# xvals = np.array([curr])
# yvals = np.array([curr])
# for i in range(8):
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, g.subs(x, curr))
#     curr = yvals[-1]
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, curr)
# result = np.transpose(np.vstack((xvals, yvals)))
# np.savetxt("./tables/table_18_02_08.csv", result, delimiter=",", fmt="%.6f")

# g = x**4 - 0.12

# curr = -0.2
# xvals = np.array([curr])
# yvals = np.array([curr])
# for i in range(10):
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, g.subs(x, curr))
#     curr = yvals[-1]
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, curr)
# result = np.transpose(np.vstack((xvals, yvals)))
# np.savetxt("./tables/table_18_02_09.csv", result, delimiter=",", fmt="%.6f")

# g = 1 / cosh(x)

# curr = 0.85
# xvals = np.array([curr])
# yvals = np.array([curr])
# for i in range(20):
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, g.subs(x, curr))
#     curr = yvals[-1]
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, curr)
# result = np.transpose(np.vstack((xvals, yvals)))
# np.savetxt("./tables/table_18_02_10_b.csv", result, delimiter=",", fmt="%.6f")

# g = (1 + (x**4 / 64) - (x**6 / 2304)) / (0.25 * x)

# curr = 2.3
# xvals = np.array([curr])
# yvals = np.array([curr])
# for i in range(8):
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, g.subs(x, curr))
#     curr = yvals[-1]
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, curr)
# result = np.transpose(np.vstack((xvals, yvals)))
# np.savetxt("./tables/table_18_02_11.csv", result, delimiter=",", fmt="%.6f")

# g = (2 * x**3 + 2 * x**2 + 4) / (3 * x**2 + 4 * x - 3)

curr = 1.5
xvals = np.array([curr])
yvals = np.array([curr])
for i in range(4):
    xvals = np.append(xvals, curr)
    yvals = np.append(yvals, g.subs(x, curr))
    curr = yvals[-1]
    xvals = np.append(xvals, curr)
    yvals = np.append(yvals, curr)
result = np.transpose(np.vstack((xvals, yvals)))
np.savetxt("./tables/table_18_02_12_f.csv", result, delimiter=",", fmt="%.6f")

# g = x - ((x**3 - 7) / (3 * x**2))

# curr = 2
# xvals = np.array([curr])
# yvals = np.array([curr])
# for i in range(4):
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, g.subs(x, curr))
#     curr = yvals[-1]
# result = np.transpose(yvals)
# np.savetxt("./tables/table_18_02_14.csv", result, delimiter=",", fmt="%.6f")


# g = x - ((2 * x - cos(x)) / (2 + sin(x)))

# curr = 100
# xvals = np.array([curr])
# yvals = np.array([curr])
# for i in range(10):
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, g.subs(x, curr).evalf())
#     curr = yvals[-1]
# result = np.transpose(yvals)
# np.savetxt("./tables/table_18_02_16.csv", result, delimiter=",", fmt="%.6f")

# f = x**3 - 5 * x**2 + 1.01 * x + 1.88
# g = x - (f / diff(f, x, 1))

# curr = -3
# xvals = np.array([curr])
# yvals = np.array([curr])
# for i in range(10):
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, g.subs(x, curr).evalf())
#     curr = yvals[-1]
# result = np.transpose(yvals)
# np.savetxt("./tables/table_18_02_17_d.csv", result, delimiter=",", fmt="%.6f")

# f = (63 * x**5 - 70 * x**3 + 15 * x) / 8
# g = x - (f / diff(f, x, 1))

# curr = 1
# xvals = np.array([curr])
# yvals = np.array([curr])
# for i in range(10):
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, g.subs(x, curr).evalf())
#     curr = yvals[-1]
# result = np.transpose(yvals)
# np.savetxt("./tables/table_18_02_18.csv", result, delimiter=",", fmt="%.6f")

# f = (7.5) * (-7 * x**4 + 8 * x**2 - 1)
# g = x - (f / diff(f, x, 1))

# curr = 0.3
# xvals = np.array([curr])
# yvals = np.array([curr])
# for i in range(10):
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, g.subs(x, curr).evalf())
#     curr = yvals[-1]
# result = np.transpose(yvals)
# np.savetxt("./tables/table_18_02_19.csv", result, delimiter=",", fmt="%.6f")

# f = x + log(x) - 2
# g = x - (f / diff(f, x, 1))

# curr = 2
# xvals = np.array([curr])
# yvals = np.array([curr])
# for i in range(10):
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, g.subs(x, curr).evalf())
#     curr = yvals[-1]
# result = np.transpose(yvals)
# np.savetxt("./tables/table_18_02_20.csv", result, delimiter=",", fmt="%.6f")

# f = x**3 - 5 * x + 3
# g = x - (f / diff(f, x, 1))

# curr = 2
# xvals = np.array([curr])
# yvals = np.array([curr])
# for i in range(10):
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, g.subs(x, curr).evalf())
#     curr = yvals[-1]
# result = np.transpose(yvals)
# np.savetxt("./tables/table_18_02_21_a.csv", result, delimiter=",", fmt="%.6f")

# f = 100 * (x**20 - 1) + 40 * x
# g = x - (f / diff(f, x, 1))

# curr = 2
# xvals = np.array([curr])
# yvals = np.array([curr])
# for i in range(20):
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, g.subs(x, curr).evalf(4))
#     curr = yvals[-1]
# result = np.transpose(yvals)
# np.savetxt("./tables/table_18_02_22.csv", result, delimiter=",", fmt="%.4f")

# f = cos(x) * cosh(x) - 1
# g = x - (f / diff(f, x, 1))

# curr = 4.5
# xvals = np.array([curr])
# yvals = np.array([curr])
# for i in range(5):
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, g.subs(x, curr).evalf())
#     curr = yvals[-1]
# result = np.transpose(yvals)
# np.savetxt("./tables/table_18_02_23.csv", result, delimiter=",", fmt="%.6f")

# f = x + log(x) - 2

# a = 1
# b = 2
# avals = np.array([a])
# bvals = np.array([b])
# cvals = np.array([])
# for i in range(7):
#     c = ((a * f.subs(x, b).evalf()) - (b * f.subs(x, a).evalf())) / (
#         f.subs(x, b).evalf() - f.subs(x, a).evalf()
#     )
#     if f.subs(x, c).evalf() == 0:
#         cvals = np.append(cvals, c)
#         break
#     elif f.subs(x, c).evalf() * f.subs(x, a).evalf() < 0:
#         cvals = np.append(cvals, c)
#         b = c
#     elif f.subs(x, c).evalf() * f.subs(x, a).evalf() > 0:
#         cvals = np.append(cvals, c)
#         a = c

# np.savetxt(
#     "./tables/table_18_02_24_c.csv", np.transpose(cvals), delimiter=",", fmt="%.6f"
# )

# f = x - cos(x)

# a = 0
# b = 2
# cvals = np.array([])
# for i in range(25):
#     c = 0.5 * (a + b)
#     if f.subs(x, c).evalf() == 0:
#         cvals = np.append(cvals, c)
#         break
#     elif (f.subs(x, c).evalf() * f.subs(x, a).evalf() < 0) and (
#         f.subs(x, c).evalf() * f.subs(x, b).evalf() > 0
#     ):
#         cvals = np.append(cvals, c)
#         b = c
#     elif (f.subs(x, c).evalf() * f.subs(x, a).evalf() > 0) and (
#         f.subs(x, c).evalf() * f.subs(x, b).evalf() < 0
#     ):
#         cvals = np.append(cvals, c)
#         a = c

# np.savetxt(
#     "./tables/table_18_02_25_a.csv", np.transpose(cvals), delimiter=",", fmt="%.6f"
# )

# f = cos(x) - x
# g = x - (f / diff(f, x, 1))

# curr = 0.5
# xvals = np.array([curr])
# yvals = np.array([curr])
# for i in range(5):
#     xvals = np.append(xvals, curr)
#     yvals = np.append(yvals, g.subs(x, curr).evalf())
#     curr = yvals[-1]
# result = np.transpose(yvals)
# np.savetxt("./tables/table_18_02_25_b.csv", result, delimiter=",", fmt="%.6f")

# f = E ** (x) + x**4 + x - 2

# a = 0
# b = 1
# cvals = np.array([])
# for i in range(25):
#     c = 0.5 * (a + b)
#     if f.subs(x, c).evalf() == 0:
#         cvals = np.append(cvals, c)
#         break
#     elif (f.subs(x, c).evalf() * f.subs(x, a).evalf() < 0) and (
#         f.subs(x, c).evalf() * f.subs(x, b).evalf() > 0
#     ):
#         cvals = np.append(cvals, c)
#         b = c
#     elif (f.subs(x, c).evalf() * f.subs(x, a).evalf() > 0) and (
#         f.subs(x, c).evalf() * f.subs(x, b).evalf() < 0
#     ):
#         cvals = np.append(cvals, c)
#         a = c

# np.savetxt(
#     "./tables/table_18_02_25_d.csv", np.transpose(cvals), delimiter=",", fmt="%.6f"
# )

# f = E ** (-x) - tan(x)
# past = 1
# curr = 0.7
# yvals = np.array([curr])
# for i in range(10):
#     next = curr - f.subs(x, curr).evalf() * (
#         (curr - past) / (f.subs(x, curr).evalf() - f.subs(x, past).evalf())
#     )
#     if abs(f.subs(x, curr).evalf() - f.subs(x, past).evalf()) > 0.000000000000001:
#         past = curr
#         curr = next
#         yvals = np.append(yvals, curr)
#     else:
#         break
# result = np.transpose(yvals)
# np.savetxt("./tables/table_18_02_26.csv", result, delimiter=",", fmt="%.6f")

# f = x**3 - 5 * x + 3
# past = 1
# curr = 2
# yvals = np.array([curr])
# for i in range(10):
#     next = curr - f.subs(x, curr).evalf() * (
#         (curr - past) / (f.subs(x, curr).evalf() - f.subs(x, past).evalf())
#     )
#     if abs(f.subs(x, curr).evalf() - f.subs(x, past).evalf()) > 0.000000000000001:
#         past = curr
#         curr = next
#         yvals = np.append(yvals, curr)
#     else:
#         break
# result = np.transpose(yvals)
# np.savetxt("./tables/table_18_02_27.csv", result, delimiter=",", fmt="%.6f")

# f = x - cos(x)
# past = 0.5
# curr = 1
# yvals = np.array([curr])
# for i in range(10):
#     next = curr - f.subs(x, curr).evalf() * (
#         (curr - past) / (f.subs(x, curr).evalf() - f.subs(x, past).evalf())
#     )
#     if abs(f.subs(x, curr).evalf() - f.subs(x, past).evalf()) > 0.000000000000001:
#         past = curr
#         curr = next
#         yvals = np.append(yvals, curr)
#     else:
#         break
# result = np.transpose(yvals)
# np.savetxt("./tables/table_18_02_28.csv", result, delimiter=",", fmt="%.6f")

f = sin(x) - cot(x)
past = 1
curr = 0.5
yvals = np.array([curr])
for i in range(10):
    next = curr - f.subs(x, curr).evalf() * (
        (curr - past) / (f.subs(x, curr).evalf() - f.subs(x, past).evalf())
    )
    if abs(f.subs(x, curr).evalf() - f.subs(x, past).evalf()) > 0.000000000000001:
        past = curr
        curr = next
        yvals = np.append(yvals, curr)
    else:
        break
result = np.transpose(yvals)
np.savetxt("./tables/table_18_02_29.csv", result, delimiter=",", fmt="%.6f")
