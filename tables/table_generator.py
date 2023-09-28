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
)
from sympy.abc import x, y, f, g, h, k, i, w, m, n, z
from scipy.special import jn_zeros

# # 01_01_17
# # f(x, y) = y
# # y = e^(x)

# df = pd.DataFrame(columns=["$n$", "$x_{n}$", "$y_{n}$", "$y(x_{n})$", "Error"])
# df.loc[len(df)] = [0, 0.0, 1.0, 1.0, 0.0]
# h = 0.1
# for i in range(10):
#     x, y = df.iloc[-1]["$x_{n}$"], df.iloc[-1]["$y_{n}$"]
#     print(x, y)
#     x_new, y_new = x + h, y + h * y
#     df.loc[len(df.index)] = [
#         (i + 1),
#         np.round(x_new, 4),
#         np.round(y_new, 4),
#         np.round(math.e ** (x_new), 4),
#         np.round(math.e ** (x_new) - y_new, 4),
#     ]

# # Convert n column to integer
# df["$n$"] = df["$n$"].astype(int)
# print(df)
# df.to_csv("./tables/table_01_01_17.csv", index=False)

# # 01_01_18
# # f(x, y) = y
# # y = e^(x)

# df = pd.DataFrame(columns=["$n$", "$x_{n}$", "$y_{n}$", "$y(x_{n})$", "Error"])
# df.loc[len(df)] = [0, 0.0, 1.0, 1.0, 0.0]
# h = 0.01
# for i in range(10):
#     x, y = df.iloc[-1]["$x_{n}$"], df.iloc[-1]["$y_{n}$"]
#     print(x, y)
#     x_new, y_new = x + h, y + h * y
#     df.loc[len(df.index)] = [
#         (i + 1),
#         np.round(x_new, 4),
#         np.round(y_new, 4),
#         np.round(math.e ** (x_new), 4),
#         np.round(math.e ** (x_new) - y_new, 4),
#     ]

# # Convert n column to integer
# df["$n$"] = df["$n$"].astype(int)
# print(df)
# df.to_csv("./tables/table_01_01_18.csv", index=False)

# # 01_01_19
# # f(x, y) = (y - x)^2
# # y = x - tanh x

# df = pd.DataFrame(columns=["$n$", "$x_{n}$", "$y_{n}$", "$y(x_{n})$", "Error"])
# df.loc[len(df)] = [0, 0.0, 0.0, 0.0, 0.0]
# h = 0.1
# for i in range(10):
#     x, y = df.iloc[-1]["$x_{n}$"], df.iloc[-1]["$y_{n}$"]
#     print(x, y)
#     x_new, y_new = x + h, y + h * ((y - x) ** 2)
#     df.loc[len(df.index)] = [
#         (i + 1),
#         np.round(x_new, 4),
#         np.round(y_new, 4),
#         np.round(x_new - math.tanh(x_new), 4),
#         np.round(x_new - math.tanh(x_new) - y_new, 4),
#     ]

# # Convert n column to integer
# df["$n$"] = df["$n$"].astype(int)
# print(df)
# df.to_csv("./tables/table_01_01_19.csv", index=False)

# 01_01_20
# f(x, y) = -5 * x^4 * y^2
# y = (1 + x)^(-5)

# df = pd.DataFrame(columns=["$n$", "$x_{n}$", "$y_{n}$", "$y(x_{n})$", "Error"])
# df.loc[len(df)] = [0, 0.0, 1.0, 1.0, 0.0]
# h = 0.2
# for i in range(10):
#     x, y = df.iloc[-1]["$x_{n}$"], df.iloc[-1]["$y_{n}$"]
#     print(x, y)
#     x_new, y_new = x + h, y + h * (-5 * (x**4) * (y**2))
#     df.loc[len(df.index)] = [
#         (i + 1),
#         np.round(x_new, 4),
#         np.round(y_new, 4),
#         np.round((1 + x_new**5)**(-1), 4),
#         np.round((1 + x_new**5)**(-1) - y_new, 4),
#     ]

# # Convert n column to integer
# df["$n$"] = df["$n$"].astype(int)
# print(df)
# df.to_csv("./tables/table_01_01_20.csv", index=False)


# f = integrate((2 / pi) * sin(w) * cos(x * w) / (w), (w, 0, 16))
# g = integrate((2 / pi) * sin(w) * cos(x * w) / (w), (w, 0, 64))
# xval = np.linspace(-2, 2, 401)
# yval = np.array([])
# zval = np.array([])
# for val in xval:
#     yval = np.append(yval, f.subs(x, val).evalf(4)).astype("float")
#     zval = np.append(zval, g.subs(x, val).evalf(4)).astype("float")


# result = np.transpose(np.vstack((xval, yval, zval)))

# np.savetxt("./tables/table_11_07_07.csv", result, delimiter=",", fmt="%.4f")


# f = integrate(exp(-w) * cos(x * w), (w, 0, 2))
# g = integrate(exp(-w) * cos(x * w), (w, 0, 16))
# xval = np.linspace(-2, 2, 401)
# yval = np.array([])
# zval = np.array([])
# for val in xval:
#     yval = np.append(yval, f.subs(x, val).evalf(4)).astype("float")
#     zval = np.append(zval, g.subs(x, val).evalf(4)).astype("float")


# result = np.transpose(np.vstack((xval, yval, zval)))

# np.savetxt("./tables/table_11_07_09.csv", result, delimiter=",", fmt="%.4f")


# f = integrate((2 / pi) * (1 / (1 - w**2)) * (1 + cos(w)) * cos(x * w), (w, 0, 4))
# print(f)
# g = integrate((2 / pi) * (1 / (1 - w**2)) * (1 + cos(w)) * cos(x * w), (w, 0, 32))
# xval = np.linspace(1, 5, 5)
# yval = np.array([])
# zval = np.array([])
# for val in xval:
#     print(f"{val=}")
#     yval = np.append(yval, f.subs(x, val).evalf(4)).astype("float")
#     zval = np.append(zval, g.subs(x, val).evalf(4)).astype("float")


# result = np.transpose(np.vstack((xval, yval, zval)))

# np.savetxt("./tables/table_11_07_11.csv", result, delimiter=",", fmt="%.4f")


# f = Si(x)
# print(f)
# xval = np.linspace(0, 64, 3200)
# yval = np.array([])
# for val in xval:
#     yval = np.append(yval, f.subs(x, val).evalf(4)).astype("float")


# result = np.transpose(np.vstack((xval, yval)))

# np.savetxt("./tables/table_11_07_15.csv", result, delimiter=",", fmt="%.4f")


# f = (2 / sqrt(pi)) * Sum(
#     (-1) ** k * x ** (2 * k + 1) / (factorial(k) * (2 * k + 1)), (k, 0, 20)
# )
# print(f.doit())
# xval = np.linspace(0, 3, 301)
# yval = np.array([])
# for val in xval:
#     yval = np.append(yval, f.subs(x, val).evalf(4)).astype("float")


# result = np.transpose(np.vstack((xval, yval)))

# np.savetxt("./tables/table_12_07_10.csv", result, delimiter=",", fmt="%.4f")


# f = (2 / sqrt(pi)) * integrate(E ** (-(y**2)), (y, 0, x))
# print(f.doit())
# xval = np.linspace(0, 3, 301)
# yval = np.array([])
# for val in xval:
#     yval = np.append(yval, f.subs(x, val).evalf(4)).astype("float")


# result = np.transpose(np.vstack((xval, yval)))

# np.savetxt("./tables/table_12_07_11.csv", result, delimiter=",", fmt="%.4f")

# xval = np.linspace(0, 100, 101)
# yval = np.array([])
# for val in xval:
#     yval = np.append(yval, erf(10**val).evalf(12)).astype("float")


# result = np.transpose(np.vstack((xval, yval)))

# np.savetxt("./tables/table_12_07_12.csv", result, delimiter=",", fmt="%.12f")


# limit = 5
# for a in range(1, limit + 1):
#     for b in range(a, limit + 1):
#         for c in range(b, limit + 1):
#             for d in range(c, limit + 1):
#                 if (
#                     (a * c != b * d)
#                     and (a * d != b * c)
#                     and ((a * c + b * d) != (a * d + b * c))
#                 ):
#                     print(
#                         f"{a}\t{b}\t{c}\t{d}\t{(a**2 + b**2)*(c**2 + d**2)}\t{abs(a*c-b*d)}\t{abs(a*d+b*c)}\t{abs(a*c+b*d)}\t{abs(a*d-b*c)}"
#                     )


# df = pd.DataFrame(
#     columns=["$m$", "$\\alpha_{m}$", "$A_{m}$", "$(m-0.25)\\pi$", "Error"]
# )
# df.loc[len(df)] = [0, 0.0, 1.0, 1.0, 0.0]
# all_zeros = jn_zeros(0, 15)
# print(f"{all_zeros}")
# for idx, val in enumerate(all_zeros):
#     m = df.iloc[-1]["$m$"]
#     print(f"{m=}")
#     df.loc[len(df.index)] = [
#         (idx + 1),
#         np.round(val, 5),
#         (8 / (val**3 * besselj(1, val))).evalf(5),
#         ((idx + 0.75) * pi).evalf(5),
#         ((idx + 0.75) * pi - val).evalf(5),
#     ]

# Convert n column to integer
# df["$m$"] = df["$m$"].astype(int)
# print(df)
# df[1:].to_csv(
#     "./tables/table_12_10_12.csv", index=False, sep="&", lineterminator="\\\\\r\n"
# )


# f = (
#     55
#     * ((2 * n + 1) / (2**n))
#     * Sum(
#         (-1) ** m
#         * factorial(2 * n - 2 * m)
#         / (factorial(m) * factorial(n - m) * factorial(n - 2 * m + 1)),
#         (m, 0, floor(n / 2)),
#     )
# )

# for val in range(0, 11):
#     print(f"{val}\t{f.subs(n, val).evalf(5)}")


# result = np.transpose(np.vstack((xval, yval)))

# np.savetxt("./tables/table_12_07_11.csv", result, delimiter=",", fmt="%.4f")

# xval = np.linspace(0, 100, 101)
# yval = np.array([])
# for val in xval:
#     yval = np.append(yval, erf(10**val).evalf(12)).astype("float")


# result = np.transpose(np.vstack((xval, yval)))

# np.savetxt("./tables/table_12_11_05.csv", result, delimiter=",", fmt="%.12f")

# order = 10
# xval = np.empty([])
# yval = np.empty([])
# for k in range(0, order):
#     xval = np.append(xval, cos(2 * pi * k / order))
#     yval = np.append(yval, sin(2 * pi * k / order))
# result = np.transpose(np.vstack((xval, yval)))

# np.savetxt("./tables/table_13_01_19.csv", result, delimiter=",", fmt="%.4f")

# z = [rootof(x**5 + 1, i).evalf(5) for i in range(5)]
# for item in z:
#     print(item)

# xs = np.arange(1, 21)
# print(xs)
# xval = np.array([1, 1])
# yval = np.empty([])
# for k in range(19):
#     xval = np.append(xval, xval[-1] + xval[-2])
#     yval = np.append(yval, xval[-1] / xval[-2])

# result = np.transpose(np.vstack((xs, yval)))
# print(yval)
# np.savetxt("./tables/table_15_03_20.csv", result, delimiter=",", fmt="%.4f")

# xs = np.arange(0, 21, 2)
# print(xs)
# f = (cos(x)) ** (-1)
# yval = np.empty([])
# for k in xs:
#     # yval = np.append(yval, xval[-1] / xval[-2])
#     g = diff(f, x, k)
#     # print(f"{k=}\t{(I)**k * ((-1) ** k) * g.subs(x, 0)}")
#     yval = np.append(yval, (I) ** k * ((-1) ** k) * g.subs(x, 0))

# result = np.transpose(np.vstack((xs, yval[1:])))
# print(yval)
# np.savetxt("./tables/table_15_04_15.csv", result, delimiter=",", fmt="%d")

xs = np.arange(0, 21, 2)
print(xs)
f = (cos(x)) ** (-1)
yval = np.empty([])
for k in xs:
    # yval = np.append(yval, xval[-1] / xval[-2])
    g = diff(f, x, k)
    # print(f"{k=}\t{(I)**k * ((-1) ** k) * g.subs(x, 0)}")
    yval = np.append(yval, (I) ** k * ((-1) ** k) * g.subs(x, 0))

result = np.transpose(np.vstack((xs, yval[1:])))
print(yval)
np.savetxt("./tables/table_15_04_16.csv", result, delimiter=",", fmt="%d")
