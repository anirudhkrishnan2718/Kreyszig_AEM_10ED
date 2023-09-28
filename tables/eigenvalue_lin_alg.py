import numpy as np
from sympy import Matrix
from sympy.abc import t, p, q


def power_method(a, x):
    max_iter = 10
    for j in range(max_iter):
        y = a @ x
        m_0 = np.dot(x.transpose(), x)
        m_1 = np.dot(x.transpose(), y)
        m_2 = np.dot(y.transpose(), y)
        q = m_1 / m_0
        delta = np.sqrt((m_2 / m_0) - q**2)
        print(f"{j=} \t {q=} \t {delta=}")
        x = y


def power_method_scaling(a, x):
    max_iter = 10
    for j in range(max_iter):
        y = a @ x
        m_0 = np.dot(x.transpose(), x)
        m_1 = np.dot(x.transpose(), y)
        m_2 = np.dot(y.transpose(), y)
        q = m_1 / m_0
        delta = np.sqrt((m_2 / m_0) - q**2)
        print(f"{j=} \t {q=} \t {delta=}")
    return q


def householder_tridiag(a):
    n = np.shape(a)[0]
    v = np.zeros((n, 1))
    for r in range(n - 2):
        S = np.sqrt(np.inner(a[r + 1 :, r], a[r + 1 :, r]))
        v = np.zeros((n, 1))
        v[r + 1, 0] = np.sqrt(0.5 * (1 + abs(a[r + 1, r]) / S))
        for j in range(r + 2, n):
            v[j, 0] = (a[j, r] * np.sign(a[r + 1, r])) / (2 * S * v[r + 1, 0])
        # print(f"{r=} \n {np.round(v, 4)}")
        P = np.identity(n) - 2 * (v @ np.transpose(v))
        a = P @ a @ P
        # print(np.round(a, 4))
    return a


def QR_factorization(b):
    n = np.shape(b)[0]
    max_iter = 10
    for m in range(max_iter):
        C_cumulative = np.identity(n)
        for r in range(1, n):
            C = np.identity(n)
            c = 1 / np.sqrt(1 + (b[r, r - 1] / b[r - 1, r - 1]) ** 2)
            s = (b[r, r - 1] / b[r - 1, r - 1]) / np.sqrt(
                1 + (b[r, r - 1] / b[r - 1, r - 1]) ** 2
            )
            C[r - 1 : r + 1, r - 1 : r + 1] = np.array([[c, s], [-s, c]])
            b = C @ b
            C_cumulative = C @ C_cumulative
        b = b @ np.linalg.inv(C_cumulative)
        error = np.max(np.abs(b - np.diag(np.diag(b))))
        print(f"{m=}\t{np.diag(b)}\t{error}")


# a = np.array([[5, 2, 4], [-2, 0, 2], [2, 4, 7]])
# t = np.array([[100, 0, 0], [0, 1, 0], [0, 0, 1]])
# b = np.linalg.inv(t) @ a @ t
# print(b)
# eigvals = np.linalg.eigvals(a)
# for item in eigvals:
#     print(item)


# p = Matrix([[10, 0.1, -0.2], [0.1, 6, 0], [-0.2, 0, 3]])
# q = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, t]])
# print(q.inv() * p * q)

# a_in = np.array([[1, 1], [0, 1]])
# x_in = np.ones((a_in.shape[0], 1))
# x_in = np.array([[3], [1]])
# power_method(a_in, x_in)
# print(np.linalg.eigvals(a_in))

# a = np.array([[3, 52, 10, 42], [52, 59, 44, 80], [10, 44, 39, 42], [42, 80, 42, 35]])
# b = householder_tridiag(a)

b_in = np.array([[14.2, -0.1, 0], [-0.1, -6.3, 0.2], [0, 0.2, 2.1]])
print(np.linalg.eigvals(b_in))
QR_factorization(b_in)
