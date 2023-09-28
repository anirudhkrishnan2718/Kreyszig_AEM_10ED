import numpy as np
from sympy import Matrix, Rational
from scipy.interpolate import interp1d


def Gauss_elim(aug):
    n = aug.shape[0]
    for pivot in range(0, 2):
        temp = aug[pivot:, pivot:]
        sorted = temp[abs(temp[:, 0]).argsort()[::-1]]
        aug[pivot:, pivot:] = sorted

        for j in range(pivot + 1, n):
            subt = (aug[j, pivot] / aug[pivot, pivot]) * aug[pivot, :]
            aug[j, :] = aug[j, :] - subt
    sol = np.zeros(n)
    if aug[-1, -2] == 0:
        print("no unique solution")
        return None
    else:
        sol[-1] = aug[-1, -1] / aug[-1, -2]
        for p in range(n - 2, -1, -1):
            sol[p] = (aug[p, -1] - np.sum(sol[-1:p:-1] * aug[p, -2:p:-1])) / aug[p, p]
        return sol


def LU_factors_doo(a, b):
    # Resolve into LU factors
    n = np.shape(a)[0]
    L = np.identity(a.shape[0])
    U = np.zeros(a.shape)
    U[0, :] = a[0, :]
    L[:, 0] = a[:, 0] / U[0, 0]

    for j in range(1, n):
        for k in range(n):
            if k < j:
                L[j, k] = (1 / U[k, k]) * (
                    a[j, k] - np.sum(np.multiply(L[j, :k], U[:k, k]))
                )
            else:
                U[j, k] = a[j, k] - np.sum(np.multiply(L[j, :j], U[:j, k]))

    # Solve Ly = b using forward substitution
    y = np.zeros(n)
    y[0] = b[0]
    for p in range(1, n):
        y[p] = b[p] - np.sum(np.multiply(L[p, :p], y[:p]))

    # # Solve Ux = y using backward substitution
    x = np.zeros(n)
    x[-1] = y[-1] / U[-1, -1]
    for q in range(n - 2, -1, -1):
        x[q] = (y[q] - np.sum(np.multiply(U[q, q + 1 :], x[q + 1 :]))) / U[q, q]
    return L, U, x, y


def LU_factors_crout(a, b):
    L_doo, U_doo, x, y = LU_factors_doo(a, b)
    U = U_doo
    D = np.diag(np.diag(U_doo))
    for k in range(U.shape[0]):
        U[k, :] = U_doo[k, :] / U_doo[k, k]
    L = np.matmul(L_doo, D)
    return L, U, x, y


def LU_factors_cholesky(a, b):
    L_doo, U_doo, x, y = LU_factors_doo(a, b)
    U = 0 * U_doo
    for k in range(U.shape[0]):
        U[k, :] = U_doo[k, :] / (U_doo[k, k] ** (1 / 2))
    L = np.transpose(U)
    return L, U, x, y


def gauss_seidel(a, b, x):
    if np.any(np.diag(a) == 0):
        print("Invalid matrix")
        return None
    else:
        n = a.shape[0]
        for k in range(n):
            b[k] = b[k] / a[k, k]
            a[k, :] = a[k, :] / a[k, k]

    max_iter = 100
    R = a - np.diag(np.diag(a))
    all_iter = np.zeros((max_iter, n))
    for iter in range(max_iter):
        for k in range(n):
            x[k] = b[k] - np.sum(np.multiply(R[k, :], x))
        all_iter[iter, :] = x
    print(x)
    return all_iter


def Jacobi(a, b, x):
    if np.any(np.diag(a) == 0):
        print("Invalid matrix")
        return None
    else:
        n = a.shape[0]
        for k in range(n):
            b[k] = b[k] / a[k, k]
            a[k, :] = a[k, :] / a[k, k]

    max_iter = 100
    R = a - np.diag(np.diag(a))
    all_iter = np.zeros((max_iter, n))
    for iter in range(max_iter):
        new = np.zeros(n)
        for k in range(n):
            new[k] = b[k] - np.sum(np.multiply(R[k, :], x))
        all_iter[iter, :] = new
        x = new
    print(x)
    return all_iter


def gauss_seidel_overrel(a, b, x):
    if np.any(np.diag(a) == 0):
        print("Invalid matrix")
        return None
    else:
        C = np.matmul(-np.linalg.inv(np.tril(a)), np.triu(a, 1))
        rho = np.max(abs(np.linalg.eigvals(C)))
        omega = 2 / (1 + np.sqrt(1 - rho))
        n = a.shape[0]
        for k in range(n):
            b[k] = b[k] / a[k, k]
            a[k, :] = a[k, :] / a[k, k]

    max_iter = 100
    R = a - np.diag(np.diag(a))
    all_iter = np.zeros((max_iter, n))
    L = np.tril(a, -1)
    U = np.triu(a, 1)
    I = np.identity(n)
    for iter in range(max_iter):
        for k in range(n):
            old = x
            old[k] = x[k] + omega * (
                b[k]
                - np.sum(np.multiply(L[k, :], old))
                - np.sum(np.multiply(U[k, :] + I[k, :], x))
            )
            x[k] = old[k]
        all_iter[iter, :] = x
    print(x)
    return all_iter


# a = np.array([[0, 3, 5, 1.20736], [3, -4, 0, -2.34066], [5, 0, 6, -0.329193]]).astype(
#     "float64"
# )
# answer = Gauss_elim(a)
# for idx, val in enumerate(answer):
#     print(f"x_{idx+1} = {val}")

# a = np.array([[2, 1, 0], [1, 4, 1], [0, 1, 2]])
# b = np.array([0, 0, 0])
# L, U, x, y = LU_factors_cholesky(a, b)
# print(f"y = {np.transpose(y)} \n x = {np.transpose(x)}")
# print(f"L \n {L} \n U \n {U}")

# a = np.array(
#     [
#         [1, -0.25, -0.25, 0],
#         [-0.25, 1, 0, -0.25],
#         [-0.25, 0, 1, -0.25],
#         [0, -0.25, -0.25, 1],
#     ]
# ).astype("float")
# b = np.array([50, 50, 25, 25]).astype("float")
# guess = np.array([100, 100, 100, 100]).astype("float")
# result = gauss_seidel(a, b, guess)
# np.savetxt("./tables/table_20_03_01.csv", result, delimiter=",", fmt="%.6f")

# a = np.array(
#     [
#         [10, 1, 1],
#         [1, 10, 1],
#         [1, 1, 10],
#     ]
# ).astype("float")
# b = np.array([6, 6, 6]).astype("float")
# guess = np.array([0, 0, 0]).astype("float")
# result = gauss_seidel(a, b, guess)
# np.savetxt(
#     "./tables/table_20_03_11_a.csv", np.round(result, 5), delimiter=",", fmt="%.6G"
# )

# A = np.array([[1, 1, 10], [10, 1, 1], [1, 10, 1]])
# C = np.matmul(-np.linalg.inv(np.tril(A)), np.triu(A, 1))
# print(C)
# print(np.linalg.norm(C))

# t = 0.8
# a = np.array(
#     [
#         [1, t, t],
#         [t, 1, t],
#         [t, t, 1],
#     ]
# ).astype("float")
# b = np.array([2, 2, 2]).astype("float")
# guess = np.array([0, 0, 0]).astype("float")
# result = gauss_seidel(a, b, guess)
# np.savetxt(
#     "./tables/table_20_03_13_c.csv", np.round(result, 9), delimiter=",", fmt="%.6G"
# )


# t = 0.8
# a = np.array(
#     [
#         [1, t, t],
#         [t, 1, t],
#         [t, t, 1],
#     ]
# ).astype("float")
# b = np.array([2, 2, 2]).astype("float")
# guess = np.array([0, 0, 0]).astype("float")
# result = gauss_seidel_overrel(a, b, guess)
# np.savetxt(
#     "./tables/table_20_03_13_f.csv", np.round(result, 9), delimiter=",", fmt="%.6G"
# )

# a_g = np.array(
#     [
#         [8, 2, 1],
#         [1, 6, 2],
#         [4, 0, 5],
#     ]
# ).astype("float")
# b_g = np.array([-11.5, 18.5, 12.5]).astype("float")
# guess_g = np.array([1, 1, 1]).astype("float")
# result = gauss_seidel(a_g, b_g, guess_g)
# np.savetxt("./tables/table_20_03_16_a.csv", result, delimiter=",", fmt="%.15G")

# a_J = np.array(
#     [
#         [8, 2, 1],
#         [1, 6, 2],
#         [4, 0, 5],
#     ]
# ).astype("float")
# b_J = np.array([-11.5, 18.5, 12.5]).astype("float")
# guess_J = np.array([1, 1, 1]).astype("float")
# resultJ = Jacobi(a_J, b_J, guess_J)
# np.savetxt("./tables/table_20_03_16_b.csv", resultJ, delimiter=",", fmt="%.15G")

# A = np.array([[1, 1 / 4, 1 / 8], [1 / 6, 1, 1 / 3], [4 / 5, 0, 1]])
# C = A - np.identity(A.shape[0])
# print(C)
# print(np.linalg.eigvals(C))

# A = np.array([[10, 1, 1], [1, 10, 1], [1, 1, 10]])
# print(f"Frobenius\t {np.linalg.norm(A)}")
# print(f"Row\t\t {np.linalg.norm(A, np.inf)}")
# print(f"Column\t\t {np.linalg.norm(A, 1)}")

# a = Matrix(
#     [
#         [21, Rational(21, 2), 7, Rational(21, 4)],
#         [Rational(21, 2), 7, Rational(21, 4), Rational(42, 10)],
#         [7, Rational(21, 4), Rational(42, 10), Rational(7, 2)],
#         [Rational(21, 4), Rational(42, 10), Rational(7, 2), 3],
#     ]
# )

# a = np.array([[3, 1.7], [1.7, 1]])
# b_1 = np.array([4.7, 2.7])
# b_2 = np.array([4.7, 2.71])
# print(np.linalg.solve(a, b_1))
# print(np.linalg.solve(a, b_2))
# print(np.linalg.cond(a))

# xvals = np.arange(2, 11)
# yvals = np.zeros(xvals.shape[0])
# for idx, n in enumerate(xvals):
#     h = np.zeros((n, n))
#     for j in range(n):
#         for k in range(n):
#             h[j, k] = 1 / (j + k + 1)
#     yvals[idx] = np.log(np.linalg.cond(h, np.inf))

# print(np.polyfit(xvals, yvals, 1))
# print(yvals)

xvals = np.array([0, 1, 2, 3, 4])
yvals = np.array([1, 1, 2, 3, 4])
result = np.polyfit(xvals, yvals, 2, full=True)
for item in result[:2]:
    print(item)
