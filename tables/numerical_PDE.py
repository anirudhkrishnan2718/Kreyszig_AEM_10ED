import numpy as np
import scipy


def gauss_seidel(a, b, x):
    if np.any(np.diag(a) == 0):
        print("Invalid matrix")
        return None
    else:
        n = a.shape[0]
        for k in range(n):
            b[k] = b[k] / a[k, k]
            a[k, :] = a[k, :] / a[k, k]
    max_iter = 20
    R = a - np.diag(np.diag(a))
    all_iter = np.zeros((max_iter, n))
    for iter in range(max_iter):
        for k in range(n):
            x[k] = b[k] - np.sum(np.multiply(R[k, :], x))
        all_iter[iter, :] = x
    return all_iter


# a_g = np.array(
#     [
#         [-4, 1, 0, 1, 0, 0, 0, 0, 0],
#         [1, -4, 0, 1],
#         [1, 0, -4, 1],
#         [0, 1, 1, -4],
#     ]
# ).astype("float")
# b_g = np.array([-200, -200, -100, -100]).astype("float")
# guess_g = np.array([100, 100, 100, 100]).astype("float")
# result = gauss_seidel(a_g, b_g, guess_g)
# np.savetxt("./tables/table_21_04_2.csv", result, delimiter=",", fmt="%.15G")


def make_liebmann_matrix(m):
    n_row, n_col = m.shape
    n_mesh = (n_row - 2) * (n_col - 2)

    coeffs_new = np.array([])
    for i in range(1, n_row - 1):
        for j in range(1, n_col - 1):
            m_clone = np.zeros((n_row, n_col))
            m_clone[i, j] = -4
            if i != 1:
                m_clone[i - 1, j] = 1
            if i != n_row - 2:
                m_clone[i + 1, j] = 1
            if j != 1:
                m_clone[i, j - 1] = 1
            if j != n_col - 2:
                m_clone[i, j + 1] = 1
            coeffs_new = np.concatenate((coeffs_new, np.ravel(m_clone[1:-1, 1:-1])))

    bs = np.zeros((n_row - 2) * (n_col - 2))
    point = 0
    for i in range(1, n_row - 1):
        for j in range(1, n_col - 1):
            b = 0
            if i == 1:
                b -= m[0, j]
            if i == n_row - 2:
                b -= m[n_row - 1, j]
            if j == 1:
                b -= m[i, 0]
            if j == n_col - 2:
                b -= m[i, n_col - 1]
            bs[point] = b
            point += 1
    return coeffs_new.reshape((n_mesh, -1)).astype("float"), bs


def ADI_method(m):
    n_row, n_col = m.shape
    star = 2 * np.sin(np.pi / max(n_row - 1, n_col - 1))
    # star = 2
    trid = scipy.sparse.diags(
        [1, -(2 + star), 1], [-1, 0, 1], shape=(n_col, n_col)
    ).toarray()
    max_iter = 1
    for iter in range(max_iter):
        for j in range(1, n_row - 1):
            bs = np.zeros(n_col)

            for i in range(1, n_col - 1):
                bs[i] = -m[j - 1, i] - m[j + 1, i] + (2 - star) * m[j, i]
                if i == 1:
                    bs[i] -= m[j, 0]
                if i == n_col - 2:
                    bs[i] -= m[j, n_col - 1]
            m[j, 1 : n_col - 1] = np.linalg.solve(trid[1:-1, 1:-1], bs[1 : n_col - 1])

        for q in range(1, n_col - 1):
            bs = np.zeros(n_row)

            for p in range(1, n_row - 1):
                bs[p] = -m[p, q - 1] - m[p, q + 1] + (2 - star) * m[p, q]
                if p == 1:
                    bs[p] -= m[0, q]
                if p == n_row - 2:
                    bs[p] -= m[n_row - 1, q]
            m[1 : n_row - 1, q] = np.linalg.solve(trid[1:-1, 1:-1], bs[1 : n_row - 1])

    print(m)


def heat_eq_explicit(m_in, h_in, r_in=0.25):
    max_iter = 5
    n = m_in.shape[0]
    xvals = np.linspace(0, 1, n)
    print(f"{xvals=}")
    result, analytical = np.zeros((max_iter + 1, n)), np.zeros((max_iter + 1, n))
    # analytical[0, :] = heat(xvals, 0)
    k_in = r_in * h_in ** (2)
    time_array = np.arange(0, k_in * (max_iter + 1), k_in)
    result[0] = m_in

    for j in range(max_iter):
        # analytical[j + 1, :] = heat(xvals, time_array[j + 1])
        # for i in range(1, n - 1):
        result[j + 1, 1 : n - 1] = (1 - 2 * r) * result[j, 1 : n - 1] + r * (
            result[j, 2:] + result[j, : n - 2]
        )
    for show in range(0, max_iter + 1):
        print(f"{time_array[show]:<6.2f} {np.round(result[show,:],6)}")
    # print(f"error \t {np.amax(abs(analytical - result))}")
    # save = np.vstack((np.transpose(xvals), result[::4]))
    # np.savetxt(
    #     "./tables/table_21_06_04_a.csv", np.transpose(save), delimiter=",", fmt="%.3G"
    # )


def Crank_Nicholson(m_in, h_in, r_in=1):
    max_iter = 5
    n = m_in.shape[0]
    xvals = np.linspace(0, 1, n)
    result, analytical = np.zeros((max_iter + 1, n)), np.zeros((max_iter + 1, n))
    # analytical[0, :] = heat(xvals, 0)
    result[0] = m_in
    k_in = r_in * h_in ** (2)
    time_array = np.arange(0, k_in * (max_iter + 1), k_in)

    trid = scipy.sparse.diags(
        [-r_in, (2 + 2 * r_in), -r_in], [-1, 0, 1], shape=(n - 2, n - 2)
    ).toarray()

    for j in range(max_iter):
        # analytical[j + 1, :] = heat(xvals, time_array[j + 1])
        bs = (2 - 2 * r_in) * result[j, 1 : n - 1] + r_in * (
            result[j, : n - 2] + result[j, 2:]
        )
        result[j + 1, 1 : n - 1] = np.linalg.solve(trid, bs)

    for show in range(1, max_iter + 1):
        print(f"{time_array[show]:<6.2f} {np.round(result[show,:],6)}")
    # print(f"error \t {np.amax(abs(analytical - result))}")
    # save = np.vstack((np.transpose(xvals), result))
    # np.savetxt(
    #     "./tables/table_21_06_04_b.csv", np.transpose(save), delimiter=",", fmt="%.3G"
    # )


def hyperbolic_PDE(m_in, h_in, k_in):
    if h_in == k_in:
        max_iter = 5
        n = m_in.shape[0]
        result = np.zeros((max_iter + 1, n))
        time_array = np.arange(0, k_in * (max_iter + 1), k_in)
        result[:, -1] = (1 + time_array[:-1]) ** 2
        result[0, 0] = np.vectorize(init_disp)(m_in[0])
        result[1, 1:-1] = k_in * np.vectorize(init_velo)(m_in[1:-1]) + 0.5 * (
            result[0, :-2] + result[0, 2:]
        )
        for j in range(1, max_iter):
            result[j + 1, 0] = result[j, 1] + 0.4 * j * h_in - result[j - 1, 0]
            result[j + 1, 1:-1] = result[j, :-2] + result[j, 2:] - result[j - 1, 1:-1]
        for show in range(1, max_iter + 1):
            print(f"{time_array[show]:<6.2f} {np.round(result[show,:],6)}")


def heat(x):
    return x * (1 - x)


# m_in = np.array([0, 0.588, -0.951, 0.951, 0.588, 0])

# m_in[:, 0] = np.array([-np.sin(np.pi * i / 4) for i in range(rows)])
# m_in[:, cols - 1] = np.array([-np.sin(np.pi * i / 4) for i in range(rows)])
# m_in[0, :] = np.array([np.sin(np.pi * j / 4) for j in range(cols)])
# m_in[rows - 1, :] = np.array([np.sin(np.pi * j / 4) for j in range(cols)])

# m_in[:, 0] = 110
# m_in[:, cols - 1] = -10
# m_in[0, :] = 220
# m_in[rows - 1, :] = 220
# m_in[1:-1, 1:-1] = 0

# make_liebmann_matrix(m_in)
# coeff_out, b_out = make_liebmann_matrix(m_in)
# print(coeff_out, "\n", b_out)
# guess = 0 * np.ones((rows - 2) * (cols - 2))
# x_out = gauss_seidel(coeff_out, b_out, guess)
# answer = np.linalg.solve(coeff_out, b_out)
# print(f"{answer.reshape(rows-2, cols-2)}")
# print(x_out[-1, :].reshape(rows - 2, cols - 2))

# print(f"error \t {np.amax(abs(answer - x_out[-1,:])):.2e}")

# ADI_method(m_in)


# m_in = np.zeros((4, 4))
# h = 0.5
# rows, cols = m_in.shape
# m_in[:, cols - 1] = np.array([0, 0.375, 3, 0])

# A = np.array([[-4, 1, 1, 0], [1, -4, 0, 1], [1, 0, -4, 1], [0, 4, 4, -24]])
# B = np.array([-3, -12, 0, -28])
# print(np.linalg.solve(A, B))

# start = 0
# end = 1
# h = 0.2
# r = 1
# xvals = np.arange(start, end + h / 2, h)
# print(xvals)

# m_in = np.array([heat(item) for item in xvals])

# Crank_Nicholson(m_in, h, r)


def init_disp(x):
    # if x >= 0 and x < 0.2:
    #     return x
    # elif x >= 0.2 and x <= 1:
    #     return 0.25 * (1 - x)
    return 1 - np.cos(2 * np.pi * x)


def init_velo(x):
    return x * (1 - x)


start, stop = 0, 1
h, k = 0.2, 0.2
m_in = np.arange(start, stop + h / 2, h).astype("float")
hyperbolic_PDE(m_in, h, k)
