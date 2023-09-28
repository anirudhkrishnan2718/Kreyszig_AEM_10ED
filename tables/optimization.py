import numpy as np


def gradient_descent(start_x):
    n = start_x.shape[0]
    max_iter = 7
    all_points = np.zeros((max_iter + 1, n))
    all_points[0, :] = start_x
    for iter in range(max_iter):
        c = all_points[iter, :]
        t = t_star(c)
        all_points[iter + 1, :] = np.array([c[0] * (1 - 2 * t), c[1] * (1 - 8 * t)])
        print(
            f"{iter+1:<4d} \t {all_points[iter+1,0]:>12.3f} {all_points[iter+1,1]:>12.3f}"
        )
    return all_points


def t_star(x):
    return ((x[0]) ** 2 + 16 * x[1] ** 2) / (2 * (x[0]) ** 2 + 128 * x[1] ** 2)


def simplex(m_in, vars_in, minimize=False):
    print(m_in)
    iter = 0
    obj = -1 * m_in[0, :vars]
    while np.any(m_in[0, :-1] < 0):
        # while iter < 6:
        n_row, n_col = m_in.shape
        pivot_col = np.argwhere(m_in[0, :-1] < 0)[0, 0]
        quotients = np.divide(
            m_in[1:, -1],
            m_in[1:, pivot_col],
            out=np.zeros_like(m_in[1:, -1]),
            where=m_in[1:, pivot_col] != 0,
        )
        quotients[quotients <= 0] = np.inf
        pivot_row = np.argmin(quotients) + 1
        p = m_in[pivot_row, pivot_col]
        print(f"{p=}")
        for j in range(n_row):
            if j != pivot_row:
                m_in[j, :] -= m_in[pivot_row, :] * (m_in[j, pivot_col]) / p
        print(m_in)
        iter += 1
    if minimize is True:
        return -1 * m_in[0, -1]
    else:
        return m_in[0, -1]


# start_x = np.array([3, 0.1]).astype("float")
# result = gradient_descent(start_x)

# np.savetxt("./tables/table_22_01_10_a.csv", result, delimiter=",", fmt="%.3f")
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)
m = np.array(
    [
        [-2, -3, -2, 0, 0, 0],
        [1, 2, -4, 1, 0, 2],
        [1, 2, 2, 0, 1, 5],
    ]
).astype("float")
vars = 2
ans = simplex(m, vars)
print(ans)
