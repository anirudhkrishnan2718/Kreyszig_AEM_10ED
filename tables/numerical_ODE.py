import numpy as np
from scipy.special import fresnel, airy


def Euler_ode(h, x_init, y_init):
    N = 25
    x_end = x_init + (N + 1) * h
    xvals = np.arange(x_init, x_end, h)
    yvals = np.zeros(xvals.shape[0])
    yvals[0] = y_init
    for i in range(1, N + 1):
        yvals[i] = yvals[i - 1] + h * f(xvals[i - 1], yvals[i - 1])
    error = yvals - g(xvals)
    for show in [5, 15, 25]:
        print(
            f"{show:<6d} {xvals[show]:<6.2f} {yvals[show]:<20.9f}{error[show]:<20.9e}"
        )


def Euler_ode_improved(h, x_init, y_init):
    N = 10
    x_end = x_init + (N + 1) * h
    xvals = np.arange(x_init, x_end, h)
    yvals = np.zeros(xvals.shape[0])
    yvals[0] = y_init
    for i in range(1, N + 1):
        k_1 = h * f(xvals[i - 1], yvals[i - 1])
        k_2 = h * f(xvals[i], yvals[i - 1] + k_1)
        yvals[i] = yvals[i - 1] + 0.5 * (k_1 + k_2)
    error = yvals - g(xvals)
    for show in [5, 10]:
        print(
            f"{show:<6d} {xvals[show]:<6.2f} {yvals[show]:<20.9f}{error[show]:<20.9e}"
        )
    return xvals, yvals


def RK_classic(h, x_init, y_init):
    N = 20
    x_end = x_init + (N + 1) * h
    xvals = np.arange(x_init, x_end, h)
    yvals = np.zeros(xvals.shape[0])
    yvals[0] = y_init
    for i in range(1, N + 1):
        k_1 = h * f(xvals[i - 1], yvals[i - 1])
        k_2 = h * f(xvals[i - 1] + 0.5 * h, yvals[i - 1] + 0.5 * k_1)
        k_3 = h * f(xvals[i - 1] + 0.5 * h, yvals[i - 1] + 0.5 * k_2)
        k_4 = h * f(xvals[i - 1] + h, yvals[i - 1] + k_3)
        yvals[i] = yvals[i - 1] + ((k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6)
    error = yvals - g(xvals)
    for show in [1, 5, 10]:
        print(
            f"{show:<6d} {xvals[show]:<6.2f} {yvals[show]:<20.9f}{error[show]:<20.9e}"
        )
    return xvals, yvals


def RK_third(h, x_init, y_init):
    N = 25
    x_end = x_init + (N + 1) * h
    xvals = np.arange(x_init, x_end, h)
    yvals = np.zeros(xvals.shape[0])
    yvals[0] = y_init
    for i in range(1, N + 1):
        k_1 = h * f(xvals[i - 1], yvals[i - 1])
        k_2 = h * f(xvals[i - 1] + 0.5 * h, yvals[i - 1] + 0.5 * k_1)
        k_3 = h * f(xvals[i], yvals[i - 1] - k_1 + 2 * k_2)
        yvals[i] = yvals[i - 1] + ((k_1 + 4 * k_2 + k_3) / 6)
    error = yvals - g(xvals)
    for show in [5, 15, 25]:
        print(
            f"{show:<6d} {xvals[show]:<6.2f} {yvals[show]:<20.9f}{error[show]:<20.9e}"
        )


def RKF_45(h, x_init, y_init):
    N = 3
    x_end = x_init + (N + 1) * h
    xvals = np.arange(x_init, x_end, h)
    yvals = np.zeros(xvals.shape[0])
    zvals = np.zeros(xvals.shape[0])
    yvals[0] = y_init
    zvals[0] = y_init
    for i in range(1, N + 1):
        k_1 = h * f(xvals[i - 1], yvals[i - 1])
        k_2 = h * f(xvals[i - 1] + 0.25 * h, yvals[i - 1] + 0.25 * k_1)
        k_3 = h * f(xvals[i - 1] + (3 / 8) * h, yvals[i - 1] + (3 * k_1 + 9 * k_2) / 32)
        k_4 = h * f(
            xvals[i - 1] + (12 / 13) * h,
            yvals[i - 1] + (1932 * k_1 - 7200 * k_2 + 7296 * k_3) / 2197,
        )
        k_5 = h * f(
            xvals[i - 1] + h,
            yvals[i - 1]
            + (439 / 216) * k_1
            - 8 * k_2
            + (3680 / 513) * k_3
            - (845 / 4104) * k_4,
        )
        k_6 = h * f(
            xvals[i - 1] + 0.5 * h,
            yvals[i - 1]
            - (8 / 27) * k_1
            + 2 * k_2
            - (3544 / 2565) * k_3
            + (1859 / 4104) * k_4
            - (11 / 40) * k_5,
        )
        yvals[i] = (
            yvals[i - 1]
            + (16 / 135) * k_1
            + (6656 / 12825) * k_3
            + (28561 / 56430) * k_4
            - (9 / 50) * k_5
            + (2 / 55) * k_6
        )
        zvals[i] = (
            zvals[i - 1]
            + (25 / 216) * k_1
            + (1408 / 2565) * k_3
            + (2197 / 4104) * k_4
            - (1 / 5) * k_5
        )
    error = yvals - zvals
    # for show in [1, 5, 10]:
    #     print(
    #         f"{show:<6d} {xvals[show]:<6.2f} {yvals[show]:<20.9f}{error[show]:<20.9e}"
    #     )
    return xvals, yvals


def adams_moulton(h, x_init, y_init):
    N = 7
    x_end = x_init[-1] + (N + 1) * h
    xvals = np.concatenate((x_init, np.arange(x_init[-1], x_end, h)[1:]))
    yvals = np.zeros(xvals.shape[0])
    yvals[0:4] = y_init
    for i in range(3, N + 3):
        z = yvals[i] + (h / 24) * (
            55 * f(xvals[i], yvals[i])
            - 59 * f(xvals[i - 1], yvals[i - 1])
            + 37 * f(xvals[i - 2], yvals[i - 2])
            - 9 * f(xvals[i - 3], yvals[i - 3])
        )
        yvals[i + 1] = yvals[i] + (h / 24) * (
            9 * f(xvals[i + 1], z)
            + 19 * f(xvals[i], yvals[i])
            - 5 * f(xvals[i - 1], yvals[i - 1])
            + f(xvals[i - 2], yvals[i - 2])
        )
    error = yvals - g(xvals)
    for show in [5, 10]:
        print(
            f"{show:<6d} {xvals[show]:<6.2f} {yvals[show]:<20.9f}{error[show]:<20.9e}"
        )


def adams_moulton_simple(h, x_init, y_init):
    N = 18
    x_end = x_init[-1] + (N + 1) * h
    xvals = np.concatenate((x_init, np.arange(x_init[-1], x_end, h)[1:]))
    yvals = np.zeros(xvals.shape[0])
    yvals[0:3] = y_init
    for i in range(2, N + 2):
        z = yvals[i] + (h / 12) * (
            23 * f(xvals[i], yvals[i])
            - 16 * f(xvals[i - 1], yvals[i - 1])
            + 5 * f(xvals[i - 2], yvals[i - 2])
        )
        yvals[i + 1] = yvals[i] + (h / 12) * (
            5 * f(xvals[i + 1], z)
            + 8 * f(xvals[i], yvals[i])
            - 1 * f(xvals[i - 1], yvals[i - 1])
        )
    error = yvals - g(xvals)
    for show in [10, 20]:
        print(
            f"{show:<6d} {xvals[show]:<6.2f} {yvals[show]:<20.9f}{error[show]:<20.9e}"
        )


def systems_euler(h, x_init, y_init):
    d = y_init.shape[0]
    N = 20
    xvals = np.array([x_init + r * h for r in range(N + 1)])
    yvals = np.zeros((N + 1, d))
    yvals[0, :] = y_init
    for i in range(1, N + 1):
        yvals[i, :] = yvals[i - 1, :] + h * f(xvals[i - 1], yvals[i - 1, :])
    error = yvals - np.transpose(g(xvals))
    for show in range(0, N + 1, 2):
        print(f"{xvals[show]:<3f}\t {yvals[show,:]}\t{error[show,:]}")
    result = np.transpose(np.vstack((xvals, yvals[:, 0])))
    np.savetxt("./tables/table_21_03_15_b5.csv", result, delimiter=",", fmt="%.4f")


def systems_RK(h, x_init, y_init):
    d = y_init.shape[0]
    N = 10
    xvals = np.array([x_init + r * h for r in range(N + 1)])
    yvals = np.zeros((N + 1, d))
    yvals[0, :] = y_init
    for i in range(1, N + 1):
        k_1 = h * f(xvals[i - 1], yvals[i - 1, :])
        k_2 = h * f(xvals[i - 1] + 0.5 * h, yvals[i - 1, :] + 0.5 * k_1)
        k_3 = h * f(xvals[i - 1] + 0.5 * h, yvals[i - 1, :] + 0.5 * k_2)
        k_4 = h * f(xvals[i - 1] + h, yvals[i - 1, :] + k_3)
        yvals[i, :] = yvals[i - 1, :] + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
    error = yvals - np.transpose(g(xvals))
    for show in range(1, N + 1):
        print(f"{xvals[show]:<6f}\t {yvals[show,:]}\t{error[show,:]}")
    result = np.transpose(np.vstack((xvals, yvals[:, 0])))
    np.savetxt("./tables/table_21_03_15_c6.csv", result, delimiter=",", fmt="%.4f")


def systems_RK_Nystrom(h, x_init, y_init, z_init):
    N = 10
    x_end = x_init + (N + 1) * h
    xvals = np.arange(x_init, x_end, h)
    yvals = np.zeros(xvals.shape[0])
    zvals = np.zeros(xvals.shape[0])
    yvals[0] = y_init
    zvals[0] = z_init
    for i in range(1, N + 1, 2):
        k_1 = 0.5 * h * f(xvals[i - 1], yvals[i - 1])
        k_2 = (
            0.5
            * h
            * f(
                xvals[i - 1] + (h / 2),
                yvals[i - 1] + (h / 2) * (zvals[i - 1] + 0.5 * k_1),
            )
        )
        k_4 = 0.5 * h * f(xvals[i - 1] + h, yvals[i - 1] + h * (zvals[i - 1] + k_2))
        yvals[i] = yvals[i - 1] + h * (zvals[i - 1] + (1 / 3) * (k_1 + 2 * k_2))
        zvals[i] = zvals[i - 1] + ((k_1 + 4 * k_2 + k_4) / 3)
    error = yvals - g(xvals)
    for show in [1, 2, 3, 4, 5]:
        print(
            f"{show:<6d} {xvals[show]:<6.2f} {yvals[show]:<20.9f}{error[show]:<20.9e}"
        )


def systems_backward_euler(h, x_init, y_init):
    d = y_init.shape[0]
    N = 10
    D = 1 + 11 * h + 10 * h**2
    xvals = np.array([x_init + r * h for r in range(N + 1)])
    yvals = np.zeros((N + 1, d))
    yvals[0, :] = y_init
    for i in range(1, N + 1):
        yvals[i, 0] = (1 / D) * (
            (1 + 11 * h) * yvals[i - 1, 0]
            + h * yvals[i - 1, 1]
            + 10 * h**2 * xvals[i - 1]
            + 11 * h**2
            + 10 * h**3
        )
        yvals[i, 1] = (1 / D) * (
            (-10 * h) * yvals[i - 1, 0]
            + yvals[i - 1, 1]
            + 10 * h * xvals[i - 1]
            + 11 * h
            + 10 * h**2
        )
    error = yvals - np.transpose(g(xvals))
    for show in range(0, N + 1):
        print(f"{xvals[show]:<3f}\t {yvals[show,:]}\t{error[show,:]}")
    result = np.transpose(np.vstack((xvals, yvals[:, 0])))
    np.savetxt("./tables/table_21_03_15_d4.csv", result, delimiter=",", fmt="%.4f")


# h = 0.2
# x_in = np.array([1 + item * h for item in range(4)])
# y_in = np.array([3, 3.07245829914744, 3.15594676761190, 3.24961536185438])
# x_start = 0
# y_start = 0


def f(x_f, y_f):
    # the derivative as a function of x and y
    # return np.array([0 * y_f[0] + 1 * y_f[1], x_f * y_f[0] - 0 * y_f[1]])
    return np.array([y_f[1], -10 * y_f[0] - 11 * y_f[1] + 10 * x_f + 11])


def g(x_g):
    # the ODE solution as a function of x
    return np.array(
        [
            np.exp(-x_g) + np.exp(-10 * x_g) + x_g,
            -np.exp(-x_g) - 10 * np.exp(-10 * x_g) + 1,
        ]
    )


# x_in, y_in = Euler_ode_improved(h, x_start, y_start)
# adams_moulton(h, x_in[:4], y_in[:4])


# Euler_ode(h, x_0, y_0)
# Euler_ode_improved(h, x_0, y_0)
# RK_classic(h, x_start, y_start)
# RKF_45(h, x_0, y_0)
# print(g(1), g(3), g(5))


# def plot_g(x_p, a):
#     return 0.01 * x_p**2 - (2 * a) / (
#         a * np.sqrt(2 * np.pi) * fresnel(x_p * np.sqrt(2 / np.pi))[0] - 2
#     )


# xv = np.arange(0, 5, 0.05)
# result = xv
# for initial in np.arange(-1, 1, 0.2):
#     yv = plot_g(xv, initial)
#     result = np.vstack((result, yv))

# np.savetxt(
#     "./tables/table_21_01_15_b.csv", np.transpose(result), delimiter=",", fmt="%.6f"
# )

x_in = 0.0
y_in = np.array([2.0, -10.0])
h = 0.5
systems_backward_euler(h, x_in, y_in)
# systems_RK_Nystrom(h, x_in, y_in, z_in)
# print(airy(0.2))
