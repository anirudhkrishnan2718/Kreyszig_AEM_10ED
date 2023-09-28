import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm, t, chi2, f, poisson, hypergeom, binom
from scipy.special import jv

# df = pd.read_csv("./tables/table_24_01_08.csv", names=["Values"])
# df2 = df.sort_values(by="Values")[1:]
# print(df2.describe())

# df = pd.DataFrame(columns=["Error"])
# for i in range(1, 21):
#     a, b = np.math.factorial(i), np.sqrt(2 * np.pi * i) * (i / np.e) ** i
#     df.at[i, "Error"] = (a - b) / a

# print(df.to_latex(float_format="{:0.2f}".format))
# df.to_csv("./tables/table_24_04_13_b.csv", header=False)
# print(df.index.to_numpy())


# def func(x, a, b):
#     return (np.e ** (a / x)) + b


def mean_z_known_var(data, stddev, conf):
    xbar = np.mean(data)
    c = norm.ppf((conf + 1) / 2)
    k = c * stddev / np.sqrt(len(data))
    results = pd.DataFrame(
        [xbar, conf, c, k, xbar - k, xbar + k],
        index=[
            "\\bar{x}",
            "\\gamma",
            "c",
            "k",
            "\\text{Lower limit}",
            "\\text{Upper limit}",
        ],
    )
    print(results.to_latex(escape=False, float_format="%.2f"))


def mean_z_unknown_var(data, conf):
    xbar = np.mean(data)
    c = t.ppf((conf + 1) / 2, len(data) - 1)
    k = c * np.std(data, ddof=1) / np.sqrt(len(data))
    results = pd.DataFrame(
        [xbar, conf, c, k, xbar - k, xbar + k],
        index=[
            "\\bar{x}",
            "\\gamma",
            "c",
            "k",
            "\\text{Lower limit}",
            "\\text{Upper limit}",
        ],
    )
    print(np.var(data))
    print(results.to_latex(escape=False, float_format="%.4f"))


def var_z(data, conf, sample_mean=0, sample_var=1):
    c1 = chi2.ppf((1 - conf) / 2, len(data) - 1)
    c2 = chi2.ppf((1 + conf) / 2, len(data) - 1)
    k1 = ((len(data) - 1) * sample_var) / c1
    k2 = ((len(data) - 1) * sample_var) / c2
    results = pd.DataFrame(
        [conf, c1, c2, k2, sample_var, k1],
        index=[
            "\\gamma",
            "c_1",
            "c_2",
            "\\text{Lower limit}",
            "s",
            "\\text{Upper limit}",
        ],
    )
    print(np.var(data))
    print(results.to_latex(escape=False, float_format="%.4f"))


def mean_test_unknown_var(
    data,
    side,
    null,
    alpha=0.05,
    s=None,
):
    xbar = np.mean(data)
    if s == None:
        s = np.std(data, ddof=1)
    n = len(data)
    if side == "right":
        statistic = (xbar - null) / (s / np.sqrt(n))
        limit = t.ppf(1 - alpha, df=n - 1)
        hypo = "Rejected" if (statistic > limit) else "Accepted"
        results = pd.DataFrame(
            [alpha, xbar, s, n, statistic, limit],
            index=[
                "\\alpha",
                "\\bar{x}",
                "s",
                "n",
                "\\text{Statistic}",
                "\\text{Limit}",
            ],
        )
    if side == "two":
        statistic = (xbar - null) / (s / np.sqrt(n))
        lower_limit = t.ppf(0.5 * (1 - alpha), df=n - 1)
        upper_limit = t.ppf(0.5 * (1 + alpha), df=n - 1)
        if (statistic < lower_limit) or (statistic > upper_limit):
            hypo = "Rejected"
        else:
            hypo = "Accepted"
        results = pd.DataFrame(
            [alpha, xbar, s, n, statistic, lower_limit, upper_limit],
            index=[
                "\\alpha",
                "\\bar{x}",
                "s",
                "n",
                "\\text{Statistic}",
                "\\text{Lower Limit}",
                "\\text{Upper Limit}",
            ],
        )
    if side == "left":
        statistic = (xbar - null) / (s / np.sqrt(n))
        limit = t.ppf(alpha, df=n - 1)
        if statistic < limit:
            hypo = "Rejected"
        else:
            hypo = "Accepted"
        results = pd.DataFrame(
            [alpha, xbar, s, n, statistic, limit],
            index=[
                "\\alpha",
                "\\bar{x}",
                "s",
                "n",
                "\\text{Statistic}",
                "\\text{Limit}",
            ],
        )
    print(np.var(data))
    print(results.to_latex(escape=False, float_format="%4.4f"))
    print(f"{hypo=}")


def mean_test_two_sample_unknown_var(
    datax, datay, side, null=0, alpha=0.05, sx=None, sy=None
):
    n1, n2 = len(datax), len(datay)
    xbar = np.mean(datax) - np.mean(datay)
    if sx == None:
        sx = np.std(datax, ddof=1)
    if sy == None:
        sy = np.std(datay, ddof=1)
    s = np.sqrt((n1 - 1) * sx**2 + (n2 - 1) * sy**2)

    if side == "two":
        if n1 == n2:
            statistic = np.sqrt(n1) * (xbar - null) / np.sqrt(sx**2 + sy**2)
        else:
            statistic = (xbar - null) / s
        lower_limit = t.ppf(0.5 * alpha, df=n1 + n2 - 2)
        upper_limit = t.ppf(1 - 0.5 * alpha, df=n1 + n2 - 2)
        if (statistic < lower_limit) or (statistic > upper_limit):
            hypo = "Rejected"
        else:
            hypo = "Accepted"
        results = pd.DataFrame(
            [alpha, xbar, sx, sy, n1, n2, statistic, lower_limit, upper_limit],
            index=[
                "\\alpha",
                "\\bar{x} - \\bar{y}",
                "s_x",
                "s_y",
                "n_1",
                "n_2",
                "\\text{Statistic}",
                "\\text{Lower Limit}",
                "\\text{Upper Limit}",
            ],
        )
    elif side == "left":
        if n1 == n2:
            statistic = np.sqrt(n1) * (xbar - null) / np.sqrt(sx**2 + sy**2)
        else:
            statistic = (xbar - null) / s
        limit = t.ppf(alpha, df=n1 + n2 - 2)
        if statistic < limit:
            hypo = "Rejected"
        else:
            hypo = "Accepted"
        results = pd.DataFrame(
            [alpha, xbar, sx, sy, n1, n2, statistic, limit],
            index=[
                "\\alpha",
                "\\bar{x} - \\bar{y}",
                "s_x",
                "s_y",
                "n_1",
                "n_2",
                "\\text{Statistic}",
                "\\text{Limit}",
            ],
        )

    elif side == "right":
        if n1 == n2:
            statistic = np.sqrt(n1) * (xbar - null) / np.sqrt(sx**2 + sy**2)
        else:
            statistic = (xbar - null) / s
        limit = t.ppf(1 - alpha, df=n1 + n2 - 2)
        if statistic > limit:
            hypo = "Rejected"
        else:
            hypo = "Accepted"
        results = pd.DataFrame(
            [alpha, xbar, sx, sy, n1, n2, statistic, limit],
            index=[
                "\\alpha",
                "\\bar{x} - \\bar{y}",
                "s_x",
                "s_y",
                "n_1",
                "n_2",
                "\\text{Statistic}",
                "\\text{Limit}",
            ],
        )
    print(results.to_latex(escape=False, float_format="%4.4f"))
    print(f"{hypo=}")


def var_test_two_sample(datax, datay, side, null=0, alpha=0.05, sx=None, sy=None):
    n1, n2 = len(datax), len(datay)
    if sx == None:
        sx = np.std(datax, ddof=1)
    if sy == None:
        sy = np.std(datay, ddof=1)

    if side == "two":
        statistic = (sx / sy) ** 2
        lower_limit = f.ppf(0.5 * (1 - alpha), n1 - 1, n2 - 1)
        upper_limit = f.ppf(0.5 * (1 + alpha), n1 - 1, n2 - 1)
        if (statistic < lower_limit) or (statistic > upper_limit):
            hypo = "Rejected"
        else:
            hypo = "Accepted"
        results = pd.DataFrame(
            [alpha, sx, sy, n1, n2, statistic, lower_limit, upper_limit],
            index=[
                "\\alpha",
                "s_x",
                "s_y",
                "n_1",
                "n_2",
                "\\text{Statistic}",
                "\\text{Lower Limit}",
                "\\text{Upper Limit}",
            ],
        )

    elif side == "left":
        statistic = (sx / sy) ** 2
        limit = f.ppf(alpha, n1 - 1, n2 - 1)
        if statistic < limit:
            hypo = "Rejected"
        else:
            hypo = "Accepted"
        results = pd.DataFrame(
            [alpha, sx, sy, n1, n2, statistic, limit],
            index=[
                "\\alpha",
                "s_x",
                "s_y",
                "n_1",
                "n_2",
                "\\text{Statistic}",
                "\\text{Limit}",
            ],
        )

    elif side == "right":
        statistic = (sx / sy) ** 2
        limit = f.ppf(1 - alpha, n1 - 1, n2 - 1)
        if statistic > limit:
            hypo = "Rejected"
        else:
            hypo = "Accepted"
        results = pd.DataFrame(
            [alpha, sx, sy, n1, n2, statistic, limit],
            index=[
                "\\alpha",
                "s_x",
                "s_y",
                "n_1",
                "n_2",
                "\\text{Statistic}",
                "\\text{Limit}",
            ],
        )
    print(results.to_latex(escape=False, float_format="%4.4f"))
    print(f"{hypo=}")


def mean_test_known_var(
    data,
    side,
    null,
    var,
    alternative=0,
    alpha=0.05,
):
    xbar = np.mean(data)
    n = len(data)

    if side == "two":
        statistic = (xbar - null) / (np.sqrt(var / n))
        upper_limit = norm.ppf(1 - 0.5 * alpha)
        lower_limit = norm.ppf(0.5 * alpha)
        if (statistic < lower_limit) or (statistic > upper_limit):
            hypo = "Rejected"
        else:
            hypo = "Accepted"
        results = pd.DataFrame(
            [alpha, xbar, var, n, statistic, lower_limit, upper_limit],
            index=[
                "\\alpha",
                "\\bar{x}",
                "\\sigma",
                "n",
                "\\text{Statistic}",
                "\\text{Lower Limit}",
                "\\text{Upper Limit}",
            ],
        )
        print(
            lower_limit * np.sqrt(var / n) + null, upper_limit * np.sqrt(var / n) + null
        )

    if side == "left":
        statistic = (xbar - null) / (np.sqrt(var / n))
        limit = norm.ppf(alpha)
        c = limit * np.sqrt(var / n) + null
        alt_stat = (c - alternative) / (np.sqrt(var / n))
        if statistic < limit:
            hypo = "Rejected"
        else:
            hypo = "Accepted"
        results = pd.DataFrame(
            [alpha, xbar, var, n, statistic, limit],
            index=[
                "\\alpha",
                "\\bar{x}",
                "\\sigma^2",
                "n",
                "\\text{Statistic}",
                "\\text{Limit}",
            ],
        )

    if side == "right":
        statistic = (xbar - null) / (np.sqrt(var / n))
        limit = norm.ppf(1 - alpha)
        c = limit * np.sqrt(var / n) + null
        alt_stat = (c - alternative) / (np.sqrt(var / n))
        if statistic > limit:
            hypo = "Rejected"
        else:
            hypo = "Accepted"
        results = pd.DataFrame(
            [alpha, xbar, var, n, statistic, limit],
            index=[
                "\\alpha",
                "\\bar{x}",
                "\\sigma^2",
                "n",
                "\\text{Statistic}",
                "\\text{Limit}",
            ],
        )
    print(results.to_latex(escape=False, float_format="%4.4f"))
    print(f"{hypo=}")
    # return hypo[0] == "A"


def var_test(
    data,
    side,
    null,
    alternative=0,
    alpha=0.05,
    std=None,
):
    n = len(data)
    if std == None:
        var = np.var(data, ddof=1)
    else:
        var = std**2
    statistic = (n - 1) * (var / null**2)
    if side == "two":
        lower_limit = chi2.ppf(0.5 * (1 - alpha), df=n - 1)
        upper_limit = chi2.ppf(0.5 * (1 + alpha), df=n - 1)
        if (statistic < lower_limit) or (statistic > upper_limit):
            hypo = "Rejected"
        else:
            hypo = "Accepted"
        results = pd.DataFrame(
            [
                alpha,
                null,
                std,
                n,
                statistic,
                lower_limit,
                upper_limit,
            ],
            index=[
                "\\alpha",
                "\\sigma",
                "s",
                "n",
                "\\text{Statistic}",
                "\\text{Lower Limit}",
                "\\text{Upper Limit}",
            ],
        )
        print(results.to_latex(escape=False, float_format="%4.4f"))
        print(f"{hypo=}")
        # return hypo[0] == "A"

    elif side == "left":
        limit = chi2.ppf(alpha, df=n - 1)
        if statistic < limit:
            hypo = "Rejected"
        else:
            hypo = "Accepted"
        results = pd.DataFrame(
            [alpha, null, std, n, statistic, limit],
            index=[
                "\\alpha",
                "\\sigma",
                "s",
                "n",
                "\\text{Statistic}",
                "\\text{Limit}",
            ],
        )
        print(results.to_latex(escape=False, float_format="%4.4f"))
        print(f"{hypo=}")
        # return hypo[0] == "A"

    elif side == "right":
        limit = chi2.ppf(1 - alpha, df=n - 1)
        if statistic > limit:
            hypo = "Rejected"
        else:
            hypo = "Accepted"
        results = pd.DataFrame(
            [alpha, null, std, n, statistic, limit],
            index=[
                "\\alpha",
                "\\sigma",
                "s",
                "n",
                "\\text{Statistic}",
                "\\text{Limit}",
            ],
        )
        print(results.to_latex(escape=False, float_format="%4.4f"))
        print(f"{hypo=}")
        # return hypo[0] == "A"


def control_chart_mean(xvals, mean=None, std=None, alpha=0.01):
    if mean == None:
        mean = np.mean(xvals)
    if std == None:
        std = np.std(xvals, ddof=1)
    UCL = mean + (std / np.sqrt(len(xvals))) * norm.ppf(1 - alpha / 2)
    LCL = mean + (std / np.sqrt(len(xvals))) * norm.ppf(alpha / 2)
    print(f"{LCL=:.4f}\t\t{UCL=:.4f}")
    print(f"{mean=:.4f}\t\t{std=:.4f}")


def OC_curve(c: int, n: int, pdf="Poisson", basket=None):
    if pdf == "Poisson":
        OC = np.zeros((3, n + 1))
        OC[0, :] = (1 / n) * np.arange(n + 1)
        for defective in range(n + 1):
            OC[1, defective] = poisson.cdf(c, defective)
        OC[2, :] = OC[0, :] * OC[1, :]
        return OC

    elif pdf == "Hypergeom":
        OC = np.zeros((3, n + 1))
        OC[0, :] = (1 / n) * np.arange(n + 1)
        for defective in range(n + 1):
            OC[1, defective] = hypergeom.cdf(c, n, defective, basket)
        OC[2, :] = OC[0, :] * OC[1, :]
        return OC

    elif pdf == "Binomial":
        OC = np.zeros((3, n + 1))
        OC[0, :] = (1 / n) * np.arange(n + 1)
        for defective in range(n + 1):
            OC[1, defective] = binom.cdf(c, basket, defective / n)
        OC[2, :] = OC[0, :] * OC[1, :]
        return OC


def goodness_of_fit(vals, bins, dist="Normal", alpha=0.05):
    if dist == "Normal":
        true_bins = np.hstack((np.array([-np.inf]), bins, np.array([np.inf])))
        f_tilde = norm(loc=np.mean(vals), scale=np.std(vals))
        b_vals, bin_edges = np.histogram(vals, true_bins)
        e_vals = len(vals) * np.diff(f_tilde.cdf(true_bins))
        chi_sq = np.sum((b_vals - e_vals) ** 2 / e_vals)
        limit = chi2.ppf(1 - alpha, len(true_bins) - 4)
        if chi_sq > limit:
            hypo = "Rejected"
        else:
            hypo = "Accepted"
        results = pd.DataFrame(
            [alpha, len(true_bins) - 1, f_tilde.mean(), f_tilde.std(), chi_sq, limit],
            index=[
                "\\alpha",
                "K",
                "\\wt{\\mu}",
                "\\wt{\\sigma}",
                "\\text{Statistic}",
                "\\text{Limit}",
            ],
        )
        print(results.to_latex(escape=False, float_format="%4.2f"))
        print(f"{hypo=}")

    elif dist == "Uniform":
        e_val = np.sum(vals) / len(vals)
        chi_sq = np.sum((vals - e_val) ** 2 / e_val)
        limit = chi2.ppf(1 - alpha, len(vals) - 1)
        if chi_sq > limit:
            hypo = "Rejected"
        else:
            hypo = "Accepted"
        results = pd.DataFrame(
            [alpha, len(vals), chi_sq, limit],
            index=[
                "\\alpha",
                "K",
                "\\text{Statistic}",
                "\\text{Limit}",
            ],
        )
        # print(results.to_latex(escape=False, float_format="%4.2f"))
        # print(f"{hypo=}")
        return chi_sq


def goodness_of_fit_Poisson(xvals, counts, alpha=0.05):
    xbar = np.sum(np.multiply(xvals, counts)) / np.sum(counts)
    print(xbar)
    e_val = sum(counts) * poisson.pmf(xvals, xbar)
    chi_sq = np.sum((counts - e_val) ** 2 / e_val)
    limit = chi2.ppf(1 - alpha, len(xvals) - 2)
    if chi_sq > limit:
        hypo = "Rejected"
    else:
        hypo = "Accepted"
    results = pd.DataFrame(
        [alpha, len(xvals), chi_sq, limit],
        index=[
            "\\alpha",
            "K",
            "\\text{Statistic}",
            "\\text{Limit}",
        ],
    )
    print(results.to_latex(escape=False, float_format="%4.2f"))
    print(f"{hypo=}")
    return chi_sq


def linear_regression(xvals, yvals, gamma=0.95):
    n = len(xvals)
    # print(xvals, yvals)
    coeffs = np.array(
        [[len(xvals), np.sum(xvals)], [np.sum(xvals), np.sum(xvals**2)]]
    )
    const = np.array([np.sum(yvals), np.sum(np.multiply(xvals, yvals))])

    sx2, sy2 = np.var(xvals, ddof=1), np.var(yvals, ddof=1)
    xbar, ybar = np.mean(xvals), np.mean(yvals)
    sxy = np.sum(np.multiply(xvals - xbar, yvals - ybar)) / (n - 1)
    k1 = sxy / sx2
    q0 = (n - 1) * (sy2 - (k1**2) * sx2)
    c = t.ppf(0.5 * (1 + gamma), df=n - 2)
    K = c * np.sqrt(q0 / ((n - 1) * (n - 2) * sx2))
    print(np.linalg.solve(coeffs, const))
    results = pd.DataFrame(
        [
            gamma,
            k1,
            q0,
            n,
            c,
            k1 - K,
            k1 + K,
        ],
        index=[
            "\\gamma",
            "k_1",
            "q_0",
            "n",
            "c",
            "\\text{Lower Limit}",
            "\\text{Upper Limit}",
        ],
    )
    # print(results.to_latex(escape=False, float_format="%4.4f"))


# result = curve_fit(func, df.index.to_numpy(), df["Error"].to_numpy())
# print(result)

# rng = np.random.default_rng()
# dice = rng.integers(1, 7, size=(1000000, 20))
# print(np.histogram(np.sum(dice, axis=1), bins=np.arange(20, 121, 10)))

# rng = np.random.default_rng()
# dice = rng.standard_normal(size=(1000, 1000000))
# print(np.mean(np.mean(dice, axis=1)))
# print(np.var(np.mean(dice, axis=1)))
# print(np.mean(np.var(dice, axis=1)))
# print(np.var(np.var(dice, axis=1)))


# xval = np.zeros(400)
# xval[:310] = 1
# var_in = 400 * 0.75 * 0.25
# conf = 0.95

# mean_test_known_var(xval, "two", 0.75, var_in)


# print(np.mean(xval))
# n, s = 24000, 12012
# c = t.ppf((conf + 1) / 2, n - 1)
# pbar = s / n
# xbar = pbar * n
# k = c * np.sqrt(n * pbar * (1 - pbar)) / np.sqrt(n)
# print(f"{pbar=}")
# print(k, c)
# print(xbar - k, xbar + k)
# print((xbar - k) / n, (xbar + k) / n)

# xvals = 58.5 * np.ones(20)

# xvals = np.array([0, 1, -1, 3, -8, 6, 1])

# var_z(np.ones(100), 0.95, sample_mean=442.5, sample_var=9.3)
# var_z(xvals, 0.95, sample_mean=np.mean(xvals), sample_var=np.var(xvals, ddof=1))

# mean_test_known_var(xvals, "two", 60, 9, 57)

# rng = np.random.default_rng()
# num = 100000
# dice = rng.normal(100, 5, size=(num, 10))
# all_results = np.zeros(num)
# # print(dice)
# for idx, item in enumerate(dice):
#     all_results[idx] = var_test(item, "two", 25, 0, 0.1)
# print(np.sum(all_results) / len(all_results))

# xvals = np.array([70, 80, 30, 70, 60, 80])
# yvals = np.array([140, 120, 130, 120, 120, 130, 120])
# var_test_two_sample(xvals, yvals, "right")

# d1 = np.array([1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4])
# d2 = np.array([0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0])
# sx, sy = 3, 3

# mean_test_two_sample_unknown_var(d1, d2, "two", 0, 0.05)
# mean_test_unknown_var(d1 - d2, "right", 0)

# xval = np.array(
#     [
#         [3.5, 3.51],
#         [3.51, 3.48],
#         [3.49, 3.50],
#         [3.52, 3.50],
#         [3.53, 3.49],
#         [3.49, 3.5],
#         [3.48, 3.47],
#         [3.52, 3.49],
#     ]
# )
# print(np.mean(xval, axis=1))
# yval = 3.5 * np.ones(2)
# s = 0.02
# control_chart_mean(yval, std=s)

# rng = np.random.default_rng()
# num = 100
# dice = rng.normal(8, 0.4, size=(num, 4))
# results = np.zeros((4, 100))
# results[0, :] = 1 + np.arange(100)
# results[1, :] = np.mean(dice, axis=1)
# results[2, :] = np.std(dice, axis=1, ddof=1)
# results[3, :] = np.max(dice, axis=1) - np.min(dice, axis=1)

# np.savetxt(
#     "./tables/table_25_05_12.csv", np.transpose(results), delimiter=",", fmt="%.2f"
# )

# accept = 0
# n_val = 5

# result = OC_curve(accept, n_val, pdf="Poisson", basket=3)
# np.savetxt(
#     "./tables/table_25_06_15.csv", np.transpose(result), delimiter=",", fmt="%.4f"
# )

# bins = np.arange(325, 406, 10)
# values = np.array(
#     [
#         320,
#         380,
#         340,
#         410,
#         380,
#         340,
#         360,
#         350,
#         320,
#         370,
#         350,
#         340,
#         350,
#         360,
#         370,
#         350,
#         380,
#         370,
#         300,
#         420,
#         370,
#         390,
#         390,
#         440,
#         330,
#         390,
#         330,
#         360,
#         400,
#         370,
#         320,
#         350,
#         360,
#         340,
#         340,
#         350,
#         350,
#         390,
#         380,
#         340,
#         400,
#         360,
#         350,
#         390,
#         400,
#         350,
#         360,
#         340,
#         370,
#         420,
#         420,
#         400,
#         350,
#         370,
#         330,
#         320,
#         390,
#         380,
#         400,
#         370,
#         390,
#         330,
#         360,
#         380,
#         350,
#         330,
#         360,
#         300,
#         360,
#         360,
#         360,
#         390,
#         350,
#         370,
#         370,
#         350,
#         390,
#         370,
#         370,
#         340,
#         370,
#         400,
#         360,
#         350,
#         380,
#         380,
#         360,
#         340,
#         330,
#         370,
#         340,
#         360,
#         390,
#         400,
#         370,
#         410,
#         360,
#         400,
#         340,
#         360,
#     ]
# )
# goodness_of_fit(values, bins)

# bins = np.arange(58.5, 61.5, 1)
# values = np.hstack(
#     (
#         57 * np.ones(4),
#         58 * np.ones(10),
#         59 * np.ones(17),
#         60 * np.ones(27),
#         61 * np.ones(8),
#         62 * np.ones(9),
#         63 * np.ones(3),
#         64 * np.ones(1),
#     )
# )
# goodness_of_fit(values, bins, alpha=0.01)

# print(chi2.ppf(0.95, df=1))

# bessel_vals = abs(np.round(jv(1, np.arange(0, 8.91, 0.1)), 4))
# collect = np.zeros(10)
# for item in bessel_vals:
#     if len(str(item)) < 6:
#         collect[0] += 1
#     else:
#         collect[int(str(item)[-1])] += 1

# print(np.sum(collect[::2]))

# choices = np.array(
#     [11, 10, 20, 8, 13, 9, 21, 9, 16, 8, 12, 8, 15, 10, 10, 9, 12, 8, 13, 9]
# )
# print(np.sum(choices[::2]))
# print(np.sum(choices[1::2]))
# print(np.sum((choices - 11.55) ** 2 / 11.55))

# num = 100000
# rng = np.random.default_rng()
# dice = rng.integers(1, 7, size=(num, 300))
# chi_sq_vals = np.zeros(num)
# for idx, item in enumerate(dice):
#     counts = np.unique(item, return_counts=True)[1]
#     chi_sq_vals[idx] = goodness_of_fit(counts, bins=None, dist="Uniform")
# print(np.sum(chi_sq_vals > chi2.ppf(0.95, df=5)))

# xvals = np.arange(0, 13)
# yvals = np.array([57, 203, 383, 525, 532, 408, 273, 139, 45, 27, 10, 4, 2])

# goodness_of_fit_Poisson(xvals, yvals)

# print(norm.sf(np.sqrt(2) * 6.5 / 3))

# xvals = np.array([121, 120, 95, 123, 140, 112, 92, 100, 102, 91])
# yvals = np.array([521, 465, 352, 455, 490, 388, 301, 395, 375, 418])

# i = np.argsort(xvals)
# vals = yvals[i]
# print(vals)
# vals = np.array([0.6, 1.1, 0.9, 1.6, 1.2, 2.0])
# transpo = 0
# for idx, item in enumerate(vals[:-1]):
#     transpo += np.sum(vals[idx + 1 :] < item)
# print(f"{transpo=}")
vals = np.loadtxt("./tables/table_25_09_11.csv", dtype="float", delimiter=",")
linear_regression(vals[:, 0], vals[:, 1])
linear_regression(vals[:, 0], vals[:, 2])
linear_regression(vals[:, 0], vals[:, 3])
linear_regression(vals[:, 0], vals[:, 4])
