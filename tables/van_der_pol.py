from numpy import linspace
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def vdp(t, z):
    x, y = z
    return [y, mu * (1 - x**2) * y - x]


a, b = 0, 200

mus = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
# mus = [2]
# styles = ["-", "--", ":"]
t = linspace(a, b, 20000)
arr = np.empty((1000), float)
print(f"{arr.shape=}")

# f1 = plt.figure(dpi = 300)
for mu in mus:
    sol = solve_ivp(vdp, [a, b], [1, 0], t_eval=t, method="DOP853")
    print(f"{arr.shape=}")
    arr = np.vstack((arr, sol.y[0, -1000:]))
    print(f"{arr.shape=}")
    arr = np.vstack((arr, sol.y[1, -1000:]))
    print(f"{arr.shape=}")
    plt.scatter(sol.y[0, -1000:], sol.y[1, -1000:], s=1)
    plt.scatter(sol.y[0, -1000:], sol.y[1, -1000:], s=1)

# #make a little extra horizontal room for legend
plt.xlim([-3, 3])
plt.legend([f"$\mu={m}$" for m in mus])
plt.gca().set_aspect(1)


plt.show()

# plt.plot([1, 1], '--')
# plt.show()

arr = np.delete(arr, 0, 0)
print(f"{arr.shape=}")
print(arr)
np.savetxt("./tables/van_der_pol_many.csv", np.transpose(arr), delimiter = ',')
