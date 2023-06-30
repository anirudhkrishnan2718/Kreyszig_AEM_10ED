from numpy import linspace
import numpy as np
from scipy import special
import matplotlib.pyplot as plt


a, b = 0, 20

mus = np.arange(0, 2.1, 0.2)
# mus = [2]
# styles = ["-", "--", ":"]
t = linspace(a, b, 200)
arr = np.empty((200), float)
print(f"{arr.shape=}")

# f1 = plt.figure(dpi = 300)
for mu in mus:
    sol = t**((1-mu)/2) * special.jv((mu-1)/2, t)
    plt.scatter(t, sol)
    arr = np.vstack((arr, t))
    arr = np.vstack((arr, sol))
# #make a little extra horizontal room for legend
# plt.xlim([-3, 3])
plt.legend([f"$k={mu}$" for mu in mus])
# plt.gca().set_aspect(1)


plt.show()

# plt.plot([1, 1], '--')
# plt.show()

arr = np.delete(arr, 0, 0)
print(f"{arr.shape=}")
print(arr)
np.savetxt("./tables/bessel_many.csv", np.transpose(arr), delimiter = ',')
