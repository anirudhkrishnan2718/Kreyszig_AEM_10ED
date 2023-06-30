import numpy as np
import pandas as pd
import math

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

df = pd.DataFrame(columns=["$n$", "$x_{n}$", "$y_{n}$", "$y(x_{n})$", "Error"])
df.loc[len(df)] = [0, 0.0, 1.0, 1.0, 0.0]
h = 0.2
for i in range(10):
    x, y = df.iloc[-1]["$x_{n}$"], df.iloc[-1]["$y_{n}$"]
    print(x, y)
    x_new, y_new = x + h, y + h * (-5 * (x**4) * (y**2))
    df.loc[len(df.index)] = [
        (i + 1),
        np.round(x_new, 4),
        np.round(y_new, 4),
        np.round((1 + x_new**5)**(-1), 4),
        np.round((1 + x_new**5)**(-1) - y_new, 4),
    ]

# Convert n column to integer
df["$n$"] = df["$n$"].astype(int)
print(df)
df.to_csv("./tables/table_01_01_20.csv", index=False)
