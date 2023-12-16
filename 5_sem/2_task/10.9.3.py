import numpy as np
import matplotlib.pyplot as plt
import utils

mu = 1000
T = 1000

n = 1000000

h = T / n
step = 0
x_arr = []
# t = np.array([step])
# u = np.array([0, 0.001])
i = 1

x = 0
y = 0.001
while step < T:
    # print(x)
    # print(u)
    # u = utils.rk4(f, step, u, h)
    x_1 = x
    x = x + y * h
    y = y + h * (mu * (1 - y ** 2) * y - x_1)
    step += h
    x_arr.append(x)
    # t = np.append(t, step)


t = np.linspace(0, T, n + 1)
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

plt.grid(True)
ax.plot(t, x_arr)

plt.savefig("task_2.png")
plt.show()
