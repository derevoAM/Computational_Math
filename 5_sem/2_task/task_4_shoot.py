import numpy as np
import matplotlib.pyplot as plt
import utils

f1 = lambda x, y, z: z
f2 = lambda x, y, z: np.sqrt(-np.exp(z) * y + np.e * y ** 2 / np.log(x) + 1 / (x ** 2))
f = lambda t, u: np.array([f(t, u[0], u[1]) for f in [f1, f2]])

y_0 = np.e
y_1 = 2*np.e ** 2
n = 10000
x_min = np.e
x_max = np.e ** 2


def calc_sol_alpha(f, y_0, z_0, x_min, x_max, n):
    h = (x_max - x_min) / n
    step = x_min
    u = np.array([y_0, z_0])

    while step < x_max:
        u = utils.rk4(f, step, u, h)
        step += h
    return u[0]


an = 1
a_h = 1e-3

F = lambda a: calc_sol_alpha(f, y_0, a, x_min, x_max, n) - y_1
dF = lambda a: (F(a + a_h) - F(a)) / a_h

for i in range(15):
    # print(an)
    an = an - (F(an) / dF(an))

print(an)


def calc_sol(f, y_0, z_0, x_min, x_max, n):
    h = (x_max - x_min) / n
    step = x_min
    y = np.array([y_0])
    x = np.array([step])
    u = np.array([y_0, z_0])

    while step < x_max:
        u = utils.rk4(f, step, u, h)
        step += h
        x = np.append(x, step)
        y = np.append(y, u[0])

    return x, y


x, y = calc_sol(f, y_0, an, x_min, x_max, n)
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

plt.grid(True)

ax.plot(x, y)

plt.savefig("task_4_shoot.png")
plt.show()
