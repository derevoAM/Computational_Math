import numpy as np
from matplotlib import pyplot as plt

mu = 0.012277471
eta = 1 - mu
A = lambda x, y: np.sqrt(((x + mu) ** 2 + y ** 2) ** 3)
B = lambda x, y: np.sqrt(((x - eta) ** 2 + y ** 2) ** 3)
f1 = lambda t, x, u, y, v: u
f2 = lambda t, x, u, y, v: x + 2 * v - eta * (x + mu) / A(x, y) - mu * (x - eta) / B(x, y)
f3 = lambda t, x, u, y, v: v
f4 = lambda t, x, u, y, v: y - 2 * u - eta * y / A(x, y) - mu * y / B(x, y)
f = lambda t, u: np.array([f(t, u[0], u[1], u[2], u[3]) for f in [f1, f2, f3, f4]])

x_0 = 0.994
y_0 = 0.0
u_0 = 0.0
v_0 = -2.00158510637908252240537862224

z = np.array([x_0, u_0, y_0, v_0])
x = np.array([])
y = np.array([])

T = 17.0652165601579625588917206249
n = 1000
# h = T / n
h = 1e-5
step = 0


def rk4(f, t, u, h):
    k1 = h * f(t, u)
    k2 = h * f(t + 0.5 * h, u + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, u + 0.5 * k2)
    k4 = h * f(t + h, u + k3)
    return u + (k1 / 6) + (k2 / 3) + k3 / 3 + (k4 / 6)


eps = 1e-5


def dormand_prince(f, t, u, h):
    k1 = h * f(t, u)
    k2 = h * f(t + (1 / 5) * h, u + (1 / 5) * k1)
    k3 = h * f(t + (3 / 10) * h, u + (3 / 40) * k1 + (9 / 40) * k2)
    k4 = h * f(t + (4 / 5) * h, u + (44 / 45) * k1 - (56 / 15) * k2 + (32 / 9) * k3)
    k5 = h * f(t + (8 / 9) * h, u + (19372 / 6561) * k1 - (25360 / 2187) * k2 + (64448 / 6561) * k3 - (212 / 729) * k4)
    k6 = h * f(t + h,
               u + (9017 / 3168) * k1 - (355 / 33) * k2 - (46732 / 5247) * k3 + (49 / 176) * k4 - (5103 / 18656) * k5)
    k7 = h * f(t + h, u + (35 / 384) * k1 + (500 / 1113) * k3 + (125 / 192) * k4 - (2187 / 6784) * k5 + (11 / 84) * k6)

    y1 = u + (35 / 384) * k1 + (500 / 1113) * k3 + (125 / 192) * k4 - (2187 / 6784) * k5 + (11 / 84) * k6
    y2 = u + (5179 / 57600) * k1 + (7571 / 16695) * k3 + (393 / 640) * k4 - (92097 / 339200) * k5 + (
            187 / 2100) * k6 + (1 / 40) * k7

    delta = (71 / 57600) * k1 - (71 / 16695) * k3 + (71 / 1920) * k4 - (17253 / 339200) * k5 + (22 / 525) * k6 - (
                1 / 40) * k7
    sum = 0
    for i in delta:
        sum += i ** 2
    s = pow((eps * h) / (2 * np.sqrt(sum)), 0.2)
    h_opt = s * h
    return y2, h_opt

x = [z[0]]
y = [z[2]]
i = 0
while step <= 3 * T:

    z = rk4(f, step, z, h)
    x.append(z[0])
    y.append(z[2])
    step += h
    i += 1

i = 0

# while step <= T:
#     print(step)
#     z, h = dormand_prince(f, step, z, h)
#     x.append(z[0])
#     y.append(z[2])
#     step += h
#     i += 1

fig, ax = plt.subplots(figsize=(16, 10), dpi=150)

plt.grid(True)
ax.plot(x, y)

# plt.savefig("task_5.png")
plt.show()
