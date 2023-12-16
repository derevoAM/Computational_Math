import numpy as np


def tridiagonal_solver(a, b, c, d):
    # a[0] = 0, c[n] = 0
    x = np.full(len(d), 0.)
    p = np.array([])
    q = np.array([])
    n = len(d)

    p = np.append(p, c[0] / b[0])
    q = np.append(q, -d[0] / b[0])

    for i in range(1, n):
        p = np.append(p, c[i] / (b[i] - a[i] * p[i - 1]))
        q = np.append(q, (a[i] * q[i - 1] - d[i]) / (b[i] - a[i] * p[i - 1]))

    x[n - 1] = q[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]

    return x


def rk4(f, t, u, h):
    k1 = h * f(t, u)
    k2 = h * f(t + 0.5 * h, u + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, u + 0.5 * k2)
    k4 = h * f(t + h, u + k3)
    return u + (k1 / 6) + (k2 / 3) + k3 / 3 + (k4 / 6)