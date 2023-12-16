import numpy as np
import utils

class FiniteDifference:
    def __init__(self, g, q, p, f, a_1, a_2, b_1, b_2, u_1, u_2, x_min, x_max, n):
        self.g = g
        self.q = q
        self.p = p
        self.f = f
        self.a_1 = a_1
        self.a_2 = a_2
        self.b_1 = b_1
        self.b_2 = b_2
        self.u_1 = u_1
        self.u_2 = u_2
        self.x_min = x_min
        self.x_max = x_max
        self.n = n
        self.h = (self.x_max - self.x_min) / self.n

    def create_tridiagonal_matrix(self):
        a = np.full(self.n + 1, 0.)
        b = np.full(self.n + 1, 0.)
        c = np.full(self.n + 1, 0.)
        d = np.full(self.n + 1, 0.)

        b[0] = self.a_1 / self.h - self.b_1
        c[0] = self.a_1 / self.h
        d[0] = self.u_1
        b[self.n] = self.a_2 / self.h + self.b_2
        a[self.n] = -self.a_2 / self.h
        d[self.n] = -self.u_2

        for i in range(1, self.n):
            a[i] = self.g(i * self.h - 0.5 * self.h) - self.q(i * self.h) * self.h / 2
            c[i] = self.g(i * self.h + 0.5 * self.h) + self.q(i * self.h) * self.h / 2
            b[i] = a[i] + c[i] + self.h ** 2 * self.p(self.h * i)
            d[i] = self.h ** 2 * self.f(self.h * i)

        return a, b, c, d

    def calc_grid(self):
        a, b, c, d = self.create_tridiagonal_matrix()
        return utils.tridiagonal_solver(a, b, c, d)