import matplotlib.pyplot as plt
import numpy as np

x_hg = np.array([2576, 2338, 2132, 2120, 1942, 1518, 856, 312])
y_hg = np.array([5770, 5461, 5401, 5341, 5331, 4916, 4358, 4047])
x_ne = np.array(
    [2608, 2576, 2516, 2502, 2476, 2454, 2444, 2406, 2396, 2378, 2368, 2354, 2332, 2308, 2300, 2280, 2270, 2250, 2226,
     2212, 2180, 2166, 1904])
y_ne = np.array(
    [7032.41, 6929.47, 6717.04, 6678.28, 6598.95, 6532.88, 6506.53, 6402.24, 6382.99, 6334.42, 6304.79, 6266.49,
     6217.28, 6163.59, 6143.06, 6096.14, 6074.34, 6030.00, 5975.53, 5944.83, 5881.89, 5852.49, 5400.56])
x_ne = np.flip(x_ne)
y_ne = np.flip(y_ne)
x_hg = np.flip(x_hg)
y_hg = np.flip(y_hg)


class CubicSpline:

    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(x)
        self.h = np.diff(x)
        self.a = self.y[1:]
        self.b = np.full(self.n - 1, 0.)
        self.c = np.full(self.n - 1, 0.)
        self.d = np.full(self.n - 1, 0.)
        self.f_ = np.zeros((self.n, 3))
        self.diag_up = np.full(self.n - 3, 0.)
        self.diag_ = np.full(self.n - 2, 2)
        self.diag_down = np.full(self.n - 3, 0.)

    # СЛАУ с трехдиагональной матрицей
    def tridiagonal_solver(self, y_):
        x = np.full(self.n, 0.)
        p = np.array([])
        q = np.array([])

        p = np.append(p, -self.diag_up[0] / self.diag_[0])
        q = np.append(q, y_[0] / self.diag_[0])

        for i in range(1, self.n - 3):
            p = np.append(p, -self.diag_up[i] / (self.diag_[i] + self.diag_down[i - 1] * p[i - 1]))
            q = np.append(q, (y_[i] - self.diag_down[i - 1] * q[i - 1]) / (
                    self.diag_[i] + self.diag_down[i - 1] * p[i - 1]))

        x[self.n - 3] = ((y_[self.n - 3] - self.diag_down[self.n - 4] * q[self.n - 4]) / (
                self.diag_[self.n - 3] + self.diag_down[self.n - 4] * p[self.n - 4]))
        for i in range(self.n - 4, -1, -1):
            x[i] = p[i] * x[i + 1] + q[i]

        return x

    def newton_coef(self):
        for i in range(self.n):
            self.f_[i][0] = self.y[i]

        for j in range(1, 3):
            for i in range(self.n - j):
                self.f_[i][j] = (self.f_[i][j - 1] - self.f_[i + 1][j - 1]) / (self.x[i] - self.x[i + j])

    def create_matrix(self):
        for i in range(len(self.diag_up)):
            self.diag_up[i] = self.h[i + 1] / (self.h[i] + self.h[i + 1])
            self.diag_down[i] = self.h[i] / (self.h[i] + self.h[i + 1])

    def calculcate_polynoms(self):
        self.newton_coef()
        u1 = self.f_[:, 1]
        u2 = self.f_[:, 2]
        u2 = u2[:self.n - 2] * 6
        self.create_matrix()
        self.c = np.array(self.tridiagonal_solver(u2))
        self.c = np.append(self.c, 0.)

        for i in range(1, self.n - 1):
            self.b[i] = self.c[i] * self.h[i] / 3 + self.c[i - 1] * self.h[i] / 6 + u1[i]
            self.d[i] = (self.c[i] - self.c[i - 1]) / self.h[i]

        self.b[0] = self.c[0] * self.h[0] / 3 + u1[0]
        self.d[0] = self.c[0] / self.h[0]

    def interpolate(self, x_):
        self.calculcate_polynoms()
        for i in range(self.n - 1):
            if x_ <= self.x[i + 1]:
                return self.a[i] + self.b[i] * (x_ - self.x[i + 1]) + self.c[i] * (x_ - self.x[i + 1]) ** 2 / 2 + \
                    self.d[i] * (
                            x_ - self.x[i + 1]) ** 3 / 6
        return self.a[self.n - 2] + self.b[self.n - 2] * (x_ - self.x[self.n - 1]) + self.c[self.n - 2] * (
                x_ - self.x[self.n - 1]) ** 2 / 2 + self.d[self.n - 2] * (x_ - self.x[self.n - 1]) ** 3 / 6


spline = CubicSpline(x_ne, y_ne)


x_arr = np.linspace(x_ne[0], x_ne[-1], 2000)
y_arr = np.array([])
for i in x_arr:
    y_arr = np.append(y_arr, spline.interpolate(i))

fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
fig.suptitle("Калибровочный график неона",
             fontsize='20')
plt.ylabel('Длина волны, A', fontsize=18)
plt.xlabel('Градусы по шкале монохрометра', fontsize=18)

plt.grid(True)

ax.plot(x_arr, y_arr)
ax.scatter(x_ne, y_ne)

# plt.savefig("Spline_interpolation.png")
plt.show()
