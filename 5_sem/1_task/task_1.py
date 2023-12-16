import numpy as np

epsilon = 1e-6


def newtonMethod(x0):
    f1 = lambda x, y: x ** 2 + y ** 2 - 1
    f2 = lambda x, y: y - np.tan(x)
    F = lambda x, y: [f1(x, y), f2(x, y)]
    J = lambda x, y: np.array([[2 * x, 2 * y], [-1 / (np.cos(x) ** 2), 1]])

    x1 = x0.copy()
    x2 = x1 - np.linalg.inv(J(*x1)) @ F(*x1)
    while abs(x1 - x2).any() > epsilon:
        x1 = x2.copy()
        x2 = x2 - np.linalg.inv(J(*x2)) @ F(*x2)
    return x2


print(newtonMethod(np.array([1, 1])))
print(newtonMethod(np.array([-1, -1])))


# def MPI(x0, y0):
#     f1x = lambda x: np.arctan(np.sqrt(1-x**2))
#     f2x = lambda x: -np.arctan(np.sqrt(1-x**2))
#     f1y = lambda x: np.tan(np.sqrt(1-x**2))
#     f2y = lambda x: -np.tan(np.sqrt(1-x**2))
#
#     x = x0
#     y = y0
#     if x >= 0:
#         x = f1x(x)
#     else:
#         x = f2x(x)
#
#     if y >= 0:
#         y = f1y(y)
#     else:
#         y = f2y(y)
#
#
#
#     while (np.sqrt((x - x0) ** 2 + (y - y0) ** 2) > epsilon):
#         if x >= 0:
#             x = f1x(x)
#         else:
#             x = f2x(x)
#
#         if y >= 0:
#             y = f1y(y)
#         else:
#             y = f2y(y)
#         print(x, y)
#
#     return np.array([x, y])
#
# print(MPI(0.1, 0.1))
