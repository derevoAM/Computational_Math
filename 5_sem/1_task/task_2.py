import numpy as np

f = lambda x: np.sin(100 * x) * np.exp(-x ** 2) * np.cos(2 * x)
a = 0
b = 3
N = 1000
h = (b - a) / N


def trapezoidal():
    I = 0
    for i in range(N):
        I += (f(i * h) + f(i * h + h)) * h / 2
    return I


def simpson():
    I = 0
    for i in range(N):
        I += (f(i * h) + f(i * h + h) + 4 * f(i * h + h / 2)) * h / 6
    return I


def three_eighth():
    I = 0
    for i in range(N):
        I += (f(i * h) + f(i * h + h) + 3 * f(i * h + h / 3) + 3 * f(i * h + 2 * h / 3)) * h / 8
    return I

print("Метод трапецией: " + str(trapezoidal()))
print("Метод Симпсона: " + str(simpson()))
print("Метод 3/8: " + str(three_eighth()))