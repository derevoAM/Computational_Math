import numpy as np

eps = 1e-3


# Функция
def f(x):
    return x * np.exp(-x ** 2)


# Значение полувысоты функции
y0 = f(np.sqrt(0.5)) / 2

# Локализация (находим начальные приближения для корней)
x_roots = []
h = np.linspace(0, 3, 31)
for i in h:
    if (f(i) - y0) * (f(i + 0.1) - y0) <= 0:
        x_roots.append(i)


# print(x)
# Итеративная функция для левого корня
def f1(x):
    return y0 * np.exp(x ** 2)


# Итеративная функция для правого корня
def f2(x):
    return np.sqrt(np.log(x / y0))


while np.abs(x_roots[0] - f1(x_roots[0])) > eps / 2:
    x_roots[0] = f1(x_roots[0])

while np.abs(x_roots[1] - f2(x_roots[1])) > eps / 2:
    x_roots[1] = f2(x_roots[1])

print("Ширина функции на полувысоте: " + str(round(x_roots[1] - x_roots[0], 3)))
