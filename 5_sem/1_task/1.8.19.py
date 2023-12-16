import matplotlib.pyplot as plt
import numpy as np

dt = 1e-3


def max_n_exp(x, x0):
    R = np.abs((x - x0) * np.exp(x))
    i = 2
    while (R > dt):
        R *= (x - x0) / i
        i += 1
    return i - 2

def max_n_exp_shift(x, x0):
    R = np.abs((x - x0) * np.exp(x0))
    i = 2
    while (R > dt):
        R *= (x - x0) / i
        i += 1
    return i - 2
def max_n_sin(x, x0):
    R = np.abs((x - x0))
    i = 2
    while (R > dt):
        R *= (x - x0) / i
        i += 1
    return i - 2

def max_n_sin_shift(x, x0):
    R = np.abs((x - x0)*(np.abs(np.sin(x0)) + np.abs(np.cos(x0))))
    i = 2
    while (R > dt):
        R *= (x - x0) / i
        i += 1
    return i - 2

print("sin Отрезок [0, 1] без сдвига:" + str(max_n_sin(1, 0)))
print("sin Отрезок [10, 11] без сдвига:" + str(max_n_sin(11, 0)))
print("sin Отрезок [10, 11] со сдвигом:" + str(max_n_sin_shift(11, 10)))
print("exp Отрезок [0, 1] без сдвига:" + str(max_n_exp(1, 0)))
print("exp Отрезок [10, 11] без сдвига:" + str(max_n_exp(11, 0)))
print("exp Отрезок [10, 11] со сдвигом:" + str(max_n_exp_shift(11, 10)))
