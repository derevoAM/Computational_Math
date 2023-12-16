import matplotlib.pyplot as plt
import numpy as np

years = np.linspace(1910, 2000, 10)
people = np.array(
    [92228496, 106021537, 123202624, 132164569, 151325798, 179323175, 203211926, 226545805, 248709873, 281421906])


# Высчитываем разделенные разности
def divided_diff(x, y):
    n = len(x)
    f_ = np.zeros((n, n))

    for i in range(n):
        f_[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            f_[i][j] = (f_[i][j - 1] - f_[i + 1][j - 1]) / (x[i] - x[i + j])
    # print(f_)

    return f_

def P(x, x_arr, y_arr):
    value_ = y_arr[0]

    f_ = divided_diff(x_arr, y_arr)
    add_coef = 1
    for i in range(1, len(x_arr)):
        add_coef = add_coef * (x - x_arr[i - 1])
        value_ += add_coef * f_[0][i]
    return value_

print("Население в 2010 году: " + str(P(2010, years, people)))

x_arr = np.linspace(1910, 2010, 101)

fig, ax = plt.subplots(figsize=(16, 10), dpi=150)

plt.ylabel('Население', fontsize=14)
plt.xlabel('Год', fontsize=14)

plt.grid(True)

ax.scatter(years, people, s=10)
ax.plot(x_arr, P(x_arr, years, people))

plt.savefig("Newton_interpolation.png")
plt.show()
