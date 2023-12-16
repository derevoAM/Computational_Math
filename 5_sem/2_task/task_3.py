import numpy as np
import utils
from finite_difference import FiniteDifference
from matplotlib import pyplot as plt
from scipy import interpolate

g = lambda x: 1
q = lambda x: x ** 2 - 3
p = lambda x: -(x ** 2 - 3) * np.cos(x)
f = lambda x: 2 - 6 * x + 2 * (x ** 3) + (x ** 2 - 3) * np.exp(x) * np.sin(x) * (1 + np.cos(x)) + np.cos(x) * (
        np.exp(x) + (x ** 2 - 1) + x ** 4 - 3 * x ** 2)

y_0 = 0
y_pi = np.pi ** 2

n = 10000
solver = FiniteDifference(g, q, p, f, 0, 0, 1, 1, y_0, y_pi, 0, np.pi, n)

y = solver.calc_grid()
x = np.linspace(0, np.pi, n+1)

f = interpolate.interp1d(x, y)
print("y(0.5)="+str(f(0.5)))
print("y(1)="+str(f(1)))
print("y(1.5)="+str(f(1.5)))
print("y(2)="+str(f(2)))
print("y(2.5)="+str(f(2.5)))
print("y(3)="+str(f(3)))

fig, ax = plt.subplots(figsize=(16, 10), dpi=150)


plt.grid(True)

ax.plot(x, y)

plt.savefig("task_3.png")
plt.show()
