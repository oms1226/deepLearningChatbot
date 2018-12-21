# Day_01_01_LinearRegression_python.py
import matplotlib.pyplot as plt
import numpy as np


def cost(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w * x[i]
        c += (hx - y[i]) ** 2
    return c / len(x)


def show_cost():
    # y = ax + b
    # y = x
    # hx = wx + b
    #      1    0
    x = [1, 2, 3]
    y = [1, 2, 3]

    print(cost(x, y, 0))
    print(cost(x, y, 1))
    print(cost(x, y, 2))
    print('-' * 50)

    for i in range(-30, 50):
        w = i / 10
        c = cost(x, y, w)
        print(w, c)

        plt.plot(w, c, 'ro')
    plt.show()


def python_basic():
    a = np.arange(5)
    print(a)

    print(a + 1)        # broadcast
    print(a + [1, 1, 1, 1, 1])
    print(a + a)        # vector





# show_cost()

python_basic()






