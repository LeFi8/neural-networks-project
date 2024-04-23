from matplotlib import pyplot as plt
import numpy as np

def perceptron_and(x: list[float]) -> float:
    return x[1] and x[2]

def calc_linear_func(x, w: list[float]):
    xf = []
    for xi in x:
        y = -(w[0] / w[2]) - ((xi * w[1]) / w[2])
        xf.append(y)
    return xf

def plot_linear_function(w):
    x = np.linspace(-0.1, 1.1, 1000)
    f = calc_linear_func(x, w)
    plt.plot(x, f, '-', color="blue")

def plot_point(x: list[float], perceptron_func):
    color = "green"
    if perceptron_func(x) == 0:
        color = "red"
    plt.plot(x[1], x[2], 'o', color=color)
    plt.text(x[1], x[2], "(" + str(x[1]) + ", " + str(x[2]) + ")")

def plot_perceptron_step(x: list[list[float]], w, perceptron_func, label: str = ""):
    if label == "":
        plt.figure()
    else:
        plt.figure(label)

    for xi in x:
        plot_point(xi, perceptron_func)
    plot_linear_function(w)