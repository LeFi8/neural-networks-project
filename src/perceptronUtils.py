from matplotlib import pyplot as plt
import numpy as np

def perceptron_xor(x: list[float]) -> float:
    return 1 if x[1] != x[2] else 0

def perceptron_xor_negation(x: list[float]) -> float:
    return 0 if x[1] != x[2] else 1

def perceptron_and(x):
    return 1 if np.all(x[1:]) else 0

def perceptron_and_x2_negation(x: list[float]) -> float:
    return 1 if x[1] == 1 and x[2] == 0 else 0

def perceptron_or(x: list[float]) -> float:
    return 1 if x[1] == 1 or x[2] == 1 else 0

def perceptron_or_negation(x: list[float]) -> float:
    return 0 if x[1] == 1 or x[2] == 1 else 1

def perceptron_or_x1_negation(x: list[float]) -> float:
    return 1 if x[1] == 0 or x[2] == 1 else 0

def perceptron_and_x1_negation(x: list[float]) -> float:
    return 1 if x[1] == 0 and x[2] == 1 else 0

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
    plt.text(x[1], x[2], f"({str(x[1])}, {str(x[2])})")

def plot_point3D(x, x3, ax, perceptron_func):
    color = "green"
    if perceptron_func(x) == 0:
        color = "red"

    ax.scatter(x[1], x[2], x3, color=color, s=100)
    ax.text(x[1], x[2], x3, f"({str(x[1])}, {str(x[2])}, {round(x3, 2)})")

def plot_plane_3d(x, w, perceptron_func):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_range = np.linspace(-0.1, 1.1, 100)
    z_range = np.linspace(-.6, .6, 100)
    X, Z = np.meshgrid(x_range, z_range)
    Y = (-w[1] * X - 0 * Z - w[0]) / w[2]

    for i in range(len(x)):
        plot_point3D(x[i], 0, ax, perceptron_func)

    ax.plot_surface(X, Y, Z, color='r', alpha=0.5)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('RBF')
    plt.title("3D RBF Perceptron Decision Boundary")

def plot_perceptron_step(x: list[list[float]], w, perceptron_func, label: str = ""):
    if label == "":
        plt.figure()
    else:
        plt.figure(label)

    for xi in x:
        plot_point(xi, perceptron_func)
    plot_linear_function(w)

