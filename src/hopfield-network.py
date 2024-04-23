import numpy as np

sync = True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def is_symmetric(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] != matrix[j][i]:
                return False

    return True


def is_diagonal_zeroes(matrix):
    for i in range(len(matrix)):
        if matrix[i][i] != 0:
            return False

    return True


def is_positively_defined(matrix):
    return True


def array_contains(array_list, array):
    for a in range(len(array_list)):
        if np.array_equal(array_list[a], array):
            return True

    return False


def ascertain_stability(matrix):
    print("Hopfield network stability")
    print("First condition: weight matrix is symmetric")
    print("PASSED") if is_symmetric(matrix) else print("FAILED")
    print("Second condition: weight matrix diagonal is zeroes")
    print("PASSED") if is_diagonal_zeroes(matrix) else print("FAILED")

    if sync is True:
        print("Third condition: weight matrix has positive definiteness")
        print("PASSED") if is_positively_defined(matrix) else print("FAILED")

    if sync is False and is_symmetric(matrix) and is_diagonal_zeroes(matrix) or sync == True and is_symmetric(
            matrix) and is_diagonal_zeroes(matrix) and is_positively_defined(matrix):
        print("The network will certainly stabilize")
    else:
        print("It is not certain if the network will stabilize")


def find_cycle(x, x_prev):
    for i in range(len(x_prev)):
        if np.array_equal(x_prev[len(x_prev) - i - 1], x):
            return i

    return -1


def synchronous_mode(x, w, f, sig):
    x_prime = x.copy()

    for i in range(len(x)):
        x_prime[i] = 0. + sig
        for j in range(len(x)):
            x_prime[i] += w[i][j] * x[j]
        x_prime[i] = f(x_prime[i])
        x[i] = x_prime[i]

    return x


X = np.random.rand(3, 1)
X_prev = []

print(X)

W = [
    [-278, -3, 15],
    [2, 432, -174],
    [-1, 1, -158]
]

#W = np.random.rand(3, 3)

print(W)

ascertain_stability(W)

for it in range(10000):
    X_prev.append(X.copy())

    if len(X_prev) > 8:
        X_prev = X_prev[1:]

    X = synchronous_mode(X, W, sigmoid, 0.)

    if array_contains(X_prev, X):
        break

cycle = find_cycle(X, X_prev)
X_prev.append(X.copy())

if cycle == 0:
    print("Network is stable")
    print(X)
elif cycle > 0:
    print("Network failed to stabilize")
    print("Cycle every " + str(cycle + 1) + " iterations")
    print(X_prev)
elif cycle == -1:
    print("Network failed to stabilize")
    print(X_prev)
