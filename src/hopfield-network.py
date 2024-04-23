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

    if sync == True:
        print("Third condition: weight matrix has positive definiteness")
        print("PASSED") if is_positively_defined(matrix) else print("FAILED")

    if sync == False and is_symmetric(matrix) and is_diagonal_zeroes(matrix) or sync == True and is_symmetric(
            matrix) and is_diagonal_zeroes(matrix) and is_positively_defined(matrix):
        print("The network will certainly stabilize")
    else:
        print("It is not certain if the network will stabilize")


def synchronous_mode(x, w, f, i):
    x_prime = x.copy()

    for i in range(len(x)):
        x_prime[i] = 0. + i
        for j in range(len(x)):
            x_prime[i] += w[i][j] * x[j]
        x_prime[i] = f(x_prime[i])
        x[i] = x_prime[i]

    return x


X = np.random.rand(3, 1)
X_prev = []

print(X)

W = [
    [0, -3, 0],
    [2, 0, 1],
    [-1, 1, 0]
]

print(W)

ascertain_stability(W)

for i in range(10000):
    print(i)
    X_prev.append(X.copy())

    if len(X_prev) > 8:
        X_prev = X_prev[1:]

    X = synchronous_mode(X, W, sigmoid, 2)

    if array_contains(X_prev, X):
        break

if np.array_equal(X, X_prev[len(X_prev) - 1]):
    print("Network is stable")
    print(X)
else:
    print("Network failed to stabilize")
    for i in range(len(X_prev)):
        if np.array_equal(X_prev[len(X_prev) - i - 1], X):
            print("Cycle every " + str(i+1) + " iterations")

    print(X)
    print(X_prev)
