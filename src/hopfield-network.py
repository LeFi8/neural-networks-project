import numpy as np


def f1(x):
    if (x > 0):
        return 1
    else:
        return 0


def generate_weights(size, X_array):
    W = np.zeros((size, size))

    for X in X_array:
        X = X.reshape((-1, 1))
        W += np.dot(X, X.T)  #Hebb’s rule
        np.fill_diagonal(W, 0)

    return W


def is_symmetric(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] != matrix[j][i]:
                return False

    return True


def is_diagonal_greater_equal_zero(matrix):
    for i in range(len(matrix)):
        if matrix[i][i] < 0:
            return False

    return True


def is_positively_defined(matrix):
    matrix_values, _ = np.linalg.eig(matrix)
    return all(val > 0 for val in matrix_values)


def array_contains(array_list, array):
    for a in range(len(array_list)):
        if np.array_equal(array_list[a], array):
            return True

    return False


def ask_for_options():
    print("Use synchronous or asynchronous mode? [S, a]")
    mode = input()
    sync = True if mode[0] == "S" or mode[0] == "s" else False
    print("How many neurons in the network?")
    neuron_num = int(input())
    print("How many vectors to test?")
    vector_num = int(input())

    return sync, neuron_num, vector_num


def ascertain_stability(matrix, sync_mode):
    print("Hopfield network stability")
    print("First condition: weight matrix is symmetric", end=": ")
    print("PASSED") if is_symmetric(matrix) else print("FAILED")
    print("Second condition: weight matrix diagonal is zeroes", end=": ")
    print("PASSED") if is_diagonal_greater_equal_zero(matrix) else print("FAILED")

    if sync_mode is True:
        print("Third condition: weight matrix has positive definiteness", end=": ")
        print("PASSED") if is_positively_defined(matrix) else print("FAILED")

    if sync_mode is False and is_symmetric(matrix) and is_diagonal_greater_equal_zero(matrix) or sync_mode is True and is_symmetric(
            matrix) and is_diagonal_greater_equal_zero(matrix) and is_positively_defined(matrix):
        print("The network will certainly stabilize")
    else:
        print("It is not certain if the network will stabilize")


def print_array(array):
    for i in range(len(array)):
        print(array[i])


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

    x = x_prime

    return x


def asynchronous_mode(x, w, f, sig):
    for i in range(len(x)):
        x_prime = 0. + sig
        for j in range(len(x)):
            x_prime += w[i][j] * x[j]
        x_prime = f(x_prime)
        x[i] = x_prime

    return x


sync, neuron_num, vector_num = ask_for_options()

X_array = np.random.randint(2, size=(vector_num, neuron_num))

print("Vectors")
print_array(X_array)

print("Vectors used for Hebb’s rule")
print_array(X_array[0:1])

W = generate_weights(neuron_num, X_array[0:1])

print("Weights")
print(W)

ascertain_stability(W, sync)

for x in range(len(X_array)):
    print("Testing vector number " + str(x + 1) + "…")

    X = X_array[x]
    X_prev = []

    for it in range(10000):
        X_prev.append(X.copy())

        if len(X_prev) > 8:
            X_prev = X_prev[1:]

        if sync is True:
            X = synchronous_mode(X, W, f1, 0.)
        else:
            X = asynchronous_mode(X, W, f1, 0.)

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
    elif cycle == -1:
        print("Network failed to stabilize")
