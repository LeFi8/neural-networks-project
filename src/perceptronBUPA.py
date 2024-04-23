from perceptronUtils import *

class PerceptronBUPA:
    def __init__(self, x: list[list[float]], w0: list[float], perceptron_func, ro: int = 1):
        self.x = x
        self.initial_weights = w0
        self.ro = ro
        self.y = [0, 0, 0, 0]
        self.perceptron_func = perceptron_func

        self.d = [0, 0, 0, 0]
        for i in range(len(x)):
            for j in range(len(x[i])):
                self.d[i] = perceptron_func(x[i])

    @classmethod
    def activation_function(cls, x):
        return 1 if x > 0 else 0

    def train(self):
        iteration = 1
        curr_weights = self.initial_weights
        curr_errors = np.ones(len(self.x))

        while True:
            weight_sums = np.zeros(len(self.x[0]))
            curr_errors = np.zeros(len(self.x))
            for i in range(len(self.x)):
                v = self.calc_v(x[i], curr_weights)
                y = self.activation_function(v)
                error = self.d[i] - y
                curr_errors[i] = error

                print(f"Current v: {v}")
                print(f"Current f(v): {int(y)}")
                print(f"Expected f(v): {int(self.d[i])}")
                print(f"Current error: {error}")
                print(f"Current weights: {curr_weights}")

                self.update_weight_sums(weight_sums, error, x[i])
                iteration += 1
                print(f"")

            if self.is_not_solution(curr_errors):
                curr_weights = self.update_weights(curr_weights, weight_sums)
                print(f"###############")
                print(f"Updated weights: {curr_weights}")
                print(f"###############\n")
            else:
                print(f"Final weights: {curr_weights}")
                break

    def update_weight_sums(self, weight_sums, error, x):
        for i in range(len(weight_sums)):
            weight_sums[i] += error * x[i]

    def update_weights(self, curr_weights, weight_sums):
        updated_weights = np.copy(curr_weights)
        for i in range(len(updated_weights)):
            updated_weights[i] += self.ro * weight_sums[i]
        return updated_weights

    def is_not_solution(self, errors):
        for error in errors:
            if error != 0:
                return True
        return False

    def calc_v(self, x, w) -> float:
        sum = 0
        for i in range(len(x)):
            sum += x[i] * w[i]
        return sum

if __name__ == "__main__":
    x1 = [1, 0, 0]
    x2 = [1, 0, 1]
    x3 = [1, 1, 0]
    x4 = [1, 1, 1]
    x = [x1, x2, x3, x4]

    # wagi
    w0 = [1, 0, 1]

    perceptron = PerceptronBUPA(x, w0, perceptron_func=perceptron_and_x2_negation)
    perceptron.train()

    plt.show()


