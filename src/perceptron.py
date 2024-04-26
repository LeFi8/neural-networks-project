from perceptronUtils import *

class Perceptron:
    def __init__(self, x: list[list[float]], w0: list[float], ro: int = 1):
        self.x = x
        self.w = w0
        self.ro = ro
        self.y = [0, 0, 0, 0]

        self.d = [0, 0, 0, 0]
        for i in range(len(x)):
            for j in range(len(x[i])):
                self.d[i] = x[i][1] and x[i][2]

    @classmethod
    def activation_function(cls, x):
        return 1 if x > 0 else 0

    def train_and(self):
        iteration = 2  # FIXME: why 2?
        while True:
            i = iteration % len(self.x)
            print(f"\nIteration: {iteration}")
            print(f"Current weights: {self.w}")

            self.set_y(self.x, self.w, i)
            self.y[i] = self.activation_function(self.y[i])
            print(f"Current y: {self.y}")
            print(f"Expected values (d): {self.d}")

            self.update_weights(i)
            print(f"Updated weights: {self.w}")

            plot_perceptron_step(self.x, self.w, perceptron_and, f"Iteration: {iteration}")

            if iteration % len(self.x) == 0 and self.y == self.d:
                print(f"Final weights: {self.w}")
                break

            iteration += 1

    def update_weights(self, iteration) -> list[float]:
        for i in range(len(self.w)):
            self.w[i] += (
                self.ro * (self.d[iteration] - self.y[iteration]) * self.x[iteration][i]
            )

    def set_y(self, x, w, iteration) -> float:
        for i in range(len(x[iteration])):
            self.y[iteration] += x[iteration][i] * w[i]


if __name__ == "__main__":
    x1 = [1, 0, 0]
    x2 = [1, 0, 1]
    x3 = [1, 1, 0]
    x4 = [1, 1, 1]
    x = [x1, x2, x3, x4]

    # wagi
    w0 = [0.5, 0, 1]

    perceptron = Perceptron(x, w0)
    perceptron.train_and()

    plt.show()


