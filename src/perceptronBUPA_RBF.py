from perceptronUtils import *
import mathUtils as mu

class PerceptronBUPA_RBF:
    def __init__(self, x: list[list[float]], w0: list[float], perceptron_func, ro: int = 1, gamma: float = 1.0):
        self.x = np.array(x)
        self.initial_weights = np.array(w0)
        self.final_weights = np.zeros((len(w0), len(w0)))
        self.ro = ro
        self.gamma = gamma
        self.perceptron_func = perceptron_func
        self.K = np.zeros((len(self.x), len(self.x)))
        self.rbf_transform_manual()
        print(f"RBF: \n{self.K}\n")

        self.d = [perceptron_func(xi) for xi in self.x]

    def rbf_transform_manual(self):
        n_samples = len(self.x)
        for i in range(n_samples):
            for j in range(n_samples):
                self.K[i, j] = np.exp(-self.gamma * np.sum((self.x[i] - self.x[j])**2))

    @classmethod
    def activation_function(cls, x):
        return 1 if x > 0 else 0

    def train(self):
        iteration = 1
        curr_weights = self.initial_weights.copy()

        while True:
            weight_sums = np.zeros(len(self.K[0]))
            curr_errors = np.zeros(len(self.K))
            for i, ki in enumerate(self.K):
                v = self.calc_v(ki, curr_weights)
                y = self.activation_function(v)
                error = self.d[i] - y
                curr_errors[i] = error

                print(f"Current v: {v}")
                print(f"Current f(v): {int(y)}")
                print(f"Expected f(v): {int(self.d[i])}")
                print(f"Current error: {error}")
                print(f"Current weights: {curr_weights}")

                weight_sums += error * ki
                print("")

            if np.any(curr_errors != 0):
                curr_weights += self.ro * weight_sums
                print(f"###############")
                print(f"Iteration: {iteration}")
                print(f"Updated weights: {curr_weights}")
                print(f"###############\n")
                iteration += 1
            else:
                self.final_weights = curr_weights
                print(f"Final weights: {curr_weights}")
                print(f"Learning process took: {iteration-1} iterations!")
                break

    def calc_v(self, ki, w):
        return np.dot(w, ki)

    def apply_rbf(self, xi):
        x_no_bias = [xi[1:] for xi in self.x]
        rbf = mu.rbf_kernel(x_no_bias, [xi], 1.0)
        rbf_flatten = rbf.flatten()
        return rbf_flatten

    def predict(self, X):
        Y_pred = [self.activation_function(self.calc_v(self.apply_rbf(xi), self.final_weights)) for xi in X]
        return np.array(Y_pred)

    def plot_rbf_boundaries(self):
        grid_x = np.linspace(-0.5, 1.5, 200)
        grid_y = np.linspace(-0.5, 1.5, 200)
        grid = np.array([[xg, yg] for xg in grid_x for yg in grid_y])
        Z = self.predict(grid)
        Z = Z.reshape((200, 200))

        plt.contourf(grid_x, grid_y, Z, levels=[-1, 0, 1], alpha=0.2, colors=['red', 'green'])
        for xi in self.x:
            plot_point(xi, self.perceptron_func)

        plt.title("Decision Boundary of RBF Perceptron")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

if __name__ == "__main__":
    x1 = [1, 0, 0]
    x2 = [1, 0, 1]
    x3 = [1, 1, 0]
    x4 = [1, 1, 1]
    x = [x1, x2, x3, x4]

    # w0 = [-30.3, -2.5, 10, 1.2]
    w0 = [.5, 1, 1, 0]

    perceptron = PerceptronBUPA_RBF(x, w0, perceptron_func=perceptron_and)
    perceptron.train()
    perceptron.plot_rbf_boundaries()