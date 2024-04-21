class Perceptron:
    def __init__(self, x: list[list[float]], w0: list[float], ro: int = 1):
        self.x = x
        self.w = w0
        self.ro = ro
        self.d_and = [0, 0, 0, 1]
        self.d_xor = [0, 1, 1, 0]
        self.y = [0, 0, 0, 0]

    def activation_function(self, x):
        return 1 if x > 0 else 0

    def train_and(self):
        iteration = 0
        while True:
            i = iteration % 4

            if iteration % 4 == 0:
                self.get_y(self.x, self.w, i)

            print(f"\nIteration: {iteration + 1}")
            print(f"Current weights: {self.w}")
            new_w = self.new_w(i)
            self.w = new_w

            if iteration % 4 == 0 and self.y == self.d_and:
                print(f"Final weights: {self.w}")
                break
            
            iteration += 1
            

    def new_w(self, iteration) -> list[float]:
        new_w = [0, 0, 0]
        for i in range(len(self.w)):
            new_w[i] = self.w[i] + self.ro * (self.d_and[iteration] - self.y[iteration]) * self.x[iteration][i]
        return new_w
    
    def get_y(self, x, w, iteration) -> float:
        for i in range(len(x[iteration])):
            self.y[iteration] += x[iteration][i] * w[i]
        
        self.y[iteration] = self.activation_function(self.y[iteration])
        print(f"Current y: {self.y}")
    

if __name__ == "__main__":
    x = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    w0 = [0.5, 0, 1]
    perceptron = Perceptron(x, w0)
    perceptron.train_and()
    print(perceptron.w)