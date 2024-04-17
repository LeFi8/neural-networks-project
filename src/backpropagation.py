import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

epochs = 10000
learning_rate = 0.05

input_layer_size = 2
hidden_layer_size = 2
output_layer_size = 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

#np.random.seed(0)
W1 = np.random.randn(input_layer_size, hidden_layer_size)
b1 = np.zeros((1, hidden_layer_size))
W2 = np.random.randn(hidden_layer_size, output_layer_size)
b2 = np.zeros((1, output_layer_size))

loss_values = []

for epoch in range(epochs):
    # Forward propagation
    hidden_layer = sigmoid(np.dot(X, W1) + b1)
    output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)
    # loss
    loss = np.mean(0.5 * (y - output_layer) ** 2)
    loss_values.append(loss)
    # Backward propagation
    output_delta = (y - output_layer) * sigmoid_derivative(output_layer)
    hidden_delta = output_delta.dot(W2.T) * sigmoid_derivative(hidden_layer)
    # weights and biases
    W2 += hidden_layer.T.dot(output_delta) * learning_rate
    b2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    W1 += np.dot(X.T, hidden_delta) * learning_rate
    b1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

def calculate_xor_output(input):
    hidden_layer = sigmoid(np.dot(input, W1) + b1)
    output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)
    return output_layer

test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_output = calculate_xor_output(test_input)
print("XOR Input:")
print(test_input);
print("XOR Output:")
print(xor_output.round())

x = np.linspace(-0.5, 1.5, 100)
y1 = (-b1[0, 0] - W1[0, 0] * x) / W1[1, 0]
y2 = (-b1[0, 1] - W1[0, 1] * x) / W1[1, 1]

# Plotting the loss graph
plt.plot(range(epochs), loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plotting the XOR data and decision lines
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr')
plt.plot(x, y1, 'g-', label='Decision Line 1')
plt.plot(x, y2, 'r-', label='Decision Line 2')
plt.title('XOR Gate Decision Lines')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.show()