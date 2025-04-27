#Write a python program in python program for creating a Back Propagation Feed-forward neural
#network

import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize weights
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Forward pass
        self.hidden_layer_activation = np.dot(X, self.weights_input_hidden)
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_activation)
        self.output_layer_activation = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        self.output = self.sigmoid(self.output_layer_activation)
        return self.output

    def backward(self, X, y):
        # Backward pass
        output_error = y - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)

        hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(self.hidden_layer_output)

        # Update weights
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_layer_delta) * self.learning_rate

    def train(self, X, y, epochs=10000):
        for _ in range(epochs):
            self.forward(X)
            self.backward(X, y)

# Example usage
if __name__ == "__main__":
    # Input data (XOR problem)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    # Output data
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    # Create and train the neural network
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=2, output_size=1)
    nn.train(X, y)

    # Test the neural network
    print("Predictions:")
    for x in X:
        print(f"Input: {x} => Output: {nn.forward(x)}")