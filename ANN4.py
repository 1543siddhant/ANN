#4. With a suitable example demonstrate the perceptron learning law with its decision regions using
#python. Give the output in graphical form.


import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        self.weights = np.zeros(input_size + 1)  # Include bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        """ReLU actiation function"""
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)  # Add bias term
        return self.activation(np.dot(self.weights, x))

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                xi = np.insert(xi, 0, 1)  # Add bias term
                prediction = self.activation(np.dot(self.weights, xi))
                self.weights += self.learning_rate * (target - prediction) * xi


# Generate a simple linearly separable dataset
def generate_data():
    np.random.seed(1)
    X1 = np.random.randn(10, 2) + np.array([2, 2])  # Class 1
    X2 = np.random.randn(10, 2) + np.array([-2, -2])  # Class -1
    X = np.vstack((X1, X2))
    y = np.hstack((np.ones(10), -np.ones(10)))
    return X, y

# Function to plot decision boundary
def plot_decision_boundary(X, y, w):
    plt.figure(figsize=(6,6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    
    # Plot decision boundary
    x_vals = np.linspace(-4, 4, 100)
    y_vals = -(w[1] / w[2]) * x_vals - (w[0] / w[2])
    plt.plot(x_vals, y_vals, 'k--', label="Decision Boundary")
    
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Perceptron Decision Boundary")
    plt.show()

# Main execution
X, y = generate_data()
perceptron = Perceptron(2)
perceptron.fit(X, y)
plot_decision_boundary(X, y, perceptron.weights)