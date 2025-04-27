# Write a python program to illustrate ART neural network.
# import numpy as np

import numpy as np

class ART:
    def __init__(self, input_size, rho=0.5):
        self.input_size = input_size
        self.rho = rho
        self.weights = np.random.rand(input_size, input_size)  # Initialize weights

    def train(self, input_vector):
        # Normalize input vector
        input_vector = input_vector / np.linalg.norm(input_vector)
        
        # Calculate the activation of the weights
        activation = np.dot(self.weights, input_vector)
        
        # Find the winning neuron
        winner_index = np.argmax(activation)
        
        # Check if the activation exceeds the vigilance criterion
        if activation[winner_index] >= self.rho:
            # Update weights for the winning neuron
            self.weights[winner_index] += input_vector
            self.weights[winner_index] /= np.linalg.norm(self.weights[winner_index])  # Normalize weights
        else:
            # Create a new neuron if no winner is found
            self.weights = np.vstack([self.weights, input_vector])
            self.weights[-1] /= np.linalg.norm(self.weights[-1])  # Normalize new weights

    def predict(self, input_vector):
        input_vector = input_vector / np.linalg.norm(input_vector)
        activation = np.dot(self.weights, input_vector)
        return np.argmax(activation)

# Example usage
if __name__ == "__main__":
    art = ART(input_size=3)

    # Training data (3-dimensional binary patterns)
    training_data = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([1, 1, 0]),
        np.array([0, 1, 1]),
        np.array([1, 0, 1]),
        np.array([1, 1, 1])
    ]

    # Train the ART network
    for pattern in training_data:
        art.train(pattern)

    # Test the ART network
    test_pattern = np.array([1, 0, 1])
    predicted_class = art.predict(test_pattern)
    print(f"Predicted class for {test_pattern}: {predicted_class}")