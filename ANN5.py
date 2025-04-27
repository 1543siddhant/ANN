#5. Write a python Program for Bidirectional Associative Memory with two pairs of vectors.


import numpy as np

class BAM:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # Initialize weight matrix (input_size x output_size) with zeros
        self.weights = np.zeros((input_size, output_size))

    def train(self, input_vectors, output_vectors):
        """
        Train the BAM using the Hebbian learning rule.
        We assume input_vectors and output_vectors are numpy arrays.
        """
        for x, y in zip(input_vectors, output_vectors):
            # Hebbian learning rule: w = x * y^T
            self.weights += np.outer(x, y)

    def recall_output(self, input_vector):
        """
        Recall the output vector given an input vector.
        """
        return np.sign(np.dot(input_vector, self.weights))

    def recall_input(self, output_vector):
        """
        Recall the input vector given an output vector.
        """
        return np.sign(np.dot(output_vector, self.weights.T))

# Example of training BAM with two pairs of vectors

# Define input and output vectors
input_vectors = np.array([[1, 1], [-1, 1]])
output_vectors = np.array([[1, -1], [1, 1]])

# Initialize BAM with input_size = 2 and output_size = 2
bam = BAM(input_size=2, output_size=2)

# Train BAM
bam.train(input_vectors, output_vectors)

# Test recall with input
print("Recall output for input [1, 1]:", bam.recall_output([1, 1]))  # Expected: [1, -1]
print("Recall output for input [-1, 1]:", bam.recall_output([-1, 1]))  # Expected: [1, 1]

# Test recall with output
print("Recall input for output [1, -1]:", bam.recall_input([1, -1]))  # Expected: [1, 1]
print("Recall input for output [1, 1]:", bam.recall_input([1, 1]))    # Expected: [-1, 1]