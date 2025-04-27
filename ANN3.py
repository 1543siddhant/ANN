# Write a Python Program using Perceptron Neural Network to recognise even and odd numbers.
# Given numbers are in ASCII form 0 to 9


import numpy as np
from sklearn.linear_model import Perceptron

def ascii_to_binary_vector(char):
    """Convert ASCII representation of a digit (0-9) into a binary vector."""
    binary_str = format(ord(char), '08b')  # Convert ASCII to 8-bit binary
    return np.array([int(bit) for bit in binary_str])

def generate_training_data():
    """Generate training data for digits 0-9 with labels for even (0) and odd (1)."""
    digits = [str(i) for i in range(10)]
    X = np.array([ascii_to_binary_vector(d) for d in digits])
    y = np.array([0 if int(d) % 2 == 0 else 1 for d in digits])  # 0 for even, 1 for odd
    return X, y

def main():
    # Generate training data
    X_train, y_train = generate_training_data()
    
    # Initialize and train Perceptron
    perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
    perceptron.fit(X_train, y_train)
    
    # Test the perceptron
    test_digits = [str(i) for i in range(10)]
    for digit in test_digits:
        binary_vector = ascii_to_binary_vector(digit).reshape(1, -1)
        prediction = perceptron.predict(binary_vector)
        print(f"Digit: {digit}, Predicted: {'Odd' if prediction[0] else 'Even'}")

if __name__ == "__main__":
    main()
    