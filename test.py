import numpy as np
from sklearn.linear_model import Perceptron

def ascii_to_binary_vector(char):
    """ Convert the ASCII representation of the digit (0-9) to its 8-bit Binary Vector """
    binary_str = format(ord(char), '08b') #Convert ascii to 8bit binary
    return np.array([int(bit) for bit in binary_str])


def generate_trani_data():
    """ Generate training data for digits from 0 to 9 with labels for even(0) and odd(1) """
    digits = [str(i) for i in range(10)]
    X = np.array([ascii_to_binary_vector(d) for d in digits])
    y = np.array([0 if int(d) % 2 == 0 else 1 for d in digits])
    return X,y


def main():

    X_train,y_train = generate_training_data()

    perceptron = Perceptron(max_iter=1000, tol=le-3, random_state=42)
    prece.fit(X_train,y_train)

    #test the preceptron
    test_digits = [str(i) for i in range(10)]
    for digit in test_digits:
        binary_vector = ascii_to_binary_vector(digit).reshape(-1,1)
        prediction = perceptron.predict(binary_vector)
        print(f"Digit: {digit}, Predicted: {'Odd' if prediction[0] else 'Even'}")


if __name__ == "__main__":
    main()

    