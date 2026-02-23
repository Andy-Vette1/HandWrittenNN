import numpy as np
import pickle


class FNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize network parameters
        self.params = {}

        # 1. Initialize first-layer weights W1 and bias b1
        # Use He initialization: np.random.randn(...) * np.sqrt(2 / input_size)
        self.params['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.params['b1'] = np.zeros((1, hidden_size))

        # 2. Initialize second-layer weights W2 and bias b2
        # Continue using He initialization
        self.params['W2'] = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.params['b2'] = np.zeros((1, output_size))

    def relu(self, Z):
        # ReLU sets values below 0 to 0 and keeps positive values unchanged
        return np.maximum(0, Z)

    def softmax(self, Z):
        # Softmax formula: exp(Z) / sum(exp(Z))
        # 1. Subtract the row-wise max to avoid exp overflow (numerical stability)
        shift_Z = Z - np.max(Z, axis=1, keepdims=True)

        # 2. Compute exponentials
        exp_Z = np.exp(shift_Z)

        # 3. Compute denominator: row-wise sum
        # axis=1 means summing values across each row
        sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)

        # 4. Compute probabilities
        return exp_Z / sum_exp_Z

    def forward(self, X):
        # 1. Layer 1
        Z1 = np.dot(X, self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)

        # 2. Layer 2
        Z2 = np.dot(A1, self.params['W2']) + self.params['b2']
        A2 = self.softmax(Z2)

        # Key point: cache intermediate values Z1 and A1
        # They are needed when computing gradients in backpropagation
        self.cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'X': X}

        return A2

    def backward(self, X, y, learning_rate=0.1):
        """
           Backpropagation algorithm.
            :param X: Input data.
            :param y: Ground-truth labels (one-hot encoded).
            :param learning_rate: Learning rate.
        """
        # 0. Setup
        m = X.shape[0]
        # Retrieve cached intermediate values from forward propagation
        # Use self.params directly for parameters
        W2 = self.params['W2']
        A1 = self.cache['A1']
        Z1 = self.cache['Z1']

        # 1. Output-layer gradients
        A2 = self.cache['A2']
        dZ2 = A2 - y  # Cross-entropy + softmax derivative

        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # 2. Hidden-layer gradients
        # 2.1 Backpropagate error: dZ2 -> dA1
        dA1 = np.dot(dZ2, W2.T)

        # 2.2 Activation derivative (ReLU derivative)
        # Gradient flows only where Z1 > 0; otherwise it is zero
        dZ1 = dA1 * (Z1 > 0)

        # 2.3 Compute first-layer parameter gradients
        # X.T is required here
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # 3. Update parameters (gradient descent)
        # W = W - learning_rate * dW
        self.params['W1'] -= learning_rate * dW1
        self.params['b1'] -= learning_rate * db1
        self.params['W2'] -= learning_rate * dW2
        self.params['b2'] -= learning_rate * db2

    def save(self, filename):
        """
        Save model parameters to file.
        """
        print(f"Saving model to {filename} ...")
        with open(filename, 'wb') as f:
            pickle.dump(self.params, f)
        print("Model saved successfully.")

    def load(self, filename):
        """
        Load model parameters from file.
        """
        print(f"Loading model from {filename} ...")
        with open(filename, 'rb') as f:
            self.params = pickle.load(f)
        print("Model loaded successfully.")
