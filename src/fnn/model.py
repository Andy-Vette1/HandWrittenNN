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

    def relu(self, linear_values):
        # ReLU sets values below 0 to 0 and keeps positive values unchanged
        return np.maximum(0, linear_values)

    def softmax(self, logits):
        # Softmax formula: exp(Z) / sum(exp(Z))
        # 1. Subtract the row-wise max to avoid exp overflow (numerical stability)
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)

        # 2. Compute exponentials
        exp_logits = np.exp(shifted_logits)

        # 3. Compute denominator: row-wise sum
        # axis=1 means summing values across each row
        row_sums = np.sum(exp_logits, axis=1, keepdims=True)

        # 4. Compute probabilities
        return exp_logits / row_sums

    def forward(self, X):
        # 1. Layer 1
        hidden_linear = np.dot(X, self.params['W1']) + self.params['b1']
        hidden_activation = self.relu(hidden_linear)

        # 2. Layer 2
        output_logits = np.dot(hidden_activation, self.params['W2']) + self.params['b2']
        output_probs = self.softmax(output_logits)

        # Key point: cache intermediate values for backpropagation
        # They are needed when computing gradients in backpropagation
        self.cache = {
            'hidden_linear': hidden_linear,
            'hidden_activation': hidden_activation,
            'output_logits': output_logits,
            'output_probs': output_probs,
            'inputs': X
        }

        return output_probs

    def backward(self, X, y, learning_rate=0.1):
        """
           Backpropagation algorithm.
            :param X: Input data from the latest forward pass (consistency check only).
            :param y: Ground-truth labels (one-hot encoded).
            :param learning_rate: Learning rate.
        """
        # 0. Setup
        cached_inputs = self.cache['inputs']
        if X.shape != cached_inputs.shape or not np.array_equal(X, cached_inputs):
            raise ValueError("Input X in backward() must match the latest forward() batch.")

        batch_size = cached_inputs.shape[0]
        # Retrieve cached intermediate values from forward propagation
        # Use self.params directly for parameters
        output_weights = self.params['W2']
        hidden_activation = self.cache['hidden_activation']
        hidden_linear = self.cache['hidden_linear']

        # 1. Output-layer gradients
        output_probs = self.cache['output_probs']
        output_delta = output_probs - y  # Cross-entropy + softmax derivative

        grad_W2 = np.dot(hidden_activation.T, output_delta) / batch_size
        grad_b2 = np.sum(output_delta, axis=0, keepdims=True) / batch_size

        # 2. Hidden-layer gradients
        # 2.1 Backpropagate output delta to hidden activation gradient
        hidden_activation_grad = np.dot(output_delta, output_weights.T)

        # 2.2 Activation derivative (ReLU derivative)
        # Gradient flows only where hidden pre-activation > 0; otherwise it is zero
        hidden_linear_delta = hidden_activation_grad * (hidden_linear > 0)

        # 2.3 Compute first-layer parameter gradients
        # cached_inputs.T is required here
        grad_W1 = np.dot(cached_inputs.T, hidden_linear_delta) / batch_size
        grad_b1 = np.sum(hidden_linear_delta, axis=0, keepdims=True) / batch_size

        # 3. Update parameters (gradient descent)
        self.params['W1'] -= learning_rate * grad_W1
        self.params['b1'] -= learning_rate * grad_b1
        self.params['W2'] -= learning_rate * grad_W2
        self.params['b2'] -= learning_rate * grad_b2

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
