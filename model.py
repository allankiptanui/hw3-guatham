import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, sigmoid_param=1.0, initial_weights=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.sigmoid_param = sigmoid_param

        # Initialize weights and biases
        if initial_weights is None:
            self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
            self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        else:
            self.weights_input_hidden = initial_weights['input_hidden']
            self.weights_hidden_output = initial_weights['hidden_output']

        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def bipolar_sigmoid(self, x):
        # try:
        scaled_x = x * 0.1
        return 2 / (1 + np.exp(-self.sigmoid_param * scaled_x)) - 1
        # except OverflowError:
          # return  np.sign(x)

    def sigmoid_derivative(self, x):
        return self.sigmoid_param * (1 - np.power(self.bipolar_sigmoid(x), 2))

    def forward(self, X):
        # Input to hidden layer
        self.hidden_output = self.bipolar_sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)

        # Hidden to output layer
        self.output = self.bipolar_sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)

        return self.output

    def backward(self, X, y, output):
        # Calculate error
        error = y - output

        # Backpropagation
        d_output = error * self.sigmoid_derivative(output)
        error_hidden = d_output.dot(self.weights_hidden_output.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * self.learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        return self.forward(X)
