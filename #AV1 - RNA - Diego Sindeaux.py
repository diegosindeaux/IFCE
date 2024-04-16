#AV1 - RNA - Diego Sindeaux

#Questão 8
import numpy as np
from sklearn.model_selection import train_test_split

def linear_activation(x):
    return x

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class Adaline:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = linear_activation(linear_output)
                error = y[idx] - y_predicted

                self.weights += self.learning_rate * error * x_i
                self.bias += self.learning_rate * error

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = linear_activation(linear_output)
        return y_predicted

#Questão 9
import numpy as np
from sklearn.model_selection import train_test_split

def step_function(x):
    return np.where(x >= 0, 1, -1)

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else -1 for i in y])

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = step_function(linear_output)
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = step_function(linear_output)
        return y_predicted

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
predictions = perceptron.predict(X_test)

accuracy = np.mean(predictions == Y_test)
print(f'Acurácia: {accuracy}')

#Questão 10

import numpy as np
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class MLP:
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate=0.1, epochs=10000):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.hidden_weights = np.random.rand(self.n_inputs, self.n_hidden)
        self.hidden_bias = np.random.rand(self.n_hidden)
        self.output_weights = np.random.rand(self.n_hidden, self.n_outputs)
        self.output_bias = np.random.rand(self.n_outputs)

    def feedforward(self, X):
        self.hidden_layer_activation = np.dot(X, self.hidden_weights) + self.hidden_bias
        self.hidden_layer_output = sigmoid(self.hidden_layer_activation)

        self.output_layer_activation = np.dot(self.hidden_layer_output, self.output_weights) + self.output_bias
        output = sigmoid(self.output_layer_activation)
        return output

    def backpropagation(self, X, y, output):
        error = y - output
        d_predicted_output = error * sigmoid_derivative(self.output_layer_activation)

        error_hidden_layer = d_predicted_output.dot(self.output_weights.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(self.hidden_layer_activation)

        self.output_weights += self.hidden_layer_output.T.dot(d_predicted_output) * self.learning_rate
        self.output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * self.learning_rate
        self.hidden_weights += X.T.dot(d_hidden_layer) * self.learning_rate
        self.hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y):
        for _ in range(self.epochs):
            output = self.feedforward(X)
            self.backpropagation(X, y, output)

X = np.array([[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1],
              [1, 1, 1, 1, 1]])

Y = np.array([[0],
              [1],
              # ...
              [1]])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

mlp = MLP(n_inputs=5, n_hidden=4, n_outputs=1)

mlp.train(X_train, Y_train)

predictions = mlp.feedforward(X_test)
predictions = np.where(predictions > 0.5, 1, 0)

accuracy = np.mean(predictions == Y_test)
print(f'Acurácia: {accuracy}')

#Questão 11
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def linear(x):
    return x

class MLPRegressor:
    def __init__(self, n_inputs, n_hidden, learning_rate=0.01, epochs=10000):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_weights = np.random.randn(self.n_inputs, self.n_hidden)
        self.hidden_bias = np.zeros(self.n_hidden)
        self.output_weights = np.random.randn(self.n_hidden)
        self.output_bias = 0

    def feedforward(self, X):
        self.hidden_layer_input = np.dot(X, self.hidden_weights) + self.hidden_bias
        self.hidden_layer_output = relu(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.output_weights) + self.output_bias
        output = linear(self.output_layer_input)
        return output

    def backpropagation(self, X, y, output):
        d_output_error = output - y
        d_output_layer_input = d_output_error * 1 
        d_hidden_layer_error = d_output_layer_input.dot(self.output_weights.T)
        d_hidden_layer_input = d_hidden_layer_error * relu_derivative(self.hidden_layer_input)
        self.output_weights -= self.learning_rate * self.hidden_layer_output.T.dot(d_output_layer_input)
        self.output_bias -= self.learning_rate * np.sum(d_output_layer_input, axis=0)
        self.hidden_weights -= self.learning_rate * X.T.dot(d_hidden_layer_input)
        self.hidden_bias -= self.learning_rate * np.sum(d_hidden_layer_input, axis=0)

    def train(self, X, y):
        for _ in range(self.epochs):
            output = self.feedforward(X)
            self.backpropagation(X, y, output)

    def predict(self, X):
        return self.feedforward(X)

X = np.array([2, 4, -2, 1, 2, 4, 4, 6, 7, 8, 10, 15, 16, 18, 20]).reshape(-1, 1)
y = np.array([0.1, 0.6, 2.1, 6.0, 6.5, 7.1, 7.2, 8.1, 8.1, 9.2, 10.1, 14.7, 15.2, 16.5, 17.3])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

mlp_regressor = MLPRegressor(n_inputs=1, n_hidden=10)

mlp_regressor.train(X_train, y_train)

y_pred = mlp_regressor.predict(X_test)

eqm = mean_squared_error(y_test, y_pred)
print(f'Erro Quadrático Médio (EQM): {eqm}')
