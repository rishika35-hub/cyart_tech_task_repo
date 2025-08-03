import numpy as np

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Layer definition
class LinearLayer:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.1
        self.b = np.zeros((1, out_features))
        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dY):
        self.dW = self.X.T @ dY
        self.db = np.sum(dY, axis=0, keepdims=True)
        return dY @ self.W.T

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]

# Model definition
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = LinearLayer(input_size, hidden_size)
        self.fc2 = LinearLayer(hidden_size, output_size)
        self.cache = {}

    def forward(self, X):
        self.cache['z1'] = self.fc1.forward(X)
        self.cache['a1'] = relu(self.cache['z1'])
        self.cache['z2'] = self.fc2.forward(self.cache['a1'])
        return self.cache['z2']

    def backward(self, dY):
        dz2 = dY
        da1 = self.fc2.backward(dz2)
        dz1 = relu_derivative(self.cache['z1']) * da1
        self.fc1.backward(dz1)

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

# Mean Squared Error Loss
class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true)**2)

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]

# SGD Optimizer
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param, grad in self.parameters:
            param -= self.lr * grad
