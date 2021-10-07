from ast import Param
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
data = np.array(data)
m, n = data.shape
# np.random.shuffle(data)

# reserve first 1000 samples for testing
dataTest = data[0:1000].T
dataTrain = data[1000:m].T

YTrain = dataTrain[0]
XTrain = dataTrain[1:n]
YTest = dataTest[0]
XTest = dataTest[1:n]

# divide by 255
XTrain = XTrain / 255
XTest = XTest / 255

print(data, m, n, dataTrain, XTrain)

np.random.seed(0)


def randomInit(j, k):
    # j is number of neurons in THIS layer and k is number of neurons in last layer
    W = np.random.randn(j, k)
    B = np.random.randn(j)
    return W, B


def blankActivation(Z):
    return Z


def blankActivationDerivative(Z):
    return 1


def relu(Z):
    # if Z is smaller than zero, then the maximum of Z and 0 is 0 and if Z is bigger than 0 then the maximum of Z and 0 is Z = ReLU
    return np.maximum(Z, 0)


def reluDerivative(Z):
    # the slope is one if Z is bigger than 0 and 0 if Z is smaller than 0 (and non differentiable for 0)
    return Z > 0


class Layer:
    def __init__(self, j, k, activation_function, activation_function_derivative):
        W, B = randomInit(j, k)
        self.W = W
        self.B = B
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def forward(self, X):
        """
        Calculate Activations of this layer and return Z and A

        :param np.array<shape: 784 x m> X: Inputs of previous layer
        :return: Z, A
        """
        self.X = X
        Z = np.dot(self.W, X)+self.B
        self.Z = Z
        A = self.activation_function(Z)
        return Z, A

    def backward(self, partial_error, learning_rate):
        """
        Update parameters of this layer and return partial derivative of the cost with respect to the activation of the previous layer L-1.

        :param np.array<shape: n_L x m> partial_error: The partial derivative of the cost with respect to the activation of this layer
        :param float learning_rate: The learning rate of the gradient descent alpha
        :return: The partial derivative of the cost with respect to the activation of the previous layer
        :rtype: np.array<shape: n_L-1 x m>
        """

        #
        error = partial_error*self.activation_function_derivative(self.Z)

        dW = 1/m*np.dot(error, self.X.T)
        dB = 1/m*np.sum(error, axis=0)
        dA = np.dot(self.W.T, error)
        self.W -= dW*learning_rate
        self.B -= dB*learning_rate
        return dA


def gradientDescent():
    pass


X = [1.0, -2.0, 1.5]

layer1 = Layer(4, 3, relu, reluDerivative)
layer2 = Layer(6, 4, relu, reluDerivative)
print(X)
print(layer1.W)
print(layer1.B)
Z, A = layer1.forward(X)
print(Z, A)
print(layer2.W)
print(layer2.B)
print(layer2.forward(A))
