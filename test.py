from ast import Param
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

data = pd.read_csv('train.csv')
data = np.array(data)
# np.random.shuffle(data)

# reserve first 1000 samples for testing
dataTest = data[:1000].T
dataTrain = data[1000:].T

YTrain = dataTrain[0]
XTrain = dataTrain[1:]
YTest = dataTest[0]
XTest = dataTest[1:]

# m=1000, n=784=28*28
n, m = XTest.shape

# divide by 255
XTrain = XTrain / 255
XTest = XTest / 255

print(XTest, m, n)


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


def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))


class Layer:
    def __init__(self, k, j, activation_function, activation_function_derivative):
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

        # see https://stackoverflow.com/a/19602209/15045364
        Z = np.dot(self.W, X)+self.B[:, None]
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
        dB = 1/m*np.sum(error, axis=1)
        dA = np.dot(self.W.T, error)
        self.W -= dW*learning_rate
        self.B -= dB*learning_rate
        return dA


def gradientDescent():
    pass


def oneHot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    for i in range(Y.size):
        one_hot_Y[i][Y[i]] = 1
    return one_hot_Y.T


def getAccuracy(A2, Y):
    predictions = np.argmax(A2, axis=0)
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


X = XTest
Y = oneHot(YTest)

layer1 = Layer(n, 10, relu, reluDerivative)
layer2 = Layer(10, 10, softmax, blankActivationDerivative)
learning_rate = 0.3

for i in range(10000):
    Z, A = layer1.forward(X)
    Z2, A2 = layer2.forward(A)
    partial_error = A2-Y
    layer2_error = layer2.backward(partial_error, learning_rate)
    layer1_error = layer1.backward(layer2_error, learning_rate)
    if i % 10 == 0:
        print("Iteration: ", i, "; Accuracy: ", getAccuracy(A2, YTest))


# print(X.shape)
# print(layer1.W.shape)
# print(layer1.B.shape)
# Z, A = layer1.forward(X)
# print(Z.shape, A.shape)
# Z2, A2 = layer2.forward(A)
# print(Z2.shape, A2.shape)

# partial_error = A2-Y
# print(partial_error.shape, A2.shape, Y.shape)
# print(Y)

# layer2_error = layer2.backward(partial_error, 0.1)
# print(layer2_error.shape)

# layer1_error = layer1.backward(layer2_error, 0.1)
# print(layer1_error.shape)
