from ast import Param
from functools import partial
import numpy as np
from numpy.core.defchararray import replace
import pandas as pd
import matplotlib.pyplot as plt
import json

# np.random.seed(0)

# get under https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/3004/861823/compressed/train.csv.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1633948773&Signature=Wg%2FzUFK8kw%2FQCP%2BPelFcluT5tkTHNZ4AA1qmF2Pq5ROghGbW9qJoDVH%2FesI%2Belwkcu%2Bq3tQWLppUmzkBDsJ22Shw4AHCt9vvDEJgHRq1QOquiffP4pg06NRcKmQbACbL0BnRUSqYA2w9o2r7aBhWScDFWyQGhDokVdENZAAyx9f5GMQzqIDHRSDXiJ4MAuqYKAeuchbKw5TtdomQ7DyxgUOIS%2BT%2Fioc3I3MKqFYrphszpXPRg9QKa8GfM8xXAP8WL3q%2BNtwkq%2FUNDNIDwUVheBC4rY7HcIKpc7hMbz9x05%2B6ZpcBtYTfjNu2VFDfx1R%2BiD%2BqfjZ0UFj2TCuOjCGXgA%3D%3D&response-content-disposition=attachment%3B+filename%3Dtrain.csv.zip
data = pd.read_csv(__file__.replace('\\', '/')
                   [:__file__.rfind('\\')+1]+'train.csv')
data = np.array(data)
np.random.shuffle(data)


def randomInit(j, k):
    # j is number of neurons in THIS layer and k is number of neurons in last layer
    W = np.random.randn(j, k)
    B = np.random.randn(j)
    return W, B


def blankActivation(Z):
    return Z


def blankActivationDerivative(Z):
    return np.ones(Z.shape)


def relu(Z):
    # if Z is smaller than zero, then the maximum of Z and 0 is 0 and if Z is bigger than 0 then the maximum of Z and 0 is Z = ReLU
    return np.maximum(Z, 0)


def relu_derivative(Z):
    # the slope is one if Z is bigger than 0 and 0 if Z is smaller than 0 (and non differentiable for 0)
    return Z > 0


def softmax(Z):
    # mathematical trick to prevent overflow (multiply both the numerator and denominator by e^K and choose K=-max(Z))
    # j x m
    #print(np.exp(Z-np.max(Z, axis=0, keepdims=True)) / np.sum(np.exp(Z-np.max(Z, axis=0, keepdims=True)), axis=0, keepdims=True))

    # normalized_Z = Z/np.max(np.abs(Z), axis=0, keepdims=True)
    # print("lol", np.max(np.abs(Z), axis=0, keepdims=True), Z)

    # normalized_Z = normalized_Z-np.max(normalized_Z, axis=0, keepdims=True)
    # return np.exp(normalized_Z) / np.sum(np.exp(normalized_Z), axis=0, keepdims=True)

    return np.exp(Z-np.max(Z, axis=0, keepdims=True))/np.sum(np.exp(Z-np.max(Z, axis=0, keepdims=True)), axis=0, keepdims=True)


def softmax_derivative(A):
    # TODO

    # see http://saitcelebi.com/tut/output/part2.html
    j, m = A.shape
    e = np.ones((j, 1))
    d = np.zeros((m, j, j))
    # j x j
    I = np.identity(j)
    for i in range(m):
        # j x 1
        a = np.reshape(A[:, i], (j, 1))
        # j x j
        temp = np.multiply(np.dot(a, e.T), (I-np.dot(e, a.T)))
        d[i] = temp
    # m x j x j
    return d


class Layer:
    def __init__(self, k, j, activation_function, activation_function_derivative):
        W, B = randomInit(j, k)
        self.W = W
        self.B = B
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.k = k
        self.j = j

    def forward(self, X):
        """
        Calculate Activations of this layer and return Z and A

        :param np.array<shape: 784 x m> X: Inputs of previous layer
        :return: Z, A
        """
        # 784 x m
        self.X = X

        # see https://stackoverflow.com/a/19602209/15045364
        # j x m
        Z = np.dot(self.W, X)+self.B[:, None]
        self.Z = Z
        A = self.activation_function(Z)
        self.A = A
        return Z, A

    def backward(self, partial_error, learning_rate, batch_size):
        """
        Update parameters of this layer and return partial derivative of the cost with respect to the activation of the previous layer L-1.

        :param np.array<shape: n_L x m> partial_error: The partial derivative of the cost with respect to the activation of this layer
        :param float learning_rate: The learning rate of the gradient descent alpha
        :return: The partial derivative of the cost with respect to the activation of the previous layer
        :rtype: np.array<shape: n_L-1 x m>
        """

        if(self.activation_function_derivative == softmax_derivative):
            error_per_sample = np.zeros((batch_size, self.j))
            partial_derivative = self.activation_function_derivative(self.A)
            for i in range(batch_size):
                error_per_sample[i] = np.dot(
                    partial_derivative[i], partial_error[:, i])
            error = error_per_sample.T
        else:
            error = partial_error*self.activation_function_derivative(self.Z)

        dW = 1/batch_size*np.dot(error, self.X.T)
        dB = 1/batch_size*np.sum(error, axis=1)
        dA = np.dot(self.W.T, error)
        self.W -= dW*learning_rate
        self.B -= dB*learning_rate
        return dA


def mean_squared_error_derivative(Y, A):
    return A-Y


def cross_entropy_derivative(Y, A):
    return -(Y/(A*np.log(2)))


class NeuralNetwork:
    def __init__(self, cost_function_derivative):
        self.layers = []
        self.cost_function_derivative = cost_function_derivative

    def add_layer(self, k, j, activation_function, activation_function_derivative):
        layer = Layer(k, j, activation_function,
                      activation_function_derivative)
        self.layers.append(layer)

    def remove_layer(self, n=-1):
        self.layers.remove(n)

    def gradient_descent(self, dataTrain, learning_rate, mini_batch_size, epochs):
        accuracies = []

        for i in range(epochs):
            # m x n
            np.random.shuffle(dataTrain)
            for j in range(0, len(dataTrain), mini_batch_size):
                # n+1 x mini_batch_size
                mini_batch = dataTrain[j:j+mini_batch_size].T

                # 1 x mini_batch_size
                mini_batch_Y = mini_batch[0]

                # n x mini_batch_size = 784 x mini_batch_size
                mini_batch_X = (mini_batch[1:])/255

                current_A = mini_batch_X
                for layer in self.layers:
                    currentZ, current_A = layer.forward(current_A)

                current_error = self.cost_function_derivative(
                    one_hot(mini_batch_Y), current_A)
                for l in range(len(self.layers), 0, -1):
                    current_error = self.layers[l-1].backward(
                        current_error, learning_rate, mini_batch_size)

            if i % 25 == 0:
                current_accuracy = getAccuracy(current_A, mini_batch_Y)
                print("Iteration: ", i, "; Accuracy: ", current_accuracy)
                accuracies.append(current_accuracy)
        print("Max accuracy: ", np.max(accuracies))


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    for i in range(Y.size):
        one_hot_Y[i][Y[i]] = 1
    return one_hot_Y.T


def getAccuracy(A, Y):
    predictions = np.argmax(A, axis=0)
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


m = 1000
n = 784

# m x n+1
dataTrain = data[:m]
neural_network = NeuralNetwork(cross_entropy_derivative)
neural_network.add_layer(n, 10, relu, relu_derivative)
neural_network.add_layer(10, 10, softmax, softmax_derivative)
neural_network.gradient_descent(
    dataTrain, learning_rate=0.3, mini_batch_size=100, epochs=2000)
