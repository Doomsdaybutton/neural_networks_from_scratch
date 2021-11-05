from ast import Param
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# np.random.seed(0)

# get under https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/3004/861823/compressed/train.csv.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1633948773&Signature=Wg%2FzUFK8kw%2FQCP%2BPelFcluT5tkTHNZ4AA1qmF2Pq5ROghGbW9qJoDVH%2FesI%2Belwkcu%2Bq3tQWLppUmzkBDsJ22Shw4AHCt9vvDEJgHRq1QOquiffP4pg06NRcKmQbACbL0BnRUSqYA2w9o2r7aBhWScDFWyQGhDokVdENZAAyx9f5GMQzqIDHRSDXiJ4MAuqYKAeuchbKw5TtdomQ7DyxgUOIS%2BT%2Fioc3I3MKqFYrphszpXPRg9QKa8GfM8xXAP8WL3q%2BNtwkq%2FUNDNIDwUVheBC4rY7HcIKpc7hMbz9x05%2B6ZpcBtYTfjNu2VFDfx1R%2BiD%2BqfjZ0UFj2TCuOjCGXgA%3D%3D&response-content-disposition=attachment%3B+filename%3Dtrain.csv.zip
data = pd.read_csv(__file__.replace('\\', '/')[:__file__.rfind('\\')+1]+'train.csv')
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
        self.k = k
        self.j = j

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

    def backward(self, partial_error, learning_rate, batch_size):
        """
        Update parameters of this layer and return partial derivative of the cost with respect to the activation of the previous layer L-1.

        :param np.array<shape: n_L x m> partial_error: The partial derivative of the cost with respect to the activation of this layer
        :param float learning_rate: The learning rate of the gradient descent alpha
        :return: The partial derivative of the cost with respect to the activation of the previous layer
        :rtype: np.array<shape: n_L-1 x m>
        """

        #
        error = partial_error*self.activation_function_derivative(self.Z)

        dW = 1/batch_size*np.dot(error, self.X.T)
        dB = 1/batch_size*np.sum(error, axis=1)
        dA = np.dot(self.W.T, error)
        self.W -= dW*learning_rate
        self.B -= dB*learning_rate
        return dA

class NeuralNetwork:
    def __init__(self):
        self.layers=[]
    def add_layer(self, k, j, activation_function, activation_function_derivative):
        layer = Layer(k, j, activation_function, activation_function_derivative)
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

                current_error = current_A-oneHot(mini_batch_Y)
                for p in range(len(self.layers), 0, -1):
                    current_error = self.layers[p-1].backward(current_error, learning_rate, len(mini_batch))

                if i % 25 == 0:
                    current_accuracy = getAccuracy(current_A, mini_batch_Y)
                    print("Iteration: ", i, "; Accuracy: ", current_accuracy)
                    accuracies.append(current_accuracy)

        print("Max accuracy: ", np.max(accuracies))


def oneHot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    for i in range(Y.size):
        one_hot_Y[i][Y[i]] = 1
    return one_hot_Y.T


def getAccuracy(A2, Y):
    predictions = np.argmax(A2, axis=0)
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

m = 1000
n = 784

# m x n+1
dataTrain = data[:m]
neural_network = NeuralNetwork()
neural_network.add_layer(n, 10, relu, reluDerivative)
neural_network.add_layer(10, 10, softmax, blankActivationDerivative)
neural_network.gradient_descent(dataTrain, learning_rate=0.3, mini_batch_size=100, epochs=2000)
