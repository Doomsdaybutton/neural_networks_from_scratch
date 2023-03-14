from ast import Param
from functools import partial
import time
import numpy as np
from numpy.core.defchararray import replace
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import json
import cv2

from logger import Logger

# np.random.seed(0)


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

    def fforward(self, X):
        """
        Calculate Activations of this layer and return Z and A

        :param np.array<shape: 784 x m> X: Inputs of previous layer
        :return: Z, A
        """
        # 784 x m
        self.X = X

        # see https://stackoverflow.com/a/19602209/15045364
        # j x m
        Z = np.dot(self.W, X)+self.B
        self.Z = Z
        A = self.activation_function(Z)
        self.A = A
        return Z, A

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
    return -(Y/((A+0.00001)*np.log(2)))


class NeuralNetwork:
    def __init__(self, logger, cost_function_derivative):
        self.iterations = 0
        self.layers = []
        self.cost_function_derivative = cost_function_derivative
        self.logger = logger

    def add_layer(self, k, j, activation_function, activation_function_derivative):
        layer = Layer(k, j, activation_function,
                      activation_function_derivative)
        self.layers.append(layer)

    def remove_layer(self, n=-1):
        self.layers.remove(n)

    def forward(self, x):
        current_A = x
        for layer in self.layers:
            currentZ, current_A = layer.forward(current_A)
        return current_A

    def gradient_descent(self, dataTrain, learning_rate, mini_batch_size, epochs):
        accuracies = []

        for i in range(epochs):
            tic = time.perf_counter()
            self.iterations += 1
            # m x n
            np.random.shuffle(dataTrain)
            for j in range(0, len(dataTrain), mini_batch_size):
                # n+1 x mini_batch_size
                mini_batch = dataTrain[j:j+mini_batch_size].T

                # if number of samples not divisible by mini_batch_size: see below

                # 1 x mini_batch_size
                mini_batch_Y = mini_batch[0]

                # n x mini_batch_size = 784 x mini_batch_size
                mini_batch_X = mini_batch[1:]

                current_A = mini_batch_X
                for layer in self.layers:
                    currentZ, current_A = layer.forward(current_A)

                current_error = self.cost_function_derivative(
                    one_hot(mini_batch_Y, self.layers[len(self.layers)-1].j), current_A)
                for l in range(len(self.layers), 0, -1):
                    current_error = self.layers[l-1].backward(
                        current_error, learning_rate, mini_batch.shape[1])
            toc = time.perf_counter()
            self.logger.train_acc_at_iteration(
                self.iterations, getAccuracy(current_A, mini_batch_Y))
            self.logger.train_time_at_iteration(self.iterations, toc-tic)

    def single_test(self, dataTest):
        dataTest = dataTest.T
        Y = dataTest[0]
        X = dataTest[1:]
        acc = getAccuracy(self.forward(X), Y)
        self.logger.test_acc_at_iteration(self.iterations, acc)
        return acc

    def test(self, dataTest, dataTrain, learning_rate,
             mini_batch_size, epochs, freq):
        dataTest = dataTest.T
        Y = dataTest[0]
        X = dataTest[1:]

        self.logger.test_acc_at_iteration(
            self.iterations, getAccuracy(self.forward(X), Y))
        for i in range(0, epochs, freq):
            self.gradient_descent(
                dataTrain, learning_rate, mini_batch_size, freq)
            self.logger.test_acc_at_iteration(
                self.iterations, getAccuracy(self.forward(X), Y))


def one_hot(Y, j):
    one_hot_Y = np.zeros((Y.size, j))
    for i in range(Y.size):
        # or try one_hot_Y[i][int(Y[i])] = 1
        one_hot_Y[i][int(Y[i])-1] = 1
    return one_hot_Y.T


def getAccuracy(A, Y):
    predictions = np.argmax(A, axis=0)
    #print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

# while(True):
#     print("Enter your image to be analyzed!")
#     try:
#         arr_input = list(map(int, input().split(",")))
#     except:
#         print("Input invalid!")
#         continue
#     arr = np.array(arr_input)

#     plt.title("Your digit:")
#     plt.imshow(arr.reshape(28, 28).T, cmap=cm.binary)
#     plt.show()

#     arr= arr.reshape((n, 1))

#     y = np.multiply(100, neural_network.forward(arr))
#     y_format = list(map(np.format_float_positional, y.round(4)))
#     for i in range(0, len(y_format)):
#         print(str(i), ": ", y_format[i], "% certainty")

#     print("Continue? [y/n]")
#     if(input() == "y"):
#         continue
#     else:
#         break


# file = r'M:/five.jpg'

# test_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
# test_image_resized = cv2.bitwise_not(cv2.resize(
#     test_image, (28, 28), interpolation=cv2.INTER_LINEAR))
# plt.title("should be white on black background")
# plt.imshow(test_image_resized, cmap='gray')
# plt.show()

# arr = np.array(test_image_resized)/255
# arr = arr.reshape((n, 1))

# y = np.multiply(100, neural_network.forward(arr))
# y_format = list(map(np.format_float_positional, y.round(4)))
# for i in range(0, len(y_format)):
#     print(str(i), ": ", y_format[i], "% certainty")


# demonstration
# counter = 0
# number_of_examples = 10
# for i in range(0, number_of_examples):
#     current_img = data[m+np.random.randint(0, len(data[m:]))]
#     label = current_img[0]
#     current_img = current_img[1:]

#     current_img = current_img.reshape((28, 28))
#     plt.title("MNIST Image. Label:"+str(label))
#     plt.imshow(current_img, cmap='gray')
#     plt.show()

#     current_img = current_img.reshape((n, 1))
#     y = np.multiply(100, neural_network.forward(current_img))
#     y_format = list(map(np.format_float_positional, y.round(4)))
#     for i in range(0, len(y_format)):
#         print(str(i), ": ", y_format[i], "% certainty")

#     print("Label: ", str(label), " Argmax: ", np.argmax(y))
#     if label == np.argmax(y):
#         counter += 1
#     input()
# print("Testing Accuracy: ", np.round(counter/number_of_examples, 2))

# while True:
#     cin = input()
#     if(cin == '-1'):
#         break

#     file = r'M:/test.jpg'

#     test_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
#     test_image_resized = cv2.bitwise_not(cv2.resize(
#         test_image, (28, 28), interpolation=cv2.INTER_LINEAR))
#     plt.title("should be white on black background")
#     plt.imshow(test_image_resized, cmap='gray')
#     plt.show()

#     arr = np.array(test_image_resized)/255
#     arr = arr.reshape((n, 1))

#     y = np.multiply(100, neural_network.forward(arr))
#     y_format = list(map(np.format_float_positional, y.round(4)))
#     for i in range(0, len(y_format)):
#         print(str(i), ": ", y_format[i], "% certainty")
