import numpy as np
import pandas as pd
import csv

import neural_network as nn
import logger as l


def sex_encode(x):
    if(x == "M"):
        return 2.0
    if(x == "F"):
        return 1.0
    if(x == "I"):
        return 0.0


def sex_encode_arr(arr):
    return np.array([[sex_encode(x) for x in arr[0]]])


dataTrain = pd.read_csv(__file__.replace('\\', '/')
                        [:__file__.rfind('\\')+1]+'data/abalone/abalone.csv')
dataTrain = np.array(dataTrain)
# 42000x785
dataTrain = np.concatenate(
    (dataTrain.T[8, None], sex_encode_arr(dataTrain.T[0, None]), (dataTrain.T[1:8]))).T.astype('float64')


np.random.shuffle(dataTrain)

dataTest = dataTrain[:500]
dataTrain = dataTrain[1000:6000]

print(dataTrain.shape)
print(dataTest.shape)

logger1 = l.Logger('abalone', 'architecture1')

for i in range(100):
    model = nn.NeuralNetwork(logger1, nn.mean_squared_error_derivative)
    model.add_layer(8, 10, nn.relu, nn.relu_derivative)
    model.add_layer(10, 10, nn.relu, nn.relu_derivative)
    model.add_layer(10, 29, nn.softmax, nn.softmax_derivative)
    model.test(dataTest, dataTrain, learning_rate=0.03,
               mini_batch_size=100, epochs=2500, freq=1)

logger2 = l.Logger('abalone', 'architecture2')

for i in range(100):
    model = nn.NeuralNetwork(logger2, nn.mean_squared_error_derivative)
    model.add_layer(8, 10, nn.relu, nn.relu_derivative)
    model.add_layer(10, 10, nn.relu, nn.relu_derivative)
    model.add_layer(10, 29, nn.softmax, nn.softmax_derivative)
    model.test(dataTest, dataTrain, learning_rate=0.6,
               mini_batch_size=100, epochs=2500, freq=1)
