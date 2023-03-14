from ast import Param
from functools import partial
import numpy as np
from numpy.core.defchararray import replace
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import json
import cv2

import neural_network as nn
import logger as l

# get under https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/3004/861823/compressed/train.csv.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1633948773&Signature=Wg%2FzUFK8kw%2FQCP%2BPelFcluT5tkTHNZ4AA1qmF2Pq5ROghGbW9qJoDVH%2FesI%2Belwkcu%2Bq3tQWLppUmzkBDsJ22Shw4AHCt9vvDEJgHRq1QOquiffP4pg06NRcKmQbACbL0BnRUSqYA2w9o2r7aBhWScDFWyQGhDokVdENZAAyx9f5GMQzqIDHRSDXiJ4MAuqYKAeuchbKw5TtdomQ7DyxgUOIS%2BT%2Fioc3I3MKqFYrphszpXPRg9QKa8GfM8xXAP8WL3q%2BNtwkq%2FUNDNIDwUVheBC4rY7HcIKpc7hMbz9x05%2B6ZpcBtYTfjNu2VFDfx1R%2BiD%2BqfjZ0UFj2TCuOjCGXgA%3D%3D&response-content-disposition=attachment%3B+filename%3Dtrain.csv.zip
dataTrain = pd.read_csv(__file__.replace('\\', '/')
                        [:__file__.rfind('\\')+1]+'data/train.csv')
dataTrain = np.array(dataTrain)
# 42000x785
dataTrain = np.concatenate((dataTrain.T[0, None], (dataTrain.T[1:])/255)).T
np.random.shuffle(dataTrain)


# dataTest = pd.read_csv(__file__.replace('\\', '/')
#                        [:__file__.rfind('\\')+1]+'data/test.csv')
# dataTest = np.array(dataTest)
# dataTest = np.concatenate((dataTest.T[0, None], (dataTest.T[1:])/255)).T
# np.random.shuffle(dataTest)

dataTest = dataTrain[:1000]
dataTrain = dataTrain[1000:6000]
logger7 = l.Logger('mnist', 'architecture5')

for i in range(100):
    model = nn.NeuralNetwork(logger5, nn.cross_entropy_derivative)
    model.add_layer(784, 10, nn.relu, nn.relu_derivative)
    model.add_layer(10, 10, nn.relu, nn.relu_derivative)
    model.add_layer(10, 10, nn.softmax, nn.softmax_derivative)
    model.test(dataTest, dataTrain, learning_rate=0.3,
               mini_batch_size=100, epochs=2500, freq=1)

logger8 = l.Logger('mnist', 'architecture6')
for i in range(100):
    model = nn.NeuralNetwork(logger6, nn.mean_squared_error_derivative)
    model.add_layer(784, 10, nn.relu, nn.relu_derivative)
    model.add_layer(10, 10, nn.relu, nn.relu_derivative)
    model.add_layer(10, 10, nn.softmax, nn.softmax_derivative)
    model.test(dataTest, dataTrain, learning_rate=0.3,
               mini_batch_size=100, epochs=2500, freq=1)
