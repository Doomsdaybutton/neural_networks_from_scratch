import numpy as np

np.random.seed(0)


def randomInit(j, k):
    W = np.random.randn(j, k)
    B = np.random.randn(j)
    return W, B


def relu(Z):
    # if Z is smaller than zero, then the maximum of Z and 0 is 0 and if Z is bigger than 0 then the maximum of Z and 0 is Z = ReLU
    return np.maximum(Z, 0)


def reluDerivative(Z):
    # the slope is one if Z is bigger than 0 and 0 if Z is smaller than 0 (and non differentiable for 0)
    return Z > 0


class Layer:
    def __init__(self, j, k):
        W, B = randomInit(j, k)
        self.W = W
        self.B = B

    def forward(self, X):
        Z = np.dot(self.W, X)+self.B
        A = relu(Z)
        return Z, A


def backprop():
    pass


X = [1.0, -2.0, 1.5]

layer1 = Layer(4, 3)
layer2 = Layer(6, 4)
print(X)
print(layer1.W)
print(layer1.B)
Z, A = layer1.forward(X)
print(Z, A)
print(layer2.W)
print(layer2.B)
print(layer2.forward(A))
