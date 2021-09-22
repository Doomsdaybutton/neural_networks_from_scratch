import numpy as np

np.random.seed(0)

def randomInit(j, k):
    W = np.random.randn(j, k)
    B = np.random.randn(j)
    return W, B

class Layer:
    def __init__(self, j,k,activationFunction):
        if(activationFunction):
            self.activationFunction = activationFunction
        else:
            def defaultActivationFunction(x):
                return x
            self.activationFunction = defaultActivationFunction
        W, B = randomInit(j,k)
        self.W = W
        self.B = B
    
    def forward(self):
        Z=np.dot(self.W, self)
