import numpy as np
from scipy import optimize

from neuralnetwork import SimpleNeuralNetwork
from neuralnetwork import trainer

X = np.array([[1,0],[0,1],[0,0],[1,1]])
Y = np.array([[1],[1],[0],[2]])

neural_net = SimpleNeuralNetwork(2, 10, 1)
T = trainer(neural_net)
T.train(X, Y)
print(neural_net.forward_propagate(X))



