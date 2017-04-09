import numpy as np
from scipy import optimize

from neuralnetwork import SimpleNeuralNetwork
from neuralnetwork import trainer

X = np.array([[1,0]])
Y = np.array([[1]])

neural_net = SimpleNeuralNetwork(2, 3, 1)
T = trainer(neural_net)
T.train(X, Y)
print(neural_net.forward_propagate(X))



