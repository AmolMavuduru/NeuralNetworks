import numpy as np

# This class implements a simple neural network with just one hidden layer
class SimpleNeuralNetwork(object):


    def __init__(self, num_inputs, hidden_layer_size, num_outputs):
        self.num_inputs = num_inputs + 1
        self.num_outputs = num_outputs
        self.hidden_layer_size = hidden_layer_size

        self.first_weights = np.random.randn(self.num_inputs, self.hidden_layer_size)
        self.second_weights = np.random.randn(self.num_inputs, self.hidden_layer_size)


    def sigmoid(z):

        return 1/(1+np.exp(-z))


    def forward_propagate(self, X):

        self.z2 = np.dot(X, self.first_weights)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.second_weights)
        output = self.sigmoid(self.z3)

        return output

    def sigmodPrime(z):
        # Derivative of Sigmoid Activation Function
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunctionPrime(self, X, y):

        # Returns partial derivatives with respect to the two weight layers

        # Compute derivative with respect to both weight sets
        self.yHat = self.forward_propagate(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.second_weights.T)*self.sigmodPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2








