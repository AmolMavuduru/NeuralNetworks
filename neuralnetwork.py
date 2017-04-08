import numpy as np
from scipy import optimize

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


    def getParams(self):
        params = np.concatenate((self.first_weights.ravel(), self.second_weights.ravel()))
        return params

    def setParams(self, params):
        first_weights_start = 0
        first_weights_end = self.hiddenLayerSize * self.inputLayerSize
        self.first_weights = np.reshape(params[first_weights_start:first_weights_end], (self.num_inputs, self.hidden_layer_size))
        second_weights_end = first_weights_end + self.hidden_layer_size * self.num_outputs
        self.second_weights = np.reshape(params[first_weights_end:second_weights_end], (self.hidden_layer_size, self.num_outputs))


    def computeGradients(self, X, y):
         dJdW1, dJdW2 = self.costFunctionPrime(X, y)
         return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

    def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            # Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)

            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            # Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2 * e)

            # Return the value we changed to zero:
            perturb[p] = 0

        # Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad

  # Separate class for training neural networks
    class trainer(object):
        def __init__(self, N):
            # Make Local reference to network:
            self.N = N

        def callbackF(self, params):
            self.N.setParams(params)
            self.J.append(self.N.costFunction(self.X, self.y))

        def costFunctionWrapper(self, params, X, y):
            self.N.setParams(params)
            cost = self.N.costFunction(X, y)
            grad = self.N.computeGradients(X, y)

            return cost, grad

        def train(self, X, y):
            # Make an internal variable for the callback function:
            self.X = X
            self.y = y

            # Make empty list to store costs:
            self.J = []

            params0 = self.N.getParams()

            options = {'maxiter': 200, 'disp': True}
            _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                     args=(X, y), options=options, callback=self.callbackF)

            self.N.setParams(_res.x)
            self.optimizationResults = _res




