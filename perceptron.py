import numpy as np

class Perceptron(object):
     """Perceptron classifier built using numpy's matrix functions.

     Parameters

     ------------------
     learn_rate: float
     Learning rate: takes a value between 0.0 and 1.0

     n_iter: int
        Number of passes over the training dataset.

     Attributes:
     -----------

     w_ : 1d-array
        Weights after fitting
     errors_ : list
        Number of misclassifications at every epoch.

      """

     def __init__(self, learn_rate=0.01, n_iter=10):
         self.learn_rate = learn_rate
         self.n_iter = n_iter


     def fit(self, X, Y):

         """Fit training data.

         Parameters
         ----------
         X : {array-like}, shape = [n_samples, n_features]
         Training vectors, where n_samples = number of samples
         and n_features is the number of features.

         y: array-like, shape = [n_samples]
         Target values for training set.

         Returns self: object

         """

         self.w_ = np.zeros(1 + X.shape[1])
         self.errors_ = []

         for _ in range(self.n_iter):
             errors = 0

             for xi, target in zip(X, Y):
                 update - self.learn_rate * (target - self.predict(xi))
                 self.w_[1:] += update * xi
                 self.w_[0] += update
                 errors += int(update  != 0.0)
             self.errors_.append(errors)

         return self

     def net_input(self, X):
         """Calculates the net input by multiplying X (contains input values) by the weights.
         This is the value that results just before applying the activation function."""

         return np.dot(X, self.w_[1:]) + self.w_[0]

     def predict(self, X):
         """
         :param X: input data
         :return: Class label after unit step by applying Heaveside step function to net input.
         """

         # Returns 1 is the net input is greater than 0.0 and -1 otherwise.

         return np.where(self.net_input(X) >= 0.0, 1, -1)

