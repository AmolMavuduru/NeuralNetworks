import pandas as pd
from perceptron import Perceptron

iris_dataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

print(iris_dataframe.tail())

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Gets the first 100 class labels of the Iris data set, which consists of 50 Iris-Setosa and 50 Iris_versicolor flowers.
y = iris_dataframe.iloc[0:100, 4]

# Sets up y as a matrix of 0 and 1 values where a zero occurs when the label is 'Iris-setosa' and one otherwise.

y = np.where(y == 'Iris-setosa', 0, 1)

X = iris_dataframe.iloc[0:100, [0,2]].values

# Uncomment the code below to display a scatterplot of the values from the Iris dataset.
""""

plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='o', label='setosa')

plt.scatter(X[:50, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')


plt.show()
"""

perceptron = Perceptron(learn_rate=0.1, n_iter=10)
perceptron.fit(X,y)

plt.plot(range(1,len(perceptron.errors_) + 1),perceptron.errors_, marker='o')

plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')

plt.show()

def plot_decision_regions(X, y, classifer, resolution=0.02):

    # Generates setup marker and color map

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')