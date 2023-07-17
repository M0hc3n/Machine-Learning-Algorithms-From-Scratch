import numpy as np

class Metric:

    # utility method to calculate the MSE model metric
    def mean_squared_error(self, y_true, y_hat):
        return np.mean((y_true - y_hat) ** 2)

    # this function calculates the ratio of true classificated
    # samples out of the global distribution
    def basic_accuracy_score(self, y_true, y_hat ):
        return np.sum(y_true == y_hat) / len(y_true)

    # this function calculates the entropy value of a given vector
    def calculate_entropy(self,x):
        # returns a list of the number of occurences of each element in x
        hist = np.bincount(x)

        # gives the pi (i.e. frequency) of each class
        probabilities = hist / len(x)

        # calculates the entropy formula 
        return - np.sum([prob * np.log2(prob) for prob in probabilities if prob > 0 ])


