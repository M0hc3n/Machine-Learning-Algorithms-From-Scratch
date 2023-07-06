import numpy as np

class Metric:

    # utility method to calculate the MSE model metric
    def mean_squared_error(self, y_true, y_hat):
        return np.mean((y_true - y_hat) ** 2)

    # this function calculates the ratio of true classificated
    # samples out of the global distribution
    def basic_accuracy_score(self, y_true, y_hat ):
        return np.sum(y_true == y_hat) / len(y_true)
