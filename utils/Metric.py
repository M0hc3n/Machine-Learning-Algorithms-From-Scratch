import numpy as np

class Metric:

    # utility method to calculate the MSE model metric
    def mean_squared_error(self, y_true, y_hat):
        return np.mean((y_true - y_hat) ** 2)