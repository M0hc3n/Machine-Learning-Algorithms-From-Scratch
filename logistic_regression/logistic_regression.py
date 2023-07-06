import numpy as np
from utils.Loss import Loss


# this article helps to understand the theory and the implementation 
# https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc
class LogisticRegression:

    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        self.weights = None
        self.bias = None
        # utility Loss class instance to calculate sigmoid method values
        self.loss_calculator = Loss()

    # an implementation of the fit method
    def fit(self, X, y):
        n_samples , n_features = X.shape

        # initialize model's parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # apply the interative Gradient Descent algorithm
        for _ in range(self.n_iterations):
            # step 1: get y_hat's value
            y_hat = np.dot(X, self.weights) + self.bias
            #   we need to bound y_hat values to be among [0, 1]
            #   therefore, we apply the Sigmoid function
            y_hat = self.loss_calculator.sigmoid_function(y_hat)

            # step 2: calculate the gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_hat - y))
            db = (1 / n_samples) * np.sum(y_hat - y)

            # step 3: update the params accordingly
            self.weights = self.weights - (self.learning_rate * dw)
            self.bias = self.bias - (self.learning_rate * db)

    # an implementation of the predict method
    def predict(self, X_sample):

        # first calculate: y = w*x + b
        y_predicated = np.dot(X_sample,self.weights) + self.bias

        # then bound the value of y into [0,1]
        y_predicated = self.loss_calculator.sigmoid_function(y_predicated)

        # we need to round (or floor) the values, 
        # so that we output either 0 or 1 (rather than float values between 0 and 1)
        y_predicated_fixed = [1 if i > 0.5 else 0 for i in y_predicated]

        return np.array(y_predicated_fixed) 
    