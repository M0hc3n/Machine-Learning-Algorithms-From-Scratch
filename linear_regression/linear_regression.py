import numpy as np

class LinearRegression:
    
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        # run Gradient Descent algorithm for given n_iterations
        for _ in range(self.n_iterations):
            # calculate y_hat = w * x + bias
            y_hat = np.dot(X, self.weights) + self.bias

            # compute df/dw
            dw = np.dot(X.T, (y_hat - y)) / n_samples

            # compute df/db
            db = np.sum(y_hat - y) / n_samples
            
            # update weights and bias
            self.weights = self.weights - (self.learning_rate * dw)
            self.bias = self.bias - (self.learning_rate * db)

    # method to predict y given a sample of data
    def predict(self, X_sample):
        y_predicted = np.dot(X_sample, self.weights) + self.bias

        return y_predicted

        


