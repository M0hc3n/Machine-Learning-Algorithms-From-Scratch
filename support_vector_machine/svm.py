import numpy as np

from utils.Loss import Loss

# this class implements the Support Vector Machine Algorithm
# the latter resources had been used:
# https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
# https://www.youtube.com/watch?v=_YPScrckx28
# https://www.youtube.com/watch?v=efR1C6CvhmE
# https://www.youtube.com/watch?v=lDwow4aOrtg
class SVM:

    def __init__(self, learning_rate=0.001, n_iterations=10000, regularization_param=0.001):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization_param = regularization_param

        self.weights = None
        self.bias = None

        self.loss_calculator = Loss()

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # re-format y to only contain {-1,1} as class labels
        y_ = np.where(y<=0, -1,1) # y_i <= 0 ? -1 : 1

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # step 1: calculate y_hat (y_hat = w * x - b)
                y_hat = np.dot(x_i, self.weights) - self.bias

                # step 2: caculate the loss value
                hinge_loss_value = self.loss_calculator.hinge_loss(y_[idx], y_hat)
                
                # step 3: calculate the gradients based on the loss value

                if(hinge_loss_value == 0):
                    # means that: y_hat * y_true >= 1
                    dw = 2 * self.regularization_param * self.weights

                    self.weights = self.weights - (self.learning_rate * dw)
                else: 
                    dw = (2 * self.regularization_param * self.weights) - np.dot(y_[idx] ,x_i)
                    db = y_[idx]

                    self.weights = self.weights - (self.learning_rate * dw)
                    self.bias = self.bias - (self.learning_rate * db)           

    def predict(self,X_sample):
        pred = np.dot(X_sample,self.weights) - self.bias  
        # returns -1 if pred < 0, 0 if pred == 0, and 1 if pred > 0
        return np.sign(pred)   


