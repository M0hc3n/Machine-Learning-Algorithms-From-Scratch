import numpy as np

from utils.factory.ActivationFunctionFactory import ActivationFunctionFactory

# implementation of single perceptron unit
# check out this enriching article:
# https://towardsdatascience.com/perceptron-algorithm-in-python-f3ac89d2e537
class Perceptron:

    def __init__(self,learning_rate=0.001, epochs=1000, activation='relu'):
        
        self.learning_rate = learning_rate
        self.epochs = epochs
         
        # get the corresponding activation function object
        self.activation = ActivationFunctionFactory().getActivationFunction(activation_function=activation)

        # initialize the model's parameters
        self.weights = None
        self.bias = None

    def fit(self, X,y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        # create two categories from given y
        y_ = np.array([1 if i > 0 else 0 for i in y])

        # outermost loop -> for number of epochs
        for _ in range(self.epochs):

            # repeate the sample algorithm for all the dataset
            for idx, x in enumerate(X):

                # step 1: calculate the linear model: y = w * x + b
                linear_model = np.dot(x, self.weights) + self.bias

                # step 2: apply the activation function to the linear model
                y_hat = self.activation.unit_step_function(linear_model)

                # step 3: update the model's parameters

                # --> recall: delta_w = lr * (y_hat - y_true) 
                delta_w = self.learning_rate * ( y_[idx] - y_hat ) 

                self.weights = self.weights + ( delta_w * x )
                self.bias = self.bias + delta_w

    def predict(self, X_sample):
        y_hat = np.dot(X_sample, self.weights) + self.bias
        y_hat = self.activation.unit_step_function(y_hat)

        return y_hat

