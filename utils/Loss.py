import numpy as np

class Loss:

    # this article help to understand the implementation 
    # https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e
    def sigmoid_function(self, x):
        return 1 / (1 + np.exp(-x))

    # this function implements the Hinge Loss
    # those articles help to understand its concept and relation with SVM
    # https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
    # https://en.wikipedia.org/wiki/Hinge_loss
    def hinge_loss(self, y_true, y_hat):
        return 0 if y_hat * y_true >= 1 else ( 1 - (y_hat * y_true) )