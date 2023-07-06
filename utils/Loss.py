import numpy as np

class Loss:

    # this article help to understand the implementation 
    # https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e
    def sigmoid_function(self, x):
        return 1 / (1 + np.exp(-x))