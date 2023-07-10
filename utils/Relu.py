import numpy as np

from utils.interfaces.IActivationFunction import IActivationFunction

# utility class that implements Relu function
class Relu(IActivationFunction):

    # unit step function that calculated relu value for a given sample x
    def unit_step_function(self, x):
        return np.where(x >= 0 , 1, 0)