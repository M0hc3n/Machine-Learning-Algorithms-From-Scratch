
from utils.Relu import Relu

class ActivationFunctionFactory:

    def getActivationFunction(self, activation_function):
        if(activation_function is None):
            return ValueError("You must provide an activation function")
        
        if(activation_function == 'relu'):
            return Relu()