import numpy as np


# this is a utility class to calculate the Probability Density Function
class PDF:

    # calculate the gaussian function value for given x, mean and variance
    def calculate_gaussian_density(self,x, mean, variance):
        numerator = np.exp(-(( x - mean) ** 2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)

        return numerator / denominator