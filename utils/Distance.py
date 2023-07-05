import numpy 

class Distance:

    def calculate_euclidean_distance(self, x,y):
        return numpy.sqrt(numpy.sum((x-y)**2))