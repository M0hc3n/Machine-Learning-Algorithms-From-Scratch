import numpy as np
from collections import Counter

from utils.Distance import Distance


# class to implemtent KNN algorithm
class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X,y):
        self.X_train = X
        self.y_train = y

    # helper function to run the knn algorithm for one single sample
    def predict_helper(self, x):
        distances_obj = Distance()
        
        # step 1: calculate distance between the point and the rest of the dataset
        distances = [distances_obj.calculate_euclidean_distance(x, x_train) for x_train in self.X_train]

        # step 2: sort the distances
        distances_idx = np.argsort(distances)

        # step 3: only keep the first k samples
        distances_idx = distances_idx[:self.k]

        # step 4: grab the class labels from the first k calculated distances
        first_k_classes = [self.y_train[i] for i in distances_idx]

        # step 5: return the most common class label
        # the Counter.most_common method return have the following format:
            # [ 
            #   (class 1, number_of_occurences), 
            #   (class 2, number_of_occurences),
            #   ...
            # ]
        # we only need the most common class, so we return the firt tuple 
        # element of the first array element.
        return Counter(first_k_classes).most_common(1)[0][0]

    # method to calculate the predicted vaues of the given sample 
    def predict(self, sample):
        predicted = [self.predict_helper(x) for x in sample]

        return np.array(predicted)