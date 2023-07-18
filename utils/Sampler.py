import numpy as np

# this utility class serves as a wrapper for the existing sampling methods
class Sampler:

    # an implementation of the standard bootstrap sampling method
    def bootstrap_sample(self, X,y):
        n_samples = X.shape[0]
        # this will choose n_samples from the range [0,n_samples]
        # when replace is set to True, the choice can select the same integer more than once
        # which means, it drops other numbers in the same range 
        selected_idxs = np.random.choice(n_samples, n_samples , replace=True) 

        return X[selected_idxs] , y[selected_idxs]