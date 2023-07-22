import numpy as np

# implementation of Decision Stump (a weak learner used in Adaboost)
class DecisionStump:

    def __init__(self):
        # refers to the branching order, by default we branch by <
        self.polarity = 1

        self.feature = None
        self.threshold = None
        
        # also referred to as alpha in the literature
        self.amount_of_say = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_targeted_column = X[: , self.feature]

        # by default, we fill the result by 1, then tweak it to -1 according to the polarity
        y_hat = np.ones(n_samples)

        # we can see the utility of branching order here:
        if self.polarity == 1:
            y_hat[X_targeted_column < self.threshold] = -1
        else:
            y_hat[X_targeted_column > self.threshold] = -1

        return y_hat