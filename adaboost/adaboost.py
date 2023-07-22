import numpy as np

from adaboost.decision_stump import DecisionStump

# an implementation of Adaboost algorithm
# this youtube video clearly explains the intuition and steps behind it
# https://www.youtube.com/watch?v=LsK-xG1cLYA
class Adaboost:

    def __init__(self, n_clf):
        # refers to the number of classifiers (i.e. number of weak learners)
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X,y):
        n_samples, n_features = X.shape

        # step 1: initialize the weights to 1/N for each sample
        w = np.full(n_samples, (1/n_samples))

        # step 2: we build n_clf classifiers 
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_err = float('inf')

            # step 2*: apply greedy search to find best feature and threshold
            #          as done previously with Decision Trees
            for feature in range(n_features):
                X_selected = X[: , feature]
                possible_thresholds = np.unique(X_selected)

                for threshold in possible_thresholds:
                    # default branching order (polarity) is 1
                    p = 1
                    y_hat = np.ones(n_samples)
                    y_hat[X_selected < threshold] = -1

                    # calculate the error
                    # recall that the error is the sum of the missclassifed samples weights
                    miss_classified = w[y != y_hat]
                    err = sum(miss_classified)

                    # a creative change: if the error is higher than half
                    # then, inverse the branching order. Thus, the error would be 
                    # 1 - the previous error
                    if err > 0.5:
                        err = 1 - err
                        p = -1
                    
                    # if the error is minimal, we update the model's params accordingly
                    if err < min_err:
                        min_err = err
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature = feature
            
            # calculate the amount of say, we use epsilon value to avoid division by 0
            EPSILON = 1e-10
            clf.amount_of_say = 0.5 * np.log((1 - min_err + EPSILON) / (min_err + EPSILON))

            # calculate the prediction to know the error
            y_hat = clf.predict(X)

            # update the weights
            # recall that the formula to update the weights is:
            # w = old_weight * (exp(-alpha * y_hat * y_true) / sum_of_old_weights)
            # note that the use of: y_hat * y is to detect whether the new weight
            # is going to increase, or to decrease
            # remember that y and y_hat always belong to {-1,1}
            updated_w = np.exp(-clf.amount_of_say * y * y_hat)
            updated_w = updated_w / np.sum(w)

            w = w * updated_w 

            self.clfs.append(clf)
    

    def predict(self, X_test):
        # recall that the prediction formula is:
        # sum(clf.amound_of_say * prediction_result)
        # we then apply the sign() method to output 1 for positive class and -1 for negative one
        clf_predictions = [clf.amount_of_say * clf.predict(X_test) for clf in self.clfs]

        y_hat = np.sum(clf_predictions, axis=0)
        y_hat = np.sign(y_hat)

        return y_hat
                    

