
import numpy as np

from decision_tree.decision_tree import DecisionTree, most_common_class
from utils.Sampler import Sampler

class RandomForest:

    def __init__(self,number_of_trees,min_num_of_samples=5, max_depth=100, n_features=None ):
        self.number_of_trees = number_of_trees

        self.min_num_of_samples = min_num_of_samples
        self.max_depth = max_depth
        self.n_features = n_features

        self.sampler = Sampler()

        self.trees = []

    def fit(self, X,y):
        # loop over all the given number of trees
        for _ in range(self.number_of_trees):
            tree = DecisionTree(min_num_of_samples=self.min_num_of_samples, max_depth=self.max_depth, n_features=self.n_features)
            
            # get a sample of the whole dataset
            sampled_X, sampled_y = self.sampler.bootstrap_sample(X,y)

            # fit the tree with the selected sample
            tree.fit(sampled_X, sampled_y)

            # append the tree to the forest
            self.trees.append(tree)
    
    def predict(self, X):
        # gather all the predictions of all the trees in
        trees_predictions = np.array([tree.predict(X) for tree in self.trees])

        # now, trees_predictions have the following format:
        # [
        #   [ X_1, X_2 , X_3 , ... , X_n] -> of tree_1 ,
        #   [ X_1, X_2 , X_3 , ... , X_n] -> of tree_2 ,
        #   .
        #   .
        #   .
        #   [ X_1, X_2 , X_3 , ... , X_n] -> of tree_n
        # ] 
        # BUT, we want to vote the best predicted value among the n X_1's and n X_2's and ... n X_n's
        # that's why, we need to swap the axis (i.e. column -> row )
        # the result would look like so:
        # [
        #   [ X_1, X_1 , X_1 , ... , X_1] -> predicted values of X_1 of all trees ,
        #   [ X_2, X_2 , X_2 , ... , X_2] -> predicted values of X_2 of all trees ,
        #   .
        #   .
        #   .
        #   [ X_n, X_n , X_n , ... , X_n] -> predicted values of X_n of all trees
        # ]  
        trees_predictions = np.swapaxes(trees_predictions, 0,1)
        
        # now, in each row, we need to find out the most likely prediction
        y_hat = [most_common_class(y) for y in trees_predictions]

        return np.array(y_hat)

