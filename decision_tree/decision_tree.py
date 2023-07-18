import numpy as np

from decision_tree.node import Node
from utils.Metric import Metric

from collections import Counter

def most_common_class(y):
    if(len(y) == 0): return 0 
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common

# this class is an implementation of the Decision Tree ML Algorithm
# the following article explains a key part of it: Entropy
# https://towardsdatascience.com/entropy-how-decision-trees-make-decisions-2946b9c18c8
class DecisionTree():

    def __init__(self,min_num_of_samples=5, max_depth=100, n_features=None):
        self.min_num_of_samples = min_num_of_samples
        self.max_depth = max_depth
        self.n_features = n_features

        # initialize root to None
        self.root = None

        # intialize object to get Entropy value
        self.entropy_calculator = Metric()

    def fit(self, X,y):
            # if not given in the constructor, initialize it  
        if(not self.n_features):
            self.n_features = X.shape[1]
        else:
            # else, take the minimum one
            self.n_features = min(self.n_features, X.shape[1])

        self.root = self._build_decision_tree(X,y)

    # recall that: at the end, prediction in Decision Trees are nothing but traversal to the tree
    # in a "Depth First" Manner.
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _build_decision_tree(self, X,y, depth=0):

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # step 1: have a base case ! 
        #        -> there is only one class in the given sample for a node
        #        -> there are less than the minimum number of samples in the node
        #        -> max depth reached
        # return the leaf node, (with its value being the most common value in that node sample)
        if( 
            n_samples < self.min_num_of_samples 
            or n_classes == 1 or depth >= self.max_depth
        ):
            leaf_value = most_common_class(y)
            return Node(value=leaf_value)

        # step 1 bis: find the best criteria to split the data
        # this involves looking for both: the best feature and the best threshold
        # e.g. we could say that the split condition at a particular node is (X_2 <= 40)
        # here: X_2 is the best split feature, and 40 is the best threshold corresponding to X_2    
        features_idxs = np.random.choice(n_features, self.n_features, replace=False) 

        best_feature, best_threshold = self._find_best_split_criteria(X,y, features_idxs)

        # step 2: split the given data to left and right children
        left_idxs, right_idxs = self._split(X[: , best_feature], best_threshold)

        # step 3: recursively construct the decision tree

        # in the left child, we only consider the data samples that meets the condition of the node
        # since those indexes are saved in left_idxs, we utilize them
        left = self._build_decision_tree(X[left_idxs, : ], y[left_idxs], depth+1)

        # similarly, in the right child, we only consider the data samples that do not meet the condition of the node
        right = self._build_decision_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    # utility function to find the best split criteria
    # i.e. to find the best feature and threshold
    def _find_best_split_criteria(self, X, y, features_idxs):
        best_gain = -1
        best_feature, best_threshold = None, None

        for features_idx in features_idxs:
            # first, we fix one feature (i.e. column)
            X_features_idx_col = X[: , features_idx]

            all_thresholds = np.unique(X_features_idx_col)

            for threshold in all_thresholds:
                gain = self._information_gain(y, X_features_idx_col, threshold)

                # if IG (Information Gain) returns a better IG, then update func return
                if gain > best_gain:
                    best_gain = gain
                    best_feature = features_idx
                    best_threshold = threshold

        return best_feature , best_threshold

    # utility function to calculate the information gain for a given feature
    # recall that the formula is as follows: 
    #     Entropy(Parent) - Entropy(child)
    # <=> Entropy(Parent) - 
    def _information_gain(self, y, X_selected, threshold):

        # first calculate the parent's entropy value
        parent_entropy = self.entropy_calculator.calculate_entropy(y)
        
        # then calculate the children's entropy value

        # to do so, we need to simulate a split 
        # i.e. a construction of two children: left and right
        left_idxs , right_idxs = self._split(X_selected, threshold)

        if len(left_idxs) == 0 or len(right_idxs):
            return 0

        total_num = len(y)
        
        left_child_num, right_child_num = len(left_idxs), len(right_idxs)

        left_child_entropy = self.entropy_calculator.calculate_entropy(y[left_idxs])
        right_child_entropy = self.entropy_calculator.calculate_entropy(y[right_idxs])

        # recall that the children's entropy is the weighted sum of: 
        # the frequency of (left / right) child data sample and 
        # the entropy of the corresponding child
        children_entropy = (1 / total_num) * ( (left_child_num * left_child_entropy) + (right_child_entropy * right_child_num) )

        return parent_entropy - children_entropy 

    def _split(self, X_selected, threshold):
        left_idxs = np.argwhere(X_selected <= threshold).flatten()
        rights_idxs = np.argwhere(X_selected > threshold).flatten()
        return left_idxs, rights_idxs


    def _traverse_tree(self, X, node):
        if node.is_leaf_node():
            return node.value

        if(X[node.feature] <= node.threshold):
            return self._traverse_tree(X, node.left)   

        return self._traverse_tree(X, node.right)
    
    