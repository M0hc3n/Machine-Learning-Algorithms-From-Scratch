
# this class is an implementation of single node of a Decision tree
class Node:

    # recall that a node of a decision tree needs to have the feature (i.e. which column it's imposing its condition on)
    # it needs to have also threshold values, (i.e. the bounding value of the condition)
    # it needs to keep the left and right nodes
    # it needs to have a value, if it doesn't have one, the it's not a decision node. Otherwise, it is.
    def __init__(self,feature=None ,threshold=None,left=None,right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

        self.value = value

    # every leaf node needs to have a value (that justifies the use of bare asterisks in the constructor)
    def is_leaf_node(self):
        return self.value is not None