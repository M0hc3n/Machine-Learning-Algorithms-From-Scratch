
import numpy as np


# this class is an implementation of the Linear Discriminant Analysis technique
# this video helps to get the intuition behind it
# youtube.com/watch?v=azXCzI57Yfc
# Note that PCA makes it easier to understan LDA
class LDA:

    def __init__(self, n_components):
        self.n_components = n_components

        # initialize the main components
        self.linear_discriminants = None
    
    def fit(self, X,y):
        n_features = X.shape[1]
        class_labels = np.unique(y)
        
        mean_overall = np.mean(X, axis=0)

        # step 1: calculate S_w and S_b
        # recall: that S_w means the scatter within all classes
        # it is calculated as the sum of scatter within individual classes
        # AND, S_b means the scatter between all classes
        
        S_w = np.zeros((n_features, n_features))
        S_b = np.zeros((n_features, n_features))
        
        for class_label in class_labels:
            X_c = X[ y == class_label]
            mean_of_c = np.mean(X_c, axis=0)

            # recall: the formula of scatter within class c is:
            #         sum( (X_i - mean_of_c) * (X_i - mean_of_c)^T  )
            S_c = np.dot((X_c - mean_of_c).T, X_c - mean_of_c )

            S_w = S_w + S_c

            n_c = X_c.shape[0]
            mean_diff = (mean_of_c - mean_overall).reshape(n_features, 1) # sets the shape to (n_feature, 1)

            S_b = S_b + (n_c * np.dot(mean_diff.T, mean_diff))
        
        # step 2: calculate our final matrix: A = S_w^-1 * S_b
        A = np.dot(np.linalg.inv(S_w), S_b)

        # step 3: get the eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(A)

        # step 4: we sort the eigenvalues in descending order
        # this helps afterwards to select the k main components
        sorted_idx = np.argsort(abs(eigenvalues))[::-1]

        eigenvalues = eigenvalues[sorted_idx]
        # ofc, we need also to sort the eigenvectors accordingly to match the order of eigenvalues
        eigenvectors = eigenvectors[: ,sorted_idx]

        # now, we save the first k components
        self.linear_discriminants = eigenvectors[:, 0:self.n_components] 

    # recall that the transformation is projecting the dataset into the new components
    def transform(self, X):
        return np.dot(X, self.linear_discriminants)
