import numpy as np

# this is an implementation of the Principal Component Analysis techniques (aka PCA)
# this article helps to clarify the implementation steps:
# https://www.askpython.com/python/examples/principal-component-analysis
# for further readings, I recommend this article (lenghty and a bit heavy in terms of information, but worth it !)
# https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491

class PCA:

    def __init__(self, num_of_components=2):
        self.num_of_components = num_of_components
        
        self.components = None # stands for the Covariance Matrix eigenvectors
        self.mean = None # stands for X_bar

    def fit(self, X):
        # step 1: center the dataset
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # step 2: calculate the covariance matrix
        # note that the numpy cov function expects the features in rows
        # and the observations in the column (the inverse what we used to have)
        # that's why, we use the transpose
        cov_matrix = np.cov(X.T)

        # step 3: get the eigenvectors and eigenvalues
        # note the use of np.linalg.eigh, since the cov matrix is a symmetric one
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix) 

        # step 4: we sort the eigenvalues in descending order
        # this helps afterwards to select the k main components
        sorted_idx = np.argsort(eigenvalues)[::-1]

        print('index: ', sorted_idx)

        eigenvalues = eigenvalues[sorted_idx]
        # ofc, we need also to re-sort the eigenvectors accordingly to match the order of eigenvalues
        eigenvectors = eigenvectors[: ,sorted_idx]

        # now, we save the first k components
        self.components = eigenvectors[:, 0:self.num_of_components]

    # the following function would transform the given data to the new basis
    def transform(self, X):
        X = X - self.mean

        return np.dot(X, self.components)


