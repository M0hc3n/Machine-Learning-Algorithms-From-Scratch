import numpy as np
import matplotlib.pyplot as plt

from utils.Distance import Distance

from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# this class is an implementation of the KMeans clustering algorithm
# the following article gives insights about this algorithm
# https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a
class KMeans:

    def __init__(self,K=5,max_iterations=1000, live_plot=False):
        self.K = K
        self.max_iterations = max_iterations
        self.live_plot = live_plot

        # here, we will save the label/class of each dataset sample
        self.clusters = [[] for _ in range(self.K)]

        # here, we will save the centroids of each label/class
        self.centroids = []

        self.distance_calculator = Distance()


    # since the KMeans algorithm is an unsupervised learning algorithm
    #  we won't need a "fit" function, rather, we will be using "predict" directly
    def predict(self, X):

        self.n_samples, self.n_features = X.shape
        
        # step 1: we standardize the data
        X_standardized = StandardScaler().fit_transform(X)
        self.X = X_standardized

        # step 2: we initialize the centroids (randomly from the dataset)
        random_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_idxs]

        # we optimize the result
        for _ in range(self.max_iterations):

            # step 3: we assign each dataset to the nearest centroid 
            self.clusters = self._update_clusters()

            if self.live_plot:
                self.plot()

            # step 4: update the centroids based on the new data/cluster assignment
            old_centroids = self.centroids
            self.centroids = self._update_centroids()

            if self.live_plot:
                self.plot()
            
            # check wether our clustering is improving, or has stabilized in a local optimum
            if self.has_converged(old_centroids):
                break
        
        return self.get_clustering_result()

    # a utility method to update the clusters based on the new centroids
    def _update_clusters(self):

        clusters = [[] for _ in range(self.K)]

        for idx, sample in enumerate(self.X):
            # get the closes centroid index, then append it to the corresponding cluster 
            nearest_centroid_idx = self._get_closest_centroid(sample)
            clusters[nearest_centroid_idx].append(idx) 
        
        return clusters

    def _get_closest_centroid(self, sample):
        # return the centroid with the smallest distance to the sample data
        distances = [self.distance_calculator.calculate_euclidean_distance(sample, centroid) for centroid in self.centroids]

        return np.argmin(distances)

    # a utility function to update the centroids, it would set them to the mean of each cluster
    def _update_centroids(self):

        centroids = np.zeros((self.K, self.n_features))

        for cluster_idx, cluster in enumerate(self.clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)

            # set the centroid to the cluster mean
            centroids[cluster_idx] = cluster_mean

        return centroids

    # a utility function to check whether we've reached a local optimum or not
    def has_converged(self, old_centroids):
        distances = [self.distance_calculator.calculate_euclidean_distance(self.centroids[j], old_centroids[j]) for j in range(self.K)]

        return sum(distances) == 0 

    # a utility function to return the final labels of each sample
    def get_clustering_result(self):
        labels = np.zeros(self.n_samples)

        for cluster_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        
        return labels

    # a utility function to plot the clustering result at each step
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()
