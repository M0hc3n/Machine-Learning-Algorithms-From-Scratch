from sklearn.datasets import make_blobs

import numpy as np

from kmeans.kmeans import KMeans

X, y = make_blobs(
    centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
)

clusters = len(np.unique(y))
print(clusters)

model = KMeans(K=clusters, max_iterations=150, live_plot=True)
y_pred = model.predict(X)

model.plot()