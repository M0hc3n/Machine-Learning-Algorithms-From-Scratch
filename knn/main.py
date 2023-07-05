from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn.knn import KNN
import numpy as np

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=1234
)

model = KNN(k=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("KNN classification accuracy : ", accuracy(y_test, predictions))