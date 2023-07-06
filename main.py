from sklearn.model_selection import train_test_split
from sklearn import datasets

import numpy as np

from utils.Metric import Metric

from logistic_regression.logistic_regression import LogisticRegression

# we will be testing our model on the well known breast cancer dataset
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# initialize the model with the given parameters
model = LogisticRegression(learning_rate=0.0001, n_iterations=1000)
model.fit(X_train, y_train)

# test out the model
predictions = model.predict(X_test)

print("LR classification accuracy:", Metric().basic_accuracy_score(y_test, predictions))