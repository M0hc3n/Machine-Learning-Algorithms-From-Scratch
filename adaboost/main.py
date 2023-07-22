from sklearn import datasets
from sklearn.model_selection import train_test_split

from utils.Metric import Metric

from adaboost.adaboost import Adaboost

data = datasets.load_breast_cancer()
X, y = data.data, data.target

# re-format the y so that it is comform with Adaboost implementation
y[y == 0] = -1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

# Adaboost classification with 5 weak classifiers
# change the number of classifiers and notice the increasing accuracy
clf = Adaboost(n_clf=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = Metric().basic_accuracy_score(y_test, y_pred)
print("Accuracy Of Adaboost Model is:", acc)