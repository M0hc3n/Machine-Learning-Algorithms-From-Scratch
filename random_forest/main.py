from sklearn import datasets
from sklearn.model_selection import train_test_split

from utils.Metric import Metric

from random_forest.random_forest import RandomForest

data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

model = RandomForest(number_of_trees=3, max_depth=10)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = Metric().basic_accuracy_score(y_test, y_pred)

print("Random Forest Accuracy is : ", acc)