from sklearn import datasets
from sklearn.model_selection import train_test_split

from utils.Metric import Metric

from decision_tree.decision_tree import DecisionTree

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

model = DecisionTree(max_depth=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = Metric().basic_accuracy_score(y_test, y_pred)

print("Decision Tree Accuracy is : ", acc)