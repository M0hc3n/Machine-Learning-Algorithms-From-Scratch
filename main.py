from sklearn.model_selection import train_test_split
from sklearn import datasets

# importing our model
from naive_bayes.naive_bayes import NaiveBayes 

# importing the Metric class
from utils.Metric import Metric

X, y = datasets.make_classification(
    n_samples=10000, n_features=10, n_classes=2, random_state=123
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

model = NaiveBayes()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

print("Naive Bayes classification accuracy", Metric().basic_accuracy_score(y_test, y_hat))