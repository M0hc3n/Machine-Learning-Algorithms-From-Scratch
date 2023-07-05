import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets

from utils.Metric import Metric
from utils.Score import Score

from linear_regression.linear_regression import LinearRegression 

X, y = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=4
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# initializing the model with appropriate parameters
model = LinearRegression(learning_rate=0.0001, n_iterations=100000)
# train it
model.fit(X_train, y_train)
# make predictions to test the model
predictions = model.predict(X_test)

# calculate the MSE
mse = Metric().mean_squared_error(y_test, predictions)
print("MSE:", mse)

# calculate Score value
score = Score().r2_score(y_test, predictions)
print("Accuracy:", score)

# try to plot the final prediction (of all dataset)
# and compare it with original plot of true y values
# to check whether our model is accurate enough
y_pred_line = model.predict(X)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
plt.show()