import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib import pyplot as plt

from linear_regression import LinearRegression

def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

X, y = datasets.make_regression(n_samples=100,
                                n_features=1,
                                noise=20,
                                random_state=4)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                        test_size=0.2,
                                        random_state=1234)

fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, color = 'b', marker = 'o', s = 30)
plt.show()

print(f'X_train.shape = {X_train.shape}')
print(f'y_train.shape = {y_train.shape}')

regressor = LinearRegression()
regressor.fit(X_train, y_train)
predicted_vals = regressor.predict(X_test)
mse_val = mse(y_test, predicted_vals)
print(mse_val)

y_pred_line = regressor.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color = cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
plt.show()