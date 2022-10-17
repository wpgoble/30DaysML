import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn import KNN

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=1234)

print(f'X_train shape = {X_train.shape}')
print(f'{X_train[0]}')

print(f'y_train shape = {y_train.shape}')
print(f'{y_train}')

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = y, cmap=cmap, edgecolors='k', s=20)
plt.show()

clf = KNN(k = 5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

accuracy = np.sum(predictions == y_test) / len(y_test)
print(f'Accuracy = {accuracy}')
